import os
import threading
import queue
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
from utils.encoding_utils import Encoder
from tiling.EncodingDatasets import TilePreprocessing_nosaving

# collate FN
def collate_fn(batch):
    batch = [item for item in batch if not (item[0][0] == -1 and item[0][1] == -1)]
    if not batch:
        return None

    coords = torch.stack([item[0] for item in batch])
    images = torch.stack([item[1] for item in batch])
    vars = torch.stack([item[2] for item in batch])
    x = coords[:, 0]
    y = coords[:, 1]
    return x, y, images, vars

class H5Writer:
    def __init__(self, output_path, high_qual=False):
        self.output_path = output_path
        self.high_qual = high_qual
        self.queue = queue.Queue(maxsize=64)
        self.thread = threading.Thread(target=self._write_worker)
        self.thread.daemon = True
        self.thread.start()

    def _write_worker(self):
        with h5py.File(self.output_path, 'w') as hdf:
            while True:
                task = self.queue.get()
                if task is None:
                    break

                key, data = task
                if key == 'finalize':
                    for k, v in data.items():
                        hdf.create_dataset(k, data=v, compression='gzip')
                else:
                    if key not in hdf:
                        maxshape = (None,) + data.shape[1:]
                        hdf.create_dataset(key, data=data, maxshape=maxshape,
                                           compression='gzip', chunks=True)
                    else:
                        hdf[key].resize((hdf[key].shape[0] + data.shape[0]), axis=0)
                        hdf[key][-data.shape[0]:] = data
                self.queue.task_done()

    def add_data(self, key, data):
        self.queue.put((key, data))

    def finalize(self, final_data):
        self.queue.put(('finalize', final_data))
        self.queue.put(None)  # Signal end of processing
        self.thread.join()

def encode_tiles(patient_id, tile_dataset, result_path, device=None, batch_size=512,
                 encoder_model="resnet50", high_qual=False, number_of_augmentations=0, token=None):
    # Setup
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Prepare output
    os.makedirs(result_path, exist_ok=True)
    output_path = os.path.join(result_path, f"{patient_id}.h5")
    writer = H5Writer(output_path, high_qual)

    # Model loading
    encoder, _ = Encoder(device, encoder_model, token).get_model_and_transform()
    encoder.eval()

    # Adjust batch size for augmentations
    if number_of_augmentations > 0:
        batch_size = max(1, math.ceil(batch_size / (number_of_augmentations + 1)))

    # DataLoader setup
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': os.cpu_count(),
        'pin_memory': str(device) != "cpu",
        'persistent_workers': True
    }

    if isinstance(tile_dataset, TilePreprocessing_nosaving):
        loader_kwargs['collate_fn'] = collate_fn

    data_loader = DataLoader(tile_dataset, **loader_kwargs)

    all_tile_paths = []

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc=f"Encoding {patient_id}"):
            if batch is None:
                continue

            x, y, images, tile_paths = batch
            all_tile_paths.extend(tile_paths)

            # Process main features
            images = images.to(device, non_blocking=True)

            if images.ndim == 4:  # No augmentations
                features = encoder(images).flatten(start_dim=1).cpu().numpy()
                writer.add_data('features', features)
            else:  # With augmentations
                batch_size, num_versions = images.shape[:2]
                features = encoder(images.flatten(0, 1)).flatten(start_dim=1)
                features = features.view(batch_size, num_versions, -1).cpu().numpy()
                writer.add_data('features', features)

            # Write coordinates
            writer.add_data('x', x.numpy())
            writer.add_data('y', y.numpy())

            # Process high quality if needed
            if high_qual:
                encoder_hq = torch.nn.Sequential(*list(encoder.children())[:-1])
                with torch.no_grad():
                    hq_features = encoder_hq(images).flatten(start_dim=1).cpu().numpy()
                    writer.add_data('high_quality', hq_features)

    # Finalize with paths and cleanup
    writer.finalize({
        'tile_path': np.array(all_tile_paths, dtype='S')
    })

    del encoder, data_loader
    if high_qual:
        del encoder_hq
    torch.cuda.empty_cache()