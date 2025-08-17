import torch
torch.backends.cudnn.benchmark = True
from tiling.EncodingDatasets import TileEncoding_h5, TilePreprocessing_png
from utils.encoding_utils import Encoder
from functools import partial
from torch.utils.data import DataLoader
import h5py
import math
import tqdm
import os
import threading
import queue
import numpy as np
import multiprocessing
from tiling.no_saving.cpu import CPUTileDataset
from concurrent.futures import ThreadPoolExecutor

class H5Writer:
    def __init__(self, output_path, high_qual=False):
        self.output_path = output_path
        self.high_qual = high_qual
        self.queue = queue.Queue(maxsize=64)
        self.thread = threading.Thread(target=self._write_worker)
        self.stop_event = threading.Event()
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
        self.queue.join()
        self.queue.put(None)
        self.thread.join()
        self.stop_event.set()

class SlideEncoding:
    def __init__(self, config, pipeline_steps):
        self.config = config
        self.pipeline_steps = pipeline_steps
        self.device = self.config.get("device")
        self.batch_size = self.config.get("batch_size")

        self._validate_config()
        self._set_up_pipeline()
    def _validate_config(self):
       if not self.config.get("feature_extractor") in["mahmood-uni", "resnet50", "resnet50-truncated","empty"]:
           raise ValueError("Currently supported feature extractors are: "
                            "'mahmood-uni', 'resnet50', 'resnet50-truncated', 'empty'"
                            "Resnet50 outputs a 2048 feature vector, "
                            "the truncated version outputs a 1024 version")

    def _set_up_pipeline(self):
        # edit batch size to augmentations to not overload encoder
        if self.config.get("augmentations") > 0:
            self.batch_size = max(1, math.ceil(self.batch_size / (self.config.get("augmentations") + 1)))
        self.loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': os.cpu_count(),
            'pin_memory': str(self.device) != "cpu",
            'persistent_workers': True
        }
        # set up model
        self.encoder, self.transforms = Encoder(self.config.get("device"), self.config.get("feature_extractor"),
                                                      self.config.get("token")).get_model_and_transform()
        self.encoder.to(self.config.get("device"))
        self.encoder.eval()
        if self.config.get("output_format") is not None:
            self.target = self._encode_presaved
            dataset_cls = {
                "h5": TileEncoding_h5,
                "png": TilePreprocessing_png
            }.get(self.config["output_format"], None)

            if dataset_cls:
                self.dataset = partial(
                    dataset_cls,
                    model_transforms=self.transforms,
                    num_augmentations=self.config.get("augmentations")
                )
        else:
            if self.config.get("device") == "cpu":
                self.target = self._encode_nosaving_cpu
                self.dataset = partial(
                    CPUTileDataset,
                    pipeline_steps=self.config.get("pipeline_steps"),
                    num_augmentations=self.config.get("augmentations"),
                    transforms=self.transforms,
                    device=self.config.get("device", "cpu")
                )
            else:
                self.target = self._encode_no_saving_gpu
                self.dataset = partial(
                    CPUTileDataset,
                    pipeline_steps=[],
                    num_augmentations=self.config.get("augmentations"),
                    transforms=None,
                    device=self.config.get("device", "cpu"))

    def __call__(self,slide, output_path, coords= None, mask= None, adjusted_size= None, desired_size= None):
        self.target(slide,output_path,coords, mask, adjusted_size, desired_size)

    def _encode_presaved(self,input_path, output_path, coords= None, mask= None, adjusted_size= None, desired_size= None):
        writer = H5Writer(output_path)
        dataloader = DataLoader(self.dataset(input_path), **self.loader_kwargs)
        all_tile_paths =[]
        with torch.inference_mode():
            for batch in dataloader:
                if batch is None:
                    continue
                coords,images, tile_paths = batch
                images = images.to(self.device, non_blocking = True)
                all_tile_paths.extend(tile_paths)
                if images.ndim == 4:  # no augmentations
                    features = self.encoder(images).flatten(start_dim=1).cpu().numpy()
                    writer.add_data('features', features)
                else:  # with augmentations
                    batch_size, num_versions = images.shape[:2]
                    features = self.encoder(images.flatten(0, 1)).flatten(start_dim=1)
                    features = features.view(batch_size, num_versions, -1).cpu().numpy()
                    writer.add_data('features', features)
                # write coordinates
                writer.add_data('coords', coords)
                del coords, images, tile_paths
        writer.finalize({
            'tile_path': np.array(all_tile_paths, dtype='S')
        })
        del writer
        torch.cuda.empty_cache()
        del dataloader

    def _encode_nosaving_cpu(self, slide, output_path, coords, mask, adjusted_size, desired_size):
        writer = H5Writer(output_path)
        dataset = CPUTileDataset(
            slide=slide,
            coordinates=coords,
            adjusted_size=adjusted_size,
            desired_size=desired_size,
            pipeline_steps=self.pipeline_steps,
            transforms=self.transforms,
        )

        def collate_fn(batch):
            valid_batch = [item for item in batch if item[0] is not None]
            if not valid_batch:
                return None, None
            images, coords = zip(*valid_batch)
            return torch.stack(images, dim=0), torch.stack(coords, dim=0)

        loader_kwargs = {
            **self.loader_kwargs,
            'collate_fn': collate_fn,
            'persistent_workers': False,
            'pin_memory': False
        }

        for images, coords in DataLoader(dataset, **loader_kwargs):
            if images is None:
                continue
            with torch.inference_mode():
                features = self.encoder(images).flatten(1).cpu().numpy()
                writer.add_data('features', features)
                writer.add_data('coords', coords.cpu().numpy())

        # Finalize so data is actually written and file is closed
        writer.finalize({})
        torch.cuda.empty_cache()



    def _encode_no_saving_gpu(self, slide,output_path,coords, mask, adjusted_size, desired_size):
        dataset = CPUTileDataset(
            slide=slide,
            coordinates=coords,
            adjusted_size=adjusted_size,
            desired_size=desired_size,
            pipeline_steps=[],
            transforms=None,
        )
        dataloader = DataLoader(dataset, **self.loader_kwargs)
        vars_dict =  multiprocessing.Manager().dict()
        writer = H5Writer(output_path)
        for batch in dataloader:
            # load images and preprocess them
            batch_tiles, coord_batch = zip(*[(t, c) for t, c in batch if t is not None])
            if len(batch_tiles) == 0:
                continue
            batch_tiles = torch.stack(batch_tiles).to(self.device, non_blocking=True)
            coord_batch = torch.stack(coord_batch).to(self.device, non_blocking=True)


            for step in self.pipeline_steps:
                batch_tiles, coord_batch = step(batch_tiles, vars_dict, coord_batch)
                if batch_tiles.numel() == 0:
                    break
            if batch_tiles.numel() == 0:
                continue

            # apply transforms
            images = self.transforms(batch_tiles).to(self.device, non_blocking=True)
            # send to model
            with torch.inference_mode():
                if images.ndim == 4:  # No augmentations
                    features = self.encoder(images).flatten(start_dim=1).cpu().numpy()
                    writer.add_data('features', features)
                else:  # With augmentations
                    batch_size, num_versions = images.shape[:2]
                    features = self.encoder(images.flatten(0, 1)).flatten(start_dim=1)
                    features = features.view(batch_size, num_versions, -1).cpu().numpy()
                    writer.add_data('features', features.cpu().numpy())
                writer.add_data('coords', coord_batch.cpu().numpy())
            del coord_batch, images,batch_tiles
        torch.cuda.empty_cache()
        del dataloader
        return vars_dict










