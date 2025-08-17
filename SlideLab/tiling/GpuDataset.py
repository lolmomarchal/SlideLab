from openslide import OpenSlide
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import numpy as np
import cv2
from torch.utils.data import DataLoader, DataSet
import torch
import time
class GpuIterator:
    def __init__(self, slide, coordinates, adjusted_size, desired_size,batch_size = 32, prefetch = 2):
        self.slide = slide
        self.coordinates = coordinates
        self.adjusted_size = adjusted_size
        self.desired_size = desired_size
        self.batch_size = batch_size
        self.prefetch = prefetch
        # async loading
        self.executor = ThreadPoolExecutor(max_workers = 4)
        self.queue = queue.Queue(maxsize=prefetch)
        self.stop_event = threading.Event()
        # start prefetching
        self.prefetch_thread = threading.Thread(target= self._prefetch_batches)
        self.prefetch_thread.start()
    def _prefetch_batches(self):
        idx = 0
        while not self.stop_event_is_set() and idx<len(self.coordinates):
            end_idx = min(idx+ self.batch_size, len(self.coordinates))
            batch_coords = self.coordinates[idx:end_idx]
            idx = end_idx
            future = self.executor.submit(self._load_batch, batch_coords)
            self.queue.put(future)
    def _load_batch(self, batch_coords):
        batch_tiles = []
        for coord in batch_coords:
            tile = np.array(self.slide.read_region(coord, 0, self.adjusted_size).convert("RGB"))
            if self.adjusted_size[0] != self.desired_size:
                # inter area best for shrinking
                tile = cv2.resize(tile, (self.desired_size, self.desired_size), interpolation = cv2.INTER_AREA)
            batch_tiles.append(tile)
            with torch.cuda.stream(torch.cuda.Stream()):
                batch_tiles = torch.from_numpy(np.stack(batch_tiles))
                batch_tiles= batch_tiles.to("cuda", non_blocking = True)
                batch_coords = torch.tensor(batch_coords, device = "cuda")
            return batch_tiles, batch_coords
    def __iter__(self):
        return self
    def __next__(self):
        try:
            future = self.queue.get_nowait()
            return future.result()
        except queue.Empty:
            if not self.prefetch_thread.is_alive():
                raise StopIteration
            time.sleep(0.01)
            return self.__next__()
    def __del__(self):
        self.stop_event.set()
        if hasattr(self, "prefetch_thread"):
            self.prefetch_thread.join()
            self.slide.close()






