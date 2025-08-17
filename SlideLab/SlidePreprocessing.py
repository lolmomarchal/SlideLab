# external imports
import os
import time
import multiprocessing
import threading
import pandas as pd
import numpy as np
import openslide
import tqdm
from PIL import Image
import csv
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import random
from torch.cuda import is_available, get_device_name
# internal imports
from TissueMask import TissueMask
from utils.VisualizationUtils import reconstruct_slide
from utils.preprocessing_utils import *
import Reports
from config import get_args_from_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
SlidePreprocessing.py

Author: Lorenzo Olmo Marchal
Created: 3/5/2024
Last Updated: 8/9/2025

Description:
This script automates the preprocessing and normalization of Whole Slide Images (WSI) in digital histopathology. 
Input:
- slide directory path or slide path
- slide directory output path

Output:
Processed tiles are saved in the output directory. Each tile is accompanied by metadata in a csv, including its origin
within the WSI file. 

"""


class SlidePreprocessing:
    # ================================ INIT & VALIDATION ==================================
    def __init__(self, config):
        self.config = config
        self._validate_config()
        self._setup_pipeline()

    def _validate_config(self):
        """Raises error if configuration parameters are incorrect"""

        # SIZING
        if self.config.get("desired_magnification") is None and self.config.get("desired_mpp") is None:
            raise ValueError("Either a desired_magnification or desired_mpp must be specified")

        if not self.config.get("sizing_scale") in ["magnification", "mpp"]:
            raise ValueError("Currently supported sizing_scale are: 'magnification' or 'mpp'")
        if self.config.get("desired_magnification") is None:
            self.config["desired_magnification"] = mpp_to_mag(self.config.get("desired_mpp"))
        if self.config.get("desired_mpp") is None:
            self.config["desired_mpp"] = mag_to_mpp(self.config.get("desired_magnification"))

        # OUTPUT FORMAT
        if not self.config.get("output_format") in ["png", "h5", None]:
            raise ValueError(
                "Currently supported output_format are: 'png', 'h5', or 'None' (no saving, just feature extraction)")

        # DEVICES
        # double check available devices
        requested_device = self.config.get("device")
        if requested_device is None:
            if is_available():
                self.config["device"] = "cuda"
                print(f"No device specified. CUDA is available. Using {get_device_name(0)}.")
            else:
                self.config["device"] = "cpu"
                print("No device specified. CUDA is not available. Using CPU.")
        elif "cuda" in str(requested_device).lower():
            if is_available():
                self.config["device"] = "cuda"
                print(f"CUDA requested and available. Using {get_device_name(0)}.")
            else:
                self.config["device"] = "cpu"
                print(
                    f"CUDA was requested ('{requested_device}') but is not available. Switching to CPU. Please check CUDA configurations.")
        else:
            self.config["device"] = "cpu"
            print(f"Device explicitly set to '{requested_device}'. Using CPU.")
        # OVERLAP
        if self.config.get("overlap") < 1:
            raise ValueError(
                f"Overlap must be equal or greater than 1, currently it is set up as {self.config.get("overlap")}")

        # NORMALIZATION
        if not self.config.get("normalize_staining") in ["macenko", "reinhard", "stainnet", None]:
            raise ValueError(
                "Currently supported normalize_staining methods are: 'macenko', 'reinhard', 'stainnet', or None for no normalization")
        if self.config.get("normalize_staining") == "stainnet" and self.config.get("device") == "cpu":
            print(
                f"Please be aware that StainNet is not recommended for your current device (CPU) as it is designed for GPU usage. Computational efficiency may drop as a result.")
        if self.config.get("normalize_staining") == "stainnet":
            set_up_stainnet()

    def _setup_pipeline(self):
        """Set up processing pipeline based on configuration"""
        # Constants: steps, device, etc..
        self.output_format = self.config.get("output_format")
        self.normalize_staining = self.config.get("normalize_staining", None)
        self.remove_blurry_tiles = self.config.get("remove_blurry_tiles", False)
        self.annotation_directory = self.config.get("annotation_directory", None)
        self.annotation_only = self.config.get("annotation_only", False)
        self.device = self.config.get("device")
        self.sizing_scale = self.config.get("sizing_scale")
        self.max_workers = self.config.get("cpu_processes")

        # SIZING SCALE
        self.convert_sizing = {"mpp": get_best_size_mpp, "magnification": get_best_size_mag}[self.sizing_scale]

        # NORMALIZATION METHOD
        # this way we can avoid importing all of torch
        if self.normalize_staining is not None:
            if self.device == "cpu":
                if self.normalize_staining == "macenko":
                    from normalization.cpu_normalizers.macenko import macenkoNormalizer_cpu
                    self.normalization_method = macenkoNormalizer_cpu
                elif self.normalize_staining == "reinhard":
                    from normalization.cpu_normalizers.reinhard import reinhardNormalizer
                    self.normalization_method = reinhardNormalizer
                # elif self.normalize_staining == "stainnet":
                #     from normalization.gpu_normalizers i
            else:
                if self.normalize_staining == "macenko":
                    from normalization.gpu_normalizers.macenko import macenkoNormalizer
                    self.normalization_method = macenkoNormalizer
                elif self.normalize_staining == "reinhard":
                    from normalization.gpu_normalizers.reinhard import reinhardNormalizer
                    self.normalization_method = reinhardNormalizer
        # BLUR METHOD
        if self.remove_blurry_tiles:
            if self.device == "cpu":
                from tileQuality.filter_cpu import LaplaceFilter
                self.blur_method = LaplaceFilter
            else:
                from tileQuality.filter_gpu import LaplaceFilter
                self.blur_method = LaplaceFilter
        else:
            self.remove_blurry_tiles = None

        # SET UP STEPS
        self.pipeline_steps = []
        if self.device == "cpu":
            if self.remove_blurry_tiles:
                # fix params
                self.pipeline_steps.append(partial(apply_laplace_filter, blur_method=self.blur_method,
                                                   blur_threshold=self.config.get("blur_threshold")))

            if self.normalize_staining:
                self.pipeline_steps.append(
                    partial(apply_stain_normalization, normalize_staining_func=self.normalization_method))
        else:
            if self.remove_blurry_tiles:
                self.pipeline_steps.append(partial(apply_laplace_filter_gpu, blur_method=self.blur_method,
                                                   blur_threshold=self.config.get("blur_threshold")))

            if self.normalize_staining:
                self.pipeline_steps.append(
                    partial(apply_stain_normalization_gpu, normalize_staining_func=self.normalization_method))

        if self.device == "cpu":
            from tiling.TileIterator import TileIterator
            self.iterator = TileIterator
            if self.output_format == "h5":
                self.target_method = self._process_tiles_cpu_h5
            elif self.output_format == "png":
                self.target_method = self._process_tiles_cpu_png
            else:
                self.target_method = None
        else:
            from tiling.TileDataset import TileDataset
            self.iterator = TileDataset
            if self.output_format == "h5":
                self.target_method = self._process_tiles_gpu_h5
            else:
                self.target_method = self._process_tiles_gpu_png

    def _create_summary(self, patient_id, path, total_tiles, valid_tiles, blurry_tiles, time_stats, status):
        """Create summary statistics for the processing."""

        return [
            patient_id, path, total_tiles, valid_tiles, blurry_tiles,
            time_stats['mask_cpu'], time_stats['coordinates_cpu'], time_stats['patches_cpu'],
            time_stats['mask'], time_stats['coordinates'], time_stats['patches'],
            status
        ]

    # ================================== PROCESSING METHODS ===================================
    def __call__(self, slide_path, patient_id, output_path):
        error = []
        summary = []
        vars = []
        time_stats = {
            'mask': -1,
            'mask_cpu': -1,
            'coordinates': -1,
            'coordinates_cpu': -1,
            'patches': -1,
            'patches_cpu': -1
        }

        # ========================= SLIDE INITIATION & CHECKS =================================
        try:
            slide = openslide.OpenSlide(slide_path)
        except Exception as e:
            error.append((patient_id, slide_path, "Error Opening file", "Slide Opening"))
            return self._create_summary(patient_id, slide_path, -1, -1, -1, time_stats, "Error"), error

        # get slide properties
        mpp_x, mpp_y, width, height, magnification, closest_mpp = get_attributes(slide)

        # check 1: not missing any important properties
        missing_props = []
        if closest_mpp is None and magnification is None:
            if self.config.get("set_standard_magnification") is not None:
                magnification = self.config.get("set_standard_magnification")
                mpp_x = mpp_y = mag_to_mpp(magnification)
            elif self.config.get("set_standard_mpp") is not None:
                mpp_x = mpp_y = self.config.get("set_standard_mpp")
                magnification = mpp_to_mag(mpp_x)
            else:
                missing_props.append("resolution")

        if width is None or height is None:
            missing_props.append("dimensions")

        if missing_props:
            error.append((patient_id, slide_path,
                          f"Missing required properties: {', '.join(missing_props)}. If only resolution is missing,you can use either --set_standard_magnification or --set_standard_mpp to process this slide.",
                          "Property Check"))
            return self._create_summary(patient_id, slide_path, -1, -1, -1, time_stats, "Error"), error

        # check 2: will not be upsampling
        if magnification < self.config.get("desired_magnification"):
            error.append((patient_id, slide_path,
                          "Desired resolution is higher than highest available resolution. To avoid upsampling and interpolation artifacts, this slide will be skipped.",
                          "Resolution Sanity Check"))
            return self._create_summary(patient_id, slide_path, -1, -1, -1, time_stats, "Error"), error

        # =============================== TISSUE MASK & CANDIDATE PATCHES ==========================

        # create results directory
        sample_path = os.path.join(output_path, patient_id)
        os.makedirs(sample_path, exist_ok=True)

        # get tissue Mask
        mask, scale, mask_, mask_time = self._create_tissue_mask(slide, sample_path)
        time_stats['mask'] = mask_time['wall']
        time_stats['mask_cpu'] = mask_time['cpu']
        if mask is None:
            error.append((patient_id, slide_path, "Failed to create tissue mask", "Tissue Mask"))
            return self._create_summary(patient_id, slide_path, -1, -1, -1, time_stats, "Error"), error

        # translate sizes -> return size tuple
        adjusted_size = self.convert_sizing(self.config.get("desired_size"), self.config.get("desired_mpp"),
                                            self.config.get("desired_magnification"),
                                            mpp_x, mpp_y, magnification)
        adjusted_size = tuple(map(int, adjusted_size))
        # get candidate patches
        coords_data, coord_time = self._get_valid_coordinates((width, height), adjusted_size, mask, scale)
        del mask

        time_stats['coordinates'] = coord_time['wall']
        time_stats['coordinates_cpu'] = coord_time['cpu']
        # where to return if outpt is none
        if self.config.get("output_format") is None:
            summary = self._create_summary(
                patient_id, slide_path,
                coords_data['total_tiles'], coords_data['valid_tiles'],
                -1, time_stats, "Processed")
            return slide, coords_data["valid_coordinates"], mask_, magnification, adjusted_size, summary, error

        tile_iterator = self.iterator(
            slide, coordinates=coords_data["valid_coordinates"], mask=mask_, normalizer=None,
            size=self.config.get("desired_size"), magnification=self.config.get("desired_magnification"),
            adjusted_size=adjusted_size, overlap=self.config.get('overlap')
        )

        tiling_information, tiling_times = self.target_method(tile_iterator, patient_id, sample_path, magnification,
                                                              scale)
        time_stats['patches'] = tiling_times['wall']
        time_stats['patches_cpu'] = tiling_times['cpu']
        self._post_processing(slide, patient_id, sample_path, mask_, coords_data["all_coords"],
                              coords_data["valid_coordinates"], adjusted_size[0], tiling_information["tiles_df"],
                              tiling_information["vars"].values())

        slide.close()
        return (
            self._create_summary(
                patient_id, slide_path,
                coords_data['total_tiles'], coords_data['valid_tiles'],
                tiling_information['blurry_tiles'], time_stats, "Processed"
            ),
            error
        )

    # ================================ IN-CLASS HELPER METHODS =========================================

    def _create_tissue_mask(self, slide, sample_path):
        start_mask_user = time.time()
        start_mask_cpu = time.process_time()

        try:
            mask_ = TissueMask(
                slide,
                result_path=sample_path,
                blue_pen_thresh=self.config.get('blue_pen_check', 0.4),
                red_pen_thresh=self.config.get('red_pen_check', 0.4),
                SCALE=self.config.get('mask_scale'),
                remove_folds=self.config.get('remove_folds', False)
            )
            mask, scale = mask_.get_mask_attributes()
            return mask, scale, mask_, {
                'wall': time.time() - start_mask_user,
                'cpu': time.process_time() - start_mask_cpu
            }
        except Exception as e:
            print(e)
            return None, None, {
                'wall': time.time() - start_mask_user,
                'cpu': time.process_time() - start_mask_cpu
            }

    def _get_valid_coordinates(self, dimensions, adjusted_size, mask, scale):
        """Calculate valid coordinates and return with timing information."""
        start_time_coordinates_user = time.time()
        start_time_coordinates_cpu = time.process_time()
        w, h = dimensions
        all_coords, valid_coordinates, coordinates, all_coords_ = get_valid_coordinates(w, h,
                                                                                        self.config.get('overlap', 1),
                                                                                        mask,
                                                                                        adjusted_size, scale,
                                                                                        threshold=self.config.get(
                                                                                            'tissue_threshold', 0.7))
        return {
            'total_tiles': all_coords,
            'valid_tiles': valid_coordinates,
            'valid_coordinates': coordinates,
            'all_coords': all_coords_,

        }, {
            'wall': time.time() - start_time_coordinates_user,
            'cpu': time.process_time() - start_time_coordinates_cpu
        }

    def _process_tiles_cpu_h5(self, tile_iterator, patient_id, sample_path, natural_magnification, scale):
        h5_file = os.path.join(sample_path, patient_id + ".h5")
        # multiprocessing
        manager = multiprocessing.Manager()
        var_list = manager.dict()
        results_list = manager.list()
        index_queue = multiprocessing.Queue()
        save_queue = multiprocessing.Queue()

        patching_user = time.time()
        patching_cpu = time.process_time()

        processing_workers = max(1, self.max_workers - 1)
        processing_threads = []
        for i in range(processing_workers):
            p = multiprocessing.Process(
                target=worker_h5,
                args=(index_queue, tile_iterator, patient_id, sample_path, results_list, var_list, self.pipeline_steps,
                      save_queue
                      )
            )
            p.start()
            processing_threads.append(p)

        saving_thread = threading.Thread(target=save_h5, args=(save_queue, h5_file))
        saving_thread.start()

        for idx in range(len(tile_iterator)):
            index_queue.put(idx)
        for _ in range(processing_workers):
            index_queue.put(None)
        for process in processing_threads:
            process.join()

        save_queue.put(None)
        saving_thread.join()

        del save_queue
        del index_queue

        results = list(results_list)
        results_ = [result for result in results if result]
        df_tiles = pd.DataFrame(results_)
        df_tiles["original_mag"] = natural_magnification
        df_tiles["scale"] = scale
        df_tiles.to_csv(os.path.join(sample_path, patient_id + ".csv"), index=False)
        blurry_tiles = len(results_) if self.remove_blurry_tiles else None
        return {
            'tiles_df': df_tiles,
            'vars': var_list,
            'blurry_tiles': blurry_tiles
        }, {

            'wall': time.time() - patching_user,
            'cpu': time.process_time() - patching_cpu
        }

    def _process_tiles_cpu_png(self, tile_iterator, patient_id, sample_path, natural_magnification, scale):
        patching_user = time.time()
        patching_cpu = time.process_time()

        manager = multiprocessing.Manager()
        results = manager.list()
        vars_dict = manager.dict()
        queue = multiprocessing.Queue()
        self.max_workers = os.cpu_count() - 2
        tiles_path = os.path.join(sample_path, "tiles")
        os.makedirs(tiles_path, exist_ok=True)

        threads = []
        for _ in range(self.max_workers):
            thread = multiprocessing.Process(
                target=worker_png,
                args=(
                    queue, tile_iterator, patient_id, tiles_path, results, vars_dict, self.pipeline_steps
                )
            )
            thread.start()
            threads.append(thread)

        for idx in range(len(tile_iterator)):
            queue.put(idx)
        for _ in range(self.max_workers):
            queue.put(None)
        for thread in threads:
            thread.join()

        results = list(results)
        vars_df = pd.DataFrame(vars_dict)
        # print(vars_dict)
        results_ = [result for result in results if result]
        df_tiles = pd.DataFrame(results_)
        df_tiles["original_mag"] = natural_magnification
        df_tiles["scale"] = scale
        # df_tiles
        # df_tiles["laplacian_variance"]
        df_tiles.to_csv(os.path.join(sample_path, patient_id + ".csv"), index=False)
        blurry_tiles = len(results_) if self.remove_blurry_tiles else None
        # self._plot_laplacian_variance(np.array(vars_dict.values()), os.path.join(sample_path, "LaplacianVar.png"), self.config.get("blur_threshold"))
        return {
            'tiles_df': df_tiles,
            'vars': vars_dict,
            'blurry_tiles': blurry_tiles
        }, {

            'wall': time.time() - patching_user,
            'cpu': time.process_time() - patching_cpu
        }

    def _process_tiles_gpu_png(self, tile_iterator, patient_id, sample_path, natural_magnification, scale):
        from torch.utils.data import DataLoader
        from torch.cuda import empty_cache

        patching_user = time.time()
        patching_cpu = time.process_time()
        # setting up
        load_workers = 4
        saving_workers = self.max_workers - load_workers
        dataloader = DataLoader(tile_iterator, num_workers=load_workers,
                                batch_size=self.config.get("batch_size"), pin_memory=True, shuffle=False)

        # multiprocessing portions
        manager = multiprocessing.Manager()
        results = []
        vars_dict = manager.dict()
        save_queue = multiprocessing.Queue()
        # set up for saving
        tiles_path = os.path.join(sample_path, "tiles")
        os.makedirs(tiles_path, exist_ok=True)
        # start saver_threads, so they start saving the second we put something in the save_queue
        saver_threads = []
        for _ in range(saving_workers):
            thread = multiprocessing.Process(args=(save_queue,), target=worker_png_gpu)
            thread.start()
            saver_threads.append(thread)
        for batch in dataloader:
            batch_tiles, coord_batch = batch
            batch_tiles = batch_tiles.to(self.device, non_blocking = True)
            coord_batch = coord_batch.to(self.device, non_blocking = True)
            for step in self.pipeline_steps:
                batch_tiles, coord_batch = step(batch_tiles, vars_dict, coord_batch)
                if batch_tiles.numel() == 0:
                    break
            batch_tiles = batch_tiles.detach().cpu().numpy()
            coord_batch = coord_batch.detach().cpu().numpy()
            for tile, coord in zip(batch_tiles, coord_batch):
                image_path = os.path.join(tiles_path,
                                          f"{patient_id}_{coord[0]}_{coord[1]}_size_{tile_iterator.size}_mag_{tile_iterator.magnification}.png")
                save_queue.put((tile, image_path))
                results.append({
                    "Patient_ID": patient_id, "x": coord[0], "y": coord[1],
                    "tile_path": image_path, "original_size": tile_iterator.adjusted_size,
                    "desired_size": tile_iterator.size, "desired_magnification": tile_iterator.magnification
                })
            del batch_tiles,coord_batch
        for _ in range(saving_workers):
            save_queue.put(None)
        for thread in saver_threads:
            thread.join()
        del dataloader
        empty_cache()

        # return results
        results = list(results)
        vars_df = pd.DataFrame(vars_dict)
        results_ = [result for result in results if result]
        df_tiles = pd.DataFrame(results_)
        df_tiles["original_mag"] = natural_magnification
        df_tiles["scale"] = scale
        df_tiles.to_csv(os.path.join(sample_path, patient_id + ".csv"), index=False)
        blurry_tiles = len(results_) if self.remove_blurry_tiles else None
        return {
            'tiles_df': df_tiles,
            'vars': vars_dict,
            'blurry_tiles': blurry_tiles
        }, {

            'wall': time.time() - patching_user,
            'cpu': time.process_time() - patching_cpu}

    def _process_tiles_gpu_h5(self, tile_iterator, patient_id, sample_path, natural_magnification, scale):
        from torch.utils.data import DataLoader
        from torch.cuda import empty_cache
        patching_user = time.time()
        patching_cpu = time.process_time()
        # setting up
        load_workers = self.max_workers - 1
        saving_workers = self.max_workers - load_workers
        dataloader = DataLoader(tile_iterator, num_workers=load_workers,
                                batch_size=self.config.get("batch_size"), pin_memory=True, shuffle=False)

        # multiprocessing portions
        manager = multiprocessing.Manager()
        results = []
        vars_dict = manager.dict()
        save_queue = multiprocessing.Queue()
        # set up for saving
        tiles_path = os.path.join(sample_path, f"{patient_id}.h5")

        saver_threads = []
        for _ in range(saving_workers):
            thread = multiprocessing.Process(args=(save_queue, tiles_path), target=save_h5)
            thread.start()
            saver_threads.append(thread)
        for batch in dataloader:
            batch_tiles, coord_batch = batch
            batch_tiles = batch_tiles.to(self.device, non_blocking = True)
            coord_batch = coord_batch.to(self.device, non_blocking = True)
            for step in self.pipeline_steps:
                batch_tiles, coord_batch = step(batch_tiles, vars_dict, coord_batch)
                if batch_tiles.numel() == 0:
                    break
            batch_tiles = batch_tiles.detach().cpu().numpy()
            coord_batch = coord_batch.detach().cpu().numpy()
            for tile, coord in zip(batch_tiles, coord_batch):
                save_queue.put((coord, tile))
                results.append({
                    "Patient_ID": patient_id, "x": coord[0], "y": coord[1],
                    "original_size": tile_iterator.adjusted_size,
                    "desired_size": tile_iterator.size, "desired_magnification": tile_iterator.magnification
                })
            del batch_tiles,coord_batch

        for _ in range(saving_workers):
            save_queue.put(None)
        for thread in saver_threads:
            thread.join()

        # return results
        results = list(results)
        vars_df = pd.DataFrame(vars_dict)
        results_ = [result for result in results if result]
        df_tiles = pd.DataFrame(results_)
        df_tiles["original_mag"] = natural_magnification
        df_tiles["scale"] = scale
        df_tiles.to_csv(os.path.join(sample_path, patient_id + ".csv"), index=False)
        blurry_tiles = len(results_) if self.remove_blurry_tiles else None
        del dataloader
        empty_cache()
        return {
            'tiles_df': df_tiles,
            'vars': vars_dict,
            'blurry_tiles': blurry_tiles
        }, {
            'wall': time.time() - patching_user,
            'cpu': time.process_time() - patching_cpu}

    def _post_processing(self, slide, patient_id, sample_path, mask_obj, all_coords, coordinates, adjusted_size,
                         df_tiles, vars):
        """Perform post-processing tasks like reconstruction and QC."""
        if self.config.get('reconstruct_slide'):
            try:
                reconstruct_slide(
                    mask_obj.get_applied_mask(), coordinates, all_coords,
                    mask_obj.SCALE, adjusted_size,
                    save_path=os.path.join(sample_path, "included_tiles.png")
                )

                if self.remove_blurry_tiles:
                    try:
                        valid_coords = np.array([[x, y] for x, y in zip(df_tiles.x.values, df_tiles.y.values)])
                        reconstruct_slide(
                            mask_obj.get_applied_mask(), valid_coords, all_coords,
                            mask_obj.SCALE, adjusted_size,
                            save_path=os.path.join(sample_path, "included_tiles_after_QC.png")
                        )
                    except Exception as e:
                        pass
            except Exception as e:
                pass

        if self.normalize_staining and not df_tiles.empty:
            self._create_qc_samples(slide, patient_id, sample_path, adjusted_size, df_tiles, vars, coordinates)

    def _create_qc_samples(self, slide, patient_id, sample_path, adjusted_size, df_tiles, vars, coordinates):
        """Create quality control sample images."""
        QC_path = os.path.join(sample_path, "QC_pipeline")
        os.makedirs(QC_path, exist_ok=True)

        desired_size = self.config.get('desired_size', 256)
        non_blurry_coords = list(zip(df_tiles['x'], df_tiles['y']))

        # Save example non-blurry tile
        i = 0
        while i < 5:
            random_coord = non_blurry_coords[random.randint(0, len(non_blurry_coords) - 1)]
            region = slide.read_region(
                (random_coord[0], random_coord[1]), 0,
                (adjusted_size, adjusted_size)
            ).convert('RGB').resize((desired_size, desired_size), Image.BILINEAR)
            if self.device == "cpu":
                region = np.array(region)
                normalized_img = self.normalization_method(region)
            else:
                import torch
                region = torch.tensor(np.array(region)).unsqueeze(0)
                normalized_img = self.normalization_method(region)
                if isinstance(normalized_img, torch.Tensor):
                    normalized_img = normalized_img.detach().cpu().numpy()
                    if normalized_img.ndim == 4 and normalized_img.shape[0] == 1:
                        normalized_img = normalized_img[0]
                region = region.detach().cpu().numpy()
                region = region[0]

            if normalized_img is not None:
                normalized_img = np.array(normalized_img)
                normalized_img = Image.fromarray(normalized_img)
                normalized_img.save(os.path.join(QC_path, "normalized_non_blurry.png"))
                Image.fromarray(region).save(os.path.join(QC_path, "original_non_blurry.png"))
                break
            i += 1

        # Save blurry tile examples if applicable
        if self.remove_blurry_tiles:
            # Laplacian distribution plot
            distributions = os.path.join(QC_path, "tile_distributions")
            os.makedirs(distributions, exist_ok=True)
            var_path = os.path.join(distributions, "Laplacian_variance.png")
            self._plot_laplacian_variance(vars, var_path, self.config.get('blur_threshold', 0.015))

            # Example blurry tiles
            coordin = [tuple(coord) for coord in coordinates]
            different_coords = list(set(coordin) - set(non_blurry_coords))

            if different_coords:
                i = 0
                while i < 5:
                    random_coord = different_coords[random.randint(0, len(different_coords) - 1)]
                    region = slide.read_region(
                        (random_coord[0], random_coord[1]), 0,
                        (adjusted_size, adjusted_size)
                    ).convert('RGB').resize((desired_size, desired_size), Image.BILINEAR)
                    if self.device == "cpu":
                        region = np.array(region)
                        normalized_img = self.normalization_method(region)
                    else:
                        import torch
                        region = torch.tensor(np.array(region)).unsqueeze(0)
                        normalized_img = self.normalization_method(region).detach().cpu().numpy()
                        if normalized_img.ndim == 4 and normalized_img.shape[0] == 1:
                            normalized_img = normalized_img[0]
                        region = region.detach().cpu().numpy()
                        region = region[0]
                    if normalized_img is not None:
                        normalized_img = np.array(normalized_img)
                        region = Image.fromarray(np.array(region))
                        region.save(os.path.join(QC_path, "original_blurry.png"))
                        normalized_img = Image.fromarray(normalized_img)
                        normalized_img.save(os.path.join(QC_path, "normalized_blurry.png"))
                        break
                    i += 1

    def _plot_laplacian_variance(self, values, save_path, var_threshold=0.015):
        plt.figure(figsize=(6, 6))
        distribution = pd.Series(values, name="Laplacian Variance")
        sns.histplot(data=distribution, kde=True)
        plt.axvline(x=var_threshold, color="red", linestyle="--", label=f"threshold = {var_threshold}")
        plt.title("Laplacian variance distribution")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(save_path)
        plt.close()


# =========================================== FILE HANDLING ============================================================
def move_svs_files(main_directory):
    valid_extensions = (".svs", ".tif", ".tiff", ".dcm",
                        ".ndpi", ".vms", ".vmu", ".scn",
                        ".mrxs", ".svslide", ".bif")
    for root, _, files in os.walk(main_directory, topdown=False):
        for file_name in files:
            if file_name.endswith(valid_extensions):
                file_path = os.path.join(root, file_name)
                if root != main_directory:
                    shutil.move(file_path, os.path.join(main_directory, file_name))


def patient_csv(input_path, results_path):
    csv_file_path = os.path.join(results_path, "patient_files.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:

        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Patient ID", "Original Slide Path", "Preprocessing Path"])

        # check if provided with a directory of samples or a sample

        if os.path.isdir(input_path):
            for file in os.listdir(input_path):

                if file.endswith((
                        ".svs", ".tif", ".tiff", ".dcm",
                        ".ndpi", ".vms", ".vmu", ".scn",
                        ".mrxs", ".svslide", ".bif")):

                    patient_id = os.path.basename(file)
                    patient_id = patient_id.split(".")[0]
                    patient_result_path = os.path.join(results_path, patient_id)

                    if not os.path.exists(patient_result_path):
                        os.makedirs(patient_result_path)
                    csv_writer.writerow([patient_id, os.path.join(input_path, file),
                                         os.path.join(results_path, patient_id)])
        else:
            patient_id = os.path.basename(input_path)
            patient_id = patient_id.split(".")[0]
            patient_result_path = os.path.join(results_path, patient_id)

            if not os.path.exists(patient_result_path):
                os.makedirs(patient_result_path)

            csv_writer.writerow([patient_id, input_path,
                                 os.path.join(results_path, patient_id)])

    return csv_file_path


# =========================================== PARAMS ====================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SlideLab Arguments for Whole Slide Image preprocessing")
    parser.add_argument("--config_file", type=str, default="None", help="path to config file")

    # input/output
    parser.add_argument("-i", "--input_path", type=str, default="/path/to/slide(s)")
    parser.add_argument("-o", "--output_path", type=str, default="/output/path")

    # tile customization
    parser.add_argument("-s", "--desired_size", type=int, default=256,
                        help="Desired size of the tiles in pixels (default: %(default)s)")
    parser.add_argument("-m", "--desired_magnification", type=int, default=None,
                        help="Desired magnification level (default: %(default)s)")
    parser.add_argument("--desired_mpp", type=float, default=None,
                        help="Desired reference mpp (default: %(default)s)")
    parser.add_argument("--sizing_scale", type=str, default="magnification",
                        help="To choose either mpp or magnification, for more accurate results between scanners, use mpp")

    parser.add_argument("--set_standard_magnification", type=float, default=None)
    parser.add_argument("--set_standard_mpp", type=float, default=None)

    parser.add_argument("-ov", "--overlap", type=int, default=1,
                        help="Overlap between tiles (default: %(default)s)")
    parser.add_argument("--output_format", type=str, default="png")

    # preprocessing processes customization
    parser.add_argument("--no_saving", action="store_true",
                        help="flag to not save tiles. Choosing this option will lead to automatic encoding.")
    parser.add_argument("-rb", "--remove_blurry_tiles", action="store_true",
                        help="flag to enable usage of the laplacian filter to remove blurry tiles")
    parser.add_argument("-n", "--normalize_staining", type=str,default=None, 
                        help="Flag to enable normalization of tiles")
    parser.add_argument("-e", "--encode", action="store_true",
                        help="Flag to encode tiles and create associated .h5 file")
    parser.add_argument("--reconstruct_slide", action="store_true",
                        help="reconstruct slide ")

    # encoding customizationss
    parser.add_argument("--extract_high_quality", action="store_true",
                        help="extract high quality ")
    parser.add_argument("--augmentations", type=int, default=0,
                        help="augment data for training ")
    parser.add_argument("--feature_extractor", default="resnet50",
                        help="current options: resnet50,resnet50-truncated, mahmood-uni, empty")
    parser.add_argument("--token", default=None, help="required to download model weights from hugging face")

    # thresholds
    parser.add_argument("-th", "--tissue_threshold", type=float, default=0.7,
                        help="Threshold to consider a tile as Tissue(default: %(default)s)")
    parser.add_argument("-bh", "--blur_threshold", type=float, default=0.025,
                        help="Threshold for laplace filter variance (default: %(default)s)")
    parser.add_argument("--red_pen_check", type=float, default=0.4,
                        help="Sanity check for % of red pen detected. If above threshold, red_pen mask will be ignored(default: %(default)s)")
    parser.add_argument("--blue_pen_check", type=float, default=0.4,
                        help="Sanity check for % of blue pen detected,  If above threshold, blue_pen mask will be ignored(default: %(default)s)")
    parser.add_argument("--include_adipose_tissue", action="store_true", help="will include adipose tissue in mask")
    parser.add_argument("--remove_folds", action="store_true", help="will remove folded tissue in mask")
    parser.add_argument("--mask_scale", type=int, default=None,
                        help="scale at which to downscale WSI for masking. Recommended is either 64 or None which will downsize to the lowest possible downscale recommended by openslide. None will produce a higher quality mask but is slower than 64")

    # for devices + multithreading
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu_processes", type=int, default=1)
    parser.add_argument("--cpu_processes", type=int, default=os.cpu_count())
    parser.add_argument("--batch_size", type=int, default=16)

    # QC
    parser.add_argument("--min_tiles", type=float, default=0,
                        help="Number of tiles a patient should have.")
    return parser.parse_args()


# ================================================= MAIN ======================================================================
def main():
    args = parse_args()
    if args.config_file != "None":
        args = get_args_from_config(args.config_file)
    args = vars(args)

    # check input/output
    input_path = args.get("input_path")
    output_path = args.get("output_path")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file '{input_path}' does not exist.")
    if args.get("output_format") == "null":
        args["output_format"] = None 

    # making output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initiate pipeline
    preprocessor = SlidePreprocessing(args)

    # get possible
    patient_path = patient_csv(input_path, output_path)
    encoding_times = []
    patients = pd.read_csv(patient_path)
    if args.get("output_format") is not None:
        t = tqdm.tqdm(zip(patients["Original Slide Path"].values, patients["Patient ID"].values), total=len(patients))
        for slide_path, patient_id in t:
            # Update the description with the current patient_id
            t.set_description(f"Processing: {patient_id}")
            t.refresh()
            if not os.path.isfile(os.path.join(output_path, patient_id, patient_id + ".csv")):
                results = preprocessor.__call__(slide_path, patient_id, output_path)
                Reports.Reports([results[0]], [results[1]], output_path)
            t.update(1)
    else:
        t = tqdm.tqdm(zip(patients["Original Slide Path"].values, patients["Patient ID"].values), total=len(patients))
        print("No output format was flagged, and so feature extraction will take place automatically.")
        encoding_dir = os.path.join(output_path, f"{preprocessor.config.get("feature_extractor")}_features")
        os.makedirs(encoding_dir,exist_ok=True)

        # initiate the encoder module
        from SlideEncoding import SlideEncoding
        slide_encoder = SlideEncoding(preprocessor.config, preprocessor.pipeline_steps)
        # loop
        for slide_path, patient_id in t:
            t.set_description(f"Encoding: {patient_id}")
            t.refresh()
            # path to save features
            sample_path = os.path.join(encoding_dir, patient_id+".h5")
            # processed slide
            out = preprocessor.__call__(slide_path,  patient_id,output_path)
            if len(out) == 2:
                summary, error = out
            else:
                slide, coords, mask, magnification, adjusted_size, summary, error = out
            report_instance = Reports.Reports([summary], [error], output_path)
            # only continue if there was NO error -> if there was some sort of error, shouldn't continue
            start_cpu_time = time.process_time()
            start_user_time = time.time()
            if not error:

                results = slide_encoder.__call__(slide, sample_path, coords,
                                                 mask, adjusted_size,
                                                 preprocessor.config.get("desired_size"))
            t.update(1)
            encoding_times.append((patient_id, time.process_time() - start_cpu_time, time.time() - start_user_time))
            report_instance.summary_report_update_encoding(encoding_times)

        # report_instance = Reports.Reports([[]], [[]], output_path)
        # report_instance.summary_report_update_encoding(encoding_times)

    # encoding for all other cases other than no saving

    if args.get("encode") and args.get("output_format") in ["png", "h5"]:
        if args.get("min_tiles") > 0:
            filter_patients(patients, os.path.join(args.get("output_path"), "SummaryReport.csv"), args)
            patients = pd.read_csv(patient_path)

        from SlideEncoding import SlideEncoding
        import torch
        torch.backends.cudnn.benchmark = True
        extension = ".csv" if preprocessor.config.get("output_format") == "png" else ".h5"
        slide_encoder = SlideEncoding(preprocessor.config, preprocessor.pipeline_steps)
        # name after the encoder
        encoding_path = os.path.join(output_path, preprocessor.config.get("feature_extractor"))
        os.makedirs(encoding_path, exist_ok=True)
        t = tqdm.tqdm(zip(patients["Original Slide Path"].values, patients["Patient ID"].values), total=len(patients))
        for slide_path, patient_id in t:
            t.set_description(f"Encoding: {patient_id}")
            t.refresh()
            sample_path = os.path.join(encoding_path, patient_id+".h5")
            if os.path.isfile(sample_path):
                continue

            in_path = os.path.join(output_path, patient_id, patient_id + extension)
            start_cpu_time = time.process_time()
            start_user_time = time.time()
            try:
                slide_encoder.__call__(in_path, sample_path)
            except Exception as e :
                print(f"[ERROR] failed to encode {patient_id}, {e}")
            encoding_times.append((patient_id, time.process_time() - start_cpu_time, time.time() - start_user_time))

            t.update(1)

    # patient_files_encoded(patient_path)
    # patient_files_encoded(patient_path)
    report_instance = Reports.Reports([[]], [[]], output_path)
    report_instance.summary_report_update_encoding(encoding_times)
    if args.get("min_tiles") > 0:
        # filter patient_csv depending on amount of tiles
        filter_patients(patients, os.path.join(args.get("output_path"), "SummaryReport.csv"), args)

        # Dataset type


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", force=True)
    main()
