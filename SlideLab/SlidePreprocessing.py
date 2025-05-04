# outside imports
import argparse
import csv
import shutil
import tqdm
import os
import openslide
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
import time
import multiprocessing
import random
import threading


# classes/functions

import Reports, SlideEncoding
from normalization.TileNormalization import normalizeStaining, normalizeStaining_torch
from TileQualityFilters import LaplaceFilter, plot_distribution
from TissueMask import is_tissue, get_region_mask, TissueMask
from tiling.TileIterator import TileIterator

# utils
from config import get_args_from_config
from utils.VisulizationUtils import reconstruct_slide
from utils.preprocessing_utils import worker_h5, save_h5_cpu, worker_png, mpp_to_mag, mag_to_mpp, get_valid_coordinates, \
    get_best_size_mag, get_best_size_mpp,get_closest_mpp
from tiling.EncodingDatasets import TileEncoding_h5, TilePreprocessing_nosaving, TilePreprocessing_png


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
SlidePreprocessing.py

Author: Lorenzo Olmo Marchal
Created: 3/5/2024
Last Updated: 4/19/2025

Description:
This script automates the preprocessing and normalization of Whole Slide Images (WSI) in digital histopathology. 
Input:
- slide directory path or slide path
- slide directory output path

Output:
Processed tiles are saved in the output directory. Each tile is accompanied by metadata in a csv, including its origin
within the WSI file.

"""

summary = []
errors = []


############### post processing ##############

def filter_patients(patient_df, summary_df, args):
    # patient_df = pd.read_csv(patient_df)
    summary_df = pd.read_csv(summary_df)
    if not args.remove_blurry_tiles:
        column = "tiles_passing_tissue_thresh"
    else:
        column = "non_blurry_tiles"
    not_passing_QC = summary_df.loc[summary_df[column] < args.min_tiles, "sample_id"]
    # filter based on sample ID
    filtered_patients = patient_df[~patient_df["Patient ID"].isin(not_passing_QC)]
    filtered_path = os.path.join(args.output_path, "filtered_patients.csv")
    filtered_patients.to_csv(filtered_path, index=False)


def patient_files_encoded(patient_files_path):
    df = pd.read_csv(patient_files_path)
    df["Encoded Path"] = [""] * len(df)
    df["Encoded Path"] = df["Encoded Path"].astype("object")
    for i, row in df.iterrows():
        encoded_path = os.path.join(os.path.dirname(row["Preprocessing Path"]), "encoded", row["Patient ID"] + ".h5")
        df.loc[i, "Encoded Path"] = str(encoded_path)
    df.to_csv(patient_files_path)


# ----------------------- PREPROCESSING -----------------------------------
# General helper methods
def save_tiles(coord, norm_tile, output_dir, patient_id, desired_size, desired_mag):
    try:
        file_path = os.path.join(
            output_dir, f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png"
        )
        cv2.imwrite(file_path, norm_tile.numpy()[:, :, ::-1])  # Convert to BGR for saving

        metadata = {
            "Patient_ID": patient_id,
            "x": coord[0],
            "y": coord[1],
            "tile_path": file_path,
            "desired_size": desired_size,
            "desired_magnification": desired_mag,
        }
        return metadata
    except Exception as e:
        print(f"Error in CPU task for tile {coord}: {e}")
        return None


def save_tiles_QC(coord, norm_tile, output_dir, patient_id, desired_size, desired_mag, threshold):
    try:
        norm_tile = norm_tile.numpy()
        blurry, var = LaplaceFilter(norm_tile, var_threshold=threshold)
        if not blurry:
            file_path = os.path.join(output_dir,
                                     f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png")
            cv2.imwrite(file_path, norm_tile[:, :, ::-1])
            metadata = {
                "Patient_ID": patient_id,
                "x": coord[0],
                "y": coord[1],
                "tile_path": file_path,
                "desired_size": desired_size,
                "desired_magnification": desired_mag}
            return metadata, var
        return None, var
    except Exception as e:
        print(f"Error in QC or saving tile: {e}")
        return None


# Main preprocessing script
"""
- needs path to slide
- patient_id 
- device 
- args -> from there can extract most things
"""


def preprocessing(path, patient_id, args):
    # if not normalizing do NOT use gpu
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() and args.normalize_staining else "cpu"
    else:
        device = args.device
    # error and summary report
    error = []
    summary = []
    vars = []
    # instance statistics
    total_tiles = -1
    valid_tiles = -1
    blurry_tiles = -1

    # preprocessing time statistics
    time_mask = -1
    time_mask_cpu = -1
    time_coordinates = -1
    time_coordinates_cpu = -1
    time_patches = -1
    time_patches_cpu = -1

    def summary_():
        summary = [patient_id, path, total_tiles, valid_tiles, blurry_tiles,
                   time_mask_cpu, time_coordinates_cpu, time_patches_cpu, time_mask,
                   time_coordinates, time_patches]
        return summary

    # check 1: Opening Slide
    try:
        slide = openslide.OpenSlide(path)
    except Exception as e:
        error.append((patient_id, path, "Error Opening file", "Slide Opening"))
        summary = summary_()
        summary.append("Error")
        return summary, error

        # check 2: natural mag is not lesser than requested mag

    natural_magnification = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if natural_magnification is not None:
        natural_magnification = float(natural_magnification)
    else:
        natural_magnification = 40


    # mmp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    # mmp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    # if mmp_x is not None:
    #     mmp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
    # if mmp_y is not None:
    #     mmp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y))
    #
    # # if there are any missing values:
    # if natural_magnification is None:
    #     if args.sizing_scale == "magnification" and mmp_x is not None:
    #         natural_magnification = mpp_to_mag(mmp_x)
    #     elif args.sizing_scale == "mpp":
    #         if mmp_x is None and args.set_standard_mpp is not None:
    #             mmp_x = args.set_standard_mpp
    #         elif args.set_standard_magnification is not None:
    #             natural_magnification = args.set_standard_magnification
    #         else:
    #             error.append((patient_id, path,
    #                           "OpenSlide object does not have mpp or natural magnification available. "
    #                           "To bypass this error either A. Run add_metadata.py with the slide metadata or B. "
    #                           "Set a standard value for unknown values with either --set_standard_mpp or --set_standard_magnification",
    #                           "Resolution Sanity check "))
    # elif mmp_x is None:
    #         if args.sizing_scale == "mpp" and natural_magnification is not None:
    #             mmp_x = mag_to_mpp(natural_magnification)
    #         elif args.sizing_scale == "mpp" and args.set_standard_mpp is not None:
    #             mmp_x = args.set_standard_mpp
    #         else:
    #             error.append((patient_id, path,
    #                           "OpenSlide object does not have mpp or natural magnification available. "
    #                           "To bypass this error either A. Run add_metadata.py with the slide metadata or B. "
    #                           "Set a standard value for unknown values with either --set_standard_mpp or --set_standard_magnification",
    #                           "Resolution Sanity check "))
    #             summary = summary_()
    #             summary.append("Error")
    #             return summary, error
    # # conversion of desired mpp/magnification
    # if args.sizing_scale == "magnification" and args.desired_magnification is None and args.desired_mpp is not None:
    #     desired_magnification = mpp_to_mag(args.desired_mpp)
    # elif args.sizing_scale == "mpp" and args.desired_mpp is None and args.desired_magnification is not None:
    #     desired_magnification = mag_to_mpp(args.desired_magnification)
    # elif args.sizing_scale == "mpp" and args.desired_mpp is not None:
    #     desired_magnification = args.desired_mpp
    # elif args.sizing_scale == "magnification" and args.desired_magnification is not None:
    #     desired_magnification = args.desired_magnification
    #
    # # resolution sanity check, you cannot extrapolate a lower resolution to higher resolution!!
    # if args.sizing_scale == "mpp":
    #     closest_mpp = get_closest_mpp(mmp_x)
    #     # ex: slide has natural mpp of 0.5 (20x) and the desired resolution is at 0.25 mpp (40x)
    #     resolution_sanity = closest_mpp > desired_magnification
    # else:
    #     resolution_sanity = natural_magnification < desired_magnification
    desired_magnification = args.desired_magnification
    resolution_sanity = natural_magnification < desired_magnification
    if resolution_sanity:

        error.append((patient_id, path, "Desired magnification is higher than natural magnification",
                      "Magnification Sanity Check"))
        summary = summary_()
        summary.append("Error")
        return summary, error


    # Step 1: Mask (obtain tissue)
    sample_path = os.path.join(args.output_path, patient_id)
    start_mask_user = time.time()
    start_mask_cpu = time.process_time()
    try:
        mask_ = TissueMask(slide, result_path=sample_path, blue_pen_thresh=args.blue_pen_check,
                           red_pen_thresh=args.red_pen_check, SCALE=args.mask_scale, remove_folds=args.remove_folds)
        mask, scale = mask_.get_mask_attributes()
    except Exception as e:
        error.append((patient_id, path, e, "Tissue Mask"))
        summary = summary_()
        summary.append("Error")
        return summary, error

    time_mask_cpu = time.process_time() - start_mask_cpu
    time_mask = time.time() - start_mask_user
    # Step 2: Valid coordinates according to tissue mask

    # desired size in pixels
    desired_size = args.desired_size
    if args.sizing_scale == "mpp":
        adjusted_size = get_best_size_mpp(desired_size, desired_magnification, mmp_x, mmp_y)
    else:
        adjusted_size = get_best_size_mag(desired_magnification, natural_magnification, desired_size)
    adjusted_size = int(adjusted_size)

    start_time_coordinates_user = time.time()
    start_time_coordinates_cpu = time.process_time()

    overlap = args.overlap
    # candidate coordinates
    w, h = slide.dimensions
    all_coords, valid_coordinates, coordinates, all_coords_ = get_valid_coordinates(w, h, overlap, mask, adjusted_size,
                                                                                    scale,
                                                                                    threshold=args.tissue_threshold)
    time_coordinates_cpu = time.process_time() - start_time_coordinates_cpu
    time_coordinates = time.time() - start_time_coordinates_user
    total_tiles = all_coords
    valid_tiles = valid_coordinates

    # setting up process information
    if args.cpu_processes is None or args.cpu_processes > os.cpu_count():
        max_workers = os.cpu_count()
    else:
        max_workers = args.cpu_processes

    if args.gpu_processes is None:
        max_gpu_workers = os.cpu_count()
    else:
        max_gpu_workers = args.gpu_processes

    start_time_patches_user = time.time()
    start_time_patches_cpu = time.process_time()
    tiles_path = os.path.join(sample_path, patient_id + ".csv")

    # only make tiles dir if saving png versions
    if args.output_format == "png":
        tiles_dir = os.path.join(sample_path, "tiles")
        os.makedirs(tiles_dir, exist_ok=True)

    # Step 3: get tiles (separated into 2 different processes depending if available gpu or not)
    if device == "cuda":
        manager = multiprocessing.Manager()
        metadata_list = manager.list()
        save_queue = manager.Queue()
        # metadata_list = multiprocessing.Manager().list()  
        tile_iterator = TileIterator(
            slide, coordinates=coordinates, mask=mask, normalizer=None,
            size=desired_size, magnification=desired_magnification,
            adjusted_size=adjusted_size, overlap=overlap
        )

        num_gpu_workers = max_workers // 3
        num_saving_workers = max_workers - num_gpu_workers
        cuda_streams = [torch.cuda.Stream() for _ in range(num_gpu_workers)]

        # set up save queue
        # save_queue = multiprocessing.Queue() 

        def chunk_iterator(iterator, num_workers):
            """ Distribute tile indices evenly among workers. """
            return [list(range(i, len(iterator), num_workers)) for i in range(num_workers)]

        # loads + sends to cpu for normalization
        def load_normalize(tiles_chunk, iterator, worker_id):
            stream = cuda_streams[worker_id % num_gpu_workers]
            for index in tiles_chunk:
                tile, coord = iterator[index]
                with torch.cuda.stream(stream):
                    try:
                        tile_tensor = torch.tensor(np.array(tile)).to("cuda", non_blocking=True)
                        norm_tile = normalizeStaining_torch(tile_tensor)
                        if norm_tile is not None:
                            save_queue.put((coord, norm_tile))  # Push to saving queue
                    except Exception as e:
                        print(f"Error in GPU task {worker_id} for tile {coord}: {e}")

        # to save tiles
        def save_worker(output_dir, patient_id, desired_size, desired_mag, blur_threshold=None):
            while True:
                item = save_queue.get()
                if item is None:
                    break

                coord, norm_tile = item
                try:
                    if blur_threshold is not None:
                        metadata = save_tiles_QC(coord, norm_tile, output_dir, patient_id, desired_size, desired_mag,
                                                 blur_threshold)
                    else:
                        metadata = save_tiles(coord, norm_tile, output_dir, patient_id, desired_size, desired_mag)

                    if metadata:
                        metadata_list.append(metadata)
                except Exception as e:
                    print(f"Error saving tile {coord}: {e}")

        chunked_tiles = chunk_iterator(tile_iterator, num_gpu_workers)

        gpu_threads = []
        for i in range(num_gpu_workers):
            t = threading.Thread(target=load_normalize, args=(chunked_tiles[i], tile_iterator, i))
            t.start()
            gpu_threads.append(t)

        save_processes = []
        for _ in range(num_saving_workers):
            p = multiprocessing.Process(target=save_worker, args=(
                os.path.join(sample_path, "tiles"), patient_id, desired_size, desired_magnification,
                args.blur_threshold if args.remove_blurry_tiles else None))
            p.start()
            save_processes.append(p)

        for t in gpu_threads:
            t.join()

        for _ in range(num_saving_workers):
            save_queue.put(None)

        for p in save_processes:
            p.join()
        try:
            if isinstance(metadata_list[0], tuple):
                metadata_list, vars = zip(*metadata_list)
            final = [item for item in metadata_list if item is not None]
            vars = [item for item in vars if item is not None]

            df_tiles = pd.DataFrame(list(final))
            df_tiles["original_mag"] = natural_magnification
            df_tiles["scale"] = scale
            df_tiles.to_csv(tiles_path, index=False)
            blurry_tiles = len(metadata_list) if args.remove_blurry_tiles else None
        except:
            error.append((patient_id, path, "No tiles were saved.", "Tiling"))
            summary = summary_()
            summary.append("Error")
            return summary, error
    else:
        # tile Iterator
        tile_iterator = TileIterator(
            slide, coordinates=coordinates, mask=mask, normalizer=None,
            size=desired_size, magnification=desired_magnification,
            adjusted_size=adjusted_size, overlap=overlap
        )

        if args.output_format == "png":
            manager = multiprocessing.Manager()
            results = manager.list()
            vars = manager.list()
            queue = multiprocessing.Queue()
            threads = []
            for _ in range(max_workers):
                thread = multiprocessing.Process(target=worker_png,
                                                 args=(queue, tile_iterator,
                                                       patient_id,
                                                       os.path.join(sample_path, "tiles"),
                                                       results, vars, args.normalize_staining, args.remove_blurry_tiles, args.blur_threshold))
                thread.start()
                threads.append(thread)
            for idx in range(len(tile_iterator)):
                queue.put(idx)
            for _ in range(max_workers):
                queue.put(None)
            for thread in threads:
                thread.join()

            results = list(results)
            vars = list(vars)
            results_ = [result for result in results if result]
            df_tiles = pd.DataFrame(results_)
            df_tiles["original_mag"] = natural_magnification
            df_tiles["scale"] = scale
            df_tiles.to_csv(tiles_path, index=False)
        elif args.output_format == "h5":
            h5_file = os.path.join(sample_path, patient_id + ".h5")
            # set up workers + summary
            manager = multiprocessing.Manager()
            var_list = manager.list()
            results_list = manager.list()
            index_queue = multiprocessing.Queue()
            save_queue = multiprocessing.Queue()

            processing_workers = max(1, max_workers - 1)
            processing_threads = []
            for i in range(processing_workers):
                p = multiprocessing.Process(target=worker_h5, args=(index_queue,
                                                                     tile_iterator, patient_id,
                                                                    results_list, var_list, args.normalize_staining,
                                                                    args.remove_blurry_tiles, args.blur_threshold,
                                                                    save_queue))
                p.start()
                processing_threads.append(p)
            saving_thread = threading.Thread(target=save_h5_cpu, args=(save_queue, h5_file))
            saving_thread.start()
            for idx in range(len(tile_iterator)):
                index_queue.put(idx)
            for _ in range(processing_workers):
                index_queue.put(None)
            for process in processing_threads:
                process.join()
            save_queue.put(None)
            saving_thread.join()
            # get results
            results = list(results_list)
            vars = list(var_list)
            results_ = [result for result in results if result]
            df_tiles = pd.DataFrame(results_)
            df_tiles["original_mag"] = natural_magnification
            df_tiles["scale"] = scale
            df_tiles.to_csv(tiles_path, index=False)

        blurry_tiles = len(results_) if args.remove_blurry_tiles else None
    


    time_patches = time.time() - start_time_patches_user
    time_patches_cpu = time.process_time() - start_time_patches_cpu
    summary = summary_()
    summary.append("Processed")
    if "x" not in df_tiles.columns:
        error.append((patient_id, path, f"No tiles were saved. Please consider lowering the tissue threshold from {args.tissue_threshold}",
                      "Slide Tiling"))
        return summary, error
        

    # additional things like slide reconstruction + QC folder 
    if args.reconstruct_slide:
        try:
            reconstruct_slide(mask_.applied, tile_iterator.coordinates,
                              all_coords_, mask_.SCALE,
                              tile_iterator.adjusted_size,
                              save_path=os.path.join(sample_path, "included_tiles.png"))
        except Exception as e:
            error.append((patient_id, path, e, "Reconstructing slide"))
        if args.remove_blurry_tiles:
            try:
                valid_coords = np.array([[x, y] for x, y in zip(df_tiles.x.values, df_tiles.y.values)])
                reconstruct_slide(mask_.applied, valid_coords,
                                  all_coords_, mask_.SCALE,
                                  tile_iterator.adjusted_size,
                                  save_path=os.path.join(sample_path, "included_tiles_after_QC.png"))
            except Exception as e:
                error.append((patient_id, path, e, "Reconstructing slide"))

    # sanity check -> statistics for process do (1-2 examples)
    
    if args.normalize_staining:
        QC_path = os.path.join(sample_path, "QC_pipeline")
        os.makedirs(QC_path, exist_ok=True)
        # choose random coordinate
        non_blurry_coords = list(zip(df_tiles['x'], df_tiles['y']))
        i = 0
        while i < 5:
            random_coord = non_blurry_coords[random.randint(0, len(non_blurry_coords) - 1)]
            region = slide.read_region((random_coord[0], random_coord[1]), 0, (adjusted_size, adjusted_size)).convert(
                'RGB').resize(
                (desired_size, desired_size), Image.BILINEAR)
            normalized_img = Image.fromarray(normalizeStaining(np.array(region)))
            if normalized_img is not None:
                normalized_img.save(os.path.join(QC_path, "normalized_non_blurry.png"))
                region.save(os.path.join(QC_path, "original_non_blurry.png"))
                break
            i += 1

        if args.remove_blurry_tiles:
            # laplacian distribution
            distributions = os.path.join(QC_path, "tile_distributions")
            os.makedirs(distributions, exist_ok=True)
            var_path = os.path.join(distributions, "Laplacian_variance.png")
            plot_distribution(vars, var_path, args.blur_threshold)
            coordin = [tuple(coord) for coord in coordinates]

            # also get an example for QC sample of a blurry and a non-blurry tile (valid coordinates)
            different_coords = list(set(coordin) - set(non_blurry_coords))
            if different_coords:
                i = 0
                while i < 5:
                    random_coord = different_coords[random.randint(0, len(different_coords) - 1)]
                    region = slide.read_region((random_coord[0], random_coord[1]), 0,
                                               (adjusted_size, adjusted_size)).convert(
                        'RGB').resize(
                        (desired_size, desired_size), Image.BILINEAR)
                    normalized_blurry = normalizeStaining(np.array(region))
                    if normalized_blurry is not None:
                        region.save(os.path.join(QC_path, "original_blurry.png"))
                        normalized_img = Image.fromarray(normalized_blurry)
                        normalized_img.save(os.path.join(QC_path, "normalized_blurry.png"))
                        break
                    i += 1
    slide.close()
    return summary, error


# ----------------------- FILE HANDLING -----------------------------------


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


# --------------------------- MAIN ----------------------------------------------------

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
    parser.add_argument("-n", "--normalize_staining", action="store_true",
                        help="Flag to enable normalization of tiles")
    parser.add_argument("-e", "--encode", action="store_true",
                        help="Flag to encode tiles and create associated .h5 file")
    parser.add_argument("--reconstruct_slide", action="store_true",
                        help="reconstruct slide ")

    # encoding customizations
    parser.add_argument("--extract_high_quality", action="store_true",
                        help="extract high quality ")
    parser.add_argument("--augmentations", type=int, default=0,
                        help="augment data for training ")
    parser.add_argument("--encoder_model", default="resnet50",
                        help="current options: resnet50, mahmood-uni")
    parser.add_argument("--token", default=None, help="required to download model weights from hugging face")

    # thresholds 
    parser.add_argument("-th", "--tissue_threshold", type=float, default=0.7,
                        help="Threshold to consider a tile as Tissue(default: %(default)s)")
    parser.add_argument("-bh", "--blur_threshold", type=float, default=0.015,
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


def main():
    args = parse_args()
    if args.config_file != "None":
        args = get_args_from_config(args.config_file)
    input_path = args.input_path
    output_path = args.output_path

    if args.desired_magnification is None and args.desired_mpp is None:
        raise Exception("""
        No target resolution was chosen!!! You must either use the --desired_magnification or --desired_mpp options. 
        If you do not know the conversion from magnification to mpp or vice versa, it will be calculated for you. 
        """)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    # raise error if invalid input path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file '{input_path}' does not exist.")

    # making output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # making encoding directory only if encoding
    if not os.path.exists(os.path.join(output_path, "encoded")) and args.encode:
        os.makedirs(os.path.join(output_path, "encoded"))
    encoder_path = os.path.join(output_path, "encoded")

    # to handle if given directory of svs files (like getting it from the TCGA directory)
    if os.path.isdir(input_path):
        move_svs_files(input_path)

    # create a csv with information about the samples available
    patient_path = patient_csv(input_path, output_path)
    patients = pd.read_csv(patient_path)

    # if no saving, go direct to feature extraction
    if args.output_format is not None:
        for i, row in tqdm.tqdm(patients.iterrows(), total=len(patients)):
            print("---------------------------------")
            print(f"Working on: {row['Patient ID']}")
            if not os.path.isfile(os.path.join(output_path, row["Patient ID"], row["Patient ID"] + ".csv")):
                results = preprocessing(row["Original Slide Path"], row["Patient ID"], args)
                Reports.Reports([results[0]], [results[1]], output_path)

    # encode after all patients have been preprocessed
    encoding_times = []
    if args.encode:
        print("encoding")
        for i, row in tqdm.tqdm(patients.iterrows(), total=len(patients)):
            patient_id = row["Patient ID"]

            # Option 1: saving with h5
            if args.output_format == "h5":
                tile_dataset = TileEncoding_h5(os.path.join(output_path, row["Patient ID"],row["Patient ID"]+ ".h5"))
            # Option 2: saving with png
            elif args.output_format == "png":
                tile_dataset = TilePreprocessing_png(os.path.join(output_path, row["Patient ID"],row["Patient ID"]+ ".csv"))
            # # Option 3: not saving, only feature extraction!!!
            else:
                tile_dataset = TilePreprocessing_nosaving(row["Original Slide Path"], size = args.desired_size,
                                                          magnification = args.desired_magnification,
                                                          normalizer = "mackenko" if args.normalize_staining is not False else None,
                                                          remove_blurry = args.remove_blurry_tiles, blur_threshold = args.blur_threshold,
                                                          num_augmentations= args.augmentations)
            if not os.path.isfile(os.path.join(encoder_path, str(patient_id) + ".h5")):
                start_cpu_time = time.process_time()
                start_user_time = time.time()
                SlideEncoding.encode_tiles(patient_id, tile_dataset, encoder_path, device, high_qual=args.extract_high_quality,
                                           batch_size=args.batch_size, number_of_augmentation=args.augmentations,
                                           encoder_model=args.encoder_model, token=args.token)
                encoding_times.append((patient_id, time.process_time() - start_cpu_time, time.time() - start_user_time))
            else:
                encoding_times.append((patient_id, -1, -1))

        patient_files_encoded(patient_path)

    report_instance = Reports.Reports([[]], [[]], output_path)
    report_instance.summary_report_update_encoding(encoding_times)
    if args.min_tiles > 0:
        # filter patient_csv depending on amount of tiles
        filter_patients(patients, os.path.join(args.output_path, "SummaryReport.csv"), args)


if __name__ == "__main__":
    main()
