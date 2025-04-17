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
import h5py
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
# classes/functions

import Reports, SlideEncoding
from TileNormalization import normalizeStaining, normalizeStaining_torch
from TileQualityFilters import LaplaceFilter, plot_distribution
from TissueMask import is_tissue, get_region_mask, TissueMask
from tiling.TileIterator import TileIterator
from tiling.TileDataset import TileDataset
from VisulizationUtils import reconstruct_slide
from config import get_args_from_config


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
SlidePreprocessing.py

Author: Lorenzo Olmo Marchal
Created: 3/5/2024
Last Updated:  2/4/2025

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


def preprocess_patient(row, device, encoder_path, args):
    result = row["Preprocessing Path"]
    original = row["Original Slide Path"]
    patient_id = row["Patient ID"]
    s, e = preprocessing(original, result, patient_id, device, encoder_path, args)
    print(f"done with patient {patient_id}")
    return s, e


def extract_diagnosis(ID):
    ID = ID.split("-")
    diagnosis = ID[3]
    tumor_identification = ''.join([char for char in diagnosis if char.isdigit()])
    if int(tumor_identification) >= 11:
        return 0
    else:
        return 1


def patient_files_encoded(patient_files_path):
    df = pd.read_csv(patient_files_path)
    df["Encoded Path"] = [""]*len(df)
    df["Encoded Path"] = df["Encoded Path"].astype("object")
    for i, row in df.iterrows():
        encoded_path = os.path.join(os.path.dirname(row["Preprocessing Path"]), "encoded", row["Patient ID"] + ".h5")
        df.loc[i, "Encoded Path"] = str(encoded_path)
    df.to_csv(patient_files_path)


# ----------------------- PREPROCESSING -----------------------------------
# General helper methods

# getting best size for the
def best_size(desired_mag, natural_mag, desired_size) -> int:
    new_size = natural_mag / desired_mag
    return int(desired_size * new_size)


def get_valid_coordinates(width, height, overlap, mask, size, scale, threshold):
    # example overlap of 2 and size of 256 = 128 stride
    if overlap > 1:
        stride = size // overlap
    else:
        stride = size
    x_coords = np.arange(0, width - size + stride, stride)
    y_coords = np.arange(0, height - size + stride, stride)
    coordinates = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    total_cords = coordinates.shape[0]
    # valid coordinates according to mask
    valid_coords = [coord for coord in coordinates if
                    is_tissue(get_region_mask(mask, scale, coord, (size, size)), threshold=threshold)]
    return total_cords, len(valid_coords), valid_coords, coordinates

def save_tiles(coord, norm_tile, output_dir, patient_id, desired_size, desired_mag):
    try:
        file_path = os.path.join(
            output_dir, f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png"
        )
        cv2.imwrite(file_path, norm_tile[:, :, ::-1])  # Convert to BGR for saving

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
      
        norm_tile = norm_tile
        blurry, var = LaplaceFilter(norm_tile,var_threshold = threshold)
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
        print(norm_tile.shape)
        print(f"Error in QC or saving tile: {e}")
        return None


def save_tiles_to_h5(queue, h5_file,var, remove_blurry, threshold):
    tiles_list = []
    coords_list = []

    while True:
        item = queue.get()
        if item is None:
            break

        coord, norm_tile = item
        if remove_blurry:
            blur, var_ = LaplaceFilter(norm_tile, threshold)
            if blur:
                var.append(var_)
                continue
            var.append(var_)
        tiles_list.append(norm_tile)
        coords_list.append(coord)
        if len(tiles_list) >= 64:
            with h5py.File(h5_file, 'a') as f:
                tiles_dataset = f.require_dataset('tiles', shape=(len(tiles_list), *tiles_list[0].shape), dtype=tiles_list[0].dtype)
                coords_dataset = f.require_dataset('coords', shape=(len(coords_list), 2), dtype=coords_list[0].dtype)
                start_idx = tiles_dataset.shape[0]
                tiles_dataset[start_idx:start_idx+len(tiles_list)] = tiles_list
                coords_dataset[start_idx:start_idx+len(coords_list)] = coords_list
            tiles_list.clear()
            coords_list.clear()
    if tiles_list:
        with h5py.File(h5_file, 'a') as f:
            tiles_dataset = f.require_dataset('tiles', shape=(len(tiles_list), *tiles_list[0].shape), dtype=tiles_list[0].dtype)
            coords_dataset = f.require_dataset('coords', shape=(len(coords_list), 2), dtype=coords_list[0].dtype)
            start_idx = tiles_dataset.shape[0]
            tiles_dataset[start_idx:start_idx+len(tiles_list)] = tiles_list
            coords_dataset[start_idx:start_idx+len(coords_list)] = coords_list

"""
- needs path to slide
- patient_id 
- device 
- args -> from there can extract most things
"""


def preprocessing(path, patient_id, args):
    #  normalizing, don't need gpu
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
    except:  # noqa: F821
        error.append((patient_id, path, "Error Opening file", "Slide Opening"))
        summary = summary_()
        summary.append("Error")

        return summary, error

        # check 2: natural mag is not lesser than requested mag
    natural_magnification = int(slide.properties.get("openslide.objective-power", 40))
    desired_magnification = args.desired_magnification
    if natural_magnification < desired_magnification:
        error.append((patient_id, path, "Desired magnification is higher than natural magnification",
                      "Magnification Sanity Check"))
        summary = summary_()
        summary.append("Error")
        return summary, error

        # Step 1: Mask
    sample_path = os.path.join(args.output_path, patient_id)
    start_mask_user = time.time()
    start_mask_cpu = time.process_time()
    try:
        mask_ = TissueMask(slide, result_path=sample_path, blue_pen_thresh=args.blue_pen_check,
                           red_pen_thresh=args.red_pen_check)
        mask, scale = mask_.get_mask_attributes()
    except Exception as e:
        error.append((patient_id, path, e, "Tissue Mask"))
        summary = summary_()
        summary.append("Error")
        return summary, error

    time_mask_cpu = time.process_time() - start_mask_cpu
    time_mask = time.time() - start_mask_user

    # Step 2: Valid coordinates according to tissue mask
    desired_size = args.desired_size
    adjusted_size = best_size(desired_magnification, natural_magnification, desired_size)
    overlap = args.overlap

    start_time_coordinates_user = time.time()
    start_time_coordinates_cpu = time.process_time()
    w, h = slide.dimensions
    all_coords, valid_coordinates, coordinates,all_coords_  = get_valid_coordinates(w, h, overlap, mask, adjusted_size, scale,
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
    tiles_dir = os.path.join(sample_path, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    # Step 3: get tiles (separated into 2 different processes depending if available gpu or not)

    if device == "cuda":
        # initialize
        manager = multiprocessing.Manager()
        metadata_list = manager.list()
        save_queue = multiprocessing.Queue()
        tile_iterator = TileDataset(
            slide, coordinates=coordinates, mask=mask,
            size=desired_size, magnification=desired_magnification, 
            adjusted_size=adjusted_size, overlap=overlap
        )
        def save_worker(save_queue, output_dir, patient_id, desired_size, desired_mag, metadata_list, blur_threshold=None,):

            while True:
                item = save_queue.get()
                if item is None:
                    break

                coord, norm_tile = item
                try:
                    if blur_threshold is not None:
                        metadata = save_tiles_QC(coord, norm_tile, output_dir, patient_id,
                                                 desired_size, desired_mag, blur_threshold)
                    else:
                        metadata = save_tiles(coord, norm_tile, output_dir, patient_id, desired_size, desired_mag)

                    if metadata:
                        metadata_list.append(metadata)
                except Exception as e:
                    print(f"Error saving tile {coord}: {e}")

        def batch_norm(loader, queue, max_streams = 16):
            streams = [torch.cuda.Stream() for _ in range(max_streams)]
            for tiles, coords, _ in loader:
                tiles = tiles.to("cuda", non_blocking = True)
                chunk_size = min(max_streams, tiles.size(0))
                for i in range(0, tiles.size(0), chunk_size):
                    for j in range(i, min(i + chunk_size, tiles.size(0))):
                        stream = streams[j % max_streams]
                        result = normalizeStaining_torch(tiles[j])
                        if result is not None:
                            queue.put((coords[j].numpy(), result.cpu().numpy()))
            torch.cuda.synchronize()
            queue.put(None)

        blur_threshold = args.blur_threshold if args.remove_blurry_tiles else None
        if args.output_format == "png":
            num_gpu_workers = 3
            num_saving_workers = (max_workers - num_gpu_workers) 
            tile_loader = DataLoader(tile_iterator,shuffle = False,num_workers = num_gpu_workers,
                                     pin_memory = True, batch_size = args.batch_size)
            gpu_thread = threading.Thread(target=batch_norm, args=(tile_loader, save_queue, 16))
            gpu_thread.start()
            save_processes = []
            for _ in range(num_saving_workers):
                p = multiprocessing.Process(
                    target=save_worker,
                    args=(save_queue, os.path.join(sample_path, "tiles"), patient_id, desired_size, desired_magnification,metadata_list,blur_threshold)
                )
                p.start()
                save_processes.append(p)
            gpu_thread.join()
            for _ in range(num_saving_workers):
                save_queue.put(None)
            for p in save_processes:
                p.join()

     
            try:
                if isinstance(metadata_list[0], tuple):
                    metadata_list, vars = zip(*metadata_list)
                final  = [item for item in metadata_list if item is not None]
                vars = [item for item in vars if item is not None]


                df_tiles = pd.DataFrame(list(final))
                df_tiles["original_mag"] = natural_magnification
                df_tiles["scale"] = scale
                df_tiles.to_csv(tiles_path, index=False)
                blurry_tiles = len(metadata_list) if args.remove_blurry_tiles else None
            except :
                        error.append((patient_id, path, "No tiles were saved.", "Tiling"))
                        summary = summary_()
                        summary.append("Error")
                        return summary, error
        elif args.output_format == "h5":
            manager = multiprocessing.Manager()
            var_list = manager.list()
            save_queue = multiprocessing.Queue()
            num_gpu_workers = max(max_workers - 2, 1)
            tile_loader = DataLoader(
                tile_iterator,
                shuffle=False,
                num_workers=num_gpu_workers,
                pin_memory=True,
                batch_size=args.batch_size
            )
            gpu_thread = threading.Thread(target=batch_norm, args=(tile_loader, save_queue, 16))
            gpu_thread.start()
            saving_process = multiprocessing.Process(
                target=save_tiles_to_h5,
                args=(save_queue, f"{sample_path}/{patient_id}.h5",
                      var_list, args.blur_threshold, args.remove_blurry_tiles)
            )
            saving_process.start()
            gpu_thread.join()
            saving_process.join()
            try:
                with h5py.File(f"{sample_path}/{patient_id}.h5", "r") as f:
                    coords = f["coords"][:]
                vars = list(var_list)
                x, y = zip(*coords)
                df_tiles = pd.DataFrame({"x":x, "y":y})
                blurry_tiles = len(coords)
                if len(coords) == 0:
                    error.append((patient_id, path, "No tiles were saved.", "Tiling"))
                    summary = summary_()
                    summary.append("Error")
                    return summary, error

            except:
                error.append((patient_id, path, "No tiles were saved.", "Tiling"))
                summary = summary_()
                summary.append("Error")
                return summary, error

    else:

        manager = multiprocessing.Manager()
        metadata_list = manager.list()
        save_queue = multiprocessing.Queue()
        tile_iterator = TileDataset(
            slide, coordinates=coordinates, mask=mask,
            size=desired_size, magnification=desired_magnification,
            adjusted_size=adjusted_size, overlap=overlap, normalizer = "mac" if args.normalize_staining else None,
            remove_blurry= args.remove_blurry_tiles
        )
        tile_loader = DataLoader(tile_iterator, shuffle = False, num_workers = max_workers//2)
        def tile_loading(queue, data_loader, vars):
            local_vars = []
            for tiles, coords, var in data_loader:
                tiles = tiles.numpy()
                coords = coords.numpy()
                var = var.numpy()
                for i in range(len(tiles)):
                    if tiles[i] is not None:
                        queue.put((tiles[i], coords[i]))
                    if var[i] != -1:
                        local_vars.append(var[i])
            vars.extend(local_vars)
            for _ in range(max_workers):
                queue.put(None)

        def save_worker_cpu(save_queue,output_dir, patient_id, desired_size, desired_mag, metadata_list):
            while True:
                item = save_queue.get()
                if item is None:
                    break

                norm_tile, coord = item
                try:
                    metadata = save_tiles(coord, norm_tile, output_dir, patient_id, desired_size, desired_mag)

                    if metadata:
                        metadata_list.append(metadata)
                except Exception as e:
                    print(f"Error saving tile {coord}: {e}")

        # load tiles and put them in queue
        tile_thread = threading.Thread(target=tile_loading, args=(save_queue, tile_loader, metadata_list))
        tile_thread.start()

        save_processes = []
        for _ in range(max_workers//2):
            p = multiprocessing.Process(
                target=save_worker_cpu,
                args=(save_queue, os.path.join(sample_path, "tiles"), patient_id, desired_size, desired_magnification,metadata_list)
            )
            p.start()
            save_processes.append(p)
        tile_thread.join()
        for _ in range(max_workers//2):
            save_queue.put(None)
        for p in save_processes:
            p.join()
        try:
            if isinstance(metadata_list[0], tuple):
                metadata_list, vars = zip(*metadata_list)
            final  = [item for item in metadata_list if item is not None]
            vars = [item for item in vars if item is not None]


            df_tiles = pd.DataFrame(list(final))
            df_tiles["original_mag"] = natural_magnification
            df_tiles["scale"] = scale
            df_tiles.to_csv(tiles_path, index=False)
            blurry_tiles = len(metadata_list) if args.remove_blurry_tiles else None
        except :
            error.append((patient_id, path, "No tiles were saved.", "Tiling"))
            summary = summary_()
            summary.append("Error")
            return summary, error

    if args.reconstruct_slide:
        reconstruct_slide(mask_.applied, tile_iterator.coordinates,
                          all_coords_, mask_.SCALE,
                          tile_iterator.adjusted_size,
                          save_path= os.path.join(sample_path, "included_tiles.png"))
        if args.remove_blurry_tiles:
            valid_coords = np.array([[x,y] for x, y in zip(df_tiles.x.values, df_tiles.y.values)])
            reconstruct_slide(mask_.applied, valid_coords,
                              all_coords_, mask_.SCALE,
                              tile_iterator.adjusted_size,
                              save_path= os.path.join(sample_path, "included_tiles_after_QC.png"))

    time_patches = time.time() - start_time_patches_user
    time_patches_cpu = time.process_time() - start_time_patches_cpu
    summary = summary_()
    summary.append("Processed")

    # sanity check -> statistics for process do (1-2 examples)
    if args.normalize_staining:
        QC_path = os.path.join(sample_path, "QC_pipeline")
        os.makedirs(QC_path, exist_ok=True)
        # choose random coordinate
        non_blurry_coords = list(zip(df_tiles['x'], df_tiles['y']))
        random_coord = non_blurry_coords[random.randint(0, len(non_blurry_coords) - 1)]
        region = slide.read_region((random_coord[0], random_coord[1]), 0, (adjusted_size, adjusted_size)).convert(
            'RGB').resize(
            (desired_size, desired_size), Image.BILINEAR)
        region.save(os.path.join(QC_path, "original_non_blurry.png"))
        normalized_img = Image.fromarray(normalizeStaining(np.array(region)))
        normalized_img.save(os.path.join(QC_path, "normalized_non_blurry.png"))
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
            print("finished QC")

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

    # input/output
    parser.add_argument("--config_file", type=str, default = "None", help = "path to config file")

    # input/output
    parser.add_argument("-i", "--input_path", type=str, default = "/path/to/slide(s)")
    parser.add_argument("-o", "--output_path", type=str, default = "/output/path")
    # tile customization
    parser.add_argument("-s", "--desired_size", type=int, default=256,
                        help="Desired size of the tiles (default: %(default)s)")
    parser.add_argument("-m", "--desired_magnification", type=int, default=20,
                        help="Desired magnification level (default: %(default)s)")
    parser.add_argument("-ov", "--overlap", type=int, default=1,
                        help="Overlap between tiles (default: %(default)s)")
    parser.add_argument("--output_format", type =str, default= "png")


    # preprocessing processes customization
    parser.add_argument("-rb", "--remove_blurry_tiles", action="store_true",
                        help="lag to enable usage of the laplacian filter to remove blurry tiles")
    parser.add_argument("-n", "--normalize_staining", action="store_true",
                        help="Flag to enable normalization of tiles")
    parser.add_argument("-e", "--encode", action="store_true",
                        help="Flag to encode tiles and create associated .h5 file")
    parser.add_argument("--extract_high_quality", action="store_true",
                        help="extract high quality ")
    parser.add_argument("--reconstruct_slide", action="store_true",
                        help="reconstruct slide ")
    parser.add_argument("--augmentations", type=int, default=0,
                        help="augment data for training ")
    parser.add_argument("--encoder_model", default="resnet50",
                        help="current options: resnet50, mahmood-uni")
    parser.add_argument("--token", default = None, help= "required to download model weights from hugging face")
    

    # thresholds
    parser.add_argument("-th", "--tissue_threshold", type=float, default=0.7,
                        help="Threshold to consider a tile as Tissue(default: %(default)s)")
    parser.add_argument("-bh", "--blur_threshold", type=float, default=0.015,
                        help="Threshold for laplace filter variance (default: %(default)s)")
    parser.add_argument("--red_pen_check", type=float, default=0.4,
                        help="Sanity check for % of red pen detected. If above threshold, red_pen mask will be ignored(default: %(default)s)")
    parser.add_argument("--blue_pen_check", type=float, default=0.4,
                        help="Sanity check for % of blue pen detected,  If above threshold, blue_pen mask will be ignored(default: %(default)s)")
    parser.add_argument("--include_adipose_tissue", action = "store_true", help = "will include adipose tissue in mask")

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

    # # process samples, will be multiprocessing the instances.
    # results = []
    for i, row in tqdm.tqdm(patients.iterrows(), total=len(patients)):
        print("---------------------------------")
        print(f"Working on: {row['Patient ID']}")
        if not os.path.isfile(os.path.join(output_path, row["Patient ID"], row["Patient ID"] + ".csv")):
          
            results = preprocessing(row["Original Slide Path"], row["Patient ID"], args)

            Reports.Reports([results[0]], [results[1]], output_path)

    # encode after all patients have been preprocessed
    encoding_times = []
    if args.encode:
        for i, row in tqdm.tqdm(patients.iterrows(), total=len(patients)):
            patient_id = row["Patient ID"]
            path = os.path.join(output_path, patient_id, patient_id + ".csv")
            if not os.path.isfile(os.path.join(encoder_path, str(patient_id) + ".h5")) and os.path.isfile(path):
                start_cpu_time = time.process_time()
                start_user_time = time.time()
                SlideEncoding.encode_tiles(patient_id, path, encoder_path, device, high_qual=args.extract_high_quality,
                                           batch_size=args.batch_size, number_of_augmentation=args.augmentations, encoder_model = args.encoder_model, token = args.token )
                encoding_times.append((patient_id, time.process_time() - start_cpu_time, time.time() - start_user_time))
            else:
                encoding_times.append((patient_id, -1, -1))

        patient_files_encoded(patient_path)

    report_instance = Reports.Reports([[]], [[]], output_path)
    report_instance.summary_report_update_encoding(encoding_times)
    if args.min_tiles >0:
        # filter patient_csv depending on amount of tiles
        filter_patients(patients, os.path.join(args.output_path, "SummaryReport.csv"), args)


if __name__ == "__main__":
    main()
