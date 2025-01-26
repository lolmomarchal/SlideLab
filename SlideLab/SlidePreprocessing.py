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
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import random

# classes/functions
import Reports, SlideEncoding
from TileNormalization import normalizeStaining, normalizeStaining_torch
from TileQualityFilters import LaplaceFilter, plot_distribution
from TissueMask import is_tissue, get_region_mask, TissueMask
from tiling.TileIterator import TileIterator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
SlidePreprocessing.py

Author: Lorenzo Olmo Marchal
Created: 3/5/2024
Last Updated:  1/10/2025

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
    for i, row in df.iterrows():
        #         label = extract_diagnosis(row["Patient ID"])
        encoded_path = os.path.join(os.path.dirname(row["Preprocessing Path"]), "encoded", row["Patient ID"] + ".h5")
        #         df.loc[i, "target"] = label
        df.loc[i, "Encoded Path"] = encoded_path
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
    return total_cords, len(valid_coords), valid_coords


# general method for tiling slide   
def tile_slide_image(coord, desired_size, adjusted_size, patient_id, sample_path, slide, desired_mag):
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize(
        (desired_size, desired_size), Image.BILINEAR)
    image_path = os.path.join(sample_path,
                              f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png")
    tile.save(image_path)
    del tile
    return {"Patient_ID": patient_id, "x": coord[0], "y": coord[1], "tile_path": image_path,
            "original_size": adjusted_size, "desired_size": desired_size, "desired_magnification": desired_mag}


def tile_slide_normalize_image(coord, desired_size, adjusted_size, patient_id, sample_path, slide, desired_mag):
    # global slide
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize(
        (desired_size, desired_size), Image.BILINEAR)
    tile_np = normalizeStaining(np.array(tile))
    if tile_np is None:
        return None, 0
    image_path = os.path.join(sample_path,
                              f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png")
    cv2.imwrite(image_path, tile_np[:, :, ::-1])
    return {"Patient_ID": patient_id, "x": coord[0], "y": coord[1], "tile_path": image_path,
            "original_size": adjusted_size, "desired_size": desired_size, "desired_magnification": desired_mag}


def tile_slide_normalize_blurry_image(coord, desired_size, adjusted_size, patient_id, sample_path, slide, desired_mag):
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize(
        (desired_size, desired_size), Image.BILINEAR)
    tile_np = normalizeStaining(np.array(tile))
    if tile_np is None:
        return None, 0
    blurry, var = LaplaceFilter(tile_np)
    if not blurry:

        image_path = os.path.join(sample_path,
                                  f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png")
        cv2.imwrite(image_path, tile_np[:, :, ::-1])
        return {"Patient_ID": patient_id, "x": coord[0], "y": coord[1], "tile_path": image_path,
                "original_size": adjusted_size, "desired_size": desired_size, "desired_magnification": desired_mag}, var
    return None, var


def normalize_gpu(coord, tile):
    try:
        norm_tile = normalizeStaining_torch(
            torch.tensor(np.array(tile), dtype=torch.float32),
        )
        if norm_tile is None:
            return None
        return coord, norm_tile
    except Exception as e:
        print(f"Error in GPU task for tile {coord}: {e}")
        return None
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


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


def save_tiles_QC(coord, norm_tile, output_dir, patient_id, desired_size, desired_mag):
    try:
        norm_tile = norm_tile.numpy()
        blurry, var = LaplaceFilter(norm_tile)
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
    # if not normalizing, don't need gpu
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
        error.append((patient_id, path, "Desired magnification is higher than natural magnification"))
        summary = summary_()
        summary.append("Error")
        return summary, error

        # Step 1: Mask
    sample_path = os.path.join(args.output_path, patient_id)
    start_mask_user = time.time()
    start_mask_cpu = time.process_time()
    mask, scale = TissueMask(slide, result_path=sample_path).get_mask_attributes()
    time_mask_cpu = time.process_time() - start_mask_cpu
    time_mask = time.time() - start_mask_user

    # Step 2: Valid coordinates according to tissue mask
    desired_size = args.desired_size
    adjusted_size = best_size(desired_magnification, natural_magnification, desired_size)
    overlap = args.overlap

    start_time_coordinates_user = time.time()
    start_time_coordinates_cpu = time.process_time()
    w, h = slide.dimensions
    all_coords, valid_coordinates, coordinates = get_valid_coordinates(w, h, overlap, mask, adjusted_size, scale,
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
        # create tile iterator
        metadata_list = []
        tile_iterator = TileIterator(slide, coordinates=coordinates, mask=mask, normalizer=None, size=desired_size,
                                     magnification=desired_magnification, adjusted_size=adjusted_size, overlap=overlap)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_gpu_workers) as gpu_executor, concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as cpu_executor:
            gpu_futures = []
            cpu_futures = []
            for tile, coord in tile_iterator:
                gpu_futures.append(
                    gpu_executor.submit(normalize_gpu, coord, tile)
                )
            if args.normalize_staining and not args.remove_blurry_tiles:
                for future in concurrent.futures.as_completed(gpu_futures):
                    result = future.result()
                    if result is not None:
                        coord, norm_tile = result
                        cpu_futures.append(
                            cpu_executor.submit(save_tiles, coord, norm_tile, tiles_dir, patient_id, desired_size,
                                                desired_magnification))
            else:
                for future in concurrent.futures.as_completed(gpu_futures):
                    result = future.result()
                    if result is not None:
                        coord, norm_tile = result
                        cpu_futures.append(
                            cpu_executor.submit(save_tiles_QC, coord, norm_tile, tiles_dir, patient_id, desired_size,
                                                desired_magnification))
            for future in concurrent.futures.as_completed(cpu_futures):
                # check if removing blurry
                if args.normalize_staining and args.remove_blurry_tiles:
                    result, var = future.result()
                    vars.append(var)
                elif args.normalize_staining and not args.remove_blurry_tiles:
                    result = future.result()
                if result is not None:
                    metadata_list.append(result)

        df_tiles = pd.DataFrame(metadata_list)
        df_tiles["original_mag"] = natural_magnification
        df_tiles["scale"] = scale
        df_tiles.to_csv(tiles_path, index=False)
        blurry_tiles = len(metadata_list) if args.remove_blurry_tiles else None



    else:
        mult_args = [(coord, desired_size, adjusted_size, patient_id, tiles_dir, slide, desired_magnification)
                     for coord in coordinates]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if not args.normalize_staining:
                results = list(executor.map(lambda p: tile_slide_image(*p), mult_args))
            elif args.normalize_staining and not args.remove_blurry_tiles:
                results = list(executor.map(lambda p: tile_slide_normalize_image(*p), mult_args))
            else:
                results_ = list(executor.map(lambda p: tile_slide_normalize_blurry_image(*p), mult_args))
                # need to rewrite -> should be a list of tuples
                results = []
                for item in results_:
                    if item[0] is not None:
                        results.append(item[0])
                    vars.append(item[1])

        time_patches = time.time() - start_time_patches_user
        time_patches_cpu = time.process_time() - start_time_patches_cpu
        results_ = [result for result in results if result]

        df_tiles = pd.DataFrame(results_)
        df_tiles["original_mag"] = natural_magnification
        df_tiles["scale"] = scale
        df_tiles.to_csv(tiles_path, index=False)
        # how many removed
        blurry_tiles = len(results_) if args.remove_blurry_tiles else None
    summary = summary_()
    summary.append("Processed")

    # sanity check -> statistics for process do (1-2 examples)
    if args.normalize_staining:
        QC_path = os.path.join(sample_path, "QC_pipeline")
        os.makedirs(QC_path, exist_ok=True)
        # choose random coordinate
        non_blurry_coords = list(zip(df_tiles['x'], df_tiles['y']))
        random_coord = non_blurry_coords[random.randint(0, len(non_blurry_coords))]
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
                while True or i < 5:
                    random_coord = different_coords[random.randint(0, len(different_coords)-1)]
                    region = slide.read_region((random_coord[0], random_coord[1]), 0, (adjusted_size, adjusted_size)).convert(
                        'RGB').resize(
                        (desired_size, desired_size), Image.BILINEAR)
                    normalized_blurry = normalizeStaining(np.array(region))
                    if normalized_blurry is not None:
                        region.save(os.path.join(QC_path, "original_blurry.png"))
                        normalized_img = Image.fromarray(normalized_blurry)
                        normalized_img.save(os.path.join(QC_path, "normalized_blurry.png"))
                        break
                    i += 1
    return summary, error


# ----------------------- FILE HANDLING -----------------------------------


def move_svs_files(main_directory):
    """
    Used to handle directories where WSI files are stored in subdirectories (like downloading it from portal.gdc)
    :param main_directory: directory where the WSI are stored (from the allowed file extensions from OpenSlide)
    """
    for patient_directory in os.listdir(main_directory):
        patient_path = os.path.join(main_directory, patient_directory)

        if os.path.isdir(patient_path):
            # look for valid files
            for file_name in os.listdir(patient_path):
                if file_name.endswith((
                        ".svs", ".tif", ".tiff", ".dcm",
                        ".ndpi", ".vms", ".vmu", ".scn",
                        ".mrxs", ".svslide", ".bif")):
                    svs_file_path = os.path.join(patient_path, file_name)

                    # move files to main directory
                    shutil.move(svs_file_path, main_directory)


def patient_csv(input_path, results_path):
    """
    This method creates a csv file with important information about the patients being handled. This includes the sample ID,
    where the original WSI is located and where the preprocessing results are located.
    :param input_path original directory
    :param results_path final directory
    :returns csv file path with the preprocessing information
    """
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
    parser = argparse.ArgumentParser(description="WSI Preprocessing")

    # input/output
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-o", "--output_path", type=str)

    # tile customization
    parser.add_argument("-s", "--desired_size", type=int, default=256,
                        help="Desired size of the tiles (default: %(default)s)")
    parser.add_argument("-m", "--desired_magnification", type=int, default=20,
                        help="Desired magnification level (default: %(default)s)")
    parser.add_argument("-ov", "--overlap", type=int, default=1,
                        help="Overlap between tiles (default: %(default)s)")

    # preprocessing processes customization
    parser.add_argument("-rb", "--remove_blurry_tiles", action="store_true",
                        help="lag to enable usage of the laplacian filter to remove blurry tiles")
    parser.add_argument("-n", "--normalize_staining", action="store_true",
                        help="Flag to enable normalization of tiles")
    parser.add_argument("-e", "--encode", action="store_true",
                        help="Flag to encode tiles and create associated .h5 file")
    parser.add_argument("--extract_high_quality", action="store_true",
                        help="extract high quality ")

    # thresholds
    parser.add_argument("-th", "--tissue_threshold", type=float, default=0.7,
                        help="Threshold to consider a tile as Tissue(default: %(default)s)")
    parser.add_argument("-bh", "--blur_threshold", type=float, default=0.015,
                        help="Threshold for laplace filter variance (default: %(default)s)")

    # for devices + multithreading
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu_processes", type=int, default=1)
    parser.add_argument("--cpu_processes", type=int, default=os.cpu_count())

    return parser.parse_args()


def main():
    args = parse_args()

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
        print(f"Working on: {row['Patient ID']}")
        if not os.path.isfile(os.path.join(output_path, row["Patient ID"], row["Patient ID"] + ".csv")):
            results = preprocessing(row["Original Slide Path"], row["Patient ID"], args)
            Reports.Reports([results[0]], [results[1]], output_path)
            # results.append(preprocessing(row["Original Slide Path"], row["Patient ID"], args))
    # encode after all patients have been preprocessed
    if args.encode:
        for i, row in tqdm.tqdm(patients.iterrows(), total=len(patients)):
            patient_id = row["Patient ID"]
            path = os.path.join(output_path, patient_id, patient_id + ".csv")
            if not os.path.isfile(os.path.join(encoder_path, str(patient_id) + ".h5")):
                SlideEncoding.encode_tiles(patient_id, path, encoder_path, device)

        patient_files_encoded(patient_path)

    # # write summary and error report
    # global summary, errors
    # for res in results:
    #     summary.extend(res[0])
    #     errors.extend(res[1])
    # Reports.Reports(summary, errors, output_path)


if __name__ == "__main__":
    main()
