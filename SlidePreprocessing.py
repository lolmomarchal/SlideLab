# outside imports
import argparse
import csv
import gc
import shutil
import multiprocessing
import os
import openslide
import numpy as np
import pandas as pd
import torch
from PIL import Image
import h5py
import time
# import line_profiler
# from line_profiler import profile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import numba
# classes/functions
import Reports
import SlideEncoding
import TileNormalization
import TileQualityFilters
from TissueMask import is_tissue, get_region_mask
import VisulizationUtils
from TissueMask import TissueMask
from TissueSlide import TissueSlide



os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
SlidePreprocessing.py

Author: Lorenzo Olmo Marchal
Created: March 5, 2024
Last Updated:  Oct 14,  2024

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
    new_size = natural_mag/desired_mag
    return int(desired_size * new_size)

def get_valid_coordinates(width, height, overlap, mask, size, scale ,threshold):
    # example overlap of 2 and size of 256 = 128 stride
    if overlap != 1:
        stride = size/overlap
    else:
        stride = size
    x_coords = np.arange(0, width - size + stride , stride)
    y_coords = np.arange(0, height - size + stride , stride)
    coordinates = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    total_cords = coordinates.shape[0]
    # valid coordinates according to mask
    valid_coords = [coord for coord in coordinates if is_tissue(get_region_mask(mask, scale, coord, (size, size)), threshold = threshold)]
    return total_cords, len(valid_coords), valid_coords

# num_workers = cpu_count()
# if num_workers>4:
#     num_workers = 4
# pool = Pool(num_workers)
# iterable = [(coord, mask, scale, size, threshold) for coord in coordinates]
# # valid coordinates according to mask
# results = pool.starmap(process_candidate_coordinates, iterable)
# pool.close()
# valid_coords = np.array([result for result in results if result])
# return total_cords, len(valid_coords), valid_coords
# def process_candidate_coordinates(coord, mask, scale, size, threshold):
#     return is_tissue(get_region_mask(mask, scale, coord, (size, size)), threshold = threshold)
# H5 file methods

# general method for tiling slide
def tile_slide_image(coord, desired_size, adjusted_size,patient_id, sample_path, slide, desired_mag):
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize((desired_size, desired_size), Image.BILINEAR)
    image_path = os.path.join(sample_path, f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png")
    tile.save(image_path)
    return {"Patient_ID": patient_id, "x":coord[0], "y": coord[1], "tile_path": image_path,"original_size":adjusted_size, "desired_size": desired_size, "desired_magnification": desired_mag}

def tile_slide_normalize_image(coord, desired_size, adjusted_size,patient_id, sample_path, slide, desired_mag):
    #global slide
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize((desired_size, desired_size), Image.BILINEAR)
    tile_np = TileNormalization.normalizeStaining(np.array(tile))
    if tile_np is None:
        return None
    image_path = os.path.join(sample_path, f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png")
    Image.fromarray(tile_np).save(image_path)
    return {"Patient_ID": patient_id, "x":coord[0], "y": coord[1], "tile_path": image_path,"original_size":adjusted_size, "desired_size": desired_size, "desired_magnification": desired_mag}

def tile_slide_normalize_blurry_image(coord, desired_size, adjusted_size,patient_id, sample_path, slide, desired_mag):
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize((desired_size, desired_size), Image.BILINEAR)
    tile_np = TileNormalization.normalizeStaining(np.array(tile))
    if tile_np is None:
        return None
    if not TileQualityFilters.LaplaceFilter(tile_np):
        image_path = os.path.join(sample_path, f"{patient_id}_{coord[0]}_{coord[1]}_size_{desired_size}_mag_{desired_mag}.png")
        Image.fromarray(tile_np).save(image_path)
        return {"Patient_ID": patient_id, "x":coord[0], "y": coord[1], "tile_path": image_path,"original_size":adjusted_size, "desired_size": desired_size, "desired_magnification": desired_mag}
    return None
# saving images

# saving images


def tile_normalize_slide_h5(args):
    #global slide
    coord, desired_size, adjusted_size, slide = args
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize((desired_size, desired_size), Image.BILINEAR)
    tile_np = TileNormalization.normalizeStaining(np.array(tile))
    if tile_np is None:
        return (None, tile_np)
    return  (tile_np, coord)
def tile_normalize_blurry_h5(args):
    coord, desired_size, adjusted_size, slide = args
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize((desired_size, desired_size), Image.BILINEAR)
    tile_np = TileNormalization.normalizeStaining(np.array(tile))
    if tile_np is None:
        return (None, coord, None)
    if TileQualityFilters.LaplaceFilter(tile_np):
        return (tile_np, coord, True)
    return (tile_np, coord, None)

def tile_slide_h5(args):
    #global slide
    coord, desired_size, adjusted_size, slide= args
    tile = slide.read_region((coord[0], coord[1]), 0, (adjusted_size, adjusted_size)).convert('RGB').resize((desired_size, desired_size), Image.BILINEAR)
    tile_np = np.array(tile)
    return (tile_np, coord)

# Main preprocessing script
"""
- needs path to slide
- patient_id 
- device 
- args -> from there can extract most things
"""
# @profile
def preprocessing(path, patient_id, device,args):

    print(f"working on {patient_id}")
    error = []
    summary = []
    total_tiles = None
    valid_tiles = None
    blurry_tiles = None
    time_mask = None
    time_opening = None
    time_coordinates = None
    start = time.time()
   # initialize_slide(path)
    try:
     slide = openslide.OpenSlide(path)
    except:
        slide = None
    time_opening = time.time()-start
    if slide is not None:

        # if the desired magnification is greater than the slide's natural mag. cannot process
        # you cannot go from 20x to 40x
        natural_magnification =  int(slide.properties.get("openslide.objective-power"))
        desired_magnification = args.desired_magnification
        # if the natural magnification is greater or equal to the desired magnification that is ok
        # ex: 40 -> 20 or 20 -> 20
        if natural_magnification >= desired_magnification:

            # need to create corresponding directory for patient
            sample_path = os.path.join(args.output_path, patient_id)
            # get mask
            start_mask = time.time()
            mask, scale = TissueMask(slide, result_path= sample_path).get_mask_attributes()
            time_mask = time.time()-start_mask
            w, h = slide.dimensions
            # first need to get the size we want
            desired_size = args.desired_size
            adjusted_size = best_size(desired_magnification, natural_magnification, desired_size)
            overlap = args.overlap
            # getting valid coordinates according to TissueMask. Adding threshold
            all_coords, valid_coordinates, coordinates = get_valid_coordinates(w, h, overlap, mask, adjusted_size,scale, threshold = args.tissue_threshold)
            time_coordinates = time.time()-time_mask
            total_tiles = all_coords
            valid_tiles = valid_coordinates

            max_workers = os.cpu_count()
            if args.save_tiles:
                tiles_path = os.path.join(sample_path, patient_id + ".csv")
                tiles_dir = os.path.join(sample_path, "tiles")
                os.makedirs(tiles_dir, exist_ok=True)
                mult_args = [(coord, desired_size, adjusted_size,patient_id, tiles_dir, slide, desired_magnification) for coord in coordinates]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    if not args.normalize_staining:
                        results = list(executor.map(lambda p: tile_slide_image(*p), mult_args))
                    elif args.normalize_staining and not args.remove_blurry_tiles:
                        results = list(executor.map(lambda p: tile_slide_normalize_image(*p), mult_args))
                    else:
                        results = list(executor.map(lambda p: tile_slide_normalize_blurry_image(*p), mult_args))
                results =  [result for result in results if result]
                df_tiles = pd.DataFrame(results)
                df_tiles["scale"] = scale
                df_tiles["original_mag"] = natural_magnification
                df_tiles.to_csv(os.path.join(tiles_path))

            else:
                # creating h5 file + associated datasets
                # tiles_path = os.path.join(sample_path, patient_id + ".h5")
                # with h5py.File(tiles_path, 'w') as h5_file:
                #     # Creating datasets
                #     chunk_size_x = min(1024, valid_tiles)
                #     chunk_size_y = min(1024, valid_tiles)
                #
                #     tile_dataset = h5_file.create_dataset(
                #         'tiles', shape=(valid_tiles, desired_size, desired_size, 3), dtype=np.uint8,
                #         chunks=(1, desired_size, desired_size, 3), compression='gzip', compression_opts=1
                #     )
                #
                #     x_dataset = h5_file.create_dataset(
                #         'x', shape=(valid_tiles,), dtype=np.int32,
                #         chunks=(chunk_size_x,), compression='gzip', compression_opts=1
                #     )
                #
                #     y_dataset = h5_file.create_dataset(
                #         'y', shape=(valid_tiles,), dtype=np.int32,
                #         chunks=(chunk_size_y,), compression='gzip', compression_opts=1
                #     )
                #
                #     # Storing additional metadata as scalars
                #     h5_file.create_dataset('scale', data=scale, dtype=np.int32)
                #     h5_file.create_dataset('natural_mag', data=natural_magnification, dtype=np.int32)
                #     h5_file.create_dataset('desired_mag', data=desired_magnification, dtype=np.int32)
                #     h5_file.create_dataset('desired_size', data=desired_size, dtype=np.int32)
                #     h5_file.create_dataset('path_to_slide', data=np.string_(path))
                #     h5_file.create_dataset('Patient_ID', data=np.string_(patient_id))
                    mult_args = [(coord, desired_size, adjusted_size, slide) for coord in coordinates]
                    # batch_size = 512 # Adjust based on memory
                    # tiles_batch = []
                    # x_batch = []
                    # y_batch = []

                    i = 0
                    tile_function = tile_slide_h5 if not args.normalize_staining else tile_normalize_slide_h5
                    if args.normalize_staining and args.remove_blurry_tiles:
                        tile_function = tile_normalize_blurry_h5

                    # Configure the ThreadPoolExecutor with adjusted chunk size
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        results = (result for result in executor.map(tile_function, mult_args) if result[0] is not None)
                    tile_nps, x, y = [], [], []
                    for result in results:
                        tile_nps.append(result[0])
                        x.append(result[1][0])
                        y.append(result[1][1])
                    tiles_path = os.path.join(sample_path, patient_id + ".h5")
                    with h5py.File(tiles_path, 'w') as h5_file:
                        h5_file.create_dataset('scale', data=scale, dtype=np.int32)
                        h5_file.create_dataset('natural_mag', data=natural_magnification, dtype=np.int32)
                        h5_file.create_dataset('desired_mag', data=desired_magnification, dtype=np.int32)
                        h5_file.create_dataset('desired_size', data=desired_size, dtype=np.int32)
                        h5_file.create_dataset('path_to_slide', data=np.string_(path))
                        h5_file.create_dataset('Patient_ID', data=np.string_(patient_id))
                        h5_file.create_dataset('x', data=np.array(x))
                        h5_file.create_dataset('y', data=np.array(y))
                        h5_file.create_dataset('tiles', data=np.stack(tile_nps), chunks=True, compression="gzip", compression_opts=9)
                    del results
                    del x
                    del y
                    gc.collect()



                        # batch_size_accumulated = batch_size * 2  # Adjust as needed to minimize I/O
                        # tiles_batch = np.empty((batch_size_accumulated, desired_size, desired_size, 3), dtype=np.uint8)
                        # x_batch = np.empty((batch_size_accumulated,), dtype=int)
                        # y_batch = np.empty((batch_size_accumulated,), dtype=int)
                        #
                        #
                        # for idx, result in enumerate(executor.map(tile_function, mult_args, chunksize=20)):
                        #     tile_np, coord = result[:2]
                        #     if tile_np is not None:
                        #         index = idx % batch_size_accumulated
                        #         tiles_batch[index] = tile_np
                        #         x_batch[index] = coord[0]
                        #         y_batch[index] = coord[1]
                        #
                        #         # Write when reaching the accumulated batch size
                        #         if (index + 1) == batch_size_accumulated:
                        #             tile_dataset[i:i + batch_size_accumulated] = tiles_batch
                        #             x_dataset[i:i + batch_size_accumulated] = x_batch
                        #             y_dataset[i:i + batch_size_accumulated] = y_batch
                        #             i += batch_size_accumulated

            slide.close()
            gc.collect()
        else:
            error.append((patient_id, path, f"The slide's natural magnification is less than the desired magnification. {natural_magnification}x cannot be transformed into {desired_magnification}x", "Magnification Sanity Check"))

    else:
        error.append((patient_id, path, "OpenSlide had an error with opening the provided slide.", "Slide Opening"))

    end = time.time() - start
    summary.append((patient_id, path, total_tiles, valid_tiles, blurry_tiles, end, time_opening, time_mask, time_coordinates))
    print(summary)
    return summary, error
# ----------------------- FILE HANDLING -----------------------------------


def move_svs_files(main_directory):
    """
    Used to handle directories where WSI files are stored in subdirectories (like downloading it from portal.gdc)
    :param main_directory: directory where the WSI are stored (from the allowed file extensions from OpenSlide)
    """
    # Iterate through each patient directory
    for patient_directory in os.listdir(main_directory):
        patient_path = os.path.join(main_directory, patient_directory)

        if os.path.isdir(patient_path):
            # Look for SVS files in the patient directory
            for file_name in os.listdir(patient_path):
                if file_name.endswith((
                        ".svs", ".tif", ".tiff", ".dcm",
                        ".ndpi", ".vms", ".vmu", ".scn",
                        ".mrxs", ".svslide", ".bif")):
                    svs_file_path = os.path.join(patient_path, file_name)

                    # Move the SVS file to the main directory
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
    parser.add_argument("-i", "--input_path", type=str,
                        default=r"C:\Users\albao\Downloads\gdc_download_20240320_111546.230274\TCGA-S3-AA17-01A-01-TS1.E8CCA5B9-9FB4-4A1B-AE41-8745D4FAFD8B.svs",
                        help="Input path (default: %(default)s)")
    parser.add_argument("-o", "--output_path", type=str, default=r"C:\Users\albao\Masters\WSI_ok",
                        help="Result path (default: %(default)s)")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="Number of threads for multiprocessing (default: %(default)s)")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Number of workers for multiprocessing (default: %(default)s)")
    parser.add_argument("-s", "--desired_size", type=int, default=256,
                        help="Desired size of the tiles (default: %(default)s)")
    parser.add_argument("-tg", "--tile_graph", action="store_true",
                        help="Flag to enable graphing of tiles")
    parser.add_argument("-m", "--desired_magnification", type=int, default=20,
                        help="Desired magnification level (default: %(default)s)")
    parser.add_argument("-ov", "--overlap", type=int, default=1,
                        help="Overlap between tiles (default: %(default)s)")
    parser.add_argument("-th", "--tissue_threshold", type=float, default=0.7,
                        help="Threshold to consider a tile as Tissue(default: %(default)s)")
    parser.add_argument("-bh", "--blur_threshold", type=int, default=0.015,
                        help="Threshold for laplace filter variance (default: %(default)s)")
    parser.add_argument("-rb", "--remove_blurry_tiles", action="store_true",
                        help="lag to enable usage of the laplacian filter to remove blurry tiles")
    parser.add_argument("-n", "--normalize_staining", action="store_true",
                        help="Flag to enable normalization of tiles")
    parser.add_argument("--save_blurry_tiles", action="store_true",
                        help="Flag to save blurry tiles in an additional folder called 'out_focus_tiles'")
    parser.add_argument("-e", "--encode", action="store_true",
                        help="Flag to encode tiles and creae associated .h5 file")
    parser.add_argument("--save_original_tiles", action="store_true",
                        help="Flag to save the original, unnormalized tiles")
    parser.add_argument("--save_tiles", action = "store_true", help = "Flag to save the tiles in png form instead than in a h5 file. Warning this will be slower.")

    return parser.parse_args()


def main():
    args = parse_args()

    # obtain args
    # getting basic arguments for preprocessing
    input_path = args.input_path
    output_path = args.output_path
    processes = args.processes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    results = []
    for i, row in patients.iterrows():
        results.append(preprocessing(row["Original Slide Path"], row["Patient ID"], device,args))

    # after processing samples, obtain encodings. do the same thing where you multiprocess instances unless gpu is available

    # # multiprocessing of sample preprocessing
    # with multiprocessing.Pool(processes=processes, maxtasksperchild=1) as pool:
    #     results = pool.starmap(preprocess_patient,
    #                            [(row, device, encoder_path, args) for _, row in patients.iterrows()])
    # encode files
    if args.encode:
        for i, row in patients.iterrows():

            patient_id = row["Patient ID"]
            if args.save_tiles:
                    path = os.path.join(output_path, patient_id, patient_id +".csv")
            else:
                    path = os.path.join(output_path, patient_id, patient_id + ".h5")

            if not os.path.isfile(os.path.join(encoder_path, str(patient_id) + ".h5")):
                SlideEncoding.encode_tiles(patient_id, path, encoder_path, device)
    # # rewrite patient_files
    patient_files_encoded(patient_path)

    # write summary and error report
    global summary, errors
    for res in results:
        summary.extend(res[0])
        errors.extend(res[1])
    Reports.Reports(summary, errors, output_path)


if __name__ == "__main__":
    main()
