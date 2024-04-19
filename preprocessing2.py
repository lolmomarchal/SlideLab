import multiprocessing
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool
import normalization
import encoding


OPENSLIDE_PATH = r"C:\Users\albao\Downloads\openslide-win64-20231011\openslide-win64-20231011\bin"
import os
import psutil
""""
HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
"""

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import histomicstk as htk
from PIL import Image
import cv2
import shutil
import matplotlib.pyplot as plt
import gc
import torch
import csv
from torchvision import transforms
import torchstain
from tiatoolbox.wsicore.wsireader import WSIReader
import masking

"""
preprocessing.py

Author: Lorenzo Olmo Marchal
Created: March 12, 2024
Last Updated:  April 17, 2024

Description:
This script automates the preprocessing and normalization of Whole Slide Images (WSI) in digital pathology. 
It extracts tiles from WSI files and applies color normalization techniques to enhance image quality and consistency.

Input:
- slide directory path
- slide directory output path

Output:
Processed tiles are saved in the output directory. Each tile is accompanied by metadata, including its origin within the WSI file.

Future Directions:
- Integration of machine learning algorithms for automated tile selection and quality control
- Support for parallel processing on distributed computing platforms for handling large-scale WSI datasets efficiently
"""



def best_size(slide):
    natural_mag  = slide.info.objective_power
    if natural_mag == 40:
        return 512
    else:
        return 256
def whole_slide_image(slide_path):
    try:
        ts = WSIReader.open(input_img = slide_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        ts = None
    return ts
def tiling(ts, result_path, mask, overlap=False, stride=512):
    # Make tile dir
    tiles = os.path.join(result_path, "tiles")
    columns = ['patient_id', 'x', 'y', 'magnification', 'path_to_slide']
    df_list = []
    os.makedirs(os.path.join(result_path, "tiles"), exist_ok=True)

    size = best_size(ts)
    if overlap != False:
        stride = stride
    else:
        stride = size
    mag = 20
    w,h = ts.info.slide_dimensions
    for x in range(0, w - size + 1, stride):
        for y in range(0, h - size + 1, stride):
            if x <= w - size and y <= h - size:
                tissue_rgb= ts.read_rect(location = (x,y), size = (size, size), resolution = mag, units = "power")
                # get mask of tile
                tile_mask = masking.get_region_mask(mask, (x, y), (size, size))
                istissue = masking.is_tissue(tile_mask)
                if  istissue == True:
                    tile = Image.fromarray(tissue_rgb)
                    tile.save(os.path.join(tiles, f"tile_w{x}_h{y}_mag{mag}.png"))
                    df_list.append({
                        'patient_id': "result1",
                        'x': x,
                        'y': y,
                        'magnification': mag,
                        'path_to_slide': os.path.join(tiles, f"tile_w{x}_h{y}_mag{mag}.png")
                    })
    df = pd.DataFrame(df_list, columns=columns)
    df.to_csv(os.path.join(result_path, "tile_information.csv"), index=False)
    del df
    gc.collect()

def mask_tiles(tile_information, result_path):
    tiles = pd.read_csv(tile_information)
    path = os.path.join(result_path, "masked_tiles")
    df_list = []
    columns = ['patient_id', 'x', 'y', 'magnification', 'path_to_slide']
    size = 512
    os.makedirs(os.path.join(result_path, "masked_tiles"), exist_ok=True)
    for i, row in tiles.iterrows():
        path_to_tile = row["path_to_slide"]
        mag = row["magnification"]
        y = row["y"]
        x = row["x"]

        #clean up tile
        tile = np.array(Image.open(path_to_tile))
        tile = masking.clean_up_tile(tile)
        tile = Image.fromarray(tile)
        tile.save(os.path.join(path, f"tile_w{x}_h{y}_mag{mag}.png"))
        df_list.append({
            'patient_id': "result1",
            'x': x,
            'y': y,
            'magnification': mag,
            'path_to_slide': os.path.join(path, f"tile_w{x}_h{y}_mag{mag}.png")
        })
    df = pd.DataFrame(df_list, columns=columns)
    df.to_csv(os.path.join(result_path, "masked_tile_information.csv"), index=False)
    del df
    gc.collect()

def normalize_tiles(tile_information, result_path):
    tiles = pd.read_csv(tile_information)
    path = os.path.join(result_path, "normalized_tiles")
    df_list = []
    columns = ['patient_id', 'x', 'y', 'magnification', 'path_to_slide']
    os.makedirs(os.path.join(result_path, "normalized_tiles"), exist_ok=True)
    for i, row in tiles.iterrows():
        path_to_tile = row["path_to_slide"]
        mag = row["magnification"]

        y = row["y"]
        x = row["x"]
        try:
            tile = np.array(Image.open(path_to_tile))
            if tile is not None:
                tile = normalization.normalizeStaining(tile)
                tile = Image.fromarray(tile)
                tile.save(os.path.join(path, f"tile_w{x}_h{y}_mag{mag}.png"))
                df_list.append({
                    'patient_id': "result1",
                    'x': x,
                    'y': y,
                    'magnification': mag,
                    'path_to_slide': os.path.join(path, f"tile_w{x}_h{y}_mag{mag}.png")
                })

            else:
                print("Error: Input tile is None.")

        except Exception as e:
            print(f"An error occurred: {e}")
    df = pd.DataFrame(df_list, columns=columns)
    df.to_csv(os.path.join(result_path, "normalized_tile_information2.csv"), index=False)
    del df
    gc.collect()

def move_svs_files(main_directory, results_path):
    # Create a CSV file to store patient ID and SVS file paths

    # Iterate through each patient directory
    for patient_directory in os.listdir(main_directory):
        patient_path = os.path.join(main_directory, patient_directory)

        if os.path.isdir(patient_path):
            # Look for SVS files in the patient directory
            for file_name in os.listdir(patient_path):
                if file_name.endswith(".svs"):
                    svs_file_path = os.path.join(patient_path, file_name)

                    # Move the SVS file to the main directory
                    shutil.move(svs_file_path, main_directory)

def patient_csv(main_directory, results_path):
    csv_file_path = os.path.join(results_path, "patient_files.csv")

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Patient ID", "Original Slide Path", "Preprocessing Path"])
        for file in os.listdir(main_directory):

            if file.endswith(".svs"):

                patient_id = file.split("-")[0] + "-" +file.split("-")[1] + "-" +file.split("-")[2] + "-" + file.split("-")[3] + "-" + file.split("-")[4] + "-" + \
                             file.split("-")[5]
                patient_id = patient_id.split(".")[0]
                patient_result_path = os.path.join(results_path, patient_id)

                if not os.path.exists(patient_result_path):
                    os.makedirs(patient_result_path)
                csv_writer.writerow([patient_id, os.path.join(main_directory, file),
                                     os.path.join(results_path, patient_id)])
    return csv_file_path


def preprocessing(path, patient_path, encoder_path, patient_id):
    slide = whole_slide_image(path)

    print(f"processing: {path}")
    if slide is not None:
        wsi_thumb = masking.get_thumbnail(openslide.OpenSlide(path))
        #getting mask for sanity check
        mask = masking.get_whole_slide_mask(wsi_thumb, patient_path)

        tile_inf_path = os.path.join(patient_path, "tile_information.csv")
        if not os.path.isfile(tile_inf_path):
            tiling(slide, patient_path, mask)
        normalize_tiles_path = os.path.join(patient_path, "normalized_tile_information2.csv")
        if not os.path.isfile(normalize_tiles_path):
            normalize_tiles(tile_inf_path, patient_path)
        # do encoding
        encoding.encode_tiles(patient_id, normalize_tiles_path,  r"C:\Users\albao\Masters\WSI_proper\encoded")

        # masked_tiles_path = os.path.join(result_path, "masked_tile_information.csv")
        # if not os.path.isfile(masked_tiles_path):
        #     mask_tiles(tile_inf_path, result_path)

def preprocess_patient(row):
    encoder_path = os.path.join( r"C:\Users\albao\Masters\WSI_proper", "encoded")
    result = row["Preprocessing Path"]
    original = row["Original Slide Path"]
    patient_id = row["Patient ID"]
    preprocessing(original, result, encoder_path, patient_id)
    print(f"done with patient {patient_id}")

def main():

    path = input("Enter input path (or leave blank for default): ").strip()
    if not path:
        path = r"C:\Users\albao\Downloads\gdc_download_20240320_111546.230274"  # Provide default input path here
    global result_path
    result_path = input("Enter result path (or leave blank for default): ").strip()
    if not result_path:
        result_path = r"C:\Users\albao\Masters\WSI_proper"  # Provide default result path here
    processes = input("Please enter number of threads for multiprocessing(leave blank for no multiprocessing): ").strip()
    if not processes:
        processes = 1

    if not os.path.exists(os.path.join(result_path, "encoded")):
            os.makedirs(os.path.join(result_path, "encoded"))

    move_svs_files(path, result_path)
    patient_path = patient_csv(path, result_path)
    patients = pd.read_csv(patient_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with multiprocessing.Pool(processes= processes, maxtasksperchild= 1) as pool:
        results = pool.map(preprocess_patient, (row for _, row in patients.iterrows()))

if __name__ == "__main__":
    main()
