import os
import argparse

import encoding

OPENSLIDE_PATH = r"C:\Users\albao\Downloads\openslide-win64-20231011\openslide-win64-20231011\bin"
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):

        import openslide
else:
    import openslide
import multiprocessing
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool
import normalization
# import encoding
from Slide_utils import TissueSlide

from PIL import Image
import cv2
import shutil
import matplotlib.pyplot as plt
import gc
import torch
import csv
from TissueMask import TissueMask
import ArtifactRemoval

"""
preprocessing.py

Author: Lorenzo Olmo Marchal
Created: March 12, 2024
Last Updated:  May 9, 2024

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



def best_size(slide, magnitude_reference = 20, size =256):
    natural_mag  = slide.magnification
    scale = natural_mag/magnitude_reference
    return int(size * scale)
def tiling(ts, result_path, mask, overlap=False, desired_size=256):
    print("getting tiles")
    # Make tile dir
    tiles = os.path.join(result_path, "tiles")

    columns = ['patient_id', 'x', 'y', 'magnification', 'size', 'path_to_slide']
    df_list = []
    os.makedirs(os.path.join(result_path, "tiles"), exist_ok=True)

    if overlap != False:
        stride = desired_size//6
    else:
        stride = desired_size
    mag = 20
    size = best_size(ts)
    w = ts.dimensions[0]
    h = ts.dimensions[1]
    for x in range(0, w - size + 1, stride):
        for y in range(0, h - size + 1, stride):
            if x <= w - size and y <= h - size:
                tissue_rgb = ts.slide.read_region((x,y), 0, (size, size))
                tile_mask = mask.get_region_mask((x,y), (size, size))
                # get mask of tile
                istissue = mask.is_tissue(tile_mask)
                if  istissue == True:
                    tile = tissue_rgb.convert("RGB")
                    tile = tile.resize((desired_size,desired_size))
                    tile.save(os.path.join(tiles, f"{ts.id}_tile_w{x}_h{y}_mag{ts.magnification}_size{size}.png"))
                    df_list.append({
                        'patient_id': ts.id,
                        'x': x,
                        'y': y,
                        'magnification': ts.magnification,
                        'size': size,
                        'path_to_slide': os.path.join(tiles, f"{ts.id}_tile_w{x}_h{y}_mag{ts.magnification}_size{size}.png")
                    })
    df = pd.DataFrame(df_list, columns=columns)
    df.to_csv(os.path.join(result_path, "tile_information.csv"), index=False)
    del df
    gc.collect()

def normalize_tiles(tile_information, result_path):
    print("normalizing")
    tiles = pd.read_csv(tile_information)
    path = os.path.join(result_path, "normalized_tiles")
    df_list = []
    columns = ['patient_id', 'x', 'y', 'magnification', 'size', 'path_to_slide']
    os.makedirs(os.path.join(result_path, "normalized_tiles"), exist_ok=True)
    for i, row in tiles.iterrows():
        path_to_tile = row["path_to_slide"]
        mag = row["magnification"]
        size = row["size"]
        y = row["y"]
        x = row["x"]
        id = row["patient_id"]
        try:
            tile = np.array(Image.open(path_to_tile))
            if tile is not None:
                normalization.normalizeStaining(tile, os.path.join(path, f"{id}_tile_w{x}_h{y}_mag{mag}_size{size}.png"))

                df_list.append({
                    'patient_id': id,
                    'x': x,
                    'y': y,
                    'magnification': mag,
                    'size': size,
                    'path_to_slide': os.path.join(path, f"{id}_tile_w{x}_h{y}_mag{mag}_size{size}.png")
                })

            else:
                print("Error: Input tile is None.")

        except Exception as e:
            print(f"An error occurred: {e}")
    df = pd.DataFrame(df_list, columns=columns)
    df.to_csv(os.path.join(result_path, "normalized_tile_information.csv"), index=False)
    del df
    gc.collect()
def remove_artifacts(tile_information, result_path):
    tiles = pd.read_csv(tile_information)
    path = os.path.join(result_path, "infocus_tiles")
    blurry_path = os.path.join(result_path, "outfocus_tiles")
    df_list = []
    columns = ['patient_id', 'x', 'y', 'magnification', 'size', 'path_to_slide']
    os.makedirs(os.path.join(result_path, "infocus_tiles"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "outfocus_tiles"), exist_ok=True)
    for i, row in tiles.iterrows():
        path_to_tile = row["path_to_slide"]
        mag = row["magnification"]
        size = row["size"]
        y = row["y"]
        x = row["x"]
        id = row["patient_id"]
        try:
            tile = np.array(Image.open(path_to_tile))
            if tile is not None:
                not_blurry = ArtifactRemoval.LaplaceFilter(tile)
                if not_blurry == True:
                    img = Image.open(path_to_tile)
                    img.save(os.path.join(path, f"{id}_tile_w{x}_h{y}_mag{mag}_size{size}.png"))
                    df_list.append({
                        'patient_id': id,
                        'x': x,
                        'y': y,
                        'magnification': mag,
                        'size': size,
                        'path_to_slide': os.path.join(path, f"{id}_tile_w{x}_h{y}_mag{mag}_size{size}.png")
                    })
                else:
                    img = Image.open(path_to_tile)
                    img.save(os.path.join(blurry_path, f"{id}_tile_w{x}_h{y}_mag{mag}_size{size}.png"))
            else:
                print("Error: Input tile is None.")
        except Exception as e:
            print(f"An error occurred: {e}")
    df = pd.DataFrame(df_list, columns=columns)
    df.to_csv(os.path.join(result_path, "infocus_tile_information.csv"), index=False)
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
    Tissue = TissueSlide(path)


    print(f"processing: {path}")
    if Tissue.slide is not None:
        mask= TissueMask(Tissue, patient_path)
        tile_inf_path = os.path.join(patient_path, "tile_information.csv")

        if not os.path.isfile(tile_inf_path):
            tiling(Tissue, patient_path, mask)
        #
        normalize_tiles_path = os.path.join(patient_path, "normalized_tile_information.csv")
        if not os.path.isfile(normalize_tiles_path):
            normalize_tiles(tile_inf_path, patient_path)

        # remove blurry
        in_focus_path = os.path.join(patient_path, "infocus_tile_information.csv")
        if not os.path.isfile(in_focus_path):
            remove_artifacts(normalize_tiles_path, patient_path)
        # # do encoding
        # encoding.encode_tiles(patient_id, normalize_tiles_path,  encoder_path)

def preprocess_patient(row):
    encoder_path = os.path.join( r"C:\Users\albao\Masters\WSI_proper", "encoded")
    result = row["Preprocessing Path"]
    original = row["Original Slide Path"]
    patient_id = row["Patient ID"]
    preprocessing(original, result, encoder_path, patient_id)
    print(f"done with patient {patient_id}")

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("-i", "--input_path", type=str, default=r"C:\Users\albao\Downloads\gdc_download_20240320_111546.230274", help="Input path (default: %(default)s)")
    parser.add_argument("-o", "--output_path", type=str, default=r"C:\Users\albao\Masters\WSI_proper", help="Result path (default: %(default)s)")
    parser.add_argument("-p", "--processes", type=int, default=1, help="Number of threads for multiprocessing (default: %(default)s)")
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    processes = args.processes
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(os.path.join(output_path, "encoded")):
        os.makedirs(os.path.join(output_path, "encoded"))

    move_svs_files(input_path, output_path)
    patient_path = patient_csv(input_path, output_path)
    patients = pd.read_csv(patient_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with multiprocessing.Pool(processes=processes, maxtasksperchild=1) as pool:
        results = pool.map(preprocess_patient, (row for _, row in patients.iterrows()))

if __name__ == "__main__":
    main()

