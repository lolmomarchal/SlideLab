import h5py
import numpy as np
import pandas as pd
import cv2
import traceback
import os
from TissueMask import is_tissue, get_region_mask, TissueMask
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import re
import numba

def filter_patients(patient_df, summary_df, args):
    summary_df = pd.read_csv(summary_df)
    if not args.remove_blurry_tiles:
        column = "tiles_passing_tissue_thresh"
    else:
        column = "non_blurry_tiles"
    not_passing_QC = summary_df.loc[summary_df[column] < args.get("min_tiles"), "sample_id"]
    # filter based on sample ID
    filtered_patients = patient_df[~patient_df["Patient ID"].isin(not_passing_QC)]
    filtered_path = os.path.join(args.get("output_path"), "filtered_patients.csv")
    filtered_patients.to_csv(filtered_path, index=False)

############ SIZING CONVERSIONS & BEST SIZE ##########################################

def get_best_size_mpp(target_pixels, target_mpp, target_magnification,mmp_x, mmp_y, magnification):

    target_physical_width = target_mpp * target_pixels
    target_physical_height = target_mpp * target_pixels
    # mapping it to current:
    mapped_width = target_physical_width//mmp_x
    mapped_height = target_physical_height//mmp_y
    return (mapped_width, mapped_height)

def get_best_size_mag(target_pixels, target_mpp, target_magnification,mmp_x, mmp_y, magnification):
    scale = magnification//target_magnification
    dimensions = int(target_pixels*scale)
    return (dimensions, dimensions)

@numba.jit(nopython=True)
def is_tissue(masked_region, threshold=0.7):
    tissue = np.count_nonzero(masked_region)
    total_elements = masked_region.size
    if total_elements == 0:
        return False
    return (tissue / total_elements) >= threshold

@numba.jit(nopython=True)
def get_region_mask(mask, scale, coord, size):
    x, y = coord
    size_x, size_y = size
    mask_x = x // scale
    mask_y = y // scale
    mask_w = size_x // scale
    mask_h = size_y // scale
    return mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]

@numba.jit(nopython=True, parallel=True)
def compute_valid_coordinates(coordinates, mask, size, scale, threshold):
    valid = np.zeros(len(coordinates), dtype=np.bool_)
    for i in numba.prange(len(coordinates)):
        coord = coordinates[i]
        region = get_region_mask(mask, scale, coord, size)
        valid[i] = is_tissue(region, threshold)
    return valid

@numba.jit(nopython=True)
def get_valid_coordinates(width, height, overlap, mask, size, scale, threshold):
    size_x, size_y = size

    if overlap > 1:
        stride_x = size_x // overlap
        stride_y = size_y // overlap
    else:
        stride_x = size_x
        stride_y = size_y

    x_coords = np.arange(0, width - size_x + stride_x, stride_x)
    y_coords = np.arange(0, height - size_y + stride_y, stride_y)

    coordinates = np.empty((len(x_coords) * len(y_coords), 2), dtype=np.int32)
    idx = 0
    for y in y_coords:
        for x in x_coords:
            coordinates[idx, 0] = x
            coordinates[idx, 1] = y
            idx += 1

    valid = compute_valid_coordinates(coordinates, mask, size, scale, threshold)
    valid_coords = coordinates[valid]
    return len(coordinates), len(valid_coords), valid_coords, coordinates

@lru_cache(maxsize=128)
def get_closest_mpp(mpp):
    mpp_to_magnification = {
        8.0: 8.0,
        4.0: 4.0,
        2.0: 2.0,
        1.0: 1.0,
        0.5: 0.5,
        0.25: 0.25,
        0.167: 0.167,
        0.1: 0.1,
        16.0: 16.0,
        32.0: 32.0
    }
    if mpp in mpp_to_magnification:
        return mpp_to_magnification[mpp]
    keys = sorted(mpp_to_magnification.keys())
    mpp = round(mpp, 3)
    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        midpoint = (k1 + k2) / 2
        if k1 <= mpp < k2:
            if mpp < midpoint:
                return mpp_to_magnification[k1]
            else:
                return mpp_to_magnification[k2]
    return mpp_to_magnification[keys[0]] if mpp < keys[0] else mpp_to_magnification[keys[-1]]

@lru_cache(maxsize=128)
def mpp_to_mag(mpp):
    mpp_to_magnification = {
        8.0: 1.25,
        4.0: 2.5,
        2.0: 5,
        1.0: 10,
        0.5: 20,
        0.25: 40,
        0.167: 60,
        0.1: 100,
        16.0: 0.625,
        32.0: 0.3125
    }
    if mpp in mpp_to_magnification:
        return mpp_to_magnification[mpp]
    keys = sorted(mpp_to_magnification.keys())
    mpp = round(mpp, 3)
    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        midpoint = (k1 + k2) / 2
        if k1 <= mpp < k2:
            if mpp < midpoint:
                return mpp_to_magnification[k1]
            else:
                return mpp_to_magnification[k2]
    return mpp_to_magnification[keys[0]] if mpp < keys[0] else mpp_to_magnification[keys[-1]]
def mag_to_mpp(mpp):
    # closest averages, might change depending on scanner
    magnification_to_mpp = {
        1.25: 8.0,
        2.5: 4.0,
        5: 2.0,
        10: 1.0,
        20: 0.5,
        40: 0.25,
        60: 0.167,
        100: 0.1,
        0.625: 16.0,
        0.3125: 32.0}
    if mpp in magnification_to_mpp:
        return magnification_to_mpp[mpp]
    keys = sorted(magnification_to_mpp.keys())
    mpp = round(mpp, 3)
    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        midpoint = (k1 + k2) / 2

        if k1 <= mpp < k2:
            if mpp < midpoint:
                return magnification_to_mpp[k1]
            else:
                return magnification_to_mpp[k2]
    return magnification_to_mpp[keys[0]] if mpp < keys[0] else magnification_to_mpp[keys[-1]]


def extract_from_comment(comment, scanner = "aperio"):
    regex_expression = { "aperio":r"MPP\s*=\s*([\d.]+).*?OriginalWidth\s*=\s*(\d+).*?Originalheight\s*=\s*(\d+)",
                         # "dicom":
                         }
    match = re.search(regex_expression[scanner], comment)
    if match:
        mpp, width, height = match.groups()
def get_attributes(slide):
    magnification, mpp_x, mpp_y, height, width, vendor = None, None, None, None, None, None

    try:
        vendor = slide.properties.get("openslide.vendor", "").lower()
    except Exception as e:
        print(f"Could not get vendor: {e}")
    try:
        magnification = int(float(slide.properties.get("openslide.objective-power", 0)))
    except Exception as e:
        print(f"Could not get magnification: {e}")

    try:
        mpp_x = slide.properties.get("openslide.mpp-x")
        mpp_y = slide.properties.get("openslide.mpp-y")
        if mpp_x is not None:
            mpp_x = float(mpp_x)
        if mpp_y is not None:
            mpp_y = float(mpp_y)

        # if one is missing
        if mpp_x is None and mpp_y is not None:
            mpp_x = float(mpp_y)
        if mpp_y is None and mpp_x is not None:
            mpp_y = float(mpp_x)

    except Exception as e:
        print(f"Could not extract MPPs: {e}")
        mpp_x = mpp_y = None
    mpp = mpp_x or mpp_y

    # estimate missing MPP or magnification
    if mpp is None and magnification:
        mpp = mag_to_mpp(magnification)
        mpp_x = mpp_y = mpp
    if magnification is None and mpp:
        magnification = mpp_to_mag(mpp)

    # dimensions
    try:
        width, height = slide.dimensions
    except Exception as e:
        print(f"Could not get dimensions: {e}")
    closest_mpp = get_closest_mpp(mpp) if mpp else None

    # sometimes mag is wrong, so check with mpp (most accurate)
    mag_based_on_mpp = mpp_to_mag(closest_mpp) if closest_mpp else None
    if mag_based_on_mpp and mag_based_on_mpp != magnification:
        magnification = mag_based_on_mpp

    return mpp_x, mpp_y, width, height, magnification, closest_mpp


###################################################### WORKERS #####################################################################
# =============================== HELPER FUNCTION CPU ====================
def apply_laplace_filter(img_np, vars_dict, coord, *, blur_method, blur_threshold):
    blurry, var = blur_method(img_np, var_threshold=blur_threshold)
    vars_dict[tuple(coord)] = var
    return None if blurry else img_np

def apply_stain_normalization(img_np, vars_dict, coord, *, normalize_staining_func):
    return normalize_staining_func(img_np)

def worker_png(queue, tile_iterator, patient_id, sample_path, results, vars_dict, pipeline_steps):
    save_executor = ThreadPoolExecutor(max_workers=4) # more internal parallelism within the function
    while True:
        index = queue.get()
        if index is None:
            break
        img, coord = tile_iterator[index]
        img = np.array(img)
        for step in pipeline_steps:
            img = step(img, vars_dict, coord)
            if img is None :
                break
        image_path = os.path.join(sample_path, f"{patient_id}_{coord[0]}_{coord[1]}_size_{tile_iterator.size}_mag_{tile_iterator.magnification}.png")
        def _save(image_path, img_data):
            try:
                cv2.imwrite(image_path, img_data)
            except Exception as e:
                print(f"[SAVE ERROR] {image_path}: {e}")
        if img is not None and img.size > 0 and len(img.shape) == 3:
            # Prepare the image data (convert color space)
            img_data = np.array(img)[:, :, ::-1]

            # Submit the saving task to the executor
            save_executor.submit(_save, image_path, img_data)

            # Append to results
            results.append({
                "Patient_ID": patient_id, "x": coord[0], "y": coord[1],
                "tile_path": image_path, "original_size": tile_iterator.adjusted_size,
                "desired_size": tile_iterator.size, "desired_magnification": tile_iterator.magnification
            })
        else:
            if img is not None:
                print(f"Image shape: {img.shape}")

    save_executor.shutdown(wait=True)
    del save_executor
def worker_h5(queue, tile_iterator, patient_id, h5_path, results, vars_dict, pipeline_steps, save_queue):
    while True:
        index = queue.get()
        if index is None:
            break
        img, coord = tile_iterator[index]
        img = np.array(img)
        for step in pipeline_steps:
            img = step(img, vars_dict, coord)
            if img is None:
                break
        if img is not None:
            save_queue.put((coord, img))
            results.append({
                "Patient_ID": patient_id, "x": coord[0], "y": coord[1],
                "original_size": tile_iterator.adjusted_size,
                "desired_size": tile_iterator.size, "desired_magnification": tile_iterator.magnification
            })
def save_h5(save_queue, h5_file, batch_size=128):
    tiles_list = []
    coords_list = []
    # if exists, process did NOT finish
    if os.path.exists(h5_file):
        os.remove(h5_file)
    while True:
        item = save_queue.get()
        if item is None:
            break
        coord, norm_tile = item
        if norm_tile is not None:
            tiles_list.append(norm_tile)
            coords_list.append(coord)
        if len(tiles_list) >= batch_size:
            with h5py.File(h5_file, "a") as f:
                if "tiles" not in f:
                    tile_shape = tiles_list[0].shape
                    f.create_dataset("tiles", shape=(0, *tile_shape),
                                     maxshape=(None, *tile_shape),
                                     dtype=tiles_list[0].dtype, chunks=True)
                    coord_shape = coords_list[0].shape
                    f.create_dataset("coords", shape=(0, 2),
                                     maxshape=(None, 2),
                                     dtype="int32",
                                     chunks=True)
                tiles_dataset = f["tiles"]
                coords_dataset = f["coords"]
                current_len = tiles_dataset.shape[0]
                new_len = current_len + len(tiles_list)
                tiles_dataset.resize((new_len, *tiles_dataset.shape[1:]))
                coords_dataset.resize((new_len, 2))
                tiles_dataset[current_len:new_len] = tiles_list
                coords_dataset[current_len:new_len] = coords_list
                tiles_list.clear()
                coords_list.clear()
        # write any remaining in list
    if tiles_list:
        with h5py.File(h5_file, "a") as f:
            tiles_dataset = f["tiles"]
            coords_dataset = f["coords"]
            current_len = tiles_dataset.shape[0]
            new_len = current_len + len(tiles_list)
            tiles_dataset.resize((new_len, *tiles_dataset.shape[1:]))
            coords_dataset.resize((new_len, 2))
            tiles_dataset[current_len:new_len] = tiles_list
            coords_dataset[current_len:new_len] = coords_list
            tiles_list.clear()
            coords_list.clear()

# ============================ HELPER FUNCTION GPU =======================================
def apply_stain_normalization_gpu(batch_img, vars_dict, coords, *, normalize_staining_func):
    # filter for any that may be none
    batch_img = normalize_staining_func(batch_img, device = batch_img.device)
    mask = batch_img.reshape(batch_img.shape[0], -1).any(dim=1)

    filtered_batch = batch_img[mask]
    filtered_coords = coords[mask]
    return filtered_batch, filtered_coords

def apply_laplace_filter_gpu(img_np, vars_dict, coord, *, blur_method, blur_threshold):
    blurry, var = blur_method(img_np, threshold=blur_threshold, device = batch_img.device)
    var = var.cpu().numpy()
    for c, v in zip(coord, var):
        vars_dict[tuple(c)] = float(v)

    # need to filter both coords & imgs in batch to ensure that these steps dont end up getting added to queue
    keep_mask = ~blurry
    img_np = img_np[keep_mask]
    coord = coord[keep_mask]
    return img_np, coord

def worker_png_gpu(save_queue):
    save_executor = ThreadPoolExecutor(max_workers=4) # more internal parallelism within the function
    while True:
        index = save_queue.get()
        if index is None:
            break
        img, save_path = index
        def _save():
            try:
                cv2.imwrite(save_path, np.array(img)[:, :, ::-1])
            except Exception as e:
                print(f"[SAVE ERROR] {image_path}: {e}")
        if img is not None:
            save_executor.submit(_save)
