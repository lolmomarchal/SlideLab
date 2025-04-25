import h5py
import numpy as np
from normalization.TileNormalization import normalizeStaining, normalizeStaining_torch
from TileQualityFilters import LaplaceFilter, plot_distribution
import cv2
import traceback
import os
from TissueMask import is_tissue, get_region_mask, TissueMask

############ SIZING CONVERSIONS & BEST SIZE ##########################################

def get_best_size_mpp(target_pixels, target_mpp, mmp_x, mmp_y):
    target_physical_width = target_mpp * target_pixels
    target_physical_height = target_mpp * target_pixels
    # mapping it to current:
    mapped_width = target_physical_width//mmp_x
    mapped_height = target_physical_height//mmp_y
    return mapped_width
def get_best_size_mag(desired_mag, natural_mag, target_pixels):
    scale = natural_mag//desired_mag
    dimensions = int(target_pixels*scale)
    return dimensions

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


def mpp_to_mag(mpp):
    # closest averages, might change depending on scanner
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

#### CPU PROCESSING -> H5 ################################
def process_tile_h5(index, tile_iterator, patient_id, results, vars, normalize_staining, remove_blurry_tiles, blur_threshold, save_queue):
    try:
        img, coord = tile_iterator[index]
        img_np = np.array(img)
        if normalize_staining:
            img_np = normalizeStaining(img_np)
            if img_np is None:
                return
        if remove_blurry_tiles:
            blurry, var = LaplaceFilter(img_np, var_threshold = blur_threshold)
            vars.append(var)
            if blurry:
                return
        save_queue.put((coord, img_np))
        results.append({
            "Patient_ID": patient_id, "x": coord[0], "y": coord[1],
            "original_size": tile_iterator.adjusted_size,
            "desired_size": tile_iterator.size, "desired_magnification": tile_iterator.magnification
        })
    except Exception as e:
        print(f"Error processing tile {index}: {e}")
        traceback.print_exc()

def save_h5_cpu(save_queue, h5_file, batch_size=64):
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

    if tiles_dataset:
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

def worker_h5(queue, tile_iterator, patient_id, results, vars,  normalize_staining, remove_blurry_tiles, blur_threshold,save_queue):
    while True:
        index = queue.get()
        if index is None:
            break
        process_tile_h5(index, tile_iterator, patient_id, results, vars,  normalize_staining, remove_blurry_tiles, blur_threshold,save_queue)


##### CPU PROCESSING -> PNG ##############################
def process_tile_png(index, tile_iterator, patient_id,
                     sample_path, results, vars,
                     normalize_staining,remove_blurry_tiles,
                     blur_threshold ):
    try:
        img, coord = tile_iterator[index]
        img_np = np.array(img)

        if normalize_staining:
            img_np = normalizeStaining(img_np)
            if img_np is None:
                return
        if remove_blurry_tiles:
            blurry, var = LaplaceFilter(img_np, var_threshold = blur_threshold)
            vars.append(var)
            if blurry:
                return
        image_path = os.path.join(sample_path, f"{patient_id}_{coord[0]}_{coord[1]}_size_{tile_iterator.size}_mag_{tile_iterator.magnification}.png")
        try:
            cv2.imwrite(image_path, img_np[:, :, ::-1])
        except Exception as e:
            print(f"cv2.imwrite failed for {image_path}: {e}")
            return

        results.append({
            "Patient_ID": patient_id, "x": coord[0], "y": coord[1],
            "tile_path": image_path,  "original_size": tile_iterator.adjusted_size,
            "desired_size": tile_iterator.size, "desired_magnification": tile_iterator.magnification
        })
    except Exception as e:
        print(f"Error processing tile {index}: {e}")
        traceback.print_exc()
# worker
def worker_png(queue, tile_iterator, patient_id,
               sample_path, results, vars, normalize_staining,remove_blurry_tiles,
               blur_threshold):
    while True:
        index = queue.get()
        if index is None:
            break
        process_tile_png(index, tile_iterator, patient_id,
                         sample_path, results, vars,
                         normalize_staining,remove_blurry_tiles,
                         blur_threshold)