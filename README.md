
# SlideLab

This script performs Whole Slide Image (WSI) preprocessing, including masking, tiling, normalization, quality checks, encoding and optional whole slide reconstruction after tiling. It can be used as a customazible pipeline for mass WSI processing or to directly call functions to perform specific tasks. As a piepeline, it is designed to ensure that if stopped for any reason, you will be able to continue at the last step that was completed. It also includes both an error report (for any error that may occur and the location it occurred) and a summary report (with the # of extracted tiles per sample and the % of them that pass the optional quality check).

### Masking
![image](https://github.com/lolmomarchal/SlideLab/assets/114376800/2c4c98fd-a6ae-40c0-8e9a-5f9d88404e92)

### Normalization
![image](https://github.com/lolmomarchal/SlideLab/assets/114376800/532fe9f7-b44b-4da3-bf86-a979ebe19127)

### Tiling 

![image](https://github.com/lolmomarchal/SlideLab/assets/114376800/b352da2c-0276-4af6-b879-8a71d5eb7388)

### reports 
#### summary report 
![image](https://github.com/lolmomarchal/SlideLab/assets/114376800/81f7cbd2-50ab-450e-a5fd-5354b899b381)
#### error report
![image](https://github.com/lolmomarchal/SlideLab/assets/114376800/e9d7c1e6-dac0-40b2-887d-869a4f34aaa5)

# Requirements

In order to use this package, you must have python (>3.9) installed on your system. Conda is also recommended for installation.

#Installation

First clone the repository and cd into repository: 
```sh
git clone https://github.com/lolmomarchal/SlideLab.git
cd SlideLab
```
To install required dependencies you can do through conda (recommended) or through pip:
### pip
```sh
pip install -r requirements.txt 
```
### conda
```sh
conda env create -f environment.yml
conda activate SlideLab
```

# Usage

## To run whole pipeline :



Arguments
===
```sh
The script accepts several command-line arguments:

-i, --input_path:  Input path of the WSI file.
-o, --output_path: Output path where results will be saved.
-p, --processes:  Number of processes for multiprocessing. Default: 1
-s, --desired_size: Desired size of the tiles in pixels. Default: 256
-tg, --tile_graph:  Enable graphing of tiles. Flag to reconstruct slide after tiling.
-m, --desired_magnification: Desired magnification level. Default: 20
-ov, --overlap: Overlap factor between tiles. Default: 0
-th, --tissue_threshold: Threshold to consider a tile as tissue and not backgroun. Default: 0.7
-bh, --blur_threshold: Threshold for Laplace filter variance to detect blurriness. Default: 0.015
-rb, --remove_blurry_tiles:  Enable removal of blurry tiles using the Laplacian filter. Flag to perform a quality check of tiles.
-n, --normalize_staining: Enable normalization of tiles. Flag to normalize tiles.
--save_blurry_tiles: Flag to save blurry tiles in an additional folder called 'out_focus_tiles'.
-e, --encode: Flag to encode tiles and create an associated .h5 file.
--save_original_tiles: Flag to save the original, unnormalized tiles.


Required:
-i, --input_path
-o, --output_path

The rest of the parameters are optional

```
Example Command
===
```sh
python SlidePreprocessing.py -i /path/to/input -o /path/to/output -p 4 -s 512 -tg -m 40 -ov 10 -n -e
```

## To run individual classes: 
Can create related objects and use their associated methods. Please see `Example.ipynb` and  `Masking Examples.ipynb`.
