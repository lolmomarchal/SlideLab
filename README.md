
# SlideLab

This script performs Whole Slide Image (WSI) preprocessing, including masking, tiling, normalization, quality checks, encoding and optional whole slide reconstruction after tiling. It can be used as a customazible pipeline for mass WSI processing or to directly call functions to perform specific tasks. As a pipeline, it is designed to ensure that if stopped for any reason, you will be able to continue at the last step that was completed. It also includes both an error report (for any error that may occur and the location it occurred) and a summary report with statistics like % of tissue and time taken to process file.

### Masking
![image](https://github.com/lolmomarchal/SlideLab/assets/114376800/2c4c98fd-a6ae-40c0-8e9a-5f9d88404e92)

### Normalization
![image](https://github.com/lolmomarchal/SlideLab/assets/114376800/532fe9f7-b44b-4da3-bf86-a979ebe19127)

### Tiling 

![image](https://github.com/lolmomarchal/SlideLab/assets/114376800/b352da2c-0276-4af6-b879-8a71d5eb7388)

### reports 
#### summary report 
![{EB0D74CC-455F-4ECA-B98D-7C92C0BCF271}](https://github.com/user-attachments/assets/e0171dbe-db1b-4dd2-93fe-f5a965de12e2)

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
To install required dependencies you can do through conda:

### conda
```sh
conda env create -f environment.yml
conda activate slidelab
```

# Usage
==============
Arguments
===
## Input/Output (Required) 
| Argument          | Description                           | Default |
|--------------------|---------------------------------------|---------|
| `-i`, `--input_path` | Path to the input WSI file            | None    |
| `-o`, `--output_path` | Path to save the output tiles         | None    |

## Tile Customization
| Argument               | Description                                         | Default |
|-------------------------|-----------------------------------------------------|---------|
| `-s`, `--desired_size` | Desired size of the tiles (in pixels)               | 256     |
| `-m`, `--desired_magnification` | Desired magnification level  (ex: 20x | 20      
| `-ov`, `--overlap`     |  Factor of overlap between tiles                   | 1 (no overlap) |

**Overlap example**:
With a size of 256 and an overlap of 2, tiles would overlap by 128 pixels. 

## Preprocessing options 
| Argument                 | Description                                              | Default |
|---------------------------|----------------------------------------------------------|---------|
| `-rb`, `--remove_blurry_tiles` | Remove blurry tiles using a Laplacian filter           | False   |
| `-n`, `--normalize_staining` | Normalize staining of the tiles                        | False   |
| `-e`, `--encode`         | Encode tiles into an `.h5` file                          | False   |
| `--extract_high_quality` | Extract only high-quality tiles                          | False   |

## Thresholds 
| Argument               | Description                                          | Default |
|-------------------------|------------------------------------------------------|---------|
| `-th`, `--tissue_threshold` | Minimum tissue content to consider a tile valid   | 0.7     |
| `-bh`, `--blur_threshold`   | Threshold for Laplacian filter variance (blur detection) | 0.015   |

## Devices and Multiprocessing 
| Argument             | Description                          | Default           |
|-----------------------|--------------------------------------|-------------------|
| `--device`           | Specify device (e.g., GPU/CPU)       | None (will utilize gpu if available)             |
| `--gpu_processes`    | Number of GPU processes to use       | 1                 |
| `--cpu_processes`    | Number of CPU processes to use       | `os.cpu_count()`  |




## To run individual classes: 
Can create related objects and use their associated methods. Please see `Example.ipynb` and  `Masking Examples.ipynb`.
