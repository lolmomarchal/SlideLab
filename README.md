<div align="right">
  <img src="https://github.com/user-attachments/assets/e7219b41-d35b-4a31-b1c2-2eac9c71d03f" width="350" height="300">
</div>

This script performs Whole Slide Image (WSI) preprocessing, including masking, tiling, normalization, quality checks, encoding, and optional whole slide reconstruction after tiling. It can be used as a customizable pipeline for mass WSI processing or to directly call functions to perform specific tasks.

As a pipeline, it is designed to ensure that if stopped for any reason, you will be able to continue at the last step that was completed. It also includes both an error report (for any error that may occur and the location it occurred) and a summary report with statistics like % of tissue and time taken to process the file.

![slidelab_workflow](https://github.com/user-attachments/assets/5ec85991-6ed8-435a-b888-4b3d9304845d)

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

## Arguments
### Input/Output (Required) 
| Argument          | Description                           | Default |
|--------------------|---------------------------------------|---------|
| `-i`, `--input_path` | Path to the input WSI file            | None    |
| `-o`, `--output_path` | Path to save the output tiles         | None    |

### Tile Customization
| Argument               | Description                                         | Default |
|-------------------------|-----------------------------------------------------|---------|
| `-s`, `--desired_size` | Desired size of the tiles (in pixels)               | 256     |
| `-m`, `--desired_magnification` | Desired magnification level  (ex: 20x | 20      
| `-ov`, `--overlap`     |  Factor of overlap between tiles                   | 1 (no overlap) |

**Overlap example**:
With a size of 256 and an overlap of 2, tiles would overlap by 128 pixels. 

### Preprocessing options 
| Argument                 | Description                                              | Default |
|---------------------------|----------------------------------------------------------|---------|
| `-rb`, `--remove_blurry_tiles` | Remove blurry tiles using a Laplacian filter           | False   |
| `-n`, `--normalize_staining` | Normalize staining of the tiles                        | False   |
| `-e`, `--encode`         | Encode tiles into an `.h5` file                          | False   |
| `--extract_high_quality` | Extract  features for high quality heatmaps                          | False   |
| `--augmentations` | Get various augmentations for encoded tiles for model training                  | 0   |

### Thresholds 
| Argument               | Description                                          | Default |
|-------------------------|------------------------------------------------------|---------|
| `-th`, `--tissue_threshold` | Minimum tissue content to consider a tile valid   | 0.7     |
| `-bh`, `--blur_threshold`   | Threshold for Laplacian filter variance (blur detection) | 0.015   |
| `--red_pen_check`   | Sanity check for % of red pen detected. If above threshold, red_pen mask will be ignored | 0.4   |
| `--blue_pen_check`  | Sanity check for % of red pen detected. If above threshold, blue_pen mask will be ignored | 0.4   |

### Devices and Multiprocessing 
| Argument             | Description                          | Default           |
|-----------------------|--------------------------------------|-------------------|
| `--device`           | Specify device (e.g., GPU/CPU)       | None (will utilize gpu if available)             |
| `--cpu_processes`    | Number of CPU processes to use       | `os.cpu_count()`  |
| `--batch_size`    | Number of CPU processes to use       | `16. If using augmentations batch_size will be recalculated by using # of augmentations/batch size`  |

### Additional Params
| Argument             | Description                          | Default           |
|-----------------------|--------------------------------------|-------------------|
| `--min_tiles`           | Minimum number of valid tiles for a sample to be counted as "valid". Will create additional filtered sample metadata file.| 0 |

### Example command 
```sh
python SlidePreprocessing.py -i /path/to/input/-o /path/to/output/ \
  -s 512 -m 40 --remove_blurry_tiles --normalize_staining --encode \
  -th 0.8 -bh 0.02 --device cuda --batch_size 256
```



## To run individual classes: 
Can create related objects and use their associated methods. Please see `Example.ipynb` and  `Masking Examples.ipynb`.
