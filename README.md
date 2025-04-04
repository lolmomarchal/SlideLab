# Table of Contents  

<img src="https://github.com/user-attachments/assets/e7219b41-d35b-4a31-b1c2-2eac9c71d03f" align="right" width="400" style="margin-left: 15px;"/>

1. [Overview](#overview)  
2. [Preprocessing Steps](#preprocessing-steps)  
   - [Masking](#masking)  
   - [Normalization](#normalization)  
   - [Tiling](#tiling)  
3. [Reports](#reports)  
   - [Summary Report](#summary-report)  
   - [Error Report](#error-report)  
4. [Requirements](#requirements)  
5. [Installation](#installation)  
   - [Conda Installation](#conda-installation)  
6. [Usage](#usage)  
   - [Arguments](#arguments)  
     - [Input/Output (Required)](#inputoutput-required)  
     - [Tile Customization](#tile-customization)  
     - [Preprocessing Options](#preprocessing-options)  
     - [Thresholds](#thresholds)  
     - [Devices and Multiprocessing](#devices-and-multiprocessing)  
     - [Additional Parameters](#additional-params)  
   - [Example Command](#example-command)  
   - [Running Individual Classes](#to-run-individual-classes)
  
7. [Contact](#contact)
8. [References](#references)

<br clear="left"/>


# Overview 
SlideLab is a preprocessing pipeline to preprocess hematoxylin and eosin (H&E) Whole Slide Images (WSI) for computational pathology applications. This script including masking, tiling, normalization, quality checks, encoding, and optional whole slide reconstruction after tiling. It can be used as a customizable pipeline for mass WSI processing or to directly call functions to perform specific tasks. Please refer to the [arguments](#arguments) to see the potential options. As a pipeline, it is designed to ensure that if stopped for any reason, you will be able to continue at the last step that was completed. It also includes both an error report (for any error that may occur and the location in the pipeline where occurred) and a summary report with statistics like % of tissue and time taken to process the WSI.


![slidelab_workflow](https://github.com/user-attachments/assets/5ec85991-6ed8-435a-b888-4b3d9304845d)

# Preprocessing Steps

### Masking
SlideLab uses several methods to filter out artifacts such as pen marks and blots from slides and segment tissue sections according to adjustable threshold. Thresholds like the tissue percentage in a tile is used to select candidate tiles and can be adjusted. To see parameters associated with masking refer to [masking params](#Thresholds) The hematoxylin and eosin otsu adaptation was obtained from Schreiber et. al [4]. 

<img src="https://github.com/user-attachments/assets/1542dfdc-9092-4722-bb7f-b384f32c9105" alt="Image 1" width="500" height="300">
<img src="https://github.com/user-attachments/assets/094332ae-ec20-47c5-a08c-92af1568adf0" alt="Image 2" width="500" height="300">
<img src="https://github.com/user-attachments/assets/51e1883e-f2f7-4da8-9806-134d240ff126" alt="Image 3" width="500" height="300">



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

# Installation

First clone the repository and cd into repository: 
```sh
git clone https://github.com/lolmomarchal/SlideLab.git
cd SlideLab
```
To install required dependencies you can do through conda:

### Conda Installation
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
Please refer to the tutorials folder!

## Contact
For questions/requests please make an issue or email aolmomarchal@ucsd.edu under the subject "SlideLab: [insert question]"

## References 
1. M. Macenko et al., "A method for normalizing histology slides for quantitative analysis," 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, Boston, MA, USA, 2009, pp. 1107-1110, doi: 10.1109/ISBI.2009.5193250.
2. Barbano, C. A., & Pedersen, A. (2022, August). EIDOSLAB/torchstain: v1.2.0-stable (Version v1.2.0-stable) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.6979540
3. Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al. Towards a general-purpose foundation model for computational pathology. Nat Med (2024). https://doi.org/10.1038/s41591-024-02857-3
4. B. A. Schreiber, J. Denholm, F. Jaeckle, M. J. Arends, K. M. Branson, C.-B.Schönlieb, and E. J. Soilleux. Bang and the artefacts are gone! Rapid artefact removal and tissue segmentation in haematoxylin and eosin stained biopsies, 2023.
URL http://arxiv.org/abs/2308.13304.
