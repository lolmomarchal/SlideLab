
# SlidePreprocessing

This script performs Whole Slide Image (WSI) preprocessing, including tiling and optional tile graphing.

## Usage

To run whole pipeline :

To run the `SlidePreprocessing.py` script, you need to have Python installed on your system along with the required dependencies listed in `requirements.txt`.

To run individual classes: 
Can create related objects and use their associated methods. Please see the Example jupyter notebook in the github).

### Dependencies

Before running the script, install the required dependencies:

```sh
pip install -r requirements.txt 
```
Arguments
===
```sh
The script accepts several command-line arguments:

-i, --input_path: Input path for the WSI files. Can be a directory containing svs files or a singular svs file.

-o, --output_path: Output path for the resulting tiles.

-p, --processes: Number of threads for multiprocessing. Default is 1.

-s, --desired_size: Desired size of the tiles. Default is 256.

-tg, --tile_graph: Flag to enable graphing of tiles.

-m, --desired_magnification: Desired magnification level. Default is 20.

-ov, --overlap: Factor of overlap between tiles. Default is 0.

-th, --tissue_threshold: Threshold to consider a tile as Tissue. Default is 0.7.

-bh, --blur_threshold: Threshold for laplace filter variance. Default is 0.015.


Required:
-i, --input_path
-o, --output_path

The rest of the parameters are optional

```
Example Command
===
```sh
python SlidePreprocessing.py -i /path/to/input -o /path/to/output -p 4 -s 512 -tg -m 40 -ov 10
```


