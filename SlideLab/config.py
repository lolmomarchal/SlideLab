import yaml
import argparse
import os


def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    if config.get('cpu_processes') is None:
        config['cpu_processes'] = os.cpu_count()
    return config

def create_parser(config=None):
    parser = argparse.ArgumentParser()
    defaults = config if config else {}

    parser.add_argument("-i", "--input_path", type=str, default=defaults.get('input_path'),
                        help="path to input file/directory")
    parser.add_argument("-o", "--output_path", type=str, default=defaults.get('output_path'),
                        help="path to output directory")
    parser.add_argument("--output_format", type = str, default =defaults.get("output_format"))

    parser.add_argument("-s", "--desired_size", type=int, default=defaults.get('desired_size', 256),
                        help="Desired size of the tiles (default: %(default)s)")
    parser.add_argument("-m", "--desired_magnification", type=int, default=defaults.get('desired_magnification', 20),
                        help="Desired magnification level (default: %(default)s)")
    parser.add_argument("-ov", "--overlap", type=int, default=defaults.get('overlap', 1),
                        help="Overlap between tiles (default: %(default)s)")


    parser.add_argument("--desired_mpp", type=float, default=defaults.get('desired_mpp'),
                        help="Desired reference mpp (default: %(default)s)")
    parser.add_argument("--sizing_scale", type=str, default=defaults.get("sizing_scale"),
                        help="To choose either mpp or magnification, for more accurate results between scanners, use mpp")

    parser.add_argument("--set_standard_magnification", type=float, default=defaults.get("set_standard_magnification"))
    parser.add_argument("--set_standard_mpp", type=float, default=defaults.get("set_standard_mpp"))

    parser.add_argument("-rb", "--remove_blurry_tiles", action="store_true", default=defaults.get('remove_blurry_tiles', False),
                        help="flag to enable usage of the laplacian filter to remove blurry tiles")
    parser.add_argument("-n", "--normalize_staining", action="store_true", default=defaults.get('normalize_staining', False),
                        help="Flag to enable normalization of tiles")
    parser.add_argument("-e", "--encode", action="store_true", default=defaults.get('encode', False),
                        help="Flag to encode tiles and create associated .h5 file")
    parser.add_argument("--reconstruct_slide", action="store_true", default=defaults.get('reconstruct_slide', False),
                        help="reconstruct slide")

    parser.add_argument("--extract_high_quality", action="store_true", default=defaults.get('extract_high_quality', False),
                        help="extract high quality")
    parser.add_argument("--augmentations", type=int, default=defaults.get('augmentations', 0),
                        help="augment data for training")
    parser.add_argument("--feature_extractor", default=defaults.get('feature_extractor', 'resnet50'),
                        help="current options: resnet50, mahmood-uni")
    parser.add_argument("--token", default=defaults.get('token'),
                        help="required to download model weights from hugging face")

    parser.add_argument("-th", "--tissue_threshold", type=float, default=defaults.get('tissue_threshold', 0.7),
                        help="Threshold to consider a tile as Tissue(default: %(default)s)")
    parser.add_argument("-bh", "--blur_threshold", type=float, default=defaults.get('blur_threshold', 0.015),
                        help="Threshold for laplace filter variance (default: %(default)s)")
    parser.add_argument("--red_pen_check", type=float, default=defaults.get('red_pen_check', 0.4),
                        help="Sanity check for % of red pen detected. If above threshold, red_pen mask will be ignored(default: %(default)s)")
    parser.add_argument("--blue_pen_check", type=float, default=defaults.get('blue_pen_check', 0.4),
                        help="Sanity check for % of blue pen detected, If above threshold, blue_pen mask will be ignored(default: %(default)s)")
    parser.add_argument("--include_adipose_tissue", action="store_true", default=defaults.get('include_adipose_tissue', False),
                        help="will include adipose tissue in mask")
    parser.add_argument("--remove_folds", action="store_true", default=defaults.get('remove_folds', False),
                        help="will remove folded tissue in mask")
    parser.add_argument("--mask_scale", type=int, default=defaults.get('mask_scale'),
                        help="scale at which to downscale WSI for masking. Recommended is either 64 or None which will downsize to the lowest possible downscale recommended by openslide. None will produce a higher quality mask but is slower than 64")


    parser.add_argument("--device", default=defaults.get('device'),
                        help="device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--gpu_processes", type=int, default=defaults.get('gpu_processes', 1),
                        help="number of GPU processes")
    parser.add_argument("--cpu_processes", type=int, default=defaults.get('cpu_processes', os.cpu_count()),
                        help="number of CPU processes")
    parser.add_argument("--batch_size", type=int, default=defaults.get('batch_size', 16),
                        help="batch size for processing")


    parser.add_argument("--min_tiles", type=float, default=defaults.get('min_tiles', 0),
                        help="Number of tiles a patient should have.")
    parser.add_argument("--config_file", type=str, default = "None")

    return parser.parse_args()

def get_args_from_config(config_path):
    config = load_config_from_yaml(config_path)
    args = create_parser(config = config)
    return args
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default = "None")
    return parser.parse_args()

def main():
    arg = args()
    config = {
        # input/output
        "input_path": None,
        "output_path": None,

        # tile customization
        "desired_size": 256,
        "desired_magnification": 20,
        "overlap": 1,

        # preprocessing processes customization
        "remove_blurry_tiles": False,
        "normalize_staining": False,
        "encode": False,
        "reconstruct_slide": False,

        # encoding customizations
        "extract_high_quality": False,
        "augmentations": 0,
        "feature_extractor": "resnet50",
        "token": None,

        # thresholds
        "tissue_threshold": 0.7,
        "blur_threshold": 0.015,
        "red_pen_check": 0.4,
        "blue_pen_check": 0.4,
        "include_adipose_tissue": False,
        "remove_folds": False,
        "mask_scale": None,

        # for devices + multithreading
        "device": None,
        "gpu_processes": 1,
        "cpu_processes": None,  # This will be evaluated when loaded
        "batch_size": 16,

        # QC
        "min_tiles": 0
    }

    # Add comments as YAML comments
    yaml_str = """# input/output
    input_path: null  # path to input file/directory
    output_path: null  # path to output directory
    
    # tile customization 
    desired_size: 256  # Desired size of the tiles in pixels 
    overlap: 1
    output_format: null
    
    # resolution scales
    # for resolutions scales either desired_magnification or desired_mpp should have a value
    sizing_scale: magnification # options: magnification, mpp (choose mpp for reproducibility between scanner brands)
    desired_magnification: 20 
    desired_mpp: 0.5
    
    # if there are errors with a slide having no resolutions in its file:
    set_standard_magnification: null
    set_standard_mpp: null
    
    # preprocessing processes customization
    
    remove_blurry_tiles: false  # flag to enable usage of the laplacian filter to remove blurry tiles
    normalize_staining: false  # Flag to enable normalization of tiles
    encode: false  # Flag to encode tiles and create associated .h5 file
    reconstruct_slide: false  # reconstruct slide
        
    # encoding customizations
    extract_high_quality: false  # extract high quality
    augmentations: 0  # augment data for training
    feature_extractor: resnet50  # current options: resnet50, mahmood-uni, resnet50-truncated
    token: null  # required to download model weights from hugging face
    
    # thresholds 
    tissue_threshold: 0.7  # Threshold to consider a tile as Tissue
    blur_threshold: 0.015  # Threshold for laplace filter variance
    red_pen_check: 0.4  # Sanity check for % of red pen detected. If above threshold, red_pen mask will be ignored
    blue_pen_check: 0.4  # Sanity check for % of blue pen detected, If above threshold, blue_pen mask will be ignored
    include_adipose_tissue: false  # will include adipose tissue in mask
    remove_folds: false  # will remove folded tissue in mask
    mask_scale: null  # scale at which to downscale WSI for masking. Recommended is either 64 or None which will downsize to the lowest possible downscale recommended by openslide. None will produce a higher quality mask but is slower than 64
    
    # for devices + multithreading
    device: null  # device to use (e.g., 'cuda:0', 'cpu')
    gpu_processes: 1  # number of GPU processes
    cpu_processes: null  # number of CPU processes. If null will default to the available cpu count. 
    batch_size: 16  # batch size for processing
    
    # QC 
    min_tiles: 0  # Number of tiles a patient should have
    """
    if arg.save_path != "None":
        save = arg.save_path
    else:
        save = 'config.yml'

    # Write to config.yml
    with open(save, 'w') as f:
        f.write(yaml_str)
    print(f"config file saved at: {save}")
if __name__ == "__main__":
    main()