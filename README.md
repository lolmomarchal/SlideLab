# WSI_preprocessing
Whole Slide Image Preprocessing

**Currently in process:**
- masking utilities -> TissueMask.py (TissueMask Class)
- Slide metadata -> Slide_utils (TissueSlide class)
- Normalization -> normalization
- Preprocessing -> preprocessing2.py (controls whole workflow)
- encoding.py -> feature extraction for tiles


Histopathology Whole Slide Image Processor
# Histopathology Whole Slide Image Processor

This Python package is designed to facilitate the processing of whole slide images (WSIs) commonly used in histopathology research. It provides functionalities for preprocessing WSIs and supports multi-instance learning for advanced analysis.

## Features

- **WSI Preprocessing**: The package allows users to preprocess whole slide images, including tasks such as resizing, normalization, and feature extraction.
- **Multi-instance Learning Support**: Incorporates algorithms for multi-instance learning, enabling more sophisticated analysis of histopathological data.
- **Parallel Processing**: Utilizes multiprocessing to speed up computation by distributing tasks across multiple threads.

## Usage

### Installation

(currently not implemented)
```bash
pip install wsi-processor


Options
-i, --input_path: Path to the directory containing input WSIs.
-o, --output_path: Path to the directory where processed results will be saved.
-p, --processes: Number of threads for multiprocessing.


Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to help improve this package.

License
This package is licensed under the MIT License. See the LICENSE file for details.
