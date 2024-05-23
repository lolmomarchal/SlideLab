import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_size(size, scale):
    return size // scale
def SlideReconstruction(tile_information, output_file=None):
    """
    Reconstructs tissue slide based on provided tiles

    Parameters:
        tile_information (str): Path to a CSV file containing all tile information
        output_file (str): Optional path to save the composite image as a file.
    """
    tiles = pd.read_csv(tile_information)
    scale = tiles["scale"].iloc[0]
    max_x = (tiles['x'].max() + tiles['size'].max()) // scale
    max_y = (tiles['y'].max() + tiles['size'].max()) // scale

    composite_img = Image.new('RGB', (max_x, max_y), color='black')
    for _, row in tiles.iterrows():
        path_to_tile = row["path_to_slide"]
        size = row["size"]
        y = row["y"] // scale
        x = row["x"] // scale
        resized = get_size(size, scale)

        img = Image.open(path_to_tile).resize((resized, resized))

        composite_img.paste(img, (x, y))
    if output_file:
        composite_img.save(output_file)
        print(f"Composite image saved as {output_file}")
    else:
        composite_img.show()
    return composite_img
