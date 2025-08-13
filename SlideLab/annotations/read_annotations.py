import numpy as np
import pandas as pd
import pyvips
import cv2
import os
# get the annotation path
def find_annotation_file(directory, slide_path):
    items = os.listdir(directory)
    for item in items:
        # get both the slide id and the mask
        slide_id, _= os.path.basename(os.path.splitext(slide_path))
        item_id, _= os.path.splitext(item)
        if item_id in slide_id or slide_id in item_id:
            return os.path.join(directory, item)
    return None
class AnnotationReader:
    def __init__(self,annotation_file, original_dimensions, annotation_backend = None, scale = 64):
        # can be three main types of annotation_backends 1. ASAP (.xml) 2. QuPATH (.geojson) 3. Numpy binary file
        self.file = annotation_file
        self.original_dimensions = original_dimensions
        self.scale = scale
        if annotation_backend is None:
            # guess backend
            end = self.file.split(".")[-1]
            if end == "geojson":
                self.annotation_backend = "QuPATH"
            elif end == "xml":
                self.annotation_backend = "ASAP"
            elif end == "npy":
                self.annotation_backend = "numpy_array"
            elif end == "png" or end == "png" or end =="tif":
                self.annotation_backend = "pyvips"
        else:
            self.annotation_backend = annotation_backend



    def _read_annotation_xml(self):


    def _read_annotation
        




