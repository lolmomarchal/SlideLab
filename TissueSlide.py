OPENSLIDE_PATH = r"C:\Users\albao\Downloads\openslide-win64-20231011\openslide-win64-20231011\bin"
import os

# if hasattr(os, 'add_dll_directory'):
#     # Windows
#     with os.add_dll_directory(OPENSLIDE_PATH):
#         import openslide
# else:
#     import openslide
import openslide
import numpy as np


class TissueSlide:
    def __init__(self, slide_path, SCALE=32):
        """
      Initializes a TissueSlide object.
      Parameters:
          slide_path (str): The file path to the tissue slide image.
          SCALE (int): An optional parameter specifying the scale for generating the thumbnail. Default is 32.
      """
        self.path = slide_path
        try:
            self.slide = openslide.OpenSlide(self.path)
            self.magnification = int(self.slide.properties.get("openslide.objective-power"))
            #self.SCALE = int(self.slide.level_downsamples[-1])
            self.SCALE = SCALE

            self.thumbnail = self.slide.get_thumbnail(
                (self.slide.dimensions[0] // SCALE, self.slide.dimensions[1] // SCALE))
            self.dimensions = self.slide.dimensions
            self.id = self.path.split("\\")[-1].split(".")[0]
        except openslide.OpenSlideError as e:
            print(f"An error occurred while opening the slide: {e}")
            self.slide = None
            self.magnification = None
            self.thumbnail = None
            self.dimensions = None
            self.id = None
            self.SCALE = None

    def get_slide(self):
        """
       Retrieves the slide
       Returns:
           openslide.OpenSlide: An object representing the full-resolution slide image.
       """
        return self.slide

    def get_thumbnail(self):
        """
      Retrieves the thumbnail image of the slide.
      Returns:
          Numpy Array -> An object representing the thumbnail image.
      """
        return np.array(self.thumbnail)

    def get_magnification(self):
        """
       Retrieves the natural magnification level of the slide.
       Returns:
           int: The magnification level.
       """
        return self.magnification

    def get_slide_path(self):
        """
      Retrieves the file path of the slide image.
      Returns:
          str: The file path of the slide image.
      """
        return self.path

    def get_id(self):
        """
      Retrieves the ID of the slide.
      Returns:
          str: The ID of the slide. i.e 'TCGA-BH-A1EV-11A-01-TSA'
      """
        return self.id

    def get_dimensions(self):
        """
     Retrieves the dimensions of the slide image.
     Returns:
         Tuple[int, int]: A tuple representing the dimensions (width, height) of the slide image.
     """
        return self.dimensions
