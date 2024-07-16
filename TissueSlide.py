import os
import openslide
import numpy as np


class TissueSlide:
    def __init__(self, slide_path, ID=None):
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
            self.SCALE = int(self.slide.level_downsamples[-1])
            self.thumbnail = self.slide.get_thumbnail(
                (self.slide.dimensions[0] // self.SCALE, self.slide.dimensions[1] // self.SCALE))
            self.dimensions = self.slide.dimensions
            if ID is not None:
                self.id = ID
            else:
                self.id = os.path.basename(self.path).split(".")[0]
        except openslide.OpenSlideError as e:
            print(f"An error occurred while opening the slide: {e}")
            self.slide = None
            self.magnification = None
            self.thumbnail = None
            self.dimensions = None
            self.id = None
            self.SCALE = None

    def metadata(self):
        """Retrieves the different metadata for the Tissue Slide
        Returns:
            Metadata Dictionary
        """
        return {"slide": self.slide, "magnification": self.magnification, "thumbnail": self.thumbnail,
                "dimensions": self.dimensions, "id": self.id, "scale": self.SCALE}

    def get_slide(self) -> openslide.OpenSlide:
        """
       Retrieves the slide
       Returns:
           openslide.OpenSlide: An object representing the full-resolution slide image.
       """
        return self.slide

    def get_thumbnail(self) -> np.array:
        """
      Retrieves the thumbnail image of the slide.
      Returns:
          Numpy Array -> An object representing the thumbnail image.
      """
        return np.array(self.thumbnail)

    def get_magnification(self) -> int:
        """
       Retrieves the natural magnification level of the slide.
       Returns:
           int: The magnification level.
       """
        return self.magnification

    def get_slide_path(self) -> str:
        """
      Retrieves the file path of the slide image.
      Returns:
          str: The file path of the slide image.
      """
        return self.path

    def get_id(self) -> str:
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
