import numpy as np
import numpy.typing as npt
from PIL import Image

class WaveletImage:
    def __init__(self, filepath: str) -> None:
        self._image_array: npt.NDArray = self.convert_image_to_array(filepath)
        self._image_array = self.normalize_array_shape()


    @property
    def image_array(self) -> npt.NDArray:
        """
        Returns the underlying image array.
        """
        return self._image_array
    
    def convert_image_to_array(self, filepath: str) -> npt.NDArray:
        image: Image.Image = Image.open(filepath)
        return np.asarray(image)


    def normalize_array_shape(self) -> npt.NDArray:
        height: int = self._image_array.shape[0]
        width: int = self._image_array.shape[1]

        if height % 2 != 0:
            self._image_array = self._image_array[:-1, :]
            
        if width % 2 != 0:
            self._image_array = self._image_array[:, :-1]

        return self._image_array
    
    
    def save_to_file(self, filepath: str="./Resources/new_kvinna.jpg") -> None: 
        newimg: Image.Image = Image.fromarray (self._image_array)
        newimg.save (filepath)
        print(f"Image saved to {filepath}")