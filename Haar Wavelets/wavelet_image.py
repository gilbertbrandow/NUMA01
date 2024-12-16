import numpy as np
import numpy.typing as npt
from PIL import Image

class WaveletImage:
    def __init__(self, filepath: str) -> None:
        self._array: npt.NDArray = self.convert_image_to_array(filepath)
        self._array = self.normalize_array_shape()


    def convert_image_to_array(self, filepath: str) -> npt.NDArray:
        image: Image.Image = Image.open(filepath)
        return np.asarray(image)


    def normalize_array_shape(self) -> npt.NDArray:
        height: int = self._array.shape[0]
        width: int = self._array.shape[1]

        if height % 2 != 0:
            self._array = self._array[:-1, :]
            
        if width % 2 != 0:
            self._array = self._array[:, :-1]

        return self._array
    
    
    def save_image_to_file(self, filepath: str="./Resources/new_kvinna.jpg") -> None: 
        newimg: Image.Image = Image.fromarray (self._array)
        newimg.save (filepath)