import numpy as np
import numpy.typing as npt
from PIL import Image

class WaveletImage:
    def __init__(self, filepath: str) -> None:
        self._image_array: npt.NDArray = self.convert_image_to_array(filepath)


    @property
    def image_array(self) -> npt.NDArray:
        return self._image_array
    
    
    def convert_image_to_array(self, filepath: str) -> npt.NDArray:
        image: Image.Image = Image.open(filepath)
        return self.normalize_array_shape(np.asarray(image))


    def normalize_array_shape(self, array: npt.NDArray) -> npt.NDArray:
        height, width = array.shape[:2]

        if height % 2 != 0:
            array = array[:-1, :]
        if width % 2 != 0:
            array = array[:, :-1]

        return array
    
    
    def save_to_file(self, filepath: str) -> None: 
        newimg: Image.Image = Image.fromarray (self._image_array)
        newimg.save (filepath)
        print(f"Image saved to {filepath}")
        
        
    def compute_haar_wavelet_matrix(self, n: int) -> npt.NDArray:
        if n < 2 or n % 2 != 0:
            raise ValueError("n must be an even integer greater than or equal to 2.")
        
        HWT = np.zeros((n, n))

        for i in range(n // 2):
            HWT[i, 2 * i] = 1 / np.sqrt(2)
            HWT[i, 2 * i + 1] = 1 / np.sqrt(2)

        for i in range(n // 2):
            HWT[n // 2 + i, 2 * i] = 1 / np.sqrt(2)
            HWT[n // 2 + i, 2 * i + 1] = -1 / np.sqrt(2)

        return HWT
