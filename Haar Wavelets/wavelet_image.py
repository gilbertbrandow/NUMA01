import numpy as np
import numpy.typing as npt
from PIL import Image
from typing import Union

"""
class WaveletTransformManager:
    def __init__(self, filepath: str) -> None:
        self._original_image: npt.NDArray = self.convert_image_to_array(filepath)
        self._history: list[WaveletImage] = [WaveletImage(self._original_image.copy())]


    def convert_image_to_array(self, filepath: str) -> npt.NDArray:
        image: Image.Image = Image.open(filepath).convert("L")
        return self.normalize_array_shape(np.asarray(image))


    def normalize_array_shape(self, array: npt.NDArray) -> npt.NDArray:
        height, width = array.shape[:2]
        if height % 2 != 0:
            array = array[:-1, :]
        if width % 2 != 0:
            array = array[:, :-1]
        return array


    def add_compressed_image(self) -> None:
        latest_image: WaveletImage = self._history[-1]
        new_image: WaveletImage = WaveletImage(latest_image.image_array.copy())
        new_image.apply_wavelet_transform()
        self._history.append(new_image)
        

    def get_latest_image(self) -> "WaveletImage": 
        return self._history[-1]
"""

class WaveletImage:
    @staticmethod
    def normalize_array_shape( array: npt.NDArray) -> npt.NDArray:
        height, width = array.shape[:2]
        if height % 2 != 0:
            array = array[:-1, :]
        if width % 2 != 0:
            array = array[:, :-1]
        return array

    def __init__(self, image_array: Union[npt.NDArray, str]) -> None:
        self._image_array: npt.NDArray

        if isinstance(image_array, str):
            self._image_array = WaveletImage.normalize_array_shape(np.asarray(Image.open(image_array).convert("L")))
        elif isinstance(image_array, npt.NDArray):
            self._image_array = WaveletImage.normalize_array_shape(image_array)
        else:
            raise TypeError("image_array must be a numpy array or a string")
        self._iteration_count: int = 0

        #rows, cols = self._image_array.shape[:2]
        #self._row_transform_matrix: npt.NDArray = self.compute_haar_wavelet_matrix(rows)
        #self._col_transform_matrix: npt.NDArray = self.compute_haar_wavelet_matrix(cols)


    @property
    def image_array(self) -> npt.NDArray:
        return self._image_array
    

    @staticmethod
    def compute_haar_wavelet_matrix(n: int, weight: float = 1) -> npt.NDArray:
        if n < 2 or n % 2 != 0:
            raise ValueError("n must be an even integer greater than or equal to 2.")
        
        HWT: npt.NDArray = np.zeros((n, n))
        for i in range(n // 2):
            HWT[i, 2 * i] = weight / 2.0
            HWT[i, 2 * i + 1] = weight / 2.0
            
        for i in range(n // 2):
            HWT[n // 2 + i, 2 * i] = -weight / 2.0
            HWT[n // 2 + i, 2 * i + 1] = weight / 2.0
        
        return HWT

    @staticmethod
    def apply_wavelet_transform(array: npt.NDArray) -> npt.NDArray:
        rows, cols = array.shape[:2]
        transformed_rows = WaveletImage.compute_haar_wavelet_matrix(rows) @ array
        return abs((transformed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols).T)).astype(np.uint8)
    
    @staticmethod
    def apply_inverse_wavelet_transform(array: npt.NDArray) -> npt.NDArray:
        rows, cols = array.shape[:2]
        reconstructed_rows = WaveletImage.compute_haar_wavelet_matrix(rows).T @ array
        return np.clip(reconstructed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols).astype(np.uint8), 0, 255)

    def save_image(self, filepath: str) -> None:
        newimg: Image.Image = Image.fromarray(self.image_array)
        newimg.save(filepath)
        print(f"Saved file to '{filepath}'")
    
    @property
    def iteration_count(self) -> int:
        return self._iteration_count
    
    @staticmethod
    def upper_left_quadrant(array: npt.NDArray) -> npt.NDArray:
        return array[:array.shape[0] // 2, :array.shape[1] // 2]
    
    @staticmethod
    def set_upper_left_corner(array: npt.NDArray, new_corner: npt.NDArray) -> npt.NDArray:
        array[:new_corner.shape[0], :new_corner.shape[1]] = new_corner
        return array

    def next(self) -> "WaveletImage":
        corner: npt.NDArray = self._image_array
        for _ in range(self._iteration_count): # If we're e.g. one iteration deep, no corner needed (use entire image)
            corner = WaveletImage.upper_left_quadrant(corner)

        self._image_array = WaveletImage.set_upper_left_corner(self._image_array.copy(), WaveletImage.apply_wavelet_transform(WaveletImage.normalize_array_shape(corner)))
        
        self._iteration_count += 1
        return self
    
    
    def prev(self) -> "WaveletImage":
        if self._iteration_count == 0:
            raise Exception("Cannot go back further")
        
        corner: npt.NDArray = self._image_array
        for _ in range(self._iteration_count - 1):
            corner = WaveletImage.upper_left_quadrant(corner)
        
        corner = WaveletImage.apply_inverse_wavelet_transform(WaveletImage.normalize_array_shape(corner))
        self._image_array = WaveletImage.set_upper_left_corner(self._image_array.copy(), corner)

        self._iteration_count -= 1
        return self