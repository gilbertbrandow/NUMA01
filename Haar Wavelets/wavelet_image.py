import numpy as np
import numpy.typing as npt
from PIL import Image
from typing import Union

"""
class WaveletTransformManager:
    def __init__(self, filepath: str) -> None:
        self._original_image: npt.NDArray = self.convert_image_to_array(filepath)
        self._history: list[WaveletImage] = [WaveletImage(self._original_image.copy())]
=======

class WaveletTransformationError(Exception):
    pass


class WaveletImageIO(object):
    @staticmethod
    def from_file(filepath: str) -> "WaveletImage":
        image: Image.Image = Image.open(filepath).convert("L")
        array: npt.NDArray = np.asarray(image)
        return WaveletImage(array)


    @staticmethod
    def to_file(wavelet_image: "WaveletImage", filepath: str, only_subarray: bool = False) -> None:
        #TODO: Handle cases where only the subarray (upper left) image should be saved
        img: Image.Image = Image.fromarray(wavelet_image.image_array)
        img.save(filepath)
        print(f"Saved image to {filepath}")


    @staticmethod
    def from_bytes(data: bytes) -> "WaveletImage":
        pass
    
    
    @staticmethod 
    def to_bytes(wavelet_image: "WaveletImage") -> str: 
        pass
    

    @staticmethod
    def save_quadrants(self, wavelet_image: "WaveletImage") -> None:
        # TODO: The whole purpose of haar wavelet is the ability to send an image in parts, maybe we should implement this functionality?
        pass
    
    
    @staticmethod
    def reconstruct_from_quadrants(quadrants: dict[str, npt.NDArray]) -> "WaveletImage":
        # TODO: Reconstruct WaveletImage from quadrants
        pass
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
    def compute_haar_wavelet_matrix(n: int, weight: float = np.sqrt(2)) -> npt.NDArray:
        if n < 2 or n % 2 != 0:
            raise ValueError(
                "n must be an even integer greater than or equal to 2.")

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
        return transformed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols).T
    
    @staticmethod
    def apply_inverse_wavelet_transform(array: npt.NDArray) -> npt.NDArray:
        rows, cols = array.shape[:2]
        reconstructed_rows = WaveletImage.compute_haar_wavelet_matrix(rows).T @ array
        return reconstructed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols)

    def save_image(self, filepath: str) -> None:
        if not self.iteration_count is 0: 
            self._image_array *= 1/np.sqrt(2)
            pass
        
        
        newimg: Image.Image = Image.fromarray(self.image_array * 1/np.sqrt(2) ** self.iteration_count).convert("L")
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

        if array.shape == new_corner.shape:
            return new_corner
        
        array[0:new_corner.shape[0], 0:new_corner.shape[1]] = new_corner
        return array

    def next(self) -> "WaveletImage":
        corner: npt.NDArray = self._image_array.copy()
        for _ in range(self._iteration_count): # If we're e.g. one iteration deep, no corner needed (use entire image)
            corner = WaveletImage.upper_left_quadrant(corner)
    
        corner = WaveletImage.apply_wavelet_transform(WaveletImage.normalize_array_shape(corner))
        self._image_array = WaveletImage.set_upper_left_corner(self._image_array.copy(), corner)
        
        self._iteration_count += 1
        return self
    
    
    def prev(self) -> "WaveletImage":
        if self._iteration_count == 0:
            raise Exception("Cannot go back further")
        
        corner: npt.NDArray = self._image_array.copy()
        for _ in range(self._iteration_count - 1):
            corner = WaveletImage.upper_left_quadrant(corner)
        
        corner = WaveletImage.apply_inverse_wavelet_transform(WaveletImage.normalize_array_shape(corner))
        self._image_array = WaveletImage.set_upper_left_corner(self._image_array.copy(), corner)

        self._iteration_count -= 1
        return self


    def get_sub_array_dimensions(self) -> tuple[int, int]:
        rows: int = int(2**-self.current_iteration * self.image_array.shape[0])
        columns: int = int(2**-self.current_iteration * self.image_array.shape[1])
        
        rows = rows if rows % 2 == 0 else rows - 1
        columns = columns if columns % 2 == 0 else columns - 1
        
        return (rows, columns)


    def apply_wavelet_transform(self, rows: int, cols: int) -> None:
        subarray: npt.NDArray = self._image_array[:rows, :cols]
        transformed_rows = self._row_transform_matrix[:rows, :rows] @ subarray
        transformed_subarray = transformed_rows @ self._col_transform_matrix[:cols, :cols].T
        self._image_array[:rows, :cols] = np.clip(transformed_subarray, 0, 255).astype(np.uint8)
        

    def next(self) -> "WaveletImage":
        rows, cols = self.get_sub_array_dimensions()
        self.apply_wavelet_transform(rows=rows, cols=cols)

        self._current_iteration += 1
        return self


    def prev(self) -> "WaveletImage":
        if (self.current_iteration == 0):
            raise WaveletTransformationError("Can not inverse transformation beyond original image.")

        # TODO: Inverse HWT on subarray

        self._current_iteration -= 1
        return self


    def go_to_iteration(self, iteration: int) -> "WaveletImage":
        if iteration < 0:
            raise WaveletTransformationError("Can not inverse transformation beyond original image.")
        elif abs(iteration - self.current_iteration) > 10:
            raise WaveletTransformationError("Too many iterations.")

        while self.current_iteration != iteration:
            if self.current_iteration < iteration:
                self.next()
            else:
                self.prev()
        
        return self
    

    def show_image(self) -> None:
        #TODO: Maybe use pyplot? And maybe display a green border between quadrants for visibility
        pass
