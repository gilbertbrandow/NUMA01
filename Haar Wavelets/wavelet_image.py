import numpy as np
import numpy.typing as npt

class WaveletTransformationError(Exception):
    pass

class WaveletImage:
    def __init__(self, image_array: npt.NDArray) -> None:
        self._image_array: npt.NDArray = WaveletImage.normalize_array_shape(image_array).copy()
        self._image_array.setflags(write=True)
        
        self._iteration_count: int = 0
        

    @property
    def image_array(self) -> npt.NDArray:
        return self._image_array
    
    
    @property
    def iteration_count(self) -> int:
        return self._iteration_count


    @staticmethod
    def normalize_array_shape( array: npt.NDArray) -> npt.NDArray:
        height, width = array.shape[:2]
        if height % 2 != 0:
            array = array[:-1, :]
        if width % 2 != 0:
            array = array[:, :-1]
        return array


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
    
    
    def get_subarray_shape(self) -> tuple[int, int]:
        height, width = self._image_array.shape
    
        factor: float = 2 ** -self.iteration_count
        subarray_height: int = int(height * factor)
        subarray_width: int  = int(width * factor)
        
        return (subarray_height, subarray_width)

    
    def set_L_L_quadrant(self, new_corner: npt.NDArray) -> "WaveletImage":
        if self.iteration_count == 0:
            self._image_array = new_corner
            return None
        
        self._image_array[0:new_corner.shape[0], 0:new_corner.shape[1]] = new_corner
        return self


    def next(self) -> "WaveletImage":
        height, width = self.get_subarray_shape()
        corner: npt.NDArray = self._image_array.copy()[:int(height), :int(width)]
    
        corner = WaveletImage.apply_wavelet_transform(WaveletImage.normalize_array_shape(corner))
        self.set_L_L_quadrant(new_corner=corner)
        
        self._iteration_count += 1
        return self
    
    
    def prev(self) -> "WaveletImage":
        if self._iteration_count == 0:
            raise WaveletTransformationError("Cannot inverse transformation on original image")

        self._iteration_count -= 1
        
        height, width = self.get_subarray_shape()
        corner: npt.NDArray = self._image_array.copy()[:int(height), :int(width)]
        corner = WaveletImage.apply_inverse_wavelet_transform(WaveletImage.normalize_array_shape(corner))
        
        
        return self.set_L_L_quadrant(new_corner=corner)


    def go_to_iteration(self, iteration: int) -> "WaveletImage":
        if iteration < 0:
            raise WaveletTransformationError("Can not inverse transformation beyond original image.")

        while self.iteration_count != iteration:
            if self.iteration_count < iteration:
                self.next()
            else:
                self.prev()
        
        return self
