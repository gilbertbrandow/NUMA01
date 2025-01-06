import numpy as np
import numpy.typing as npt


class WaveletTransformationError(Exception):
    pass


class WaveletImage:
    def __init__(self, image_array: npt.NDArray) -> None:
        self._image_array: npt.NDArray = WaveletImage.normalize_array_shape(
            image_array).copy()
        self._image_array.setflags(write=True)

        self._iteration_count: int = 0


    @property
    def image_array(self) -> npt.NDArray:
        return self._image_array


    @property
    def iteration_count(self) -> int:
        return self._iteration_count


    @staticmethod
    def normalize_array_shape(array: npt.NDArray) -> npt.NDArray:
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
                "n must be an even integer greater than or equal to 2."
            )

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
        transformed_rows = WaveletImage.compute_haar_wavelet_matrix(
            rows) @ array
        return transformed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols).T
    
    
    @staticmethod 
    def apply_manual_wavelet_transform(array: npt.NDArray) -> npt.NDArray: 
        pass


    @staticmethod
    def apply_inverse_wavelet_transform(array: npt.NDArray) -> npt.NDArray:
        rows, cols = array.shape[:2]
        reconstructed_rows = WaveletImage.compute_haar_wavelet_matrix(
            rows).T @ array
        return reconstructed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols)


    @staticmethod 
    def apply_manual_inverse_wavelet_transform(array: npt.NDArray) -> npt.NDArray: 
        pass


    def get_subarray_shape(self) -> tuple[int, int]:
        height, width = self._image_array.shape

        factor: float = 2 ** -self.iteration_count
        subarray_height: int = int(height * factor)
        subarray_width: int = int(width * factor)

        return (subarray_height, subarray_width)


    def set_L_L_quadrant(self, new_quadrant: npt.NDArray) -> "WaveletImage":
        if self.iteration_count == 0:
            self._image_array = new_quadrant
            return None

        self._image_array[0:new_quadrant.shape[0],
                          0:new_quadrant.shape[1]] = new_quadrant
        return self


    def next(self, matrix_multiplication: bool = True) -> "WaveletImage":
        height, width = self.get_subarray_shape()

        if height < 2 or width < 2:
            raise WaveletTransformationError(
                f"Cannot perform transformation on subarray at iteration {self.iteration_count} "
                f"because its dimensions are too small {(height, width)}."
            )

        corner: npt.NDArray = self.normalize_array_shape(self._image_array.copy()[
            :int(height), :int(width)])

        corner = self.apply_wavelet_transform(corner) if matrix_multiplication else self.apply_manual_wavelet_transform(corner)
        
        self.set_L_L_quadrant(new_quadrant=corner)

        self._iteration_count += 1
        return self


    def prev(self, matrix_multiplication: bool = True) -> "WaveletImage":
        if self._iteration_count == 0:
            raise WaveletTransformationError(
                "Cannot inverse transformation on original image")

        self._iteration_count -= 1

        height, width = self.get_subarray_shape()
        corner: npt.NDArray = self.normalize_array_shape(self._image_array.copy()[
            :int(height), :int(width)])
        
        corner = self.apply_inverse_wavelet_transform(corner) if matrix_multiplication else self.apply_manual_inverse_wavelet_transform(corner)

        return self.set_L_L_quadrant(new_quadrant=corner)


    def go_to_iteration(self, iteration: int, matrix_multiplication: bool = True) -> "WaveletImage":
        if iteration < 0:
            raise WaveletTransformationError(
                "Can not inverse transformation beyond original image.")

        while self.iteration_count != iteration:
            if self.iteration_count < iteration:
                self.next(matrix_multiplication=matrix_multiplication)
            else:
                self.prev(matrix_multiplication=matrix_multiplication)

        return self
