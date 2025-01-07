import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Self
from PIL import Image

class WaveletTransformationError(Exception):
    """
    A custom exception to better understand wavelet-transformation-related errors. 
    This exception is raised when a wavelet transform operation encounters issues.
    """
    pass

class AbstractWaveletImage(ABC):
    @property
    def image_array(self) -> npt.NDArray:
        """
        Gets the underlying NumPy array for this wavelet image
               
        :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2024-07-01)
        :return: A representation of the image as a NumPy array
        """
        return np.array([])
    

    @abstractmethod
    def next(self, matrix_multiplication: bool = True) -> Self:
        """
        Advances the wavelet transform to the next iteration level.

        :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2024-07-01)
        :param matrix_multiplication: Determines whether to use matrix-based or manual transform.
        :return: Self, for method chaining.
        """
        pass

    @abstractmethod
    def prev(self, matrix_multiplication: bool = True) -> Self:
        """
        Reverses the wavelet transform by one iteration level.

        :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2024-07-01)
        :param matrix_multiplication: Determines whether to use matrix-based or manual inverse transform.
        :return: Self, for method chaining.
        """
        pass

    @abstractmethod
    def go_to_iteration(self, iteration: int, matrix_multiplication: bool = True) -> Self:
        """
        Moves the wavelet transform to a specific iteration level.

        :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2024-07-01)
        :param iteration: The target iteration level to move to.
        :param matrix_multiplication: Determines whether to use matrix-based or manual approach.
        :return: Self, for method chaining.
        """
        pass

class WaveletImage(AbstractWaveletImage):
    def __init__(self, image_array: npt.NDArray) -> None:
        self._image_array: npt.NDArray = WaveletImage.normalize_array_shape(
            image_array).copy()
        self._image_array.setflags(write=True)

        self._iteration_count: int = 0


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
    def apply_manual_wavelet_transform(array: npt.NDArray, weight: float = np.sqrt(2)) -> npt.NDArray:
        rows: int = array.shape[0]
        cols: int = array.shape[1]
        temp: npt.NDArray = np.zeros((rows, cols), dtype=float)
        out: npt.NDArray = np.zeros((rows, cols), dtype=float)
        factor: float = weight / 2.0

        for r in range(rows):
            row: npt.NDArray = array[r, :]
            half: int = cols // 2
            sums: npt.NDArray = np.zeros(half, dtype=float)
            diffs: npt.NDArray = np.zeros(half, dtype=float)
            for i in range(half):
                x: float = np.clip(row[2 * i], 0, 255)
                y: float = np.clip(row[2 * i + 1], 0, 255)
                sums[i] = factor * (x + y)
                diffs[i] = factor * (x - y)
            temp[r, :half] = sums
            temp[r, half:] = diffs

        for c in range(cols):
            col: npt.NDArray = temp[:, c]
            half = rows // 2
            sums = np.zeros(half, dtype=float)
            diffs = np.zeros(half, dtype=float)
            for i in range(half):
                x = np.clip(col[2 * i], 0, 255)
                y = np.clip(col[2 * i + 1], 0, 255)
                sums[i] = factor * (x + y)
                diffs[i] = factor * (x - y)
            out[:half, c] = sums
            out[half:, c] = diffs

        return out


    @staticmethod
    def apply_inverse_wavelet_transform(array: npt.NDArray) -> npt.NDArray:
        rows, cols = array.shape[:2]
        reconstructed_rows = WaveletImage.compute_haar_wavelet_matrix(
            rows).T @ array
        return reconstructed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols)


    @staticmethod
    def apply_manual_inverse_wavelet_transform(array: npt.NDArray) -> npt.NDArray:
        rows: int = array.shape[0]
        cols: int = array.shape[1]
        temp: npt.NDArray = np.zeros((rows, cols), dtype=float)
        out: npt.NDArray = np.zeros((rows, cols), dtype=float)
        factor: float = 1.0 / np.sqrt(2.0)

        for c in range(cols):
            col: npt.NDArray = array[:, c]
            half: int = rows // 2
            sums: npt.NDArray = col[:half]
            diffs: npt.NDArray = col[half:]
            reconstructed: npt.NDArray = np.zeros(rows, dtype=float)
            for i in range(half):
                s: float = sums[i]
                d: float = diffs[i]
                x: float = factor * (s + d)
                y: float = factor * (s - d)
                reconstructed[2 * i] = x
                reconstructed[2 * i + 1] = y
            temp[:, c] = reconstructed

        for r in range(rows):
            row: npt.NDArray = temp[r, :]
            half = cols // 2
            sums = row[:half]
            diffs = row[half:]
            reconstructed = np.zeros(cols, dtype=float)
            for i in range(half):
                s = sums[i]
                d = diffs[i]
                x = factor * (s + d)
                y = factor * (s - d)
                reconstructed[2 * i] = x
                reconstructed[2 * i + 1] = y
            out[r, :] = reconstructed

        return out


    def get_subarray_shape(self) -> tuple[int, int]:
        height, width = self._image_array.shape

        factor: float = 2 ** -self.iteration_count
        subarray_height: int = int(height * factor)
        subarray_width: int = int(width * factor)

        return (subarray_height, subarray_width)


    def set_L_L_quadrant(self, new_quadrant: npt.NDArray) -> "WaveletImage":
        if self.iteration_count == 0:
            self._image_array = new_quadrant
            return self

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

class RGBWaveletImage(AbstractWaveletImage):
    def __init__(self, image_array: npt.NDArray):
        channels: npt.NDArray = np.transpose(image_array, (2, 0, 1))
        self._channels: list[WaveletImage] = [
            WaveletImage(channels[i]) for i in range(channels.shape[0])
        ]


    @property
    def channels(self) -> list[WaveletImage]:
        return self._channels


    @property
    def image_array(self) -> npt.NDArray:
        channel_arrays: list[npt.NDArray] = []
        
        for wavelet_channel in self._channels:
            arr: npt.NDArray = np.asarray(
                Image.fromarray(wavelet_channel._image_array).convert("L")
            )
            channel_arrays.append(arr)

        return np.transpose(np.stack(channel_arrays, axis=0), (1, 2, 0))
    
    
    def next(self, matrix_multiplication: bool = True) -> Self:
        for channel in self._channels:
            channel.next(matrix_multiplication=matrix_multiplication)
        return self


    def prev(self, matrix_multiplication: bool = True) -> Self:
        for channel in self._channels:
            channel.prev(matrix_multiplication=matrix_multiplication)
        return self


    def go_to_iteration(self, iteration: int, matrix_multiplication: bool = True) -> Self:
        for channel in self._channels:
            channel.go_to_iteration(
                iteration=iteration,
                matrix_multiplication=matrix_multiplication
            )
        return self