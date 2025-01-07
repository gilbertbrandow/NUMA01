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
    """
    An abstract base class allowing us to use consistent type hints 
    for both grayscale (WaveletImage) and RGB (RGBWaveletImage) 
    instances in the WaveletImageIO class. 

    :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2025-01-07)
    """

    @property
    def image_array(self) -> npt.NDArray:
        """
        Returns a NumPy array representation of the image. This is an attempt at an abstract property.

        :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2025-01-07)
        :return: A representation of the image as a NumPy array.
        """
        return np.array([])

    @abstractmethod
    def next(self, matrix_multiplication: bool = True) -> Self:
        """
        Performs one level of the Haar-wavelet transform 
        on the image (or a subarray of it), depending on the current iteration.

        :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2025-01-07)
        :param matrix_multiplication: Determines whether to use matrix-based or manual transform.
        :return: Self, for method chaining.
        """
        pass

    @abstractmethod
    def prev(self, matrix_multiplication: bool = True) -> Self:
        """
        Reverses the Haar-wavelet transform by one iteration level, 
        essentially undoing the last forward transform.

        :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2025-01-07)
        :param matrix_multiplication: Determines whether to use matrix-based or manual inverse transform.
        :return: Self, for method chaining.
        """
        pass

    @abstractmethod
    def go_to_iteration(self, iteration: int, matrix_multiplication: bool = True) -> Self:
        """
        Moves directly to a specified iteration level, using the same 
        logic as 'next' or 'prev' for forward and inverse transforms.

        :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2025-01-07)
        :param iteration: The target iteration level to move to.
        :param matrix_multiplication: Determines whether to use 
                                      matrix-based or manual approach.
        :return: Self, for method chaining.
        """
        pass

class WaveletImage(AbstractWaveletImage):
    """
    A concrete implementation of a single-channel (grayscale) wavelet image.

    :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2025-01-07)
    """

    def __init__(self, image_array: npt.NDArray) -> None:
        """
        Initializes the WaveletImage with a normalized, writable NumPy array.

        :param image_array: The input array for the grayscale image.
        """
        self._image_array: npt.NDArray = WaveletImage.normalize_array_shape(image_array).copy()
        self._image_array.setflags(write=True)
        self._iteration_count: int = 0

    @property
    def iteration_count(self) -> int:
        """
        Returns the current iteration level.

        :return: The integer representing the current iteration count.
        """
        return self._iteration_count

    @staticmethod
    def normalize_array_shape(array: npt.NDArray) -> npt.NDArray:
        """
        Ensures both dimensions (height, width) are even by trimming a row or column if needed.

        :param array: The NumPy array to normalize.
        :return: A possibly trimmed version of the array with even dimensions.
        """
        height, width = array.shape[:2]
        if height % 2 != 0:
            array = array[:-1, :]
        if width % 2 != 0:
            array = array[:, :-1]
        return array

    @staticmethod
    def compute_haar_wavelet_matrix(n: int, weight: float = np.sqrt(2)) -> npt.NDArray:
        """
        Computes an n x n Haar wavelet transform matrix.

        :param n: The size of the matrix (must be even).
        :param weight: The scale factor (defaults to sqrt(2)).
        :return: The NumPy array representing the Haar matrix.
        """
        if n < 2 or n % 2 != 0:
            raise ValueError("n must be an even integer >= 2.")

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
        """
        Applies a single-level forward Haar transform via matrix multiplication.

        :param array: The 2D array to transform.
        :return: The transformed array (LL, LH, HL, HH).
        """
        rows, cols = array.shape[:2]
        transformed_rows = WaveletImage.compute_haar_wavelet_matrix(rows) @ array
        return transformed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols).T

    @staticmethod
    def apply_manual_wavelet_transform(array: npt.NDArray, weight: float = np.sqrt(2)) -> npt.NDArray:
        """
        Applies a single-level forward Haar transform by looping over rows & columns.

        :param array: The 2D array to transform.
        :param weight: The scale factor (defaults to sqrt(2)).
        :return: The transformed array (LL, LH, HL, HH).
        """
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
        """
        Applies a single-level inverse Haar transform via matrix multiplication.

        :param array: The array containing LL, LH, HL, HH.
        :return: The reconstructed 2D array.
        """
        rows, cols = array.shape[:2]
        reconstructed_rows = WaveletImage.compute_haar_wavelet_matrix(rows).T @ array
        return reconstructed_rows @ WaveletImage.compute_haar_wavelet_matrix(cols)

    @staticmethod
    def apply_manual_inverse_wavelet_transform(array: npt.NDArray) -> npt.NDArray:
        """
        Applies a single-level inverse Haar transform by looping over rows & columns.

        :param array: The array with LL, LH, HL, HH.
        :return: The reconstructed 2D array.
        """
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
        """
        Computes the subarray dimensions based on the current iteration level.

        :return: A tuple (height, width) for the subarray.
        """
        height, width = self._image_array.shape
        factor: float = 2 ** -self._iteration_count
        sub_h: int = int(height * factor)
        sub_w: int = int(width * factor)
        return (sub_h, sub_w)

    def set_L_L_quadrant(self, new_quadrant: npt.NDArray) -> Self:
        """
        Sets the top-left quadrant (LL) to the specified subarray.

        :param new_quadrant: The subarray to replace the LL region.
        :return: Self for method chaining.
        """
        if self._iteration_count == 0:
            self._image_array = new_quadrant
            return self
        self._image_array[: new_quadrant.shape[0], : new_quadrant.shape[1]] = new_quadrant
        return self

    def next(self, matrix_multiplication: bool = True) -> Self:
        """
        Moves one level forward in the wavelet transform.

        :param matrix_multiplication: Use matrix-based (True) or manual (False) transform.
        :return: Self, for method chaining.
        """
        height, width = self.get_subarray_shape()
        if height < 2 or width < 2:
            raise WaveletTransformationError(
                f"Cannot perform transformation on subarray at iteration {self._iteration_count} "
                f"because its dimensions are too small {(height, width)}."
            )
        corner: npt.NDArray = self.normalize_array_shape(
            self._image_array.copy()[:height, :width]
        )
        corner = (
            self.apply_wavelet_transform(corner)
            if matrix_multiplication
            else self.apply_manual_wavelet_transform(corner)
        )
        self.set_L_L_quadrant(new_quadrant=corner)
        self._iteration_count += 1
        return self

    def prev(self, matrix_multiplication: bool = True) -> Self:
        """
        Moves one level backward in the wavelet transform.

        :param matrix_multiplication: Use matrix-based (True) or manual (False) inverse transform.
        :return: Self, for method chaining.
        """
        if self._iteration_count == 0:
            raise WaveletTransformationError("Cannot inverse transform on original image.")

        self._iteration_count -= 1
        height, width = self.get_subarray_shape()
        corner: npt.NDArray = self.normalize_array_shape(
            self._image_array.copy()[:height, :width]
        )
        corner = (
            self.apply_inverse_wavelet_transform(corner)
            if matrix_multiplication
            else self.apply_manual_inverse_wavelet_transform(corner)
        )
        return self.set_L_L_quadrant(new_quadrant=corner)

    def go_to_iteration(self, iteration: int, matrix_multiplication: bool = True) -> Self:
        """
        Moves directly to a specific iteration level, using 'next' or 'prev' as needed.

        :param iteration: The target iteration level.
        :param matrix_multiplication: Determines whether to use matrix-based or manual approach.
        :return: Self, for method chaining.
        """
        if iteration < 0:
            raise WaveletTransformationError("Cannot go to negative iteration.")
        while self._iteration_count != iteration:
            if self._iteration_count < iteration:
                self.next(matrix_multiplication=matrix_multiplication)
            else:
                self.prev(matrix_multiplication=matrix_multiplication)
        return self

class RGBWaveletImage(AbstractWaveletImage):
    """
    An RGB wavelet image. Performs the wavelet transformation on each of
    the three channels separately.

    :author: Egor Jakimov.
    """

    def __init__(self, image_array: npt.NDArray):
        """
        Creates an RGB wavelet image from the array created from a
        PIL image.

        :param image_array: The image array. 
        """
        channels: npt.NDArray = np.transpose(image_array, (2, 0, 1))
        self._channels: list[WaveletImage] = [
            WaveletImage(channels[i]) for i in range(channels.shape[0])
        ]


    @property
    def channels(self) -> list[WaveletImage]:
        """
        Returns the three wavelet images comprising this RGB wavelet image

        :return: The three wavelet images
        """
        return self._channels


    @property
    def image_array(self) -> npt.NDArray:
        """
        Gets the three channels' image arrays, and converts them to a format
        supported by PIL. (PIL doesn't support floating-point RGB images, so
        we need to make them 8-bit by passing them through PIL.)

        :return: The NumPy array
        """

        channel_arrays: list[npt.NDArray] = []
        
        for wavelet_channel in self._channels:
            arr: npt.NDArray = np.asarray(
                Image.fromarray(wavelet_channel._image_array).convert("L")
            )
            channel_arrays.append(arr)

        return np.transpose(np.stack(channel_arrays, axis=0), (1, 2, 0))
    
    
    def next(self, matrix_multiplication: bool = True) -> Self:
        """
        Perform a Haar wavelet transformation on each channel.

        :param matrix_multiplication: Whether to use matrix multiplication or not
        :return: The image
        """

        for channel in self._channels:
            channel.next(matrix_multiplication=matrix_multiplication)
        return self


    def prev(self, matrix_multiplication: bool = True) -> Self:
        """
        Perform an Inverse Haar wavelet transformation on each channel.

        :param matrix_multiplication: Whether to use matrix multiplication or not
        :return: The image
        """

        for channel in self._channels:
            channel.prev(matrix_multiplication=matrix_multiplication)
        return self


    def go_to_iteration(self, iteration: int, matrix_multiplication: bool = True) -> Self:
        """
        Perform (inverse) wavelet transformations until a specific iteration level
        is reached.

        :param iteration: The target iteration level
        :param matrix_multiplication: Whether to use matrix multiplication or not
        :return: The image
        """

        for channel in self._channels:
            channel.go_to_iteration(
                iteration=iteration,
                matrix_multiplication=matrix_multiplication
            )
        return self