from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Self
from wavelet_image import WaveletImage
from PIL import Image

class AbstractWaveletImage(ABC):
    @abstractmethod
    #@property
    def image_array(self) -> npt.NDArray:
        pass
    
    @abstractmethod
    def next(self, matrix_multiplication: bool = True) -> Self:
        pass

    @abstractmethod
    def prev(self, matrix_multiplication: bool = True) -> Self:
        pass

    @abstractmethod
    def go_to_iteration(self, iteration: int, matrix_multiplication: bool = True) -> Self:
        pass

class RGBWaveletImage(AbstractWaveletImage):
    def __init__(self, image_array: npt.NDArray):
        channels = np.transpose(image_array, (2, 0, 1))
        self._red, self._green, self._blue = (WaveletImage(channel) for channel in channels)
    
    def next(self, matrix_multiplication: bool = True) -> Self:
        self._red.next(matrix_multiplication=matrix_multiplication)
        self._green.next(matrix_multiplication=matrix_multiplication)
        self._blue.next(matrix_multiplication=matrix_multiplication)
        return self
    
    def prev(self, matrix_multiplication: bool = True) -> Self:
        self._red.prev(matrix_multiplication=matrix_multiplication)
        self._green.prev(matrix_multiplication=matrix_multiplication)
        self._blue.prev(matrix_multiplication=matrix_multiplication)
        return self
    
    def go_to_iteration(self, iteration: int, matrix_multiplication: bool = True) -> Self:
        self._red.go_to_iteration(iteration=iteration, matrix_multiplication=matrix_multiplication)
        self._green.go_to_iteration(iteration=iteration, matrix_multiplication=matrix_multiplication)
        self._blue.go_to_iteration(iteration=iteration, matrix_multiplication=matrix_multiplication)
        return self
    
    #@property
    def image_array(self) -> npt.NDArray:
        r = np.asarray(Image.fromarray(self._red._image_array).convert("L"))
        g = np.asarray(Image.fromarray(self._green._image_array).convert("L"))
        b = np.asarray(Image.fromarray(self._blue._image_array).convert("L"))
        return np.transpose(np.array([r, g, b]), (1, 2, 0))