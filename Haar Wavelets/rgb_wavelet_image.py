from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Self
from wavelet_image import WaveletImage
from PIL import Image

class AbstractWaveletImage(ABC):
    @property
    def image_array(self) -> npt.NDArray:
        return np.array([])
    
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