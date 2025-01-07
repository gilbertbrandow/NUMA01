from PIL import Image
import numpy.typing as npt
import numpy as np
#from wavelet_image import WaveletImage
from rgb_wavelet_image import RGBWaveletImage

class WaveletImageIO(object):
    @staticmethod
    def from_file(filepath: str) -> RGBWaveletImage:
        image: Image.Image = Image.open(filepath).convert("RGB")
        array: npt.NDArray = np.asarray(image)
        return RGBWaveletImage(array)

    @staticmethod
    def to_file(wavelet_image: "RGBWaveletImage", filepath: str, only_compressed: bool = False) -> None:
        #TODO: Handle cases where only the subarray (upper left) image should be saved
        #wavelet_image._blue.next().prev()
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
