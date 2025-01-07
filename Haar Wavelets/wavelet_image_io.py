from PIL import Image
import numpy.typing as npt
import numpy as np
from wavelet_image import AbstractWaveletImage, WaveletImage, RGBWaveletImage

class WaveletImageIO(object):
    @staticmethod
    def from_file(filepath: str, only_grayscale=False) -> AbstractWaveletImage:
        """
        Loads image from path & instantiates either RGBWaveletImage or WaveletImage 
        based on only_grayscale argument.

        :param filepath: The path of the image to be loaded.
        :param only_grayscale: A flag for determining if preserving rgb values or not, 
        :return: an AbstractWaveletImage, either RGB or grayscale
        """
        image: Image.Image = Image.open(filepath).convert("L" if only_grayscale else "RGB")
        return WaveletImage(np.asarray(image)) if only_grayscale else RGBWaveletImage(np.asarray(image))
    
    
    @staticmethod
    def to_file(wavelet_image: AbstractWaveletImage, filepath: str) -> None:
        """
        Saves image from AbstractWaveletImage instance to the provided filepath.

        :param wavelet_image: The instance of wavelet image that should be saved.
        :param filepath: The destination to which save image
        :return: None
        """
        img: Image.Image = Image.fromarray(wavelet_image.image_array).convert("L" if isinstance(wavelet_image, WaveletImage) else "RGB")
            
        img.save(filepath)
        print(f"Saved image to {filepath}")
