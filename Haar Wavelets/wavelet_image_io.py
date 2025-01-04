from PIL import Image
from wavelet_image import WaveletImage

class WaveletImageIO(object):
    def save_image(self, wavelet_image: WaveletImage, filepath: str) -> None:
        newimg: Image.Image = Image.fromarray(wavelet_image.image_array).convert("L")
        newimg.save(filepath)
        print(f"Saved file to '{filepath}'")
        
    def show_image(self) -> None:
        #TODO: Maybe use pyplot? And maybe display a green border between quadrants for visibility
        pass


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
