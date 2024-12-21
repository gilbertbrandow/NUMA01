import numpy as np
from PIL import Image
from numpy.typing import NDArray
from wavelet_image import WaveletImage
from matplotlib import pyplot as plt

INPUT_FILEPATH: str = "./Resources/kvinna.jpg"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    try:
        wavelet_image = WaveletImage(INPUT_FILEPATH)
        wavelet_image.save_to_file(OUTPUT_FILEPATH)
    except FileNotFoundError:
        print(f"Error: The file {INPUT_FILEPATH} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    array: NDArray = convert_image_to_array("./Resources/kvinna.jpg")
    array = normalize_array_shape(array)
    
    return None

def convert_image_to_array(filepath: str) -> NDArray:
    """
    Reads an image from the given file path and returns it as a NumPy array
    """
    image: Image.Image = Image.open(filepath)
    return np.asarray(image)

def normalize_array_shape(array: np.ndarray) -> np.ndarray:
    """
    Normalizes the shape of the array
    """
    height: int = array.shape[0]
    width: int = array.shape[1]
    
    if height % 2 != 0:
        array = array[:-1, :]
    
    if width % 2 != 0:
        array = array[:, :-1]
        
    return array

if __name__ == "__main__":
    main()