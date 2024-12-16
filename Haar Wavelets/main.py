import numpy as np
from PIL import Image
from numpy.typing import NDArray
from matplotlib import pyplot as plt

def main() -> None:
    return None

def convert_image_to_array(filepath: str) -> NDArray:
    """
    Reads an image from the given file path and returns it as a NumPy array
    """
    image: Image.Image = Image.open(filepath)
    return np.asarray(image)

if __name__ == "__main__":
    main()