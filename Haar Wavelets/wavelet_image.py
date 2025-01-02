import numpy as np
import numpy.typing as npt
from PIL import Image

class WaveletImage:
    def __init__(self, filepath: str) -> None:
        self.set_image_array(self.convert_image_to_array(filepath))
        
        rows, cols = self._image_array.shape[:2]
        self._row_transform_matrix: npt.NDArray = self.compute_haar_wavelet_matrix(rows)
        self._col_transform_matrix: npt.NDArray = self.compute_haar_wavelet_matrix(cols)


    @property
    def image_array(self) -> npt.NDArray:
        return self._image_array
    

    def set_image_array(self, image_array: npt.NDArray) -> None: 
        self._image_array: npt.NDArray = image_array
        return 

    
    @property
    def row_transform_matrix(self) -> npt.NDArray:
        return self._row_transform_matrix
    
    
    @property
    def col_transform_matrix(self) -> npt.NDArray:
        return self._col_transform_matrix
    
    
    def convert_image_to_array(self, filepath: str) -> npt.NDArray:
        image: Image.Image = Image.open(filepath).convert("L")
        return self.normalize_array_shape(np.asarray(image))


    def normalize_array_shape(self, array: npt.NDArray) -> npt.NDArray:
        height, width = array.shape[:2]

        if height % 2 != 0:
            array = array[:-1, :]
        if width % 2 != 0:
            array = array[:, :-1]

        return array
    
    def save_to_file(self, filepath: str) -> None: 
        newimg: Image.Image = Image.fromarray (self._image_array)
        newimg.save (filepath)
        print(f"Image saved to {filepath}")
        
    def get_wavelet_transformed(self) -> npt.NDArray:
        transformed_image: npt.NDArray = self.apply_wavelet_transform()
        normalized_image: npt.NDArray = np.clip(transformed_image, 0, 255).astype(np.uint8)

        return normalized_image
    

    def compute_haar_wavelet_matrix(self, n: int, weight: float = np.sqrt(2)) -> npt.NDArray:
        if n < 2 or n % 2 != 0:
            raise ValueError("n must be an even integer greater than or equal to 2.")
        
        HWT: npt.NDArray = np.zeros((n, n))

        for i in range(n // 2):
            HWT[i, 2 * i] = weight / 2
            HWT[i, 2 * i + 1] = weight / 2

        for i in range(n // 2):
            HWT[n // 2 + i, 2 * i] = -weight / 2
            HWT[n // 2 + i, 2 * i + 1] = weight / 2

        return HWT
