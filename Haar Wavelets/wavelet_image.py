import numpy as np
import numpy.typing as npt
from PIL import Image

class WaveletTransformManager:
    def __init__(self, filepath: str) -> None:
        self._original_image: npt.NDArray = self.convert_image_to_array(filepath)
        self._history: list[WaveletImage] = [WaveletImage(self._original_image.copy())]


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


    def add_compressed_image(self) -> None:
        latest_image: WaveletImage = self._history[-1]
        new_image: WaveletImage = WaveletImage(latest_image.image_array.copy())
        new_image.apply_wavelet_transform()
        self._history.append(new_image)
        

    def get_latest_image(self) -> "WaveletImage": 
        return self._history[-1]

class WaveletImage:
    def __init__(self, image_array: npt.NDArray) -> None:
        self._image_array: npt.NDArray = image_array
        rows, cols = self._image_array.shape[:2]
        self._row_transform_matrix: npt.NDArray = self.compute_haar_wavelet_matrix(rows)
        self._col_transform_matrix: npt.NDArray = self.compute_haar_wavelet_matrix(cols)


    @property
    def image_array(self) -> npt.NDArray:
        return self._image_array
    

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


    def apply_wavelet_transform(self) -> None:
        transformed_rows = self._row_transform_matrix @ self._image_array
        self._image_array = transformed_rows @ self._col_transform_matrix.T

    def apply_inverse_wavelet_transform(self) -> None:
        reconstructed_rows = self._row_transform_matrix.T @ self._image_array
        self._image_array = reconstructed_rows @ self._col_transform_matrix
        
    def normalize_image(self) -> npt.NDArray:
        array = self.image_array
        normalized = (array - array.min()) / (array.max() - array.min()) * 255
        return normalized.astype(np.uint8)


    def save_image(self, filepath: str) -> None:
        normalized_array = self.normalize_image()
        newimg: Image.Image = Image.fromarray(normalized_array)
        newimg.save(filepath)
        print(f"Saved file to '{filepath}'")
            