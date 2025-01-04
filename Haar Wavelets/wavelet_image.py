import numpy as np
import numpy.typing as npt
from PIL import Image


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


class WaveletImage:
    def __init__(self, image_array: npt.NDArray, weight: int = 1.0) -> None:
        self._image_array: npt.NDArray = self.normalize_array_shape(image_array.copy())
        self._image_array.setflags(write=True)
        self._current_iteration: int = 0

        self._row_transform_matrix: npt.NDArray = self.compute_haar_wavelet_matrix(
            n=self.image_array.shape[0],
            weight=weight
        )
        
        self._col_transform_matrix: npt.NDArray = self.compute_haar_wavelet_matrix(
            n=self.image_array.shape[1],
            weight=weight
        )


    @property
    def image_array(self) -> npt.NDArray:
        return self._image_array


    @property
    def current_iteration(self) -> npt.NDArray:
        return self._current_iteration


    @staticmethod 
    def normalize_array_shape(array: npt.NDArray) -> npt.NDArray:
        height, width = array.shape[:2]
        if height % 2 != 0:
            array = array[:-1, :]
        if width % 2 != 0:
            array = array[:, :-1]
        return array


    @staticmethod
    def compute_haar_wavelet_matrix(n: int, weight: float = 1.0) -> npt.NDArray:
        if n < 2 or n % 2 != 0:
            raise ValueError(
                "n must be an even integer greater than or equal to 2.")

        HWT: npt.NDArray = np.zeros((n, n))

        for i in range(n // 2):
            HWT[i, 2 * i] = weight / 2
            HWT[i, 2 * i + 1] = weight / 2

        for i in range(n // 2):
            HWT[n // 2 + i, 2 * i] = -weight / 2
            HWT[n // 2 + i, 2 * i + 1] = weight / 2

        return HWT


    def get_sub_array_dimensions(self) -> tuple[int, int]:
        rows: int = int(2**-self.current_iteration * self.image_array.shape[0])
        columns: int = int(2**-self.current_iteration * self.image_array.shape[1])
        
        rows = rows if rows % 2 == 0 else rows - 1
        columns = columns if columns % 2 == 0 else columns - 1
        
        return (rows, columns)


    def apply_wavelet_transform(self, rows: int, cols: int) -> None:
        subarray: npt.NDArray = self._image_array[:rows, :cols]
        transformed_rows = self._row_transform_matrix[:rows, :rows] @ subarray
        transformed_subarray = transformed_rows @ self._col_transform_matrix[:cols, :cols].T
        self._image_array[:rows, :cols] = np.clip(transformed_subarray, 0, 255).astype(np.uint8)
        

    def next(self) -> "WaveletImage":
        rows, cols = self.get_sub_array_dimensions()
        self.apply_wavelet_transform(rows=rows, cols=cols)

        self._current_iteration += 1
        return self


    def prev(self) -> "WaveletImage":
        if (self.current_iteration == 0):
            raise WaveletTransformationError("Can not inverse transformation beyond original image.")

        # TODO: Inverse HWT on subarray

        self._current_iteration -= 1
        return self


    def go_to_iteration(self, iteration: int) -> "WaveletImage":
        if iteration < 0:
            raise WaveletTransformationError("Can not inverse transformation beyond original image.")
        elif abs(iteration - self.current_iteration) > 10:
            raise WaveletTransformationError("Too many iterations.")

        while self.current_iteration != iteration:
            if self.current_iteration < iteration:
                self.next()
            else:
                self.prev()
        
        return self
    

    def show_image(self) -> None:
        #TODO: Maybe use pyplot?
        pass
