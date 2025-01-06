from wavelet_image import WaveletImage
from wavelet_image_io import WaveletImageIO
import time

INPUT_FILEPATH: str = "./Resources/article-image.gif"
OUTPUT_FILEPATH: str = "./Resources/new-article-image.gif"

def main() -> None:
    
    wi: WaveletImage = WaveletImageIO.from_file(filepath=INPUT_FILEPATH)
    wi.go_to_iteration(2, matrix_multiplication=True)
    
    wi.next().next().go_to_iteration(0)
    
    wi.next(matrix_multiplication=True).prev(matrix_multiplication=True)
    
    WaveletImageIO.to_file(wavelet_image=wi, filepath=OUTPUT_FILEPATH)
    
    print_time_difference(INPUT_FILEPATH, 4)


def print_time_difference(filepath: str, number_of_iterations: int = 5, revert_back_to_original: bool = True) -> None:
    wi_matrix = WaveletImageIO.from_file(filepath=filepath)
    wi_manual = WaveletImageIO.from_file(filepath=filepath)

    start: float = time.time()
    wi_matrix.go_to_iteration(iteration=number_of_iterations, matrix_multiplication=True)
    
    if revert_back_to_original:
        wi_matrix.go_to_iteration(0, matrix_multiplication=True)
    
    matrix_time: float = time.time() - start

    start: float = time.time()
    wi_manual.go_to_iteration(iteration=number_of_iterations, matrix_multiplication=False)
    
    if revert_back_to_original:
        wi_manual.go_to_iteration(0, matrix_multiplication=False)
    
    manual_time: float = time.time() - start

    print(f"Matrix-based time: {matrix_time:.5f}s")
    print(f"Manual-based time: {manual_time:.5f}s")
    print(f"Difference (manual - matrix): {manual_time - matrix_time:.5f}s")
    
    if matrix_time and matrix_time < manual_time: 
        ratio: float = int(manual_time / matrix_time)
        print(f"Matrix multiplication is about {ratio}x faster")

    

if __name__ == "__main__":
    main()