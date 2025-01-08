from wavelet_image import WaveletImage, RGBWaveletImage, AbstractWaveletImage
from wavelet_image_io import WaveletImageIO
import time
import sys

INPUT_FILEPATH: str = "./Resources/colors.png"
OUTPUT_FILEPATH: str = "./Resources/new-colors"


def main() -> None:
    """
    Main function.

    :author: Isak Blom, Egor Jakimov, Simon Gustafsson (2024-07-01)
    :return: None
    """
    only_grayscale: bool = True
    if len(sys.argv) > 1:
        if sys.argv[1] == "-r":
            only_grayscale = False
        elif sys.argv[1] == "-g":
            only_grayscale = True
        else:
            print("Usage: python main.py [-r|-g] (r for RGB, g for grayscale)")
            exit()

    wi: AbstractWaveletImage = WaveletImageIO.from_file(
        filepath=INPUT_FILEPATH, only_grayscale=only_grayscale)

    wi.next().next().next().prev()
    
    WaveletImageIO.to_file(wavelet_image=wi, filepath=f"{OUTPUT_FILEPATH}-after-2-iterations.png")

    wi.go_to_iteration(0)
    
    WaveletImageIO.to_file(wavelet_image=wi, filepath=f"{OUTPUT_FILEPATH}-as-normal.png")

    # print_time_difference(INPUT_FILEPATH, 4)


def print_time_difference(
    filepath: str,
    number_of_iterations: int = 5,
    revert_back_to_original: bool = True
) -> None:
    """
    Function to evaluate the time needed to perform the same number of iterations of
    Haar Wavelet Transformations on an image. It compare matrix multiplication with manual multiplcation. 

    :author: Simon Gustafsson (2024-07-01)
    :param filepath: The filepath of the image to use
    :param number_of_iterations: The number of iterations to perform on the image in both cases
    :param revert_back_to_original: Flag indicating wheter or not to invert all iterations and return image back to original.
    :return: None
    """
    
    wi_matrix = WaveletImageIO.from_file(filepath=filepath)
    wi_manual = WaveletImageIO.from_file(filepath=filepath)

    start: float = time.time()
    wi_matrix.go_to_iteration(
        iteration=number_of_iterations, matrix_multiplication=True)

    if revert_back_to_original:
        wi_matrix.go_to_iteration(0, matrix_multiplication=True)

    matrix_time: float = time.time() - start

    start = time.time()
    wi_manual.go_to_iteration(
        iteration=number_of_iterations, matrix_multiplication=False)

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
