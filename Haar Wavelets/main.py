from wavelet_image import WaveletImage
from wavelet_image_io import WaveletImageIO

INPUT_FILEPATH: str = "./Resources/kvinna.jpg"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    
    wi: WaveletImage = WaveletImageIO.from_file(filepath=INPUT_FILEPATH)
    wi.go_to_iteration(2, matrix_multiplaction=False)
    
    wi.next().next()
    
    wi.next(matrix_multiplaction=False).prev(matrix_multiplaction=False)
    
    WaveletImageIO.to_file(wavelet_image=wi, filepath=OUTPUT_FILEPATH)
    
if __name__ == "__main__":
    main()