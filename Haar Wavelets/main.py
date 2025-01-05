from wavelet_image import WaveletImage
from wavelet_image_io import WaveletImageIO

INPUT_FILEPATH: str = "./Resources/kvinna.jpg"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    
    wi: WaveletImage = WaveletImageIO.from_file(filepath=INPUT_FILEPATH)
    wi.go_to_iteration(7).go_to_iteration(0)
    WaveletImageIO.to_file(wavelet_image=wi, filepath=OUTPUT_FILEPATH)
    
if __name__ == "__main__":
    main()