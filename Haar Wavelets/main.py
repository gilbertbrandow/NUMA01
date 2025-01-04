from wavelet_image import WaveletImage
from wavelet_image_io import WaveletImageIO

INPUT_FILEPATH: str = "./Resources/article-image.gif"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    
    wi: WaveletImage = WaveletImageIO.from_file(filepath=INPUT_FILEPATH)
    wi.next().next().prev().prev()
    WaveletImageIO.to_file(wavelet_image=wi, filepath=OUTPUT_FILEPATH)
    
if __name__ == "__main__":
    main()