from wavelet_image import WaveletImage

INPUT_FILEPATH: str = "./Resources/kvinna.jpg"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    wavelet_image: WaveletImage = WaveletImage(INPUT_FILEPATH)
    return None

if __name__ == "__main__":
    main()