from wavelet_image import WaveletImage

INPUT_FILEPATH: str = "./Resources/article-image.gif"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    wi: WaveletImage = WaveletImage(INPUT_FILEPATH)
    wi.next().next()
    wi.save_image(OUTPUT_FILEPATH)
    
if __name__ == "__main__":
    main()