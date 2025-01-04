from wavelet_image import WaveletImageIO, WaveletImage

INPUT_FILEPATH: str = "./Resources/article-image.gif"
OUTPUT_FILEPATH: str = "./Resources/new-article-image.jpg"

def main() -> None:
    wavelet_image: WaveletImage = WaveletImageIO.from_file(INPUT_FILEPATH)
    
    wavelet_image.next().next()
    
    print(f"This is the current iteration: {wavelet_image.current_iteration}")
    
    WaveletImageIO.to_file(wavelet_image=wavelet_image, filepath=OUTPUT_FILEPATH)

    
if __name__ == "__main__":
    main()