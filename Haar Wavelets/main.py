from wavelet_image import WaveletImageIO, WaveletImage

INPUT_FILEPATH: str = "./Resources/kvinna.jpg"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    wavelet_image: WaveletImage = WaveletImageIO.from_file(INPUT_FILEPATH)
    
    wavelet_image.next().next().next().prev().go_to_iteration(1).next()
    
    print(f"This is the current iteration: {wavelet_image.current_iteration}")
    
    WaveletImageIO.to_file(wavelet_image=wavelet_image, filepath=OUTPUT_FILEPATH)

    
if __name__ == "__main__":
    main()