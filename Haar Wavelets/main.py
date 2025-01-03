from wavelet_image import WaveletTransformManager

INPUT_FILEPATH: str = "./Resources/article-image.gif"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    manager: WaveletTransformManager = WaveletTransformManager(INPUT_FILEPATH)

    for _ in range(1):
        manager.add_compressed_image()

    manager.get_latest_image().save_image(OUTPUT_FILEPATH)
    
if __name__ == "__main__":
    main()