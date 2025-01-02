from wavelet_image import WaveletTransformManager

INPUT_FILEPATH: str = "./Resources/kvinna.jpg"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    manager: WaveletTransformManager = WaveletTransformManager(INPUT_FILEPATH)

    for _ in range(3):
        manager.add_compressed_image()

    manager.get_latest_image().apply_inverse_wavelet_transform()
    manager.get_latest_image().save_image(OUTPUT_FILEPATH)
    
if __name__ == "__main__":
    main()