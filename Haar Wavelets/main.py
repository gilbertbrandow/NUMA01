from wavelet_image import WaveletImage

INPUT_FILEPATH: str = "./Resources/kvinna.jpg"
OUTPUT_FILEPATH: str = "./Resources/new-kvinna.jpg"

def main() -> None:
    wi: WaveletImage = WaveletImage(INPUT_FILEPATH)
    wi.next().next().next().prev().prev().prev()
    #wi.prev()
    wi.save_image(OUTPUT_FILEPATH)

    """
    manager: WaveletTransformManager = WaveletTransformManager(INPUT_FILEPATH)

    for _ in range(1):
        manager.add_compressed_image()

    manager.get_latest_image().save_image(OUTPUT_FILEPATH)
    """
    
if __name__ == "__main__":
    main()