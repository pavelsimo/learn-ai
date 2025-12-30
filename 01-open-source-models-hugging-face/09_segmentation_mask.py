from transformers import pipeline
from pathlib import Path
from PIL import Image
from helpers import show_mask_on_image


if __name__ == '__main__':
    sam = pipeline("mask-generation", model="Zigeng/SlimSAM-uniform-77")
    image_path = Path("..") / "data" / "sample.png"
    output_path = Path("..") / "data" / "sample_mask.png"
    raw_image = Image.open(image_path)
    output = sam(raw_image, points_per_side=32)
    show_mask_on_image(raw_image, output, output_path=output_path)