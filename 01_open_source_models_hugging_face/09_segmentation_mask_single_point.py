from pathlib import Path
from PIL import Image
from helpers import show_mask_on_image
from transformers import AutoProcessor, AutoModelForMaskGeneration
import torch

processor = AutoProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")
model = AutoModelForMaskGeneration.from_pretrained("Zigeng/SlimSAM-uniform-77")

if __name__ == '__main__':
    # Load image
    image_path = Path("..") / "data" / "sample.png"
    raw_image = Image.open(image_path)

    # Define input point targeting toddler's torso
    input_points = [[[150, 200]]]  # Format: [batch][point_set][x, y]

    # Prepare inputs using processor
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt")

    # Run inference without gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process masks to original image dimensions
    predicted_masks = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )

    # Extract first image's masks (shape: [1, 3, H, W] or [3, H, W])
    predicted_mask = predicted_masks[0]

    # Visualize all 3 masks on single output image
    output_path = Path("..") / "data" / "sample_mask_2.png"
    show_mask_on_image(
        raw_image,
        [predicted_mask[:, i] for i in range(3)],
        output_path=output_path
    )