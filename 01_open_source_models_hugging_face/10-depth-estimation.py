from transformers import pipeline
from pathlib import Path
from PIL import Image
from helpers import show_depth_map
import torch


if __name__ == '__main__':
    # Initialize depth estimator pipeline
    # Note: First run may take 2-5 minutes to download the ~400MB model
    depth_estimator = pipeline(
        task="depth-estimation",
        model="Intel/dpt-hybrid-midas"
    )

    # Define paths
    image_path = Path("..") / "data" / "sample.png"
    output_path = Path("..") / "data" / "sample_depth.png"

    # Load image
    raw_image = Image.open(image_path)

    # Run depth estimation
    output = depth_estimator(raw_image)

    # Post-process: resize to original image size using bicubic interpolation
    # The pipeline output has shape (batch, height, width), so we need to add a channel dimension
    predicted_depth = output["predicted_depth"]

    # Add batch and channel dimensions if needed
    if len(predicted_depth.shape) == 2:
        predicted_depth = predicted_depth.unsqueeze(0).unsqueeze(0)
    elif len(predicted_depth.shape) == 3:
        predicted_depth = predicted_depth.unsqueeze(1)

    prediction = torch.nn.functional.interpolate(
        predicted_depth,
        size=raw_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Convert to numpy array
    depth_map = prediction.squeeze().cpu().numpy()

    # Visualize and save
    show_depth_map(raw_image, depth_map, output_path=output_path, show_original=False, show_colorbar=False)
