# Use a pipeline as a high-level helper
from transformers import pipeline
from pathlib import Path
from PIL import Image

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Load the image
image_path = Path("..") / "data" / "sample.png"
image = Image.open(image_path)

# Generate caption
result = pipe(image)

# Display results
print("Image Caption:")
print(result[0]["generated_text"])