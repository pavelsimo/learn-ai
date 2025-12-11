# Load model directly
from transformers import AutoProcessor, BlipForImageTextRetrieval
from PIL import Image
import torch
from pathlib import Path

processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

# Load an image
image_path = Path("..") / "data" / "sample.png"
image = Image.open(image_path)

# Define text queries to match against the image
texts = [
    "a kid playing with a ball and a dog",
    "a dog playing in the park",
    "a cat sleeping on a couch",
    "a person riding a bicycle",
    "a beach with palm trees",
    "a dog next to a couch",
]

print("Image-Text Matching Results:")
print("-" * 50)

# Process each text query
for text in texts:
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the image-text matching score
    itm_score = outputs.itm_score
    match_probability = torch.nn.functional.softmax(itm_score, dim=1)[:, 1].item()

    print(f"Text: {text}")
    print(f"Match probability: {match_probability:.4f}")
    print("-" * 50)