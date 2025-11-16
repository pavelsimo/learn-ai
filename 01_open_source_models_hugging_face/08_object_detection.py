from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.library import Library
import json

_TORCHVISION_STUB_LIB: Library | None = None


def _ensure_torchvision_nms_stub() -> None:
    """
    Some minimal PyTorch builds (CPU-only) do not define torchvision::nms,
    yet torchvision tries to register meta kernels for it on import.
    We stub the operator schema ahead of time so torchvision can be imported.
    """
    try:
        torch.ops.torchvision.nms  # type: ignore[attr-defined]
        return
    except AttributeError:
        pass

    global _TORCHVISION_STUB_LIB
    _TORCHVISION_STUB_LIB = Library("torchvision", "DEF")
    _TORCHVISION_STUB_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")


_ensure_torchvision_nms_stub()

from transformers import pipeline


def annotate_and_save(
    image: Image.Image,
    detections: Sequence[dict],
    output_image_path: Path,
    score_threshold: float = 0.5,
) -> None:
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    for detection in detections:
        score = detection.get("score", 0.0)
        if score < score_threshold:
            continue

        box = detection["box"]
        xmin, ymin, xmax, ymax = (
            box["xmin"],
            box["ymin"],
            box["xmax"],
            box["ymax"],
        )
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="lime", width=3)

        label = f"{detection['label']} {score:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        max_x_offset = max(0, annotated_image.width - text_width - 6)
        text_x = max(0, min(xmin, max_x_offset))
        text_y = ymin - text_height - 6
        if text_y < 0:
            text_y = max(
                0,
                min(annotated_image.height - text_height - 6, ymin + 3),
            )

        text_x1 = min(annotated_image.width, text_x + text_width + 6)
        text_y1 = min(annotated_image.height, text_y + text_height + 6)

        text_bg = [
            (text_x, text_y),
            (text_x1, text_y1),
        ]
        draw.rectangle(text_bg, fill="lime")
        draw.text((text_x + 3, text_y + 3), label, fill="black", font=font)

    annotated_image.save(output_image_path)


def main() -> None:
    detector = pipeline(
        "object-detection",
        model="facebook/detr-resnet-50",
    )

    image_path = Path("..") / "data" / "sample.png"
    raw_image = Image.open(image_path)
    detections = detector(raw_image)
    print(json.dumps(detections, indent=2))
    output_image_path = Path("..") / "data" / "sample_od.png"
    annotate_and_save(raw_image, detections, output_image_path)

if __name__ == "__main__":
    main()
