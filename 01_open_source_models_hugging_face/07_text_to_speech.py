import unicodedata
from pathlib import Path

import soundfile as sf
from transformers import pipeline


UNSUPPORTED_CHARS = "()[]{}"


def normalize_text(input_text: str) -> str:
    """Strip characters VITS tokenizer cannot encode (e.g. curly quotes, brackets)."""
    normalized = unicodedata.normalize("NFKD", input_text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.translate({ord(ch): None for ch in UNSUPPORTED_CHARS})

# https://huggingface.co/kakao-enterprise/vits-ljs
narrator = pipeline(
    "text-to-speech",
    model="kakao-enterprise/vits-ljs"
)

text = """
What is VITS? VITS is a type of Text-to-Speech (TTS) model. 
In simple terms: You give it text, it generates speech audio that sounds like a real person talking. 
It’s called “end-to-end” because it learns to go directly from text to sound, 
instead of using many separate steps or tools in between.
"""

sanitized_text = normalize_text(text)
narrated_text = narrator(sanitized_text)

output_path = Path("data") / "sample.wav"
output_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(output_path, narrated_text["audio"].squeeze(), narrated_text["sampling_rate"])
print(f"Saved narration to {output_path.resolve()}")
