from transformers.utils import logging
from datasets import load_dataset, Audio
from transformers import pipeline
import librosa
import io

logging.set_verbosity_error()
dataset = load_dataset("ashraq/esc50", split="train[0:10]")

# Set the audio column to not decode automatically
dataset = dataset.cast_column(
    "audio",
    Audio(decode=False, sampling_rate=48_000)
)

audio_sample = dataset[0]
audio_column = audio_sample["audio"]
audio_array = audio_column.get("array")
if audio_array is None:
    # If the dataset didn't decode the audio, load it manually from disk/bytes
    audio_path = audio_column.get("path")
    audio_bytes = audio_column.get("bytes")
    if audio_path:
        audio_file = audio_path
    elif audio_bytes:
        audio_file = io.BytesIO(audio_bytes)
    else:
        raise RuntimeError("Audio column is missing both 'path' and 'bytes' entries.")
    audio_array, _ = librosa.load(audio_file, sr=48_000)

candidate_labels = [
    "Sound of a child crying",
    "Sound of vacuum cleaner",
    "Sound of a bird singing",
    "Sound of an airplane",
    "Sound of a dog",
    "Sound of vacuum cleaner"
]

# Use a pipeline as a high-level helper
zero_shot_classifier = pipeline(
    "zero-shot-audio-classification",
    model="laion/clap-htsat-unfused"
)

print(zero_shot_classifier(audio_array, candidate_labels=candidate_labels))
