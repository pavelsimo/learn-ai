import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import librosa
from pathlib import Path

dataset = load_dataset(
    "librispeech_asr",
    split="train.clean.100",
    streaming=True
)

sample = dataset.take(5)
print(sample)

audio_path = Path("..") / "data" / "sample.wav"
if not audio_path.exists():
    raise FileNotFoundError(f"Missing audio sample at {audio_path}. Place a WAV file there.")

audio, sampling_rate = sf.read(str(audio_path))
print(f"Audio Sample Rate: {sampling_rate} Hz")

channels = 1 if audio.ndim == 1 else audio.shape[1]
channel_label = "mono" if channels == 1 else "stereo"
print(f"Audio channels: {channels} ({channel_label})")

# Whisper pipelines expect mono input; take the first channel when audio is stereo.
mono_audio = audio if channels == 1 else audio[:, 0]

target_sampling_rate = 16000
if sampling_rate != target_sampling_rate:
    mono_audio = librosa.resample(mono_audio, orig_sr=sampling_rate, target_sr=target_sampling_rate)
    sampling_rate = target_sampling_rate
    print(f"Resampled audio to {target_sampling_rate} Hz to match model expectations.")

mono_audio = mono_audio.astype("float32")

# https://huggingface.co/distil-whisper/distil-small.en
device = "cuda:0" if torch.cuda.is_available() else "cpu"
asr = pipeline(
    "automatic-speech-recognition",
    model="distil-whisper/distil-small.en",
    device=device,
)

print(f"Model Sample Rate: {asr.feature_extractor.sampling_rate}")

print(asr(
    {"array": mono_audio, "sampling_rate": sampling_rate},
    chunk_length_s=30,
    batch_size=4,
    return_timestamps=True,
)["chunks"])
