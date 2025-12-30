from transformers import pipeline
import torch

summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    torch_dtype=torch.bfloat16
)

text = """
We regularly check our sources for new videos and add them as soon as they become available. 
That said, if you are actively studying the videos by reading explanations, practicing flashcards, 
and repeating sentences, it will take you a while to get through all the content on the site, 
even with a smaller library. It is designed to help you learn deeply, not just passively watch.
"""

summary = summarizer(
    text,
    min_length=0,
    max_length=100
)

print(summary)

