from transformers import pipeline
import torch

translator = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    torch_dtype=torch.bfloat16
)

text = """
We regularly check our sources for new videos and add them as soon as they become available. 
That said, if you are actively studying the videos by reading explanations, practicing flashcards, 
and repeating sentences, it will take you a while to get through all the content on the site, 
even with a smaller library. It is designed to help you learn deeply, not just passively watch.
"""

text_translated = translator(
    text,
    src_lang="eng_Latn",
    tgt_lang="spa_Latn"
)

print(text_translated[0]["translation_text"])