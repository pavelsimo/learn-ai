from transformers import pipeline

chatbot = pipeline(
    task="text2text-generation",
    model="facebook/blenderbot-400M-distill",
)

user_message = "What are the 10 most important concepts in AI?"
out = chatbot(user_message, max_new_tokens=1024)  # prefer max_new_tokens over max_length
print(out[0]["generated_text"])
