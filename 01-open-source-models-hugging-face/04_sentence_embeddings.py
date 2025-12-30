from sentence_transformers import SentenceTransformer, util

sentences1 = [
    "The cat sits outside",
    "A man is playing guitar",
    "The movies are awesome"
]

sentences2 = [
    "The dog plays in the garden",
    "A woman watches TV",
    "The new movie is so great"
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
print(embeddings1)
print(embeddings2)

# you will get the similarities between the first sentences of both lists.
cosine_scores = util.cos_sim(embeddings1, embeddings2)
print(cosine_scores)