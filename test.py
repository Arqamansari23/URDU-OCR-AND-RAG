# from sentence_transformers import SentenceTransformer
# import torch


# sentences = ["This is an example sentence", "Each sentence is converted"]

# #model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# device = "cuda" if torch.cuda.is_available() else "cpu"
# embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=device)
# print(f"✅ Embedding model loaded on {device.upper()}")
# embeddings = embedding_model.encode(sentences)
# print(embeddings)


from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings)