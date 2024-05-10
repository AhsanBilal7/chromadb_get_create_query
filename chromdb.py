import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import numpy as np

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Embeddings) -> Embeddings:
        # embed the documents somehow
        print("getting input")
        embeddings = np.random.randn(128,1)
        return list(embeddings)

client = chromadb.PersistentClient(path="db")


# collection = client.create_collection(name="my_collection", embedding_function=MyEmbeddingFunction())
collection = client.get_collection(name="my_collection", embedding_function=MyEmbeddingFunction())



embeddings1 = np.random.randn(128,1)
embeddings2 = np.random.randn(128,1)

# Flatten NumPy arrays and convert them to lists of floats
embeddings1_list = embeddings1.flatten().tolist()
embeddings2_list = embeddings2.flatten().tolist()

collection.add(
    embeddings=[embeddings1_list, embeddings2_list], # Flatten NumPy arrays and convert to lists
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on these!
    ids=["ee", "rrr"], # unique for each doc
)



# Query embeddings
query_embedding = np.random.randn(128, 1)  # Example query embedding
query_embedding_list = query_embedding.flatten().tolist()  # Flatten NumPy array and convert to list
result = collection.query(query_embeddings=[query_embedding_list], n_results=10, include=["embeddings", 'distances',])

print(result)