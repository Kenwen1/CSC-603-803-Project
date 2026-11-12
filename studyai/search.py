import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
collection = chromadb.PersistentClient(path="db").get_collection("course")

def find_pieces(question, how_many=4):
    vector = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=vector, n_results=how_many)
    pieces = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        pieces.append({"text": text, "file": meta["file"], "page": meta["page"]})
    return pieces

if __name__ == "__main__":
    for p in find_pieces("What is a transformer?"):
        print(p["file"], "slide", p["page"])
        print(p["text"][:200])
        print("---")