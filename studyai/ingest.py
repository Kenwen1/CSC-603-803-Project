import os
import chromadb
import fitz  # reads PDFs
from sentence_transformers import SentenceTransformer
 
 

def read_pdf(path):
    pieces = []
    doc = fitz.open(path)
    for number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pieces.append((number, text))
    return pieces
 
 
def read_all(folder="materials"):
    chunks = []
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, filename)
        for number, text in read_pdf(path):
            chunks.append({"file": filename, "page": number, "text": text})
    return chunks
 
 
def build_database():
    chunks = read_all()
    if not chunks:
        print("No text found. Check that your PDFs are in the 'materials' folder.")
        return
 
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection("course")
 
    texts = [c["text"] for c in chunks]
    ids = [f'{c["file"]}-{c["page"]}' for c in chunks]
    metadata = [{"file": c["file"], "page": c["page"]} for c in chunks]
    vectors = embedder.encode(texts).tolist()
 
    collection.upsert(ids=ids, documents=texts, metadatas=metadata, embeddings=vectors)
    print("Saved", len(chunks), "pieces to the filing cabinet!")
 
 
if __name__ == "__main__":
    build_database()