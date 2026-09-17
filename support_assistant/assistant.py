import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def run_rag_pipeline(user_query: str):
    # Load knowledge base document
    doc_path = "support_assistant/docs/faq.txt"
    if not os.path.exists(doc_path):
        print(f"Error: Knowledge base file missing at {doc_path}")
        return

    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split document into chunks
    chunks = [line.strip() for line in text.split("\n") if line.strip()]

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate vector embeddings
    chunk_embeddings = model.encode(chunks)
    dimension = chunk_embeddings.shape[1]

    # Build FAISS vector index
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings).astype("float32"))

    # Vector search query
    query_embedding = model.encode([user_query])
    k = 1  # Top 1 match
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k)

    best_match = chunks[indices[0][0]]
    print(f"\nUser Query: '{user_query}'")
    print(f"Retrieved Context: '{best_match}'")
    print(f"L2 Distance Score: {distances[0][0]:.4f}")

if __name__ == "__main__":
    run_rag_pipeline("How long does delivery take?")