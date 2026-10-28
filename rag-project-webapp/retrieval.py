
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 1. Load the same embedding model used for indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to our existing ChromaDB database
database_path = Path(__file__).resolve().parent / "chroma_db"
client = chromadb.PersistentClient(path=str(database_path))

collection = client.get_collection(name="rag_documents")


def retrieve(query, top_k=5):
    if not query.strip():
        raise ValueError("Please enter a non-empty question.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    count = collection.count()
    if count == 0:
        raise ValueError("The rag_documents collection is empty. Index the PDF first.")

    # 3. Convert the user's question into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # 4. Find the most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
    )

    # 5. Return the retrieved chunks
    retrieved_chunks = []

    for i in range(len(results["documents"][0])):
        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "section": results["metadatas"][0][i]["section"],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


# 6. Test retrieval
if __name__ == "__main__":
    question = input("Ask a question about the PDF: ")

    chunks = retrieve(question)

    print(f"\nTop {len(chunks)} retrieved chunks:")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Section: {chunk['section']}")
        print(f"Cosine distance: {chunk['distance']:.4f}")
        print(f"Text:\n{chunk['text']}")
