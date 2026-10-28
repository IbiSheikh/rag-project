
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent vector database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"}
)


def index_chunks(chunks):
    # Extract text and metadata
    texts = [chunk["text"] for chunk in chunks]

    metadata = [
        {"section": chunk["section"]}
        for chunk in chunks
    ]

    # Convert chunks into numerical vectors
    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    # Store everything in ChromaDB
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=metadata
    )

    print(f"Successfully indexed {len(chunks)} chunks!")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    print(f"Total stored vectors: {collection.count()}")
