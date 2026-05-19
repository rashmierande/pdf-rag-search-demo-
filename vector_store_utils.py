from typing import List
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def build_chroma_collection(
    chunks: List[str],
    name: str = "demo-collection",
):
    """
    Build a Chroma collection from a list of text chunks.
    Returns the collection object.
    """

    # 1. Create a Chroma client (in-memory by default)
    chroma_client = chromadb.Client()

    # 2. Use a sentence-transformers embedding function
    embedding_fn = SentenceTransformerEmbeddingFunction()

    # 3. Create (or get) a collection with this embedding function
    collection = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
    )

    # 4. Generate simple string IDs for each chunk
    ids = [str(i) for i in range(len(chunks))]

    # 5. Add chunks to the collection
    collection.add(ids=ids, documents=chunks)

    return collection

def search_collection(collection, query: str, k: int = 5) -> List[str]:
    """
  Query a Chroma collection and return top-k documents.
    """
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]  # Get the list of documents for the query
    return docs


if __name__ == "__main__":
    # Small self-test: integrate with load_and_chunk_pdf
    from pdf_utils import load_and_chunk_pdf

    pdf_path = "data/microsoft-annual-report.pdf"
    print("Building chunks from:", pdf_path)
    chunks = load_and_chunk_pdf(pdf_path)
    print("Total chunks:", len(chunks))

    collection = build_chroma_collection(chunks, name="microsoft-demo")
    print("Collection count:", collection.count())

    # Simple query
    query = "What was the total revenue for the year?"
    docs = search_collection(collection, query,k=3)
    # results = collection.query(query_texts=[query], n_results=3)
    print("\nQuery:", query)
    print("Results:")
    for doc in docs:
        print("----")
        print(doc[:400])