# RAG Cheatsheet

## 1. High-level mental model

PDF -> extract text -> chunk text -> embed & store in Chroma -> query -> get top-k relevant chunks.

## 2. Core functions in this project

### load_and_chunk_pdf(path)
Meaning:
- Take a PDF file.
- Extract text from each page.
- Split the text into smaller chunks.

Mental summary:
- "Read PDF, clean text, and turn it into chunks."

### build_chroma_collection(chunks, name)
Meaning:
- Take the chunks.
- Create a Chroma vector store.
- Add each chunk with an ID.

Mental summary:
- "Index my chunks in a vector database."

### search_collection(collection, query, k)
Meaning:
- Search the Chroma collection.
- Return the top-k most relevant chunks for the query.

Mental summary:
- "Ask a question and get the closest matching chunks."

## 3. Utility classes and what they do

### PdfReader
- Reads text from PDF pages.

Mental summary:
- "PDF -> page text"

### RecursiveCharacterTextSplitter
- Splits long text into smaller chunks.
- Uses separators like paragraph breaks, line breaks, sentences, spaces.

Mental summary:
- "Big text -> smaller chunks"

### SentenceTransformerEmbeddingFunction
- Converts text into embeddings.
- Embeddings are vectors used for semantic similarity search.

Mental summary:
- "Text -> vector"

### chromadb.Client
- Creates a Chroma database client.

Mental summary:
- "Connect to vector DB"

### collection.add(ids=..., documents=...)
- Stores chunks in the Chroma collection.

Mental summary:
- "Save chunks into the database"

### collection.query(query_texts=[...], n_results=k)
- Searches the collection for the most relevant chunks.

Mental summary:
- "Find top matching chunks"

## 4. Code snippets

### Load and chunk PDF
```python
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(
    path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
) -> List[str]:
    reader = PdfReader(path)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    page_texts = [t.strip() for t in page_texts if t and t.strip()]

    full_text = "\n\n".join(page_texts)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(full_text)
```

Mental summary:
- "Load PDF, join page text, split into chunks."

### Build Chroma collection
```python
from typing import List
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def build_chroma_collection(
    chunks: List[str],
    name: str = "demo-collection",
):
    chroma_client = chromadb.Client()
    embedding_fn = SentenceTransformerEmbeddingFunction()

    collection = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
    )

    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)

    return collection
```

Mental summary:
- "Create vector DB and store my chunks."

### Search collection
```python
from typing import List

def search_collection(collection, query: str, k: int = 5) -> List[str]:
    results = 
