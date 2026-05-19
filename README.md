# PDF RAG Search Demo

Hands-on Retrieval-Augmented Generation (RAG) demo for querying financial PDFs (e.g., annual reports). The project loads and chunks a PDF, builds a Chroma vector store with sentence-transformer embeddings, and retrieves the most relevant text chunks for natural-language questions.

## What this project does

- Extracts text from a PDF using `pypdf`.
- Splits the text into manageable chunks with `RecursiveCharacterTextSplitter`.
- Builds an in-memory Chroma collection with sentence-transformer embeddings.
- Runs semantic search over the chunks to answer questions like  
  _"What was the total revenue for the year?"_

This repo focuses on a clean, modular RAG core that’s easy to understand and extend.

## Project structure

Key files:

- `pdf_utils.py`  
  - `load_and_chunk_pdf(path, chunk_size=1000, chunk_overlap=0)`  
  - Loads a PDF, extracts page text, and returns a list of text chunks.

- `vector_store_utils.py`  
  - `build_chroma_collection(chunks, name="demo-collection")`  
    - Creates a Chroma client, builds a collection, and indexes the chunks.  
  - `search_collection(collection, query, k=5)`  
    - Runs a semantic query against the collection and returns the top‑k chunks.

You can compose these functions to go from raw PDF → chunks → vector store → search.

## Quick start

1. **Create and activate a virtual environment**

```bash
cd /Users/rashmi/projects/advanced-rag-techniques

python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows
```

2. **Install dependencies**

```bash
python -m pip install pypdf langchain-text-splitters chromadb
```

3. **Add a PDF**

Place a PDF under the `data/` folder, for example:

```text
data/microsoft-annual-report.pdf
```

4. **Test the PDF loading and chunking**

```bash
python pdf_utils.py
```

You should see the total number of chunks and a preview of the first few.

5. **Build the vector store and run a query**

```bash
python vector_store_utils.py
```

This will:

- Load and chunk the PDF.
- Build a Chroma collection.
- Run a sample query such as:

```text
What was the total revenue for the year?
```

and print the top matching chunks.

## Minimal example (from Python)

You can also use the pipeline from a Python shell or notebook:

```python
from pdf_utils import load_and_chunk_pdf
from vector_store_utils import build_chroma_collection, search_collection

pdf_path = "data/microsoft-annual-report.pdf"

# 1. PDF -> chunks
chunks = load_and_chunk_pdf(pdf_path)

# 2. chunks -> Chroma collection
collection = build_chroma_collection(chunks, name="microsoft-demo")

# 3. query -> top-k chunks
query = "What was the total revenue for the year?"
docs = search_collection(collection, query, k=3)

print("Query:", query)
for i, doc in enumerate(docs, start=1):
    print(f"\nResult {i}:")
    print(doc[:400])
```

## Next steps / extensions

This project is designed to be extended with:

- Query expansion using OpenAI (multi-query RAG).
- Visualization of embeddings using UMAP + matplotlib.
- Evaluation of retrieval quality with LLM-based metrics (e.g., RAG testing).

It is intentionally kept small and modular to serve as a clear demo of the RAG basics over PDFs.
