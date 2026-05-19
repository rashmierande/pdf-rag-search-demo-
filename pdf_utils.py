from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("pdf_utils imported, __name__ =", __name__)


def load_and_chunk_pdf(
    path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
) -> List[str]:
    """
    Load a PDF from `path` and split its text into chunks.
    Returns a list of chunk strings.
    """
    # 1. Load PDF and extract text from each page
    reader = PdfReader(path)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    page_texts = [t.strip() for t in page_texts if t and t.strip()]

    # 2. Join into a single string
    full_text = "\n\n".join(page_texts)

    # 3. Split into chunks with a character-based splitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(full_text)
    return chunks


if __name__ == "__main__":
    print("starting pdf_utils test")
    pdf_path = "data/microsoft-annual-report.pdf"
    print("Using PDF path:", pdf_path)

    chunks = load_and_chunk_pdf(pdf_path)
    print("Function returned, total chunks:", len(chunks))
    for i, c in enumerate(chunks[:3]):
        print(f"--- Chunk {i} ---")
        print(c[:400])
        print()