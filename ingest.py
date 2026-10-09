"""Step 1: read documents, split into chunks, store them in a vector database."""
from pathlib import Path

import chromadb
from pypdf import PdfReader

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION = "docs"
CHUNK_SIZE = 800    # characters per chunk
OVERLAP = 150       # characters shared between neighbouring chunks


def read_file(path: Path):
    """Return a list of (page_number, text) for a PDF, TXT or MD file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """Split text into overlapping pieces so answers are not cut at chunk edges."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def main():
    files = [p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not files:
        print(f"No .pdf/.txt/.md files found in '{DOCS_DIR}'. Add some and run again.")
        return

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)   # rebuild from scratch each run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    for path in files:
        for page_no, page_text in read_file(path):
            for i, chunk in enumerate(chunk_text(page_text)):
                ids.append(f"{path.name}-p{page_no}-c{i}")
                texts.append(chunk)
                metas.append({"source": path.name, "page": page_no})

    # Chroma creates the embeddings automatically with its built-in model.
    for start in range(0, len(ids), 100):
        end = start + 100
        collection.add(ids=ids[start:end], documents=texts[start:end], metadatas=metas[start:end])

    print(f"Indexed {len(texts)} chunks from {len(files)} file(s).")


if __name__ == "__main__":
    main()
