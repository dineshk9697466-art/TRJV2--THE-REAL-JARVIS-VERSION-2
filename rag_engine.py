from pathlib import Path
import hashlib
import time

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(
    name="trjv2_documents", metadata={"hnsw:space": "cosine"}
)


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def clear_legacy_global_documents():
    """Remove pre-chat-scoped index entries once; preserve entries with chat_id."""
    try:
        stored = _collection.get(include=["metadatas"])
        ids_to_delete = []
        for doc_id, metadata in zip(stored.get("ids", []), stored.get("metadatas", [])):
            if not metadata or not metadata.get("chat_id"):
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            for start in range(0, len(ids_to_delete), 500):
                _collection.delete(ids=ids_to_delete[start:start + 500])
        return len(ids_to_delete)
    except Exception:
        # Never block app startup if an older Chroma version behaves differently.
        return 0


def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": page_number, "text": text})
    return pages


def split_text(text, chunk_size=800, overlap=120):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def ingest_pdf(pdf_path, chat_id):
    """Index a PDF within one chat; re-uploading the same file replaces its chunks there."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Please provide an existing PDF file")
    if chat_id is None or str(chat_id).strip() == "":
        raise ValueError("A chat must be selected before indexing a PDF")

    pages = extract_pdf_text(pdf_path)
    if not pages:
        raise ValueError("No extractable text found. This PDF may need OCR.")

    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    source_name = pdf_path.name
    model = get_embedding_model()
    ids, documents, metadatas = [], [], []
    uploaded_at = time.time()

    for page in pages:
        for chunk_index, chunk in enumerate(split_text(page["text"])):
            ids.append(f"chat_{chat_id}_{file_hash}_p{page['page']}_c{chunk_index}")
            documents.append(chunk)
            metadatas.append({
                "source": source_name,
                "page": int(page["page"]),
                "file_hash": file_hash,
                "chat_id": str(chat_id),
                "uploaded_at": uploaded_at,
            })

    if not documents:
        raise ValueError("No text chunks were generated")

    # Replace this same PDF only in this chat. Identical PDF in another chat is untouched.
    _collection.delete(where={"$and": [{"chat_id": str(chat_id)}, {"file_hash": file_hash}]})
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    _collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(documents)


def get_chat_documents(chat_id):
    if chat_id is None:
        return []
    result = _collection.get(where={"chat_id": str(chat_id)}, include=["metadatas"])
    latest_by_source = {}
    for metadata in result.get("metadatas", []) or []:
        if not metadata:
            continue
        source = metadata.get("source", "")
        uploaded_at = float(metadata.get("uploaded_at", 0))
        if source and (source not in latest_by_source or uploaded_at > latest_by_source[source]):
            latest_by_source[source] = uploaded_at
    return sorted(latest_by_source, key=lambda name: latest_by_source[name], reverse=True)


def search_documents(question, chat_id, top_k=4, source=None, latest_only=True):
    """Search only this chat. By default, search the latest uploaded PDF in this chat."""
    if not question or not question.strip() or chat_id is None:
        return []
    sources = get_chat_documents(chat_id)
    if not sources:
        return []
    selected_source = source or (sources[0] if latest_only else None)
    clauses = [{"chat_id": str(chat_id)}]
    if selected_source:
        clauses.append({"source": selected_source})
    where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
    existing = _collection.get(where=where, include=[])
    count = len(existing.get("ids", []))

    if count == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([question], normalize_embeddings=True).tolist()
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    matches = []
    for i, document in enumerate((results.get("documents") or [[]])[0]):
        metadata = (results.get("metadatas") or [[]])[0][i] or {}
        matches.append({
            "text": document,
            "source": metadata.get("source", "Unknown PDF"),
            "page": metadata.get("page", "?"),
            "distance": (results.get("distances") or [[]])[0][i],
        })
    return matches
