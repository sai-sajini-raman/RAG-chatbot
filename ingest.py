"""Ingest script:
- Reads the Excel file at config.EXCEL_PATH
- Converts each row to a text chunk (includes metadata)
- Chunks long text into overlapping chunks
- Generates embeddings using sentence-transformers
- Stores embeddings in Chroma with metadata
- Builds a simple LlamaIndex index (for orchestration)
"""
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
# from chromadb.config import Settings
from pathlib import Path
import time
import json
from config import EXCEL_PATH, HF_EMBEDDING_MODEL, CHROMA_DIR, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP, INDEX_DIR
from llama_index.core import Document, ServiceContext, GPTVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding




def read_excel(path: Path):
    xls = pd.ExcelFile(path)
    sheets = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, dtype=str).fillna("")
        sheets[sheet] = df
    return sheets

def row_to_text(sheet_name: str, row_idx: int, df_row: pd.Series):
    # include column headers as part of text and metadata
    parts = []
    for col in df_row.index:
        val = str(df_row[col]).strip()
        if val:
            parts.append(f"{col}: {val}")
    text = "\n".join(parts)
    metadata = {
        "sheet": sheet_name,
        "row_index": int(row_idx),
        "columns": list(df_row.index)
    }
    return text, metadata

def chunk_text(text: str, words_per_chunk=CHUNK_WORD_SIZE, overlap=CHUNK_WORD_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk_words = words[i:i+words_per_chunk]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        i += (words_per_chunk - overlap)
    return chunks

def build_embeddings_and_store(sheets: dict):
    # load model
    print("Loading embedding model...", HF_EMBEDDING_MODEL)
    model = SentenceTransformer(HF_EMBEDDING_MODEL)

    # init chroma client
    # client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(CHROMA_DIR)))
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = "MAL-Food-SC_knowledge"
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)
    collection = client.create_collection(name=collection_name)

    documents_for_llama = []

    print("Starting ingestion...")
    total_chunks = 0
    start = time.time()
    for sheet, df in sheets.items():
        for idx, row in df.iterrows():
            text, metadata = row_to_text(sheet, idx, row)
            if not text.strip():
                continue
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{sheet}::row{idx}::chunk{i}"
                # create embedding
                emb = model.encode(chunk).tolist()
                # metadata with provenance
                md = {}
                # flatten metadata: ensure only str/int/float/bool/None
                for k, v in metadata.items():
                    if isinstance(v, (list, tuple)):
                        md[k] = ", ".join(map(str, v))   # convert list → string
                    else:
                        md[k] = str(v) if not isinstance(v, (int, float, bool, type(None))) else v

                # add chunk info
                md.update({"chunk_id": chunk_id, "chunk_index": i})

                # add to chroma
                collection.add(
                    ids=[chunk_id],
                    embeddings=[emb],
                    metadatas=[md],
                    documents=[chunk]
                )
                total_chunks += 1
                # add to llama docs
                doc_metadata = {**md}
                doc = Document(text=chunk, extra_info=doc_metadata)
                documents_for_llama.append(doc)

    elapsed = time.time() - start
    print(f"Ingested {total_chunks} chunks in {elapsed:.2f}s")

    # Build a small LlamaIndex using HuggingFaceEmbedding wrapper (for orchestration)
    # print("Building LlamaIndex (service context)...")
    # hf_embed = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL)
    # service_context = ServiceContext.from_defaults(embed_model=hf_embed)
    # index = GPTVectorStoreIndex.from_documents(documents_for_llama, service_context=service_context)
    # index.storage_context.persist(persist_dir=str(INDEX_DIR))
    # print("Index built and persisted to", INDEX_DIR)
    # Build a small LlamaIndex using HuggingFaceEmbedding + GitHub Marketplace LLM
    print("Building LlamaIndex (service context)...")

    import os
    from openai import OpenAI
    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    # --- GitHub Marketplace LLM setup ---
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("Missing GITHUB_TOKEN environment variable.")

    endpoint = "https://models.github.ai/inference"
    model_name = "gpt-4o"

    # wrap GitHub Marketplace GPT-4o for LlamaIndex
    llm = LlamaOpenAI(
        api_key=token,
        base_url=endpoint,
        model=model_name,
    )

    # HuggingFace embedding wrapper
    hf_embed = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL)

    # ServiceContext with embeddings + LLM
    service_context = ServiceContext.from_defaults(embed_model=hf_embed, llm=llm)

    # build LlamaIndex
    index = GPTVectorStoreIndex.from_documents(documents_for_llama, service_context=service_context)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print("Index built and persisted to", INDEX_DIR)


if __name__ == '__main__':
    if not EXCEL_PATH.exists():
        print("ERROR: Excel file not found at:", EXCEL_PATH)
        print("Please place your Excel file at that path and re-run ingest.py")
        raise SystemExit(1)
    sheets = read_excel(EXCEL_PATH)
    build_embeddings_and_store(sheets)
