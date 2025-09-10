"""RAG pipeline:
- Retrieve top-k from Chroma
- Prepare prompt with retrieved chunks
- Call LLM (GPT-4.1) using GitHub token via Azure SDK
- Enforce hallucination control and trust layer (include sources)
"""
import os
import time
import pandas as pd
from config import CHROMA_DIR, TOP_K, GITHUB_TOKEN, HF_EMBEDDING_MODEL
import chromadb
from chromadb.config import Settings
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document

PROMPT_TEMPLATE = """You are a triage assistant.
Use ONLY the following retrieved knowledge to answer the query.
If no relevant knowledge is found, say "I don’t know, not in knowledge base."
Include the source (Excel sheet, row_index, and chunk_id) for every answer.

Context:
{retrieved_chunks}

User query:
{query}
"""

# -----------------------
# Chroma DB retrieval
# -----------------------
def init_chroma():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = "MAL-Food-SC_knowledge"
    if collection_name not in [c.name for c in client.list_collections()]:
        client.create_collection(name=collection_name)
    collection = client.get_collection(name=collection_name)
    return collection


def retrieve(query: str, top_k: int = TOP_K):
    collection = init_chroma()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    retrieved = []
    entries = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    distances = results.get("distances", [])
    if entries and len(entries) > 0:
        for doc, md, dist in zip(entries[0], metadatas[0], distances[0]):
            retrieved.append({
                "document": doc,
                "metadata": md,
                "distance": dist
            })
    return retrieved


def build_context(retrieved_chunks):
    if not retrieved_chunks:
        return ""
    parts = []
    for r in retrieved_chunks:
        md = r.get("metadata", {})
        src = f"[sheet={md.get('sheet')}, row={md.get('row_index')}, chunk={md.get('chunk_id')}]"
        parts.append(src + "\n" + r.get("document", ""))
    return "\n\n---\n\n".join(parts)


# -----------------------
# GitHub LLM client
# -----------------------
if not GITHUB_TOKEN:
    raise RuntimeError("No GITHUB_TOKEN found in environment.")

gh_client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(GITHUB_TOKEN),
)


from azure.ai.inference.models import UserMessage, SystemMessage
from azure.ai.inference.models import ChatMessage


def call_llm(system_prompt: str, user_prompt: str):
    if not GITHUB_TOKEN:
        raise RuntimeError("No GITHUB_TOKEN found in environment.")

    client = ChatCompletionsClient(
        endpoint="https://models.github.ai/inference",
        credential=AzureKeyCredential(GITHUB_TOKEN),
    )

    # Send system + user messages
    response = client.create(
        model="gpt-4.1",
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt)
        ],
    )

    # The Azure SDK returns a plain string for the chat completion
    return response


    # response is already the text
    llm_out = response.choices[0].message.content


    # Collect final response
    # if hasattr(response, "__iter__"):  # streaming
    #     full_response = ""
    #     for chunk in response:
    #         delta_text = getattr(chunk.choices[0].delta, "content", "")
    #         full_response += delta_text
    #     return full_response
    # else:
    #     return response.choices[0].message.content


# -----------------------
# RAG query function
# -----------------------
def answer_query(query: str, top_k: int = TOP_K):
    start = time.time()
    retrieved = retrieve(query, top_k=top_k)
    context = build_context(retrieved)

    # Hallucination control
    if not context.strip():
        latency = time.time() - start
        return {
            "answer": "I don’t know, not in knowledge base.",
            "sources": [],
            "retrieved": [],
            "latency": latency
        }

    system_prompt = "You are a helpful assistant. Be concise."
    user_prompt = PROMPT_TEMPLATE.format(retrieved_chunks=context, query=query)
    llm_out = call_llm(system_prompt, user_prompt)
    latency = time.time() - start

    # Always return sources for trust layer
    sources = []
    for r in retrieved:
        md = r.get("metadata", {})
        sources.append({"sheet": md.get("sheet"), "row_index": md.get("row_index"), "chunk_id": md.get("chunk_id")})

    return {
        "answer": llm_out,
        "sources": sources,
        "retrieved": retrieved,
        "latency": latency
    }
