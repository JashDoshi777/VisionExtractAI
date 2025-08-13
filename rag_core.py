# rag_core.py

import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import hashlib
import google.generativeai as genai
# NEW: Import the Supabase client library
from supabase import create_client, Client
import io as python_io

# --- Supabase Configuration ---
# IMPORTANT: Replace these with your actual Supabase URL and Public Key
SUPABASE_URL = "https://nwcyfrvkfozlzwjimhmb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im53Y3lmcnZrZm96bHp3amltaG1iIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODE0NDAsImV4cCI6MjA3MDY1NzQ0MH0.51FFi8Tk51weqnUTC5fvKLldBWcNP_eYAzJzo6sDt88"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


embedding_model = None
faiss_indexes = {}
vector_id_to_text_map = {}
# The local VECTOR_STORE_PATH is no longer needed as we use Supabase
EMBEDDING_DIM = 384


def initialize_rag_system():
    """
    Loads the embedding model into memory.
    """
    global embedding_model
    print("🧠 RAG Core: Initializing...")
    print("🧠 RAG Core: Loading embedding model (this may take a moment on first run)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ RAG Core: Embedding model loaded successfully.")


def _get_user_specific_filenames(user_api_key, mode):
    """
    Creates unique, safe filenames for a user based on a hash of their API key.
    """
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    index_filename = f'{user_hash}_{mode}_index.faiss'
    map_filename = f'{user_hash}_{mode}_map.json'
    return index_filename, map_filename


def _load_user_data(user_api_key, mode):
    """
    Loads a user's FAISS index and text map from Supabase Storage into memory.
    """
    if mode not in faiss_indexes:
        faiss_indexes[mode] = {}
    if mode not in vector_id_to_text_map:
        vector_id_to_text_map[mode] = {}

    if user_api_key in faiss_indexes[mode]: # Already loaded for this session
        return
        
    index_filename, map_filename = _get_user_specific_filenames(user_api_key, mode)

    # Load FAISS index from Supabase
    try:
        print(f"🧠 RAG Core: Downloading FAISS index '{index_filename}' from Supabase...")
        index_bytes = supabase.storage.from_("vector_store").download(index_filename)
        # FAISS needs to read from a file-like object, so we use an in-memory buffer
        index_buffer = python_io.BytesIO(index_bytes)
        faiss_indexes[mode][user_api_key] = faiss.read_index(faiss.PyCallbackIOReader(index_buffer.read))
        print("✅ RAG Core: FAISS index loaded.")
    except Exception as e:
        print(f"ℹ️ RAG Core: No index file found for '{index_filename}'. Creating a new one. Error: {e}")
        faiss_indexes[mode][user_api_key] = faiss.IndexFlatL2(EMBEDDING_DIM)

    # Load text map from Supabase
    try:
        print(f"🧠 RAG Core: Downloading map file '{map_filename}' from Supabase...")
        map_bytes = supabase.storage.from_("vector_store").download(map_filename)
        map_data = json.loads(map_bytes.decode('utf-8'))
        vector_id_to_text_map[mode][user_api_key] = {int(k): v for k, v in map_data.items()}
        print("✅ RAG Core: Map file loaded.")
    except Exception as e:
        print(f"ℹ️ RAG Core: No map file found for '{map_filename}'. Creating a new one. Error: {e}")
        vector_id_to_text_map[mode][user_api_key] = {}


def _save_user_data(user_api_key, mode):
    """
    Saves a user's FAISS index and text map from memory to Supabase Storage.
    """
    index_filename, map_filename = _get_user_specific_filenames(user_api_key, mode)
    
    # Save FAISS index to Supabase
    if mode in faiss_indexes and user_api_key in faiss_indexes[mode]:
        try:
            # Write the index to an in-memory buffer
            buffer = python_io.BytesIO()
            faiss.write_index(faiss_indexes[mode][user_api_key], faiss.PyCallbackIOWriter(buffer.write))
            buffer.seek(0)
            # Upload the buffer's content to Supabase
            supabase.storage.from_("vector_store").upload(
                file=buffer.getvalue(),
                path=index_filename,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            print(f"✅ RAG Core: Saved FAISS index to '{index_filename}' in Supabase.")
        except Exception as e:
            print(f"❌ RAG Core: Failed to save FAISS index to Supabase. Error: {e}")

    # Save text map to Supabase
    if mode in vector_id_to_text_map and user_api_key in vector_id_to_text_map[mode]:
        try:
            map_bytes = json.dumps(vector_id_to_text_map[mode][user_api_key], indent=4).encode('utf-8')
            supabase.storage.from_("vector_store").upload(
                file=map_bytes,
                path=map_filename,
                file_options={"content-type": "application/json", "upsert": "true"}
            )
            print(f"✅ RAG Core: Saved map file to '{map_filename}' in Supabase.")
        except Exception as e:
            print(f"❌ RAG Core: Failed to save map file to Supabase. Error: {e}")


def _chunk_text(text, chunk_size=350, chunk_overlap=50):
    if not isinstance(text, str): return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def add_document_to_knowledge_base(user_api_key, document_text, document_id, mode):
    _load_user_data(user_api_key, mode)
    print(f"🧠 RAG Core: Adding/Updating document '{document_id}' in '{mode}' knowledge base...")
    chunks = _chunk_text(document_text)
    if not chunks:
        print("⚠️ RAG Core: Document contains no text to add.")
        return

    chunk_embeddings = embedding_model.encode(chunks)
    index = faiss_indexes[mode][user_api_key]
    text_map = vector_id_to_text_map[mode][user_api_key]
    start_index = index.ntotal
    new_vector_ids = list(range(start_index, start_index + len(chunks)))
    index.add(np.array(chunk_embeddings, dtype=np.float32))

    for i, chunk in enumerate(chunks):
        vector_id = new_vector_ids[i]
        text_map[vector_id] = {"text": chunk, "source_doc": document_id}

    print(f"✅ RAG Core: Added {len(chunks)} chunks from '{document_id}'. Total vectors in '{mode}' mode: {index.ntotal}")
    _save_user_data(user_api_key, mode)
    
def remove_document_from_knowledge_base(user_api_key, document_id, mode):
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or not text_map:
        print(f"⚠️ RAG Core: No knowledge base found for user in '{mode}' mode.")
        return

    ids_to_remove = [
        vector_id for vector_id, meta in text_map.items() 
        if meta.get("source_doc") == document_id
    ]

    if not ids_to_remove:
        print(f"⚠️ RAG Core: No vectors found for document '{document_id}' to remove.")
        return

    index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
    
    for vector_id in ids_to_remove:
        del text_map[vector_id]
        
    print(f"✅ RAG Core: Removed {len(ids_to_remove)} vectors for document '{document_id}'.")
    _save_user_data(user_api_key, mode)


def query_knowledge_base(user_api_key, query_text, mode, history=[]):
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or not text_map or index.ntotal == 0:
        return f"The {mode} knowledge base is empty. Please upload some documents first."

    print(f"🧠 RAG Core: Received query for '{mode}' mode: '{query_text}'")
    query_embedding = embedding_model.encode([query_text])
    k = min(5, index.ntotal)
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    retrieved_chunks = [text_map[i]["text"] for i in indices[0] if i in text_map]
    if not retrieved_chunks:
        return "I couldn't find any relevant information in your documents to answer that question."

    context = "\n\n---\n\n".join(retrieved_chunks)
    print(f"🧠 RAG Core: Found {len(retrieved_chunks)} relevant chunks.")

    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        chat = model.start_chat(history=history)
        
        prompt = f"""
        You are an expert analyst. Your task is to provide a detailed and proper answer to the user's question based ONLY on the provided text snippets and the previous conversation turns.

        Follow these steps:
        1.  First, consider the ongoing conversation history to understand the full context of the user's latest question.
        2.  Carefully read the new context snippets provided below.
        3.  Think step-by-step about how the snippets and the conversation history can be combined to answer the latest question.
        4.  Formulate a comprehensive answer. If the information is not in the context or history, you must explicitly state that the information is not available in the documents.

        CONTEXT SNIPPETS:
        {context}

        LATEST QUESTION:
        {query_text}

        FINAL ANSWER:
        """
        response = chat.send_message(prompt)
        print("✅ RAG Core: Generated final answer with Gemini, considering history.")
        return response.text
    except Exception as e:
        print(f"❌ RAG Core: Error during Gemini API call for chat: {e}")
        return f"An error occurred while trying to generate an answer: {e}"
