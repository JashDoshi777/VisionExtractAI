
import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import hashlib
import google.generativeai as genai

embedding_model = None
faiss_indexes = {}
vector_id_to_text_map = {}
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_DIM = 384


def initialize_rag_system():
    """
    Loads the embedding model into memory and prepares the vector store directory.
    """
    global embedding_model
    print("🧠 RAG Core: Initializing...")
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
        print(f"✅ Created directory: {VECTOR_STORE_PATH}")
    print("🧠 RAG Core: Loading embedding model (this may take a moment on first run)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ RAG Core: Embedding model loaded successfully.")


def _get_user_specific_paths(user_api_key):
    """
    Creates a unique, safe filename prefix for a user based on a hash of their API key.
    """
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    index_path = os.path.join(VECTOR_STORE_PATH, f'{user_hash}_index.faiss')
    map_path = os.path.join(VECTOR_STORE_PATH, f'{user_hash}_map.json')
    return index_path, map_path


def _load_user_data(user_api_key):
    """
    Loads a user's FAISS index and text map from disk into memory.
    """
    if user_api_key in faiss_indexes: # Already loaded
        return
        
    index_path, map_path = _get_user_specific_paths(user_api_key)

    if os.path.exists(index_path):
        print(f"🧠 RAG Core: Loading existing FAISS index for user from {index_path}")
        faiss_indexes[user_api_key] = faiss.read_index(index_path)
    else:
        print("🧠 RAG Core: No existing index found. Creating new FAISS index for user.")
        faiss_indexes[user_api_key] = faiss.IndexFlatL2(EMBEDDING_DIM)

    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            vector_id_to_text_map[user_api_key] = {int(k): v for k, v in json.load(f).items()}
    else:
        vector_id_to_text_map[user_api_key] = {}


def _save_user_data(user_api_key):
    """
    Saves a user's FAISS index and text map from memory to disk.
    """
    index_path, map_path = _get_user_specific_paths(user_api_key)
    if user_api_key in faiss_indexes:
        faiss.write_index(faiss_indexes[user_api_key], index_path)
    if user_api_key in vector_id_to_text_map:
        with open(map_path, 'w') as f:
            json.dump(vector_id_to_text_map[user_api_key], f)


def _chunk_text(text, chunk_size=350, chunk_overlap=50):
    """
    Splits a long text into smaller, overlapping chunks.
    """
    if not isinstance(text, str): return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def add_document_to_knowledge_base(user_api_key, document_text, document_id):
    """
    Processes a document's text, chunks it, creates embeddings, and adds them to the user's knowledge base.
    """
    _load_user_data(user_api_key)
    print(f"🧠 RAG Core: Adding document '{document_id}' to knowledge base...")
    chunks = _chunk_text(document_text)
    if not chunks:
        print("⚠️ RAG Core: Document contains no text to add.")
        return

    chunk_embeddings = embedding_model.encode(chunks)
    index = faiss_indexes[user_api_key]
    text_map = vector_id_to_text_map[user_api_key]
    start_index = index.ntotal
    new_vector_ids = list(range(start_index, start_index + len(chunks)))
    index.add(np.array(chunk_embeddings, dtype=np.float32))

    for i, chunk in enumerate(chunks):
        vector_id = new_vector_ids[i]
        text_map[vector_id] = {"text": chunk, "source_doc": document_id}

    print(f"✅ RAG Core: Added {len(chunks)} chunks from '{document_id}'. Total vectors: {index.ntotal}")
    _save_user_data(user_api_key)


def query_knowledge_base(user_api_key, query_text):
    """
    Searches the user's knowledge base for relevant information and generates a contextual answer.
    """
    _load_user_data(user_api_key)
    index = faiss_indexes.get(user_api_key)
    text_map = vector_id_to_text_map.get(user_api_key)

    if not index or not text_map or index.ntotal == 0:
        return "The knowledge base is empty. Please upload some documents first."

    print(f"🧠 RAG Core: Received query: '{query_text}'")
    query_embedding = embedding_model.encode([query_text])

    k = 3
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)

    retrieved_chunks = []
    for i in indices[0]:
        if i in text_map: # Check if the index is valid
            retrieved_chunks.append(text_map[i]["text"])
    
    if not retrieved_chunks:
        return "I couldn't find any relevant information in your documents to answer that question."

    context = "\n\n---\n\n".join(retrieved_chunks)
    print(f"🧠 RAG Core: Found {len(retrieved_chunks)} relevant chunks.")

    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a helpful assistant. Your task is to answer the user's question based ONLY on the context provided below.
        Do not use any external knowledge. If the answer is not present in the context, state that you cannot find the information in the provided documents.

        CONTEXT:
        {context}

        QUESTION:
        {query_text}

        ANSWER:
        """
        
        response = model.generate_content(prompt)
        print("✅ RAG Core: Generated final answer with Gemini.")
        return response.text
    except Exception as e:
        print(f"❌ RAG Core: Error during Gemini API call for chat: {e}")
        return f"An error occurred while trying to generate an answer: {e}"