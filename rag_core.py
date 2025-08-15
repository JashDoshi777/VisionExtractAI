import os
os.environ['FAISS_NO_AVX2'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import resource
resource.setrlimit(resource.RLIMIT_AS, (350 * 1024 * 1024, 350 * 1024 * 1024))

# Lazy-loaded components
_loaded = False
np = None
faiss = None
SentenceTransformer = None
genai = None
create_client = None

def _load_dependencies():
    global _loaded, np, faiss, SentenceTransformer, genai, create_client
    if not _loaded:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        import google.generativeai as genai
        from supabase import create_client
        _loaded = True

class SupabaseClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            _load_dependencies()
            cls._instance = create_client(
                "https://nwcyfrvkfozlzwjimhmb.supabase.co",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im53Y3lmcnZrZm96bHp3amltaG1iIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODE0NDAsImV4cCI6MjA3MDY1NzQ0MH0.51FFi8Tk51weqnUTC5fvKLldBWcNP_eYAzJzo6sDt88"
            )
        return cls._instance

# Core RAG components
embedding_model = None
faiss_indexes = {}
vector_id_to_text_map = {}
EMBEDDING_DIM = 384

def cleanup_memory():
    import gc
    gc.collect()
    global faiss_indexes, vector_id_to_text_map
    for mode in list(faiss_indexes.keys()):
        if len(faiss_indexes[mode]) > 3:  # Keep only 3 most recent users
            oldest_key = next(iter(faiss_indexes[mode]))
            del faiss_indexes[mode][oldest_key]
            del vector_id_to_text_map[mode][oldest_key]

def initialize_rag_system():
    global embedding_model
    _load_dependencies()
    try:
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
    except Exception as e:
        print(f"⚠️ Using fallback model: {e}")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    cleanup_memory()

def _get_user_specific_filenames(user_api_key, mode):
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    index_filename = f'{user_hash}_{mode}_index.faiss'
    map_filename = f'{user_hash}_{mode}_map.json'
    return index_filename, map_filename

def _load_user_data(user_api_key, mode):
    cleanup_memory()
    
    if mode not in faiss_indexes:
        faiss_indexes[mode] = {}
    if mode not in vector_id_to_text_map:
        vector_id_to_text_map[mode] = {}

    if user_api_key in faiss_indexes[mode]:
        return
        
    index_filename, map_filename = _get_user_specific_filenames(user_api_key, mode)

    try:
        print(f"🧠 RAG Core: Downloading FAISS index '{index_filename}' from Supabase...")
        index_bytes = supabase.storage.from_("vector_store").download(index_filename)
        index_buffer = python_io.BytesIO(index_bytes)
        faiss_indexes[mode][user_api_key] = faiss.read_index(faiss.PyCallbackIOReader(index_buffer.read))
        print("✅ RAG Core: FAISS index loaded.")
    except Exception as e:
        print(f"ℹ️ RAG Core: No index file found for '{index_filename}'. Creating new. Error: {e}")
        faiss_indexes[mode][user_api_key] = faiss.IndexFlatL2(EMBEDDING_DIM)

    try:
        print(f"🧠 RAG Core: Downloading map file '{map_filename}' from Supabase...")
        map_bytes = supabase.storage.from_("vector_store").download(map_filename)
        map_data = json.loads(map_bytes.decode('utf-8'))
        vector_id_to_text_map[mode][user_api_key] = {int(k): v for k, v in map_data.items()}
        print("✅ RAG Core: Map file loaded.")
    except Exception as e:
        print(f"ℹ️ RAG Core: No map file found for '{map_filename}'. Creating new. Error: {e}")
        vector_id_to_text_map[mode][user_api_key] = {}

def _save_user_data(user_api_key, mode):
    cleanup_memory()
    
    index_filename, map_filename = _get_user_specific_filenames(user_api_key, mode)
    
    if mode in faiss_indexes and user_api_key in faiss_indexes[mode]:
        try:
            buffer = python_io.BytesIO()
            faiss.write_index(faiss_indexes[mode][user_api_key], faiss.PyCallbackIOWriter(buffer.write))
            buffer.seek(0)
            supabase.storage.from_("vector_store").upload(
                file=buffer.getvalue(),
                path=index_filename,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            print(f"✅ RAG Core: Saved FAISS index to '{index_filename}' in Supabase.")
        except Exception as e:
            print(f"❌ RAG Core: Failed to save FAISS index to Supabase. Error: {e}")

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

def _chunk_text(text, chunk_size=200, chunk_overlap=30):  # Reduced chunk size
    if not isinstance(text, str): return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def add_document_to_knowledge_base(user_api_key, document_text, document_id, mode):
    cleanup_memory()
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
    cleanup_memory()
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
    cleanup_memory()
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or not text_map or index.ntotal == 0:
        return f"The {mode} knowledge base is empty. Please upload some documents first."

    print(f"🧠 RAG Core: Received query for '{mode}' mode: '{query_text}'")
    query_embedding = embedding_model.encode([query_text])
    k = min(3, index.ntotal)  # Reduced from 5 to 3
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    retrieved_chunks = [text_map[i]["text"] for i in indices[0] if i in text_map]
    if not retrieved_chunks:
        return "I couldn't find any relevant information in your documents to answer that question."

    context = "\n\n---\n\n".join(retrieved_chunks)
    print(f"🧠 RAG Core: Found {len(retrieved_chunks)} relevant chunks.")

    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash',
                                    generation_config={
                                        "max_output_tokens": 1000,
                                        "temperature": 0
                                    })
        
        chat = model.start_chat(history=history)
        
        prompt = f"""
        You are an expert analyst. Your task is to provide a detailed answer based ONLY on the provided text snippets.

        CONTEXT SNIPPETS:
        {context}

        LATEST QUESTION:
        {query_text}

        FINAL ANSWER:
        """
        response = chat.send_message(prompt)
        print("✅ RAG Core: Generated final answer with Gemini.")
        return response.text
    except Exception as e:
        print(f"❌ RAG Core: Error during Gemini API call for chat: {e}")
        return f"An error occurred while trying to generate an answer: {e}"

