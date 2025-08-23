# rag_core.py

import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import hashlib
import google.generativeai as genai
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import spacy
import en_core_web_sm
from typing import List, Dict, Tuple
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Load spaCy for better text processing
try:
    nlp = en_core_web_sm.load()
except Exception as e:
    print("⚠️ spaCy model loading failed:", e)
    nlp = spacy.blank("en")

embedding_model = None
reranker_model = None
faiss_indexes = {}
vector_id_to_text_map = {}
keyword_indexes = {}
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_DIM = 768  # Dimension for 'all-mpnet-base-v2'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Initialize components
def initialize_rag_system():
    """
    Loads the embedding model, reranker, and prepares the vector store directory.
    """
    global embedding_model, reranker_model
    print("🧠 RAG Core: Initializing Advanced RAG System...")
    
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
        print(f"✅ Created directory: {VECTOR_STORE_PATH}")
    
    print("🧠 RAG Core: Loading advanced embedding model...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    print("🧠 RAG Core: Loading cross-encoder reranker...")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("✅ RAG Core: Advanced models loaded successfully.")


def _get_user_specific_paths(user_api_key, mode):
    """
    Creates a unique, safe filename prefix for a user based on a hash of their API key.
    """
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    index_path = os.path.join(VECTOR_STORE_PATH, f'{user_hash}_{mode}_index.faiss')
    map_path = os.path.join(VECTOR_STORE_PATH, f'{user_hash}_{mode}_map.json')
    keyword_path = os.path.join(VECTOR_STORE_PATH, f'{user_hash}_{mode}_keyword.json')
    return index_path, map_path, keyword_path


def _load_user_data(user_api_key, mode):
    """
    Loads a user's FAISS index, text map, and keyword index from disk into memory.
    """
    if mode not in faiss_indexes:
        faiss_indexes[mode] = {}
    if mode not in vector_id_to_text_map:
        vector_id_to_text_map[mode] = {}
    if mode not in keyword_indexes:
        keyword_indexes[mode] = {}

    if user_api_key in faiss_indexes[mode]:
        return
        
    index_path, map_path, keyword_path = _get_user_specific_paths(user_api_key, mode)

    # Load FAISS index
    if os.path.exists(index_path):
        print(f"🧠 RAG Core: Loading FAISS index for user in '{mode}' mode")
        faiss_indexes[mode][user_api_key] = faiss.read_index(index_path)
    else:
        print(f"🧠 RAG Core: Creating new FAISS index for user in '{mode}' mode")
        faiss_indexes[mode][user_api_key] = faiss.IndexFlatIP(EMBEDDING_DIM)

    # Load text map
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            vector_id_to_text_map[mode][user_api_key] = {int(k): v for k, v in json.load(f).items()}
    else:
        vector_id_to_text_map[mode][user_api_key] = {}

    # Load keyword index
    if os.path.exists(keyword_path):
        with open(keyword_path, 'r') as f:
            keyword_indexes[mode][user_api_key] = json.load(f)
    else:
        keyword_indexes[mode][user_api_key] = {"documents": {}, "vocabulary": {}}


def _save_user_data(user_api_key, mode):
    """
    Saves a user's FAISS index, text map, and keyword index to disk.
    """
    index_path, map_path, keyword_path = _get_user_specific_paths(user_api_key, mode)
    
    if mode in faiss_indexes and user_api_key in faiss_indexes[mode]:
        faiss.write_index(faiss_indexes[mode][user_api_key], index_path)
    
    if mode in vector_id_to_text_map and user_api_key in vector_id_to_text_map[mode]:
        with open(map_path, 'w') as f:
            json.dump(vector_id_to_text_map[mode][user_api_key], f)
    
    if mode in keyword_indexes and user_api_key in keyword_indexes[mode]:
        with open(keyword_path, 'w') as f:
            json.dump(keyword_indexes[mode][user_api_key], f)


def _semantic_chunking(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Advanced chunking that respects semantic boundaries using spaCy.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start:] + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _expand_query(query: str) -> List[str]:
    """
    Expand the query with synonyms and related terms for better retrieval.
    """
    expansion_rules = {
        r"\bhow\b": ["what", "method", "process", "way"],
        r"\bwhy\b": ["reason", "cause", "purpose"],
        r"\bwhat\b": ["which", "information about", "details about"],
        r"\bwhen\b": ["time", "date", "schedule"],
        r"\bwhere\b": ["location", "place", "address"],
        r"\bwho\b": ["person", "individual", "contact"],
        r"\bcompany\b": ["business", "organization", "firm", "corporation"],
        r"\bemail\b": ["contact", "address", "electronic mail"],
        r"\bphone\b": ["number", "telephone", "contact", "call"],
        r"\baddress\b": ["location", "place", "where"],
    }
    
    expanded_queries = {query} # Use a set to avoid duplicates
    
    for pattern, replacements in expansion_rules.items():
        if re.search(pattern, query, re.IGNORECASE):
            for replacement in replacements:
                expanded_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                expanded_queries.add(expanded_query)
    
    return list(expanded_queries)


def _build_keyword_index(text, doc_id, user_api_key, mode):
    """
    Build a simple keyword index for hybrid search.
    """
    # **FIX**: Add guard clause to prevent crash on invalid input.
    if not isinstance(text, str) or not text.strip():
        return

    if user_api_key not in keyword_indexes[mode]:
        keyword_indexes[mode][user_api_key] = {"documents": {}, "vocabulary": {}}
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    keyword_index = keyword_indexes[mode][user_api_key]
    
    for word in words:
        if word not in stop_words:
            stemmed = ps.stem(word)
            if stemmed not in keyword_index["vocabulary"]:
                keyword_index["vocabulary"][stemmed] = []
            
            if doc_id not in keyword_index["vocabulary"][stemmed]:
                keyword_index["vocabulary"][stemmed].append(doc_id)
    
    keyword_index["documents"][doc_id] = {
        "text": text,
        "length": len(text),
        "word_count": len(words)
    }


def _keyword_search(query, user_api_key, mode, top_k=5):
    """
    Perform keyword search as a fallback or for hybrid search.
    """
    if user_api_key not in keyword_indexes[mode]:
        return []
    
    keyword_index = keyword_indexes[mode][user_api_key]
    ps = PorterStemmer()
    query_terms = [ps.stem(term) for term in query.lower().split()
                   if term not in stopwords.words('english') and len(term) > 2]
    
    doc_scores = {}
    
    for term in query_terms:
        if term in keyword_index.get("vocabulary", {}):
            for doc_id in keyword_index["vocabulary"][term]:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1
    
    # **FIX**: Make scoring robust against inconsistent keyword index state.
    # This prevents KeyError crashes if a doc_id is in vocabulary but not documents.
    scored_docs = {}
    for doc_id, score in doc_scores.items():
        if doc_id in keyword_index.get("documents", {}):
            doc_length = keyword_index["documents"][doc_id].get("word_count", 1)
            scored_docs[doc_id] = score / (1 + np.log(1 + doc_length))

    sorted_docs = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [doc_id for doc_id, score in sorted_docs]


# In rag_core.py, replace the existing add_document_to_knowledge_base function with this one.

def add_document_to_knowledge_base(user_api_key, document_text, document_id, mode):
    """
    Processes a document's text and adds it to the knowledge base.
    """
    try:
        print("\n--- RAG ADD TRACE: START ---")
        _load_user_data(user_api_key, mode)
        print("--- RAG ADD TRACE: 1. User data loaded.")

        chunks = _semantic_chunking(document_text)
        print(f"--- RAG ADD TRACE: 2. Text chunked into {len(chunks)} pieces.")

        # Also build keyword index for the full text
        _build_keyword_index(document_text, document_id, user_api_key, mode)
        print("--- RAG ADD TRACE: 3. Keyword index built.")

        if not chunks:
            print("--- RAG ADD TRACE: No chunks to add to vector index. Saving keyword index.")
            _save_user_data(user_api_key, mode)
            print("--- RAG ADD TRACE: FINISHED (no vector chunks).")
            return

        print("--- RAG ADD TRACE: 4. Encoding embeddings for chunks...")
        chunk_embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
        print("--- RAG ADD TRACE: 5. Embeddings encoded successfully.")

        index = faiss_indexes[mode][user_api_key]
        text_map = vector_id_to_text_map[mode][user_api_key]
        start_index = index.ntotal

        print("--- RAG ADD TRACE: 6. Preparing to add vectors to FAISS index...")
        index.add(np.array(chunk_embeddings, dtype=np.float32))
        print("--- RAG ADD TRACE: 7. Vectors added to FAISS index successfully.")

        for i, chunk in enumerate(chunks):
            vector_id = start_index + i
            text_map[vector_id] = {
                "text": chunk, 
                "source_doc": document_id,
                "chunk_id": i,
                "length": len(chunk)
            }
        print("--- RAG ADD TRACE: 8. Text map updated.")
        
        print("--- RAG ADD TRACE: 9. Saving all RAG data to disk...")
        _save_user_data(user_api_key, mode)
        print("--- RAG ADD TRACE: 10. All RAG data saved successfully.")
        
        print(f"✅ RAG Core: Added {len(chunks)} chunks from '{document_id}'. Total vectors: {index.ntotal}")
        print("--- RAG ADD TRACE: FINISHED (with vector chunks).")

    except Exception as e:
        print(f"❌❌❌ CRITICAL CRASH INSIDE add_document_to_knowledge_base: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception to ensure the calling function knows about the failure
        raise e

def remove_document_from_knowledge_base(user_api_key, document_id, mode):
    """
    Removes all chunks associated with a document from the knowledge base by rebuilding the index.
    This is a safe and robust method that prevents assertion errors.
    """
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)
    keyword_index = keyword_indexes.get(mode, {}).get(user_api_key, {})

    if not index or not text_map:
        print(f"⚠️ RAG Core: No knowledge base found for user in '{mode}' mode")
        return

    ids_to_remove = {
        vector_id for vector_id, meta in text_map.items()
        if meta.get("source_doc") == document_id
    }

    if not ids_to_remove:
        print(f"⚠️ RAG Core: No vectors found for document '{document_id}'")
        # Attempt to clean up keyword index just in case it's inconsistent
        if keyword_index.get("documents", {}).get(document_id):
             print(f"⚠️ RAG Core: Found orphaned document '{document_id}' in keyword index. Cleaning up.")
        else:
            return # Nothing to do

    all_current_ids = sorted(text_map.keys())
    ids_to_keep = [vid for vid in all_current_ids if vid not in ids_to_remove]
    
    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    new_text_map = {}

    if ids_to_keep:
        vectors_to_keep = np.array([index.reconstruct(int(vid)) for vid in ids_to_keep], dtype=np.float32)
        if vectors_to_keep.ndim == 2 and vectors_to_keep.shape[0] > 0:
            new_index.add(vectors_to_keep)
        
        for new_id, old_id in enumerate(ids_to_keep):
            new_text_map[new_id] = text_map[old_id]
            
    new_keyword_index = {"documents": {}, "vocabulary": {}}
    remaining_doc_ids = {meta['source_doc'] for meta in new_text_map.values()}
    # Also explicitly remove the document_id being deleted
    remaining_doc_ids.discard(document_id)

    if keyword_index.get("documents"):
        new_keyword_index["documents"] = {
            doc_id: doc_meta for doc_id, doc_meta in keyword_index["documents"].items()
            if doc_id in remaining_doc_ids
        }
    
    if keyword_index.get("vocabulary"):
        new_keyword_index["vocabulary"] = {}
        for term, doc_list in keyword_index["vocabulary"].items():
            new_doc_list = [doc_id for doc_id in doc_list if doc_id in remaining_doc_ids]
            if new_doc_list:
                new_keyword_index["vocabulary"][term] = new_doc_list

    faiss_indexes[mode][user_api_key] = new_index
    vector_id_to_text_map[mode][user_api_key] = new_text_map
    keyword_indexes[mode][user_api_key] = new_keyword_index
    
    print(f"✅ RAG Core: Removed document '{document_id}'. Rebuilt index with {new_index.ntotal} vectors.")
    _save_user_data(user_api_key, mode)


def _hybrid_search(query, user_api_key, mode, top_k=10):
    """
    Perform hybrid search combining vector and keyword approaches.
    """
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or not text_map or index.ntotal == 0:
        return []

    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    vector_k = min(top_k * 2, index.ntotal)
    
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), vector_k)
    vector_results = [(i, float(score)) for i, score in zip(indices[0], distances[0]) if i != -1 and i in text_map]

    keyword_doc_ids = set(_keyword_search(query, user_api_key, mode, top_k=top_k*2))
    
    combined_scores = {}
    
    for vec_id, score in vector_results:
        combined_scores[vec_id] = score * 0.7
    
    for vec_id, meta in text_map.items():
        if meta.get("source_doc") in keyword_doc_ids:
            combined_scores[vec_id] = combined_scores.get(vec_id, 0) + 0.3
    
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [vec_id for vec_id, score in sorted_results]


def _rerank_results(query, candidate_chunks, top_k=5):
    """
    Rerank search results using a cross-encoder for better relevance.
    """
    if not candidate_chunks or not reranker_model:
        return candidate_chunks[:top_k]
    
    pairs = [(query, chunk["text"]) for chunk in candidate_chunks]
    scores = reranker_model.predict(pairs)
    reranked = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk, score in reranked[:top_k]]


def query_knowledge_base(user_api_key, query_text, mode):
    """
    Query the knowledge base with advanced retrieval and ranking.
    """
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or (index.ntotal == 0 and not keyword_indexes.get(mode, {}).get(user_api_key, {}).get("documents")):
        return "The knowledge base is empty. Please upload some documents first."

    print(f"🧠 RAG Core: Processing query: '{query_text}'")
    
    expanded_queries = _expand_query(query_text)
    print(f"🧠 RAG Core: Expanded queries: {expanded_queries}")
    
    all_candidate_ids = set()
    for query in expanded_queries:
        candidate_ids = _hybrid_search(query, user_api_key, mode, top_k=5)
        all_candidate_ids.update(candidate_ids)
    
    unique_candidates = [text_map[vec_id] for vec_id in all_candidate_ids if vec_id in text_map]
    
    top_chunks = _rerank_results(query_text, unique_candidates, top_k=3)
    
    if not top_chunks:
        return "I couldn't find any relevant information in your documents to answer that question."

    context = "\n\n---\n\n".join([chunk["text"] for chunk in top_chunks])
    print(f"🧠 RAG Core: Found {len(top_chunks)} relevant chunks after reranking.")

    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an expert assistant tasked with answering questions based on the provided context.
        
        **Instructions:**
        1. Answer the question using ONLY the context provided below.
        2. If the answer cannot be found in the context, say "I cannot find this information in the provided documents."
        3. Be precise, helpful, and concise in your response.
        4. If the question is general (not specifically about the context), you can use your general knowledge but mention that it's not from the documents.
        5. Format your response nicely with clear sections if appropriate.
        
        **Context:**
        {context}
        
        **Question:**
        {query_text}
        
        **Answer:**
        """
        
        response = model.generate_content(prompt)
        print("✅ RAG Core: Generated answer with enhanced prompting.")
        return response.text
    except Exception as e:
        print(f"❌ RAG Core: Error during Gemini API call: {e}")
        return f"An error occurred while trying to generate an answer: {e}"