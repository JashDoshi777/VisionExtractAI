# rag_core.py

import os
import sys
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
from typing import List, Dict, Tuple
import time

# --- ROBUST NLTK SETUP ---
# Point NLTK to the local 'nltk_data' directory.
local_nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if os.path.exists(local_nltk_data_path):
    nltk.data.path.insert(0, local_nltk_data_path)
else:
    raise FileNotFoundError(
        "The 'nltk_data' directory was not found. "
        "Please run the 'download_nltk.py' script once to download the necessary NLTK models."
    )
# --- END SETUP ---

embedding_model = None
reranker_model = None
faiss_indexes: Dict[str, Dict[str, faiss.Index]] = {}
vector_id_to_text_map: Dict[str, Dict[str, Dict[int, Dict]]] = {}
keyword_indexes: Dict[str, Dict[str, Dict]] = {}
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_DIM = 768
CHUNK_SIZE = 300  # Reduced for better granularity
CHUNK_OVERLAP = 50

# Initialize components
def initialize_rag_system():
    """
    Loads the embedding model, reranker, and prepares the vector store directory.
    """
    global embedding_model, reranker_model
    print("RAG Core: Initializing Advanced RAG System...")
    
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
        print(f"Created directory: {VECTOR_STORE_PATH}")
    
    print("RAG Core: Loading advanced embedding model...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    print("RAG Core: Loading cross-encoder reranker...")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("RAG Core: Advanced models loaded successfully.")


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
        print(f"RAG Core: Loading FAISS index for user in '{mode}' mode")
        faiss_indexes[mode][user_api_key] = faiss.read_index(index_path)
    else:
        print(f"RAG Core: Creating new FAISS index for user in '{mode}' mode")
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


def _smart_chunking(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Intelligent chunking that preserves context and meaning.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # First split by paragraphs to maintain logical structure
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If the paragraph is small enough, try to keep it with the current chunk
        if len(current_chunk) + len(paragraph) <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph is too big, split by sentences
            if len(paragraph) > chunk_size:
                sentences = nltk.sent_tokenize(paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += " " + sentence if temp_chunk else sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                
                current_chunk = temp_chunk
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Ensure chunks have some overlap for better context
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and chunk_overlap > 0:
            # Add some words from previous chunk for context
            prev_words = chunks[i-1].split()[-chunk_overlap:]
            if prev_words:
                chunk = " ".join(prev_words) + " " + chunk
        final_chunks.append(chunk)
    
    return final_chunks


def _enhanced_query_expansion(query: str) -> List[str]:
    """
    Advanced query expansion with business context awareness.
    """
    query_lower = query.lower()
    expanded_queries = {query}
    
    # Business-specific expansions
    business_expansions = {
        r"\bgeneral manager\b": ["GM", "manager", "head", "director", "chief"],
        r"\bCEO\b": ["chief executive officer", "president", "director"],
        r"\bCFO\b": ["chief financial officer", "finance director"],
        r"\blocation\b": ["address", "located", "office", "headquarters", "branch"],
        r"\boffice\b": ["location", "branch", "headquarters", "situated"],
        r"\bservices\b": ["offerings", "products", "solutions", "business"],
        r"\bcompany\b": ["business", "organization", "firm", "corporation", "enterprise"],
        r"\bcontact\b": ["reach", "get in touch", "communicate"],
        r"\bbranch\b": ["office", "location", "division", "subsidiary"],
        r"\bheadquarters\b": ["main office", "head office", "corporate office"],
    }
    
    # Location-specific expansions
    location_patterns = {
        r"\bhong\s*kong\b": ["HK", "hongkong"],
        r"\bsingapore\b": ["SG", "sing"],
        r"\bunited\s*states\b": ["USA", "US", "America"],
        r"\bunited\s*kingdom\b": ["UK", "Britain"],
    }
    
    # Apply business expansions
    for pattern, replacements in business_expansions.items():
        if re.search(pattern, query_lower):
            for replacement in replacements:
                expanded_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                expanded_queries.add(expanded_query)
    
    # Apply location expansions
    for pattern, replacements in location_patterns.items():
        if re.search(pattern, query_lower):
            for replacement in replacements:
                expanded_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                expanded_queries.add(expanded_query)
    
    return list(expanded_queries)


def _build_enhanced_keyword_index(text, doc_id, user_api_key, mode):
    """
    Build an enhanced keyword index with business context awareness.
    """
    if not isinstance(text, str) or not text.strip():
        return

    if user_api_key not in keyword_indexes[mode]:
        keyword_indexes[mode][user_api_key] = {"documents": {}, "vocabulary": {}, "entities": {}}
    
    keyword_index = keyword_indexes[mode][user_api_key]
    
    # Extract both regular words and business entities
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    # Business entities (preserve case and multi-word terms)
    business_entities = re.findall(r'\b[A-Z][a-zA-Z&\s]{1,30}(?:Ltd|Inc|Corp|Company|Group|Holdings|Limited|Corporation|Enterprise|Solutions)\b', text)
    locations = re.findall(r'\b[A-Z][a-zA-Z\s]{2,20}(?:Street|Road|Avenue|Lane|Drive|Plaza|Square|Center|Centre|Building|Tower|Floor)\b', text)
    
    # Index regular words
    for word in words:
        if word not in stop_words and len(word) > 2:
            stemmed = ps.stem(word)
            if stemmed not in keyword_index["vocabulary"]:
                keyword_index["vocabulary"][stemmed] = []
            
            if doc_id not in keyword_index["vocabulary"][stemmed]:
                keyword_index["vocabulary"][stemmed].append(doc_id)
    
    # Index business entities
    if "entities" not in keyword_index:
        keyword_index["entities"] = {}
    
    for entity in business_entities + locations:
        entity_key = entity.lower()
        if entity_key not in keyword_index["entities"]:
            keyword_index["entities"][entity_key] = []
        if doc_id not in keyword_index["entities"][entity_key]:
            keyword_index["entities"][entity_key].append(doc_id)
    
    keyword_index["documents"][doc_id] = {
        "text": text,
        "length": len(text),
        "word_count": len(words),
        "entities": business_entities + locations
    }


def _enhanced_keyword_search(query, user_api_key, mode, top_k=10):
    """
    Enhanced keyword search with business context awareness.
    """
    if user_api_key not in keyword_indexes[mode]:
        return []
    
    keyword_index = keyword_indexes[mode][user_api_key]
    ps = PorterStemmer()
    
    # Process query for both regular words and entities
    query_terms = [ps.stem(term) for term in query.lower().split()
                   if term not in stopwords.words('english') and len(term) > 2]
    
    # Look for business entities in query
    entity_matches = []
    if "entities" in keyword_index:
        for entity, docs in keyword_index["entities"].items():
            if any(term in entity for term in query.lower().split()):
                entity_matches.extend(docs)
    
    doc_scores: Dict[str, float] = {}
    
    # Score based on regular keyword matches
    for term in query_terms:
        if term in keyword_index.get("vocabulary", {}):
            for doc_id in keyword_index["vocabulary"][term]:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1.0
    
    # Boost scores for entity matches
    for doc_id in entity_matches:
        if doc_id not in doc_scores:
            doc_scores[doc_id] = 0
        doc_scores[doc_id] += 2.0  # Higher weight for entity matches
    
    # Normalize scores
    final_scores = {}
    for doc_id, score in doc_scores.items():
        if doc_id in keyword_index.get("documents", {}):
            doc_length = keyword_index["documents"][doc_id].get("word_count", 1)
            final_scores[doc_id] = score / (1 + np.log(1 + doc_length))

    sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_id for doc_id, score in sorted_docs]


def add_document_to_knowledge_base(user_api_key, document_text, document_id, mode):
    """
    Processes a document's text and adds it to the knowledge base with enhanced indexing.
    """
    try:
        print(f"\nRAG: Adding document '{document_id}' to knowledge base...")
        _load_user_data(user_api_key, mode)

        chunks = _smart_chunking(document_text)
        print(f"RAG: Created {len(chunks)} intelligent chunks")

        _build_enhanced_keyword_index(document_text, document_id, user_api_key, mode)
        print("RAG: Built enhanced keyword index")

        if not chunks:
            print("RAG: No chunks to vectorize, saving keyword index only")
            _save_user_data(user_api_key, mode)
            return

        chunk_embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
        print("RAG: Generated embeddings")

        index = faiss_indexes[mode][user_api_key]
        text_map = vector_id_to_text_map[mode][user_api_key]
        start_index = index.ntotal

        index.add(np.array(chunk_embeddings, dtype=np.float32))

        for i, chunk in enumerate(chunks):
            vector_id = start_index + i
            text_map[vector_id] = {
                "text": chunk, 
                "source_doc": document_id,
                "chunk_id": i,
                "length": len(chunk)
            }
        
        _save_user_data(user_api_key, mode)
        print(f"RAG: Successfully indexed document. Total vectors: {index.ntotal}")

    except Exception as e:
        print(f"CRITICAL ERROR in add_document_to_knowledge_base: {e}")
        import traceback
        traceback.print_exc()
        raise e


def remove_document_from_knowledge_base(user_api_key, document_id, mode):
    """
    Removes all chunks associated with a document from the knowledge base by rebuilding the index.
    """
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)
    keyword_index = keyword_indexes.get(mode, {}).get(user_api_key, {})

    if not index or not text_map:
        print(f"RAG: No knowledge base found for user in '{mode}' mode")
        return

    ids_to_remove = {
        vector_id for vector_id, meta in text_map.items()
        if meta.get("source_doc") == document_id
    }

    if not ids_to_remove and not keyword_index.get("documents", {}).get(document_id):
        print(f"RAG: No data found for document '{document_id}'")
        return

    # Rebuild vector index
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

    # Rebuild keyword index
    new_keyword_index = {"documents": {}, "vocabulary": {}, "entities": {}}
    remaining_doc_ids = {meta['source_doc'] for meta in new_text_map.values()}
    remaining_doc_ids.discard(document_id)

    if keyword_index.get("documents"):
        new_keyword_index["documents"] = {
            doc_id: doc_meta for doc_id, doc_meta in keyword_index["documents"].items()
            if doc_id in remaining_doc_ids
        }
    
    if keyword_index.get("vocabulary"):
        for term, doc_list in keyword_index["vocabulary"].items():
            new_doc_list = [doc_id for doc_id in doc_list if doc_id in remaining_doc_ids]
            if new_doc_list:
                new_keyword_index["vocabulary"][term] = new_doc_list

    if keyword_index.get("entities"):
        for entity, doc_list in keyword_index["entities"].items():
            new_doc_list = [doc_id for doc_id in doc_list if doc_id in remaining_doc_ids]
            if new_doc_list:
                new_keyword_index["entities"][entity] = new_doc_list

    faiss_indexes[mode][user_api_key] = new_index
    vector_id_to_text_map[mode][user_api_key] = new_text_map
    keyword_indexes[mode][user_api_key] = new_keyword_index
    
    print(f"RAG: Removed document '{document_id}'. Index now has {new_index.ntotal} vectors.")
    _save_user_data(user_api_key, mode)


def _advanced_hybrid_search(query, user_api_key, mode, top_k=10):
    """
    Advanced hybrid search combining multiple retrieval methods.
    """
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or not text_map or index.ntotal == 0:
        return []

    # Vector search with multiple query variations
    expanded_queries = _enhanced_query_expansion(query)
    vector_candidates = set()
    
    for q in expanded_queries[:3]:  # Limit to avoid too much noise
        query_embedding = embedding_model.encode([q], normalize_embeddings=True)
        vector_k = min(top_k * 2, index.ntotal)
        
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), vector_k)
        for i, score in zip(indices[0], distances[0]):
            if i != -1 and i in text_map:
                vector_candidates.add((i, float(score)))

    # Enhanced keyword search
    keyword_doc_ids = set(_enhanced_keyword_search(query, user_api_key, mode, top_k=top_k*2))
    
    # Combine and score
    combined_scores: Dict[int, float] = {}
    
    # Add vector search results
    for vec_id, score in vector_candidates:
        combined_scores[vec_id] = score * 0.6  # 60% weight for semantic similarity
    
    # Add keyword search boost
    for vec_id, meta in text_map.items():
        if meta.get("source_doc") in keyword_doc_ids:
            combined_scores[vec_id] = combined_scores.get(vec_id, 0) + 0.4  # 40% weight for keyword match
    
    # Sort and return top results
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [vec_id for vec_id, score in sorted_results]


def _intelligent_rerank(query, candidate_chunks, top_k=5):
    """
    Intelligent reranking that considers both relevance and context completeness.
    """
    if not candidate_chunks or not reranker_model:
        return candidate_chunks[:top_k]
    
    # Use cross-encoder for initial scoring
    pairs = [(query, chunk["text"]) for chunk in candidate_chunks]
    cross_encoder_scores = reranker_model.predict(pairs)
    
    # Additional scoring based on content completeness
    enhanced_scores = []
    for i, (chunk, ce_score) in enumerate(zip(candidate_chunks, cross_encoder_scores)):
        text = chunk["text"]
        
        # Bonus for chunks that seem to contain complete information
        completeness_bonus = 0
        if any(marker in text.lower() for marker in ["located", "address", "office", "branch"]):
            completeness_bonus += 0.1
        if any(marker in text.lower() for marker in ["manager", "director", "ceo", "head"]):
            completeness_bonus += 0.1
        if any(marker in text.lower() for marker in ["company", "business", "organization"]):
            completeness_bonus += 0.05
        
        final_score = ce_score + completeness_bonus
        enhanced_scores.append((chunk, final_score))
    
    # Sort by enhanced scores and return top results
    reranked = sorted(enhanced_scores, key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in reranked[:top_k]]


def query_knowledge_base(user_api_key, query_text, mode):
    """
    Advanced query processing with human-like response generation.
    """
    _load_user_data(user_api_key, mode)
    index = faiss_indexes.get(mode, {}).get(user_api_key)
    text_map = vector_id_to_text_map.get(mode, {}).get(user_api_key)

    if not index or (index.ntotal == 0 and not keyword_indexes.get(mode, {}).get(user_api_key, {}).get("documents")):
        return "I don't have any documents in my knowledge base yet. Please upload some brochures or business cards first, and I'll be happy to help you find information from them!"

    print(f"RAG: Processing query: '{query_text}'")
    
    # Advanced search with multiple strategies
    expanded_queries = _enhanced_query_expansion(query_text)
    print(f"RAG: Expanded to {len(expanded_queries)} query variations")
    
    all_candidate_ids = set()
    for query in expanded_queries[:3]:  # Use top 3 expansions
        candidate_ids = _advanced_hybrid_search(query, user_api_key, mode, top_k=8)
        all_candidate_ids.update(candidate_ids)
    
    unique_candidates = [text_map[vec_id] for vec_id in all_candidate_ids if vec_id in text_map]
    
    # Intelligent reranking
    top_chunks = _intelligent_rerank(query_text, unique_candidates, top_k=5)
    
    if not top_chunks:
        return f"I couldn't find specific information about '{query_text}' in the uploaded documents. Could you try rephrasing your question or check if the information might be in a document that hasn't been uploaded yet?"

    # Prepare context for AI model
    context = "\n\n---DOCUMENT SECTION---\n\n".join([chunk["text"] for chunk in top_chunks])
    print(f"RAG: Found {len(top_chunks)} relevant sections. Generating human-like response...")

    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Enhanced prompt for human-like responses
        prompt = f"""You are a highly knowledgeable and helpful assistant who provides natural, conversational answers based on document information.

**CRITICAL INSTRUCTIONS:**
1. Answer the user's question in a natural, human-like way as if you're having a conversation
2. Use the information from the document sections below to provide accurate, specific details
3. If the user asks about a company, person, or location, provide comprehensive information from the documents
4. Be direct and specific - if someone asks "where is X located" and you find the address, state it clearly
5. If someone asks about a person's role, provide their title and any relevant details
6. Write in a conversational tone, not like you're reading from a manual
7. If you can't find the specific information requested, be honest but mention what related information you did find

**USER'S QUESTION:**
{query_text}

**RELEVANT DOCUMENT SECTIONS:**
{context}

**YOUR NATURAL, CONVERSATIONAL RESPONSE:**"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"RAG: Error during response generation: {e}")
        return f"I found some relevant information but encountered an error while processing it: {e}"