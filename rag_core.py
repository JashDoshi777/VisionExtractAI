# rag_core.py - Chroma Cloud Integration

import os
import sys
import numpy as np
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import hashlib
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from typing import List, Dict, Tuple
import time
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

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

# Model configuration - matching app.py
MODEL_MAP = {
    'gemini': 'google/gemma-3-4b-it:free',
    'deepseek': 'google/gemma-3-27b-it:free',
    'qwen': 'mistralai/mistral-small-3.1-24b-instruct:free',
    'nvidia': 'nvidia/nemotron-nano-12b-v2-vl:free',
    'amazon': 'amazon/nova-2-lite-v1:free'
}

# Best → fallback order (OCR strength)
FALLBACK_ORDER = [
    'gemini',
    'deepseek',
    'qwen',
    'nvidia',
    'amazon'
]

# Chroma Cloud configuration
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")

embedding_model = None
reranker_model = None
chroma_client = None
collections: Dict[str, chromadb.Collection] = {}
keyword_indexes: Dict[str, Dict[str, Dict]] = {}

EMBEDDING_DIM = 768
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Initialize components
def initialize_rag_system():
    """
    Loads the embedding model, reranker, and connects to Chroma Cloud.
    """
    global embedding_model, reranker_model, chroma_client
    print("RAG Core: Initializing Advanced RAG System with Chroma Cloud...")
    
    # Validate Chroma Cloud credentials
    if not all([CHROMA_TENANT, CHROMA_DATABASE, CHROMA_API_KEY]):
        raise ValueError(
            "Chroma Cloud credentials not found. Please set CHROMA_TENANT, CHROMA_DATABASE, and CHROMA_API_KEY in your .env file"
        )
    
    # Connect to Chroma Cloud
    print("RAG Core: Connecting to Chroma Cloud...")
    chroma_client = chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY
    )
    print("RAG Core: Successfully connected to Chroma Cloud!")
    
    print("RAG Core: Loading advanced embedding model...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    print("RAG Core: Loading cross-encoder reranker...")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("RAG Core: Advanced models loaded successfully.")


def _call_openrouter_api_with_fallback(api_key, selected_model_key, prompt):
    """
    Calls OpenRouter API with fallback support for text-only requests.
    """
    # Start with the selected model, then try others in fallback order
    models_to_try = [selected_model_key]
    for model in FALLBACK_ORDER:
        if model != selected_model_key:
            models_to_try.append(model)
    
    last_error = None
    
    for model_key in models_to_try:
        model_name = MODEL_MAP.get(model_key)
        if not model_name:
            continue
            
        print(f"RAG: Attempting API call with model: {model_name}...")
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            api_response = response.json()
            
            if 'choices' not in api_response or not api_response['choices']:
                print(f"RAG: Model {model_name} returned unexpected response format")
                last_error = f"Model {model_name} returned unexpected response format"
                continue
            
            result = api_response['choices'][0]['message']['content']
            print(f"RAG: Successfully processed with model: {model_name}")
            return result

        except requests.exceptions.HTTPError as http_err:
            error_msg = f"RAG: HTTP error for model {model_name}: {http_err}"
            if hasattr(response, 'text'):
                error_msg += f"\nResponse: {response.text}"
            print(error_msg)
            last_error = f"API request failed for {model_name} with status {response.status_code}."
            continue
        except Exception as e:
            print(f"RAG: Error with model {model_name}: {e}")
            last_error = f"An unexpected error occurred with model {model_name}."
            continue
    
    # If all models failed, return a user-friendly error
    return f"I'm having trouble connecting to the AI models right now. Please check your API key and try again. Last error: {last_error}"


def _get_collection_name(user_api_key, mode):
    """
    Creates a unique collection name for a user based on a hash of their API key.
    """
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    return f"{user_hash}_{mode}"


def _get_or_create_collection(user_api_key, mode):
    """
    Gets or creates a ChromaDB collection for the user/mode combination.
    """
    collection_name = _get_collection_name(user_api_key, mode)
    
    if collection_name in collections:
        return collections[collection_name]
    
    print(f"RAG Core: Getting/creating collection '{collection_name}' in Chroma Cloud")
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    collections[collection_name] = collection
    
    # Load keyword index from collection if exists
    _load_keyword_index(user_api_key, mode)
    
    return collection


def _load_keyword_index(user_api_key, mode):
    """
    Loads keyword index from Chroma Cloud collection metadata.
    """
    collection_name = _get_collection_name(user_api_key, mode)
    
    if mode not in keyword_indexes:
        keyword_indexes[mode] = {}
    
    if user_api_key in keyword_indexes[mode]:
        return
    
    try:
        collection = collections.get(collection_name)
        if collection:
            # Try to get keyword index document
            results = collection.get(
                ids=["__keyword_index__"],
                include=["documents"]
            )
            if results and results['documents'] and results['documents'][0]:
                keyword_indexes[mode][user_api_key] = json.loads(results['documents'][0])
                print(f"RAG Core: Loaded keyword index from Chroma Cloud")
            else:
                keyword_indexes[mode][user_api_key] = {"documents": {}, "vocabulary": {}, "entities": {}}
        else:
            keyword_indexes[mode][user_api_key] = {"documents": {}, "vocabulary": {}, "entities": {}}
    except Exception as e:
        print(f"RAG Core: Could not load keyword index: {e}")
        keyword_indexes[mode][user_api_key] = {"documents": {}, "vocabulary": {}, "entities": {}}


def _save_keyword_index(user_api_key, mode):
    """
    Saves keyword index to Chroma Cloud collection.
    """
    collection_name = _get_collection_name(user_api_key, mode)
    collection = collections.get(collection_name)
    
    if not collection or mode not in keyword_indexes or user_api_key not in keyword_indexes[mode]:
        return
    
    keyword_data = json.dumps(keyword_indexes[mode][user_api_key])
    
    try:
        # Upsert the keyword index document
        collection.upsert(
            ids=["__keyword_index__"],
            documents=[keyword_data],
            metadatas=[{"type": "keyword_index"}]
        )
        print("RAG Core: Saved keyword index to Chroma Cloud")
    except Exception as e:
        print(f"RAG Core: Error saving keyword index: {e}")


def _smart_chunking(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Intelligent chunking that preserves context and meaning.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
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
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and chunk_overlap > 0:
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
    
    location_patterns = {
        r"\bhong\s*kong\b": ["HK", "hongkong"],
        r"\bsingapore\b": ["SG", "sing"],
        r"\bunited\s*states\b": ["USA", "US", "America"],
        r"\bunited\s*kingdom\b": ["UK", "Britain"],
    }
    
    for pattern, replacements in business_expansions.items():
        if re.search(pattern, query_lower):
            for replacement in replacements:
                expanded_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                expanded_queries.add(expanded_query)
    
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

    if mode not in keyword_indexes:
        keyword_indexes[mode] = {}
    
    if user_api_key not in keyword_indexes[mode]:
        keyword_indexes[mode][user_api_key] = {"documents": {}, "vocabulary": {}, "entities": {}}
    
    keyword_index = keyword_indexes[mode][user_api_key]
    
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    business_entities = re.findall(r'\b[A-Z][a-zA-Z&\s]{1,30}(?:Ltd|Inc|Corp|Company|Group|Holdings|Limited|Corporation|Enterprise|Solutions)\b', text)
    locations = re.findall(r'\b[A-Z][a-zA-Z\s]{2,20}(?:Street|Road|Avenue|Lane|Drive|Plaza|Square|Center|Centre|Building|Tower|Floor)\b', text)
    
    for word in words:
        if word not in stop_words and len(word) > 2:
            stemmed = ps.stem(word)
            if stemmed not in keyword_index["vocabulary"]:
                keyword_index["vocabulary"][stemmed] = []
            
            if doc_id not in keyword_index["vocabulary"][stemmed]:
                keyword_index["vocabulary"][stemmed].append(doc_id)
    
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
    if mode not in keyword_indexes or user_api_key not in keyword_indexes[mode]:
        return []
    
    keyword_index = keyword_indexes[mode][user_api_key]
    ps = PorterStemmer()
    
    query_terms = [ps.stem(term) for term in query.lower().split()
                   if term not in stopwords.words('english') and len(term) > 2]
    
    entity_matches = []
    if "entities" in keyword_index:
        for entity, docs in keyword_index["entities"].items():
            if any(term in entity for term in query.lower().split()):
                entity_matches.extend(docs)
    
    doc_scores: Dict[str, float] = {}
    
    for term in query_terms:
        if term in keyword_index.get("vocabulary", {}):
            for doc_id in keyword_index["vocabulary"][term]:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1.0
    
    for doc_id in entity_matches:
        if doc_id not in doc_scores:
            doc_scores[doc_id] = 0
        doc_scores[doc_id] += 2.0
    
    final_scores = {}
    for doc_id, score in doc_scores.items():
        if doc_id in keyword_index.get("documents", {}):
            doc_length = keyword_index["documents"][doc_id].get("word_count", 1)
            final_scores[doc_id] = score / (1 + np.log(1 + doc_length))

    sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_id for doc_id, score in sorted_docs]


def add_document_to_knowledge_base(user_api_key, document_text, document_id, mode):
    """
    Processes a document's text and adds it to the knowledge base with Chroma Cloud.
    """
    try:
        print(f"\nRAG: Adding document '{document_id}' to Chroma Cloud...")
        collection = _get_or_create_collection(user_api_key, mode)

        chunks = _smart_chunking(document_text)
        print(f"RAG: Created {len(chunks)} intelligent chunks")

        _build_enhanced_keyword_index(document_text, document_id, user_api_key, mode)
        print("RAG: Built enhanced keyword index")

        if not chunks:
            print("RAG: No chunks to vectorize, saving keyword index only")
            _save_keyword_index(user_api_key, mode)
            return

        chunk_embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
        print("RAG: Generated embeddings")

        # Prepare data for Chroma
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_doc": document_id,
                "chunk_id": i,
                "length": len(chunk),
                "type": "document_chunk"
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Add to Chroma Cloud
        collection.upsert(
            ids=ids,
            embeddings=chunk_embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas
        )
        
        # Save keyword index
        _save_keyword_index(user_api_key, mode)
        
        print(f"RAG: Successfully indexed document to Chroma Cloud. Total chunks: {len(chunks)}")

    except Exception as e:
        print(f"CRITICAL ERROR in add_document_to_knowledge_base: {e}")
        import traceback
        traceback.print_exc()
        raise e


def remove_document_from_knowledge_base(user_api_key, document_id, mode):
    """
    Removes all chunks associated with a document from Chroma Cloud.
    """
    try:
        collection = _get_or_create_collection(user_api_key, mode)
        
        # Delete all chunks from this document using where filter
        collection.delete(
            where={"source_doc": document_id}
        )
        
        # Update keyword index
        if mode in keyword_indexes and user_api_key in keyword_indexes[mode]:
            keyword_index = keyword_indexes[mode][user_api_key]
            
            # Remove document from vocabulary
            if "vocabulary" in keyword_index:
                for term in list(keyword_index["vocabulary"].keys()):
                    if document_id in keyword_index["vocabulary"][term]:
                        keyword_index["vocabulary"][term].remove(document_id)
                    if not keyword_index["vocabulary"][term]:
                        del keyword_index["vocabulary"][term]
            
            # Remove document from entities
            if "entities" in keyword_index:
                for entity in list(keyword_index["entities"].keys()):
                    if document_id in keyword_index["entities"][entity]:
                        keyword_index["entities"][entity].remove(document_id)
                    if not keyword_index["entities"][entity]:
                        del keyword_index["entities"][entity]
            
            # Remove document metadata
            if "documents" in keyword_index and document_id in keyword_index["documents"]:
                del keyword_index["documents"][document_id]
            
            _save_keyword_index(user_api_key, mode)
        
        print(f"RAG: Removed document '{document_id}' from Chroma Cloud")
        
    except Exception as e:
        print(f"Error removing document: {e}")
        import traceback
        traceback.print_exc()


def _advanced_hybrid_search(query, user_api_key, mode, top_k=10):
    """
    Advanced hybrid search using Chroma Cloud query.
    """
    collection = _get_or_create_collection(user_api_key, mode)
    
    # Check if collection has documents
    try:
        count = collection.count()
        if count == 0:
            return []
    except:
        return []

    # Vector search with Chroma Cloud
    expanded_queries = _enhanced_query_expansion(query)
    all_results = {}
    
    for q in expanded_queries[:3]:  # Limit to avoid too much noise
        query_embedding = embedding_model.encode([q], normalize_embeddings=True)
        
        try:
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(top_k * 2, count),
                where={"type": "document_chunk"},
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results['ids'] and results['ids'][0]:
                for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (Chroma returns L2 distance for cosine)
                    score = 1 - distance if distance else 0
                    if doc_id not in all_results or all_results[doc_id]['score'] < score:
                        all_results[doc_id] = {
                            'text': doc,
                            'source_doc': metadata.get('source_doc', ''),
                            'chunk_id': metadata.get('chunk_id', 0),
                            'length': metadata.get('length', 0),
                            'score': score
                        }
        except Exception as e:
            print(f"RAG: Search error: {e}")
            continue
    
    # Enhanced keyword search boost
    keyword_doc_ids = set(_enhanced_keyword_search(query, user_api_key, mode, top_k=top_k*2))
    
    # Add keyword boost to scores
    for doc_id, result in all_results.items():
        if result.get('source_doc') in keyword_doc_ids:
            result['score'] = result.get('score', 0) + 0.4
    
    # Sort and return top results
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
    return [result for doc_id, result in sorted_results]


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


def query_knowledge_base(user_api_key, query_text, mode, selected_model_key):
    """
    Advanced query processing with human-like response generation using selected model with fallback.
    """
    collection = _get_or_create_collection(user_api_key, mode)
    
    try:
        count = collection.count()
        # Exclude keyword index from count
        if count <= 1:
            return "I don't have any documents in my knowledge base yet. Please upload some brochures or business cards first, and I'll be happy to help you find information from them!"
    except:
        return "I don't have any documents in my knowledge base yet. Please upload some brochures or business cards first, and I'll be happy to help you find information from them!"

    print(f"RAG: Processing query: '{query_text}' with model: {selected_model_key}")
    
    # Advanced search with multiple strategies
    expanded_queries = _enhanced_query_expansion(query_text)
    print(f"RAG: Expanded to {len(expanded_queries)} query variations")
    
    all_candidates = []
    seen_texts = set()
    
    for query in expanded_queries[:3]:  # Use top 3 expansions
        candidates = _advanced_hybrid_search(query, user_api_key, mode, top_k=8)
        for candidate in candidates:
            text = candidate.get('text', '')
            if text and text not in seen_texts:
                seen_texts.add(text)
                all_candidates.append(candidate)
    
    # Intelligent reranking
    top_chunks = _intelligent_rerank(query_text, all_candidates, top_k=5)
    
    if not top_chunks:
        return f"I couldn't find specific information about '{query_text}' in the uploaded documents. Could you try rephrasing your question or check if the information might be in a document that hasn't been uploaded yet?"

    # Prepare context for AI model
    context = "\n\n---DOCUMENT SECTION---\n\n".join([chunk["text"] for chunk in top_chunks])
    print(f"RAG: Found {len(top_chunks)} relevant sections. Generating response with {selected_model_key}...")

    try:
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
        
        response = _call_openrouter_api_with_fallback(user_api_key, selected_model_key, prompt)
        return response
        
    except Exception as e:
        print(f"RAG: An unexpected error occurred during response generation: {e}")
        import traceback
        traceback.print_exc()
        return "I found relevant information but ran into an unexpected error while processing it. Please try again."