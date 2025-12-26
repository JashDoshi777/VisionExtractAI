# app.py

import os
import io
import json
import hashlib
import requests
import base64
from flask import Flask, request, jsonify, send_from_directory, render_template, session
import webbrowser
from flask_cors import CORS
from PIL import Image
import fitz  # PyMuPDF
import rag_core
from datetime import timedelta
import traceback
import time
import re

from dotenv import load_dotenv
load_dotenv()

# --- MODIFIED: Import db and models from models.py ---
from models import db, BusinessCard, Brochure, Contact


app = Flask(__name__)
CORS(app)

# Disable template caching for development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Session configuration
app.secret_key = os.environ.get("SESSION_SECRET", "a-very-secret-key-for-sessions")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# --- FOLDER CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'user_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    
# --- DATABASE CONFIGURATION ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URI',
    'sqlite:///local_crm.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- MODIFIED: Initialize the app with the database object ---
db.init_app(app)


# --- DATABASE MODEL DEFINITIONS HAVE BEEN MOVED TO models.py ---


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



# All your other functions (_call_openrouter_api_with_fallback, etc.) remain unchanged below...
def _call_openrouter_api_with_fallback(api_key, selected_model_key, prompt, images=[]):
    if images:
        vision_models = ['gemini','deepseek','qwen','nvidia','amazon']
        models_to_try = [m for m in vision_models if m == selected_model_key]
        models_to_try.extend([m for m in vision_models if m != selected_model_key])
        models_to_try.extend([m for m in FALLBACK_ORDER if m not in vision_models])
    else:
        models_to_try = [selected_model_key]
        for model in FALLBACK_ORDER:
            if model != selected_model_key:
                models_to_try.append(model)
    
    last_error = None
    
    for model_key in models_to_try:
        model_name = MODEL_MAP.get(model_key)
        if not model_name: continue
            
        print(f"Attempting API call with model: {model_name}...")
        content_parts = [{"type": "text", "text": prompt}]
        
        if images and model_key in ['gemini','deepseek','qwen','nvidia','amazon']:
            for img in images:
                buffered = io.BytesIO()
                img_format = img.format or "PNG"
                img.save(buffered, format=img_format)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                content_parts.append({
                    "type": "image_url",
                    "image_url": { "url": f"data:image/{img_format.lower()};base64,{img_base64}" }
                })
        elif images and model_key not in ['gemini','deepseek','qwen','nvidia','amazon']:
            print(f"Skipping {model_name} - no image input support")
            continue

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": [{"role": "user", "content": content_parts}]},
                timeout=30
            )
            response.raise_for_status()
            api_response = response.json()
            
            if 'choices' not in api_response or not api_response['choices']:
                print(f"Model {model_name} returned empty response")
                last_error = {"error": f"Model {model_name} returned empty response"}
                continue
                
            json_text = api_response['choices'][0]['message']['content']
            
            cleaned_json_text = re.search(r'```json\s*([\s\S]+?)\s*```', json_text)
            if cleaned_json_text:
                json_text = cleaned_json_text.group(1)
            else:
                json_text = json_text.strip()
            
            result = json.loads(json_text)
            print(f"Successfully processed with model: {model_name}")
            return result
        except requests.exceptions.HTTPError as http_err:
            error_msg = f"HTTP error occurred for model {model_name}: {http_err}"
            if hasattr(response, 'text'): error_msg += f"\nResponse: {response.text}"
            print(error_msg)
            last_error = {"error": f"API request failed for {model_name} with status {response.status_code}."}
            continue
        except requests.exceptions.Timeout:
            print(f"Timeout error for model {model_name}")
            last_error = {"error": f"Request timeout for model {model_name}"}
            continue
        except json.JSONDecodeError as json_err:
            error_msg = f"JSON Decode Error for model {model_name}: {json_err}\nMalformed response: {json_text}"
            print(error_msg)
            last_error = {"error": f"Model {model_name} returned invalid JSON."}
            continue
        except Exception as e:
            print(f"An error occurred with model {model_name}: {e}")
            traceback.print_exc()
            last_error = {"error": f"An unexpected error occurred with model {model_name}."}
            continue
    
    return last_error or {"error": "All models failed to process the request."}

def _call_openrouter_api_text_only_with_fallback(api_key, selected_model_key, prompt):
    models_to_try = [selected_model_key] + [m for m in FALLBACK_ORDER if m != selected_model_key]
    last_error = None
    for model_key in models_to_try:
        model_name = MODEL_MAP.get(model_key)
        if not model_name: continue
        print(f"Attempting text-only API call with model: {model_name}...")
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            response.raise_for_status()
            api_response = response.json()
            if 'choices' not in api_response or not api_response['choices']:
                last_error = {"error": f"Model {model_name} returned unexpected response format"}
                continue
            result = api_response['choices'][0]['message']['content']
            print(f"Successfully processed text with model: {model_name}")
            return result
        except requests.exceptions.HTTPError as http_err:
            error_msg = f"HTTP error occurred for model {model_name}: {http_err}"
            if hasattr(response, 'text'): error_msg += f"\nResponse: {response.text}"
            print(error_msg)
            last_error = {"error": f"API request failed for {model_name} with status {response.status_code}."}
            continue
        except requests.exceptions.Timeout:
            print(f"Timeout error for model {model_name}")
            last_error = {"error": f"Request timeout for model {model_name}"}
            continue
        except Exception as e:
            print(f"An error occurred with model {model_name}: {e}")
            traceback.print_exc()
            last_error = {"error": f"An unexpected error occurred with model {model_name}."}
            continue
    if isinstance(last_error, dict) and "error" in last_error:
        return last_error["error"]
    return "All models failed to process the text request."


def _extract_contact_info_from_text(text):
    if not text: return "", []
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(?:\+?\d{1,4}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    emails = re.findall(email_pattern, text, re.IGNORECASE)
    phones = re.findall(phone_pattern, text)
    clean_text = text
    clean_text = re.sub(email_pattern, '', clean_text, flags=re.IGNORECASE)
    for phone in phones:
        if len(phone.replace('-', '').replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('+', '')) >= 7:
            clean_text = clean_text.replace(phone, '')
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text)
    return clean_text, emails + phones

def _create_clean_info_text(brochure_data):
    company_name = brochure_data.get("company_name", "")
    raw_text = brochure_data.get("raw_text", "")
    info_parts = []
    if company_name and company_name != "Unknown Company":
        info_parts.append(f"Company: {company_name}")
    if raw_text:
        clean_text, _ = _extract_contact_info_from_text(raw_text)
        contact_phrases = [r'contact\s+us\s*:?', r'for\s+more\s+information\s*:?', r'reach\s+out\s+to\s*:?', r'get\s+in\s+touch\s*:?', r'phone\s*:', r'email\s*:', r'tel\s*:', r'mobile\s*:', r'call\s+us\s*:?', r'write\s+to\s+us\s*:?',]
        for phrase in contact_phrases:
            clean_text = re.sub(phrase, '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        clean_text = re.sub(r'\n\s*\n', '\n', clean_text)
        if clean_text: info_parts.append(clean_text)
    return "\n".join(info_parts) if info_parts else ""

def _get_user_data_filepath(user_api_key, mode):
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    return os.path.join(DATA_FOLDER, f'{user_hash}_{mode}_data.json')

def _load_user_data(user_api_key, mode):
    filepath = _get_user_data_filepath(user_api_key, mode)
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f: return json.load(f)
    except (IOError, json.JSONDecodeError): return []
    return []

def _save_user_data(user_api_key, mode, data):
    filepath = _get_user_data_filepath(user_api_key, mode)
    try:
        with open(filepath, 'w') as f: json.dump(data, f, indent=4)
        return True
    except IOError: return False

def _clean_and_validate_contacts(data):
    if not data or "contacts" not in data: return data
    cleaned_contacts = []
    def is_placeholder(value):
        if not isinstance(value, str): return True
        test_val = value.strip().lower()
        if not test_val: return True
        placeholders = ["n/a", "na", "none", "null"]
        if test_val in placeholders: return True
        if "not available" in test_val or "not specified" in test_val or "not applicable" in test_val: return True
        return False
    for contact in data.get("contacts", []):
        name = contact.get("Owner Name")
        if is_placeholder(name): continue
        cleaned_contacts.append({
            "Owner Name": name.strip(),
            "Email": None if is_placeholder(contact.get("Email")) else contact.get("Email").strip(),
            "Number": None if is_placeholder(contact.get("Number")) else contact.get("Number").strip()
        })
    data["contacts"] = cleaned_contacts
    return data

def extract_card_data(image_bytes, user_api_key, selected_model_key):
    print("Processing business card with OpenRouter API...")
    if not user_api_key: return {"error": "A valid OpenRouter API Key was not provided."}
    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = """You are an expert at reading business cards. Analyze the image and extract information into a structured JSON format. The JSON object must use these exact keys: "Owner Name", "Company Name", "Email", "Number", "Address". If a piece of information is not present, its value must be `null`. Your entire response MUST be a single, valid JSON object."""
        parsed_info = _call_openrouter_api_with_fallback(user_api_key, selected_model_key, prompt, images=[img])
        if "error" in parsed_info: return parsed_info
        return {"Owner Name": parsed_info.get("Owner Name"), "Company Name": parsed_info.get("Company Name"), "Email": parsed_info.get("Email"), "Number": parsed_info.get("Number"), "Address": parsed_info.get("Address")}
    except Exception as e:
        print(f"Error during OpenRouter API call for business card: {e}")
        traceback.print_exc()
        return {"error": f"Failed to parse AI response: {e}"}

def _extract_brochure_data_with_vision(image_list, user_api_key, selected_model_key):
    print(f"Vision Extraction: Analyzing {len(image_list)} images with OpenRouter...")
    if not user_api_key: return {"error": "A valid OpenRouter API Key was not provided."}
    try:
        prompt = """You are a world-class document analysis expert. Analyze the provided document images with maximum precision. CRITICAL INSTRUCTIONS: 1. Extract the company name. 2. Extract ONLY contact information (names, emails, phone numbers) and put them in the "contacts" array. 3. Extract ALL OTHER content (company description, services, mission, addresses, general information) as "raw_text". 4. DO NOT include contact details like names, emails, or phone numbers in the raw_text. 5. Focus on separating contact information from general company information. OUTPUT FORMAT: Return a SINGLE, valid JSON object with these exact keys: "company_name", "contacts", "raw_text". The "contacts" key must contain a list of objects, each with "Owner Name", "Email", and "Number". If a piece of information is missing for a contact, use `null`. The "raw_text" should contain business information, services, descriptions, but NO contact details."""
        raw_data = _call_openrouter_api_with_fallback(user_api_key, selected_model_key, prompt, images=image_list)
        if "error" in raw_data: return raw_data
        print("AI vision extraction complete. Applying bulletproof cleaning...")
        cleaned_data = _clean_and_validate_contacts(raw_data)
        return cleaned_data
    except Exception as e:
        print(f"Error during unified brochure vision extraction: {e}")
        traceback.print_exc()
        return {"error": f"Failed to parse data from brochure images: {e}"}

@app.before_request
def make_session_permanent():
    session.permanent = True

@app.route('/process_card', methods=['POST'])
def process_card_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file, user_api_key, selected_model_key = request.files['file'], request.form.get('apiKey'), request.form.get('selectedModel')
    if not user_api_key or not selected_model_key: return jsonify({'error': 'API Key and Model are required'}), 400
    if selected_model_key not in MODEL_MAP: return jsonify({'error': 'Invalid model selected'}), 400
    
    try:
        image_bytes = file.read()
        extracted_info = extract_card_data(image_bytes, user_api_key, selected_model_key)
        if "error" in extracted_info: return jsonify(extracted_info), 500

        file_id = os.urandom(8).hex()
        _, f_ext = os.path.splitext(file.filename)
        safe_ext = f_ext if f_ext.lower() in ['.png', '.jpg', '.jpeg', '.webp'] else '.png'
        image_filename = f"{file_id}{safe_ext}"
        save_path = os.path.join(UPLOAD_FOLDER, image_filename)
        with open(save_path, 'wb') as f: f.write(image_bytes)
        
        extracted_info['id'] = file_id
        extracted_info['image_filename'] = image_filename
        
        user_contacts = _load_user_data(user_api_key, 'cards')
        user_contacts.insert(0, extracted_info)
        _save_user_data(user_api_key, 'cards', user_contacts)
        
        try:
            user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()
            new_card = BusinessCard(
                json_id=file_id,
                owner_name=extracted_info.get("Owner Name"),
                company_name=extracted_info.get("Company Name"),
                email=extracted_info.get("Email"),
                phone_number=extracted_info.get("Number"),
                address=extracted_info.get("Address"),
                source_document=file.filename,
                user_hash=user_hash
            )
            db.session.add(new_card)
            db.session.commit()
            print(f"Successfully saved business card for '{extracted_info.get('Owner Name')}' to the database.")
        except Exception as e:
            db.session.rollback()
            print(f"DATABASE ERROR: Failed to save business card data. Error: {e}")
            traceback.print_exc()
        
        raw_text_for_rag = ' '.join(str(v) for k, v in extracted_info.items() if v and k not in ['id', 'image_filename'])
        rag_core.add_document_to_knowledge_base(user_api_key, raw_text_for_rag, file_id, 'cards')
        
        return jsonify(extracted_info)
    except Exception as e:
        print(f"An error occurred in process_card endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Server processing failed'}), 500

@app.route('/process_brochure', methods=['POST'])
def process_brochure_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file, user_api_key, selected_model_key = request.files['file'], request.form.get('apiKey'), request.form.get('selectedModel')
    if not user_api_key or not selected_model_key: return jsonify({'error': 'API Key and Model are required'}), 400
    if selected_model_key not in MODEL_MAP: return jsonify({'error': 'Invalid model selected'}), 400

    try:
        pdf_bytes = file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        brochure_json_id = os.urandom(8).hex()
        pdf_filename = f"{brochure_json_id}.pdf"
        save_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        with open(save_path, 'wb') as f: f.write(pdf_bytes)

        extracted_data = {}
        full_text_from_pdf = "".join(page.get_text("text") for page in pdf_doc).strip()
        
        if len(full_text_from_pdf) > 100:
            print("'Text-First' successful. Using text model.")
            try:
                prompt = """Analyze the following text and structure it into a JSON object with keys "company_name", "contacts", and "raw_text". CRITICAL INSTRUCTIONS: 1. Extract the company name. 2. Extract ONLY contact information (names, emails, phone numbers) into the "contacts" array. 3. Extract ALL OTHER content into "raw_text". 4. DO NOT include contact details in raw_text. "contacts" should be a list of objects with "Owner Name", "Email", and "Number". DOCUMENT TEXT: --- {full_text_from_pdf} ---"""
                result = _call_openrouter_api_text_only_with_fallback(user_api_key, selected_model_key, prompt)
                if isinstance(result, str) and not result.startswith("All models failed"):
                    try: extracted_data = json.loads(result)
                    except json.JSONDecodeError: extracted_data = {}
                else: extracted_data = {}
            except Exception: extracted_data = {}
        
        if "error" in extracted_data or not extracted_data:
            print("Adaptive Vision: Attempting medium resolution (150 DPI)...")
            med_res_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=150).tobytes("png"))) for page in pdf_doc]
            extracted_data = _extract_brochure_data_with_vision(med_res_images, user_api_key, selected_model_key)
            is_poor_quality = "error" in extracted_data or (not extracted_data.get("contacts") and len(extracted_data.get("raw_text", "")) < 50)
            if is_poor_quality:
                print("Medium resolution failed. Retrying with high resolution (300 DPI)...")
                high_res_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for page in pdf_doc]
                extracted_data = _extract_brochure_data_with_vision(high_res_images, user_api_key, selected_model_key)

        if "error" in extracted_data: return jsonify(extracted_data), 500

        final_brochure_object = {
            "id": brochure_json_id,
            "company_name": extracted_data.get("company_name", "Unknown Company"),
            "contacts": extracted_data.get("contacts", []),
            "raw_text": extracted_data.get("raw_text", ""),
            "image_filename": pdf_filename
        }
        for contact in final_brochure_object["contacts"]: contact["id"] = os.urandom(8).hex()
        
        user_brochures = _load_user_data(user_api_key, 'brochures')
        user_brochures.insert(0, final_brochure_object)
        _save_user_data(user_api_key, 'brochures', user_brochures)
        
        try:
            user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()
            new_brochure = Brochure(
                json_id=brochure_json_id,
                company_name=final_brochure_object.get("company_name"),
                raw_text=final_brochure_object.get("raw_text"),
                source_document=file.filename,
                user_hash=user_hash
            )
            db.session.add(new_brochure)
            
            for contact_data in final_brochure_object.get("contacts", []):
                new_contact = Contact(
                    json_id=contact_data['id'],
                    owner_name=contact_data.get("Owner Name"),
                    email=contact_data.get("Email"),
                    phone_number=contact_data.get("Number"),
                    brochure=new_brochure 
                )
                db.session.add(new_contact)
            
            db.session.commit()
            print(f"Successfully saved brochure '{new_brochure.company_name}' and {len(new_brochure.contacts)} contacts to the database.")
        except Exception as e:
            db.session.rollback()
            print(f"DATABASE ERROR: Failed to save brochure data. Error: {e}")
            traceback.print_exc()
        
        print("Indexing separated and cleaned content for high-quality RAG...")
        contacts = final_brochure_object.get("contacts", [])
        if contacts:
            contact_text_parts = [f"Contact information for {final_brochure_object.get('company_name', 'this company')}:"]
            for contact in contacts:
                name, email, number = contact.get("Owner Name"), contact.get("Email"), contact.get("Number")
                contact_info = [f"Name: {name}"]
                if email: contact_info.append(f"Email: {email}")
                if number: contact_info.append(f"Phone: {number}")
                contact_text_parts.append("- " + ", ".join(contact_info))
            contacts_document_text = "\n".join(contact_text_parts)
            rag_core.add_document_to_knowledge_base(user_api_key, contacts_document_text, f"{brochure_json_id}_contacts", 'brochures')
        clean_info_text = _create_clean_info_text(final_brochure_object)
        if clean_info_text and clean_info_text.strip():
            rag_core.add_document_to_knowledge_base(user_api_key, clean_info_text, f"{brochure_json_id}_info", 'brochures')
        print("RAG indexing completed successfully!")
        
        return jsonify(final_brochure_object)
    except Exception as e:
        print(f"An error occurred in process_brochure endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server processing failed: {e}'}), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    user_api_key, query_text, mode, selected_model_key = data.get('apiKey'), data.get('query'), data.get('mode'), data.get('selectedModel')
    if not all([user_api_key, query_text, mode, selected_model_key]): return jsonify({'error': 'API key, query, mode, and model are required.'}), 400
    if selected_model_key not in MODEL_MAP: return jsonify({'error': 'Invalid model selected'}), 400
    try:
        session['api_key'] = user_api_key
        intent = 'synthesis' if "table" in query_text.lower() or "list all" in query_text.lower() else 'research'
        print(f"Intent detected: {intent}")
        if intent == 'synthesis':
            data_source = _load_user_data(user_api_key, mode)
            synthesis_data = []
            if mode == 'brochures':
                for brochure in data_source:
                    for contact in brochure.get('contacts', []):
                        synthesis_data.append({"Company Name": brochure.get("company_name"), "Owner Name": contact.get("Owner Name"), "Email": contact.get("Email"), "Number": contact.get("Number")})
            else:
                synthesis_data = data_source
            synthesis_prompt = f"As a data analyst, create a markdown table based on the user's request from the following JSON data.\nJSON: {json.dumps(synthesis_data, indent=2)}\nRequest: {query_text}\nAnswer:"
            answer = _call_openrouter_api_text_only_with_fallback(user_api_key, selected_model_key, synthesis_prompt)
        else:
            answer = rag_core.query_knowledge_base(user_api_key, query_text, mode, selected_model_key)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /chat endpoint: {e}"); traceback.print_exc()
        return jsonify({'error': 'An internal error occurred.'}), 500

@app.route('/load_data/<mode>', methods=['POST'])
def load_data_endpoint(mode):
    user_api_key = request.json.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    user_data = _load_user_data(user_api_key, mode)
    return jsonify(user_data)

@app.route('/update_card/<mode>/<item_id>', methods=['POST'])
def update_card_endpoint(mode, item_id):
    data = request.get_json()
    user_api_key, field, value, contact_id = data.get('apiKey'), data.get('field'), data.get('value'), data.get('contactId')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    
    # Step 1: Update JSON file (Existing Logic, Unchanged)
    user_data = _load_user_data(user_api_key, mode)
    item_found_in_json = False
    if mode == 'cards':
        for card in user_data:
            if card.get('id') == item_id:
                card[field] = value
                item_found_in_json = True
                break
    elif mode == 'brochures':
        for brochure in user_data:
            if brochure.get('id') == item_id and contact_id:
                for contact in brochure.get('contacts', []):
                    if contact.get('id') == contact_id:
                        contact[field] = value
                        item_found_in_json = True
                        break
                if item_found_in_json: break
    if item_found_in_json:
        _save_user_data(user_api_key, mode, user_data)
        
        # Step 1.5: Update ChromaDB (RAG knowledge base)
        try:
            if mode == 'cards':
                # Get the updated card data
                updated_card = next((c for c in user_data if c.get('id') == item_id), None)
                if updated_card:
                    # Remove old document and re-add with updated content
                    rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
                    raw_text = ' '.join(str(v) for k, v in updated_card.items() if v and k not in ['id', 'image_filename'])
                    rag_core.add_document_to_knowledge_base(user_api_key, raw_text, item_id, mode)
                    print(f"ChromaDB: Updated document {item_id} in {mode} knowledge base")
        except Exception as e:
            print(f"ChromaDB update warning: {e}")

    # ## FINAL DATABASE CODE ##
    # Step 2: Update Database (New Logic)
    try:
        user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()
        if mode == 'cards':
            db_card = BusinessCard.query.filter_by(json_id=item_id, user_hash=user_hash).first()
            if db_card:
                field_map = {"Owner Name": "owner_name", "Company Name": "company_name", "Email": "email", "Number": "phone_number", "Address": "address"}
                db_field = field_map.get(field)
                if db_field:
                    setattr(db_card, db_field, value)
                    db.session.commit()
                    print(f"Database updated for business card json_id: {item_id}")
                    return jsonify({"success": True})
        elif mode == 'brochures' and contact_id:
            db_contact = Contact.query.filter_by(json_id=contact_id).first()
            if db_contact and db_contact.brochure.user_hash == user_hash:
                field_map = {"Owner Name": "owner_name", "Email": "email", "Number": "phone_number"}
                db_field = field_map.get(field)
                if db_field:
                    setattr(db_contact, db_field, value)
                    db.session.commit()
                    print(f"Database updated for brochure contact json_id: {contact_id}")
                    return jsonify({"success": True})
        
        if not item_found_in_json:
            return jsonify({"success": False, "message": "Item not found in JSON"}), 404
        return jsonify({"success": True, "message": "JSON updated, but item not found in DB."})

    except Exception as e:
        db.session.rollback()
        print(f"DATABASE ERROR: Failed to update record. Error: {e}")
        return jsonify({"success": False, "message": "Database update failed."}), 500
    # ## END FINAL DATABASE CODE ##


@app.route('/delete_card/<mode>/<item_id>', methods=['DELETE'])
def delete_card_endpoint(mode, item_id):
    data = request.get_json()
    user_api_key, contact_id = data.get('apiKey'), data.get('contactId')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400

    # Step 1: Delete from JSON file (Existing Logic, Unchanged)
    user_data = _load_user_data(user_api_key, mode)
    item_found_in_json = False
    original_len = len(user_data)
    if mode == 'cards':
        user_data = [c for c in user_data if c.get('id') != item_id]
        if len(user_data) < original_len: item_found_in_json = True
    elif mode == 'brochures':
        if contact_id:
            for brochure in user_data:
                if brochure.get('id') == item_id:
                    original_contacts_len = len(brochure.get('contacts', []))
                    brochure['contacts'] = [c for c in brochure.get('contacts', []) if c.get('id') != contact_id]
                    if len(brochure.get('contacts', [])) < original_contacts_len:
                        item_found_in_json = True
                        break
        else: # Delete whole brochure
             user_data = [b for b in user_data if b.get('id') != item_id]
             if len(user_data) < original_len: item_found_in_json = True
    if item_found_in_json:
        _save_user_data(user_api_key, mode, user_data)
    
    # Step 1.5: Delete from ChromaDB (RAG knowledge base)
    try:
        if mode == 'cards' or (mode == 'brochures' and not contact_id):
            # Remove document vectors from ChromaDB
            rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
            print(f"ChromaDB: Removed document {item_id} from {mode} knowledge base")
    except Exception as e:
        print(f"ChromaDB removal warning: {e}")
    
    # ## FINAL DATABASE CODE ##
    # Step 2: Delete from Database (New Logic)
    try:
        user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()
        if mode == 'cards':
            db_card = BusinessCard.query.filter_by(json_id=item_id, user_hash=user_hash).first()
            if db_card:
                db.session.delete(db_card)
                db.session.commit()
                print(f"Database record deleted for business card json_id: {item_id}")
                return jsonify({"success": True})
        elif mode == 'brochures':
            if contact_id:
                db_contact = Contact.query.filter_by(json_id=contact_id).first()
                if db_contact and db_contact.brochure.user_hash == user_hash:
                    db.session.delete(db_contact)
                    db.session.commit()
                    print(f"Database record deleted for brochure contact json_id: {contact_id}")
                    return jsonify({"success": True})
            else: # Delete whole brochure
                db_brochure = Brochure.query.filter_by(json_id=item_id, user_hash=user_hash).first()
                if db_brochure:
                    db.session.delete(db_brochure) # Cascading delete will handle linked contacts
                    db.session.commit()
                    print(f"Database record deleted for brochure json_id: {item_id}")
                    return jsonify({"success": True})

        if not item_found_in_json:
            return jsonify({"success": False, "message": "Item not found in JSON"}), 404
        return jsonify({"success": True, "message": "JSON deleted, but item not found in DB."})

    except Exception as e:
        db.session.rollback()
        print(f"DATABASE ERROR: Failed to delete record. Error: {e}")
        return jsonify({"success": False, "message": "Database delete failed."}), 500
    # ## END FINAL DATABASE CODE ##

@app.route('/')
def serve_dashboard():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    rag_core.initialize_rag_system()
    
    # This creates all the new tables based on your models if they don't exist yet.
    with app.app_context():
        db.create_all()
        print("Database tables (business_card, brochure, contact) checked and created if necessary.")

    print("--- Server is starting! ---")
    print(f"User-specific data will be saved in '{os.path.abspath(DATA_FOLDER)}'")
    print("To use the dashboard, open your web browser and go to: http://127.0.0.1:5000")
    webbrowser.open_new('http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)