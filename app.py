# app.py

import os
import io
import json
import hashlib
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory, render_template, session
import webbrowser
from flask_cors import CORS
from PIL import Image
import fitz  # PyMuPDF
import rag_core
from datetime import timedelta
import traceback
import time

app = Flask(__name__)
CORS(app)

# Session configuration
app.secret_key = os.environ.get("SESSION_SECRET", "a-very-secret-key-for-sessions")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'user_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def _create_rag_text_for_brochure(brochure):
    """Combines structured contacts and raw text into a single string for RAG."""
    rag_text_parts = []
    company_name = brochure.get("company_name")
    if company_name:
        rag_text_parts.append(f"Information about the company: {company_name}.")

    contacts = brochure.get("contacts", [])
    if contacts:
        rag_text_parts.append("\nThis document contains the following contact details:")
        for contact in contacts:
            name = contact.get("Owner Name") or "Not specified"
            email = contact.get("Email") or "Not specified"
            number = contact.get("Number") or "Not specified"
            rag_text_parts.append(f"- Name: {name}. Email: {email}. Phone: {number}.")
    
    raw_text = brochure.get("raw_text")
    if raw_text:
        rag_text_parts.append("\n---\nAdditional information from the brochure is as follows:\n" + raw_text)

    return "\n".join(rag_text_parts)

def _get_user_data_filepath(user_api_key, mode):
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    return os.path.join(DATA_FOLDER, f'{user_hash}_{mode}_data.json')

def _load_user_data(user_api_key, mode):
    filepath = _get_user_data_filepath(user_api_key, mode)
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError):
        return []
    return []

def _save_user_data(user_api_key, mode, data):
    filepath = _get_user_data_filepath(user_api_key, mode)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except IOError as e:
        print(f"❌ Could not save data to {filepath}! Error: {e}")
        return False

# ## START: BULLETPROOF CLEANING FUNCTION ##
def _clean_and_validate_contacts(data):
    """
    Final, bulletproof cleaning function.
    Aggressively finds and replaces any placeholder text with null.
    """
    if not data or "contacts" not in data:
        return data

    cleaned_contacts = []
    
    def is_placeholder(value):
        """Checks if a string is a placeholder. Case-insensitive and robust."""
        if not isinstance(value, str):
            return True # Treat non-strings (like None, integers, etc.) as placeholders
        
        test_val = value.strip().lower()
        if not test_val: # Catches empty strings
            return True
            
        # Check against a list of exact, common placeholder values
        placeholders = ["n/a", "na", "none", "null"]
        if test_val in placeholders:
            return True
            
        # Check for placeholder phrases that might be returned
        if "not available" in test_val or "not specified" in test_val or "not applicable" in test_val:
            return True
            
        return False

    for contact in data.get("contacts", []):
        name = contact.get("Owner Name")
        email = contact.get("Email")
        number = contact.get("Number")

        # If the name itself is a placeholder, it's bad data, so we skip the contact.
        if is_placeholder(name):
            continue

        # Clean each field using the robust placeholder check
        cleaned_contact = {
            "Owner Name": name.strip(),
            "Email": None if is_placeholder(email) else email.strip(),
            "Number": None if is_placeholder(number) else number.strip()
        }
        cleaned_contacts.append(cleaned_contact)
            
    data["contacts"] = cleaned_contacts
    return data
# ## END: BULLETPROOF CLEANING FUNCTION ##


def extract_card_data(image_bytes, user_api_key):
    # This function remains unchanged as it works correctly.
    print("🤖 Processing business card with Gemini Vision API...")
    if not user_api_key: return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = """
        You are an expert at reading business cards. Analyze the image and extract information into a structured JSON format.
        The JSON object must use these exact keys: "Owner Name", "Company Name", "Email", "Number", "Address".
        If a piece of information is not present, its value must be `null`.
        Your entire response MUST be a single, valid JSON object.
        """
        response = model.generate_content([prompt, img])
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        parsed_info = json.loads(json_text)
        
        return {
            "Owner Name": parsed_info.get("Owner Name"),
            "Company Name": parsed_info.get("Company Name"),
            "Email": parsed_info.get("Email"),
            "Number": parsed_info.get("Number"),
            "Address": parsed_info.get("Address"),
        }
    except Exception as e:
        print(f"❌ Error during Gemini API call for business card: {e}")
        traceback.print_exc()
        return {"error": f"Failed to parse AI response: {e}"}

def extract_brochure_contacts_and_company(image_list, user_api_key):
    """Uses a high-accuracy prompt and the bulletproof code-based cleaner."""
    print("🤖 Brochure Extraction: Using high-accuracy prompt and bulletproof cleaner...")
    if not user_api_key: return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        # ## START: FINAL PROMPT WITH HIGH-ACCURACY INSTRUCTION ##
        prompt = """
        Analyze the document with maximum precision to extract all contact information.

        **CRITICAL INSTRUCTIONS:**
        1.  **CHARACTER ACCURACY:** Pay extremely close attention to the spelling of names (e.g., "Nerissa" not "Nelson", "Syn" not "Sydd"). Double-check every character before outputting.
        2.  **FIND EVERYONE:** Identify every single contact entry listed.
        3.  **NO MERGING:** If a person's name appears more than once (e.g., Sasa Hu), treat each instance as a separate and unique contact. Do not merge them.
        4.  **EXTRACT DETAILS:** For each entry, find the "Owner Name", "Email", and the complete "Number".

        **OUTPUT:**
        Return a single JSON object with a "company_name" and a "contacts" list.
        """
        # ## END: FINAL PROMPT WITH HIGH-ACCURACY INSTRUCTION ##

        response = model.generate_content([prompt] + image_list)
        json_text = response.text.strip().replace('```json', '').replace('```', '')

        raw_data = json.loads(json_text)
        
        print("✅ AI extraction complete. Applying bulletproof cleaning function...")
        cleaned_data = _clean_and_validate_contacts(raw_data)
        
        return cleaned_data

    except Exception as e:
        print(f"❌ Error during robust brochure extraction: {e}")
        traceback.print_exc()
        return {"error": f"Failed to parse contacts from brochure: {e}"}


def extract_brochure_raw_text(image_list, contacts_to_exclude, user_api_key):
    # This function remains unchanged.
    print("🤖 Brochure Step 2: Extracting remaining raw text...")
    if not user_api_key: return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        exclude_string = json.dumps(contacts_to_exclude, indent=2)

        prompt = f"""
        You are a text summarization expert. Analyze the provided brochure images.
        You have been given a list of specific contact details that have already been extracted. This list is:
        ---
        {exclude_string}
        ---
        Your task is to extract all the OTHER text from the brochure. This includes marketing copy, slogans, mission statements, general addresses, product descriptions, etc.
        Your response MUST NOT include any of the specific names, emails, or phone numbers from the list provided above.
        Return a single JSON object with one key: "raw_text".
        """
        
        response = model.generate_content([prompt] + image_list)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)
    except Exception as e:
        print(f"❌ Error during brochure raw text extraction: {e}")
        return {"error": f"Failed to parse raw text from brochure: {e}"}

@app.before_request
def make_session_permanent():
    session.permanent = True

@app.route('/process_card', methods=['POST'])
def process_card_endpoint():
    # This function remains unchanged.
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_api_key = request.form.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        extracted_info = extract_card_data(image_bytes, user_api_key)
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
        
        raw_text_for_rag = ' '.join(str(v) for k, v in extracted_info.items() if v and k not in ['id', 'image_filename'])
        rag_core.add_document_to_knowledge_base(user_api_key, raw_text_for_rag, file_id, 'cards')
        
        return jsonify(extracted_info)
    except Exception as e:
        print(f"An error occurred in process_card endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Server processing failed'}), 500

@app.route('/process_brochure', methods=['POST'])
def process_brochure_endpoint():
    # This function remains unchanged.
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_api_key = request.form.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400

    try:
        pdf_bytes = file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        brochure_id = os.urandom(8).hex()
        pdf_filename = f"{brochure_id}.pdf"
        save_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        with open(save_path, 'wb') as f: f.write(pdf_bytes)

        image_list = [Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for page in pdf_doc]
        
        if not image_list:
            return jsonify({'error': 'No images found in the PDF brochure.'}), 400
            
        contact_data = extract_brochure_contacts_and_company(image_list, user_api_key)
        if "error" in contact_data: return jsonify(contact_data), 500
        
        image_list_for_raw_text = [Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for page in pdf_doc]
        raw_text_data = extract_brochure_raw_text(image_list_for_raw_text, contact_data.get("contacts", []), user_api_key)
        if "error" in raw_text_data: return jsonify(raw_text_data), 500

        final_brochure_object = {
            "id": brochure_id,
            "company_name": contact_data.get("company_name", "Unknown Company"),
            "contacts": contact_data.get("contacts", []),
            "raw_text": raw_text_data.get("raw_text", ""),
            "image_filename": pdf_filename
        }

        for contact in final_brochure_object["contacts"]:
            contact["id"] = os.urandom(8).hex()
        
        user_brochures = _load_user_data(user_api_key, 'brochures')
        user_brochures.insert(0, final_brochure_object)
        _save_user_data(user_api_key, 'brochures', user_brochures)
        
        full_text_for_rag = _create_rag_text_for_brochure(final_brochure_object)
        rag_core.add_document_to_knowledge_base(user_api_key, full_text_for_rag, brochure_id, 'brochures')
        
        return jsonify(final_brochure_object)

    except Exception as e:
        print(f"An error occurred in process_brochure endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server processing failed: {e}'}), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    # This function remains unchanged.
    data = request.get_json()
    user_api_key = data.get('apiKey')
    query_text = data.get('query')
    mode = data.get('mode') 

    if not all([user_api_key, query_text, mode]):
        return jsonify({'error': 'API key, query, and mode are required.'}), 400

    try:
        session['api_key'] = user_api_key
        
        if "table" in query_text.lower() or "list all" in query_text.lower():
            intent = 'synthesis'
        else:
            intent = 'research'
        
        print(f"🤖 Intent detected: {intent}")

        if intent == 'synthesis':
            data_source = _load_user_data(user_api_key, mode)
            synthesis_data = []
            
            if mode == 'brochures':
                for brochure in data_source:
                    for contact in brochure.get('contacts', []):
                        synthesis_data.append({
                            "Company Name": brochure.get("company_name"),
                            "Owner Name": contact.get("Owner Name"),
                            "Email": contact.get("Email"),
                            "Number": contact.get("Number")
                        })
            else:
                synthesis_data = data_source

            synthesis_prompt = f"""
            You are a data analysis assistant. Based on the JSON data below, fulfill the user's request. Format tables as clean Markdown.
            JSON Data: {json.dumps(synthesis_data, indent=2)}
            User Request: {query_text}
            Answer:
            """
            
            genai.configure(api_key=user_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(synthesis_prompt)
            answer = response.text
        else:
            answer = rag_core.query_knowledge_base(user_api_key, query_text, mode)

        return jsonify({'answer': answer})
    except Exception as e:
        print(f"❌ Error in /chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An internal error occurred.'}), 500

@app.route('/load_data/<mode>', methods=['POST'])
def load_data_endpoint(mode):
    # This function remains unchanged.
    user_api_key = request.json.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if mode not in ['cards', 'brochures']: return jsonify({'error': 'Invalid mode specified'}), 400
    user_data = _load_user_data(user_api_key, mode)
    return jsonify(user_data)

@app.route('/update_card/<mode>/<item_id>', methods=['POST'])
def update_card_endpoint(mode, item_id):
    # This function remains unchanged.
    data = request.get_json()
    user_api_key = data.get('apiKey')
    field, value = data.get('field'), data.get('value')
    contact_id = data.get('contactId')

    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    
    user_data = _load_user_data(user_api_key, mode)
    item_found = False
    
    if mode == 'cards':
        for contact in user_data:
            if contact.get('id') == item_id:
                contact[field] = value
                _save_user_data(user_api_key, mode, user_data)
                try:
                    rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
                    new_text = ' '.join(str(v) for k, v in contact.items() if v and k not in ['id', 'image_filename'])
                    rag_core.add_document_to_knowledge_base(user_api_key, new_text, item_id, mode)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to update document {item_id} in RAG index. Error: {e}")
                return jsonify({"success": True})
    
    elif mode == 'brochures':
        for brochure in user_data:
            if brochure.get('id') == item_id:
                if contact_id:
                    for contact in brochure.get('contacts', []):
                        if contact.get('id') == contact_id:
                            contact[field] = value
                            item_found = True
                            break
                if item_found:
                    _save_user_data(user_api_key, mode, user_data)
                    try:
                        full_text_for_rag = _create_rag_text_for_brochure(brochure)
                        rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
                        rag_core.add_document_to_knowledge_base(user_api_key, full_text_for_rag, item_id, mode)
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to update document {item_id} in RAG index. Error: {e}")
                    return jsonify({"success": True})

    return jsonify({"success": False, "message": "Item not found"}), 404

@app.route('/delete_card/<mode>/<item_id>', methods=['DELETE'])
def delete_card_endpoint(mode, item_id):
    # This function remains unchanged.
    data = request.get_json()
    user_api_key = data.get('apiKey')
    contact_id = data.get('contactId')

    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400

    user_data = _load_user_data(user_api_key, mode)
    
    if mode == 'cards':
        original_len = len(user_data)
        user_data = [c for c in user_data if c.get('id') != item_id]
        if len(user_data) < original_len:
            _save_user_data(user_api_key, mode, user_data)
            try:
                rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
            except Exception as e:
                print(f"⚠️ Warning: Failed to remove document {item_id} from RAG index. Error: {e}")
            return jsonify({"success": True})

    elif mode == 'brochures':
        if contact_id:
            for brochure in user_data:
                if brochure.get('id') == item_id:
                    original_len = len(brochure.get('contacts', []))
                    brochure['contacts'] = [c for c in brochure.get('contacts', []) if c.get('id') != contact_id]
                    if len(brochure.get('contacts', [])) < original_len:
                        _save_user_data(user_api_key, mode, user_data)
                        return jsonify({"success": True})
        else:
            original_len = len(user_data)
            user_data = [b for b in user_data if b.get('id') != item_id]
            if len(user_data) < original_len:
                _save_user_data(user_api_key, mode, user_data)
                try:
                    rag_core.remove_document_from_knowledge_base(user_api_key, item_id, mode)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to remove document {item_id} from RAG index. Error: {e}")
                return jsonify({"success": True})

    return jsonify({"success": False, "message": "Item not found"}), 404

@app.route('/')
def serve_dashboard():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    rag_core.initialize_rag_system()
    print("--- Server is starting! ---")
    print(f"User-specific data will be saved in '{os.path.abspath(DATA_FOLDER)}'")
    print("To use the dashboard, open your web browser and go to: http://127.0.0.1:5000")
    webbrowser.open_new('http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)