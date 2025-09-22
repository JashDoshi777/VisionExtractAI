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
import re

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

def _extract_contact_info_from_text(text):
    """
    Extract contact information patterns from text and return clean text without contacts.
    """
    if not text:
        return "", []
    
    # Patterns to identify contact information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(?:\+?\d{1,4}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    
    # Find all emails and phones
    emails = re.findall(email_pattern, text, re.IGNORECASE)
    phones = re.findall(phone_pattern, text)
    
    # Remove obvious contact information from text
    clean_text = text
    
    # Remove email addresses
    clean_text = re.sub(email_pattern, '', clean_text, flags=re.IGNORECASE)
    
    # Remove phone numbers (more conservative approach)
    for phone in phones:
        if len(phone.replace('-', '').replace('.', '').replace(' ', '').replace('(', '').replace(')', '').replace('+', '')) >= 7:
            clean_text = clean_text.replace(phone, '')
    
    # Clean up extra spaces and line breaks
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text)
    
    return clean_text, emails + phones

def _create_clean_info_text(brochure_data):
    """
    Create clean info text that excludes contact details for RAG indexing.
    """
    company_name = brochure_data.get("company_name", "")
    raw_text = brochure_data.get("raw_text", "")
    
    # Start with company name if available
    info_parts = []
    if company_name and company_name != "Unknown Company":
        info_parts.append(f"Company: {company_name}")
    
    if raw_text:
        # Clean contact information from raw text
        clean_text, extracted_contacts = _extract_contact_info_from_text(raw_text)
        
        # Further clean common contact-related phrases
        contact_phrases = [
            r'contact\s+us\s*:?',
            r'for\s+more\s+information\s*:?',
            r'reach\s+out\s+to\s*:?',
            r'get\s+in\s+touch\s*:?',
            r'phone\s*:',
            r'email\s*:',
            r'tel\s*:',
            r'mobile\s*:',
            r'call\s+us\s*:?',
            r'write\s+to\s+us\s*:?',
        ]
        
        for phrase in contact_phrases:
            clean_text = re.sub(phrase, '', clean_text, flags=re.IGNORECASE)
        
        # Clean up formatting
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        clean_text = re.sub(r'\n\s*\n', '\n', clean_text)
        
        if clean_text:
            info_parts.append(clean_text)
    
    return "\n".join(info_parts) if info_parts else ""

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
        print(f"Could not save data to {filepath}! Error: {e}")
        return False

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
            return True
        
        test_val = value.strip().lower()
        if not test_val:
            return True
            
        placeholders = ["n/a", "na", "none", "null"]
        if test_val in placeholders:
            return True
            
        if "not available" in test_val or "not specified" in test_val or "not applicable" in test_val:
            return True
            
        return False

    for contact in data.get("contacts", []):
        name = contact.get("Owner Name")
        email = contact.get("Email")
        number = contact.get("Number")

        if is_placeholder(name):
            continue

        cleaned_contact = {
            "Owner Name": name.strip(),
            "Email": None if is_placeholder(email) else email.strip(),
            "Number": None if is_placeholder(number) else number.strip()
        }
        cleaned_contacts.append(cleaned_contact)
            
    data["contacts"] = cleaned_contacts
    return data

def extract_card_data(image_bytes, user_api_key):
    print("Processing business card with Gemini Vision API...")
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
        print(f"Error during Gemini API call for business card: {e}")
        traceback.print_exc()
        return {"error": f"Failed to parse AI response: {e}"}

def _extract_brochure_data_with_vision(image_list, user_api_key):
    """
    Uses a single, powerful prompt to extract all brochure data from images.
    """
    print(f"Vision Extraction: Analyzing {len(image_list)} images...")
    if not user_api_key:
        return {"error": "A valid Google AI API Key was not provided."}

    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        prompt = """
        You are a world-class document analysis expert. Analyze the provided document images with maximum precision.
        
        CRITICAL INSTRUCTIONS:
        1. Extract the company name
        2. Extract ONLY contact information (names, emails, phone numbers) and put them in the "contacts" array
        3. Extract ALL OTHER content (company description, services, mission, addresses, general information) as "raw_text"
        4. DO NOT include contact details like names, emails, or phone numbers in the raw_text
        5. Focus on separating contact information from general company information
        
        OUTPUT FORMAT:
        Return a SINGLE, valid JSON object with these exact keys: "company_name", "contacts", "raw_text".
        The "contacts" key must contain a list of objects, each with "Owner Name", "Email", and "Number".
        The "raw_text" should contain business information, services, descriptions, but NO contact details.
        """
        response = model.generate_content([prompt] + image_list)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        raw_data = json.loads(json_text)
        
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

        extracted_data = {}
        
        full_text_from_pdf = "".join(page.get_text("text") for page in pdf_doc).strip()
        
        if len(full_text_from_pdf) > 100:
            print("'Text-First' successful. Using text model.")
            try:
                genai.configure(api_key=user_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                Analyze the following text and structure it into a JSON object with keys "company_name", "contacts", and "raw_text".
                
                CRITICAL INSTRUCTIONS:
                1. Extract the company name
                2. Extract ONLY contact information (names, emails, phone numbers) and put them in the "contacts" array
                3. Extract ALL OTHER content (company description, services, mission, addresses, general information) as "raw_text"
                4. DO NOT include contact details like names, emails, or phone numbers in the raw_text
                5. Focus on separating contact information from general company information
                
                "contacts" should be a list of objects with "Owner Name", "Email", and "Number".
                
                DOCUMENT TEXT:
                ---
                {full_text_from_pdf}
                ---
                """
                response = model.generate_content(prompt)
                json_text = response.text.strip().replace('```json', '').replace('```', '')
                extracted_data = json.loads(json_text)
            except Exception as e:
                print(f"Text-based extraction failed: {e}. Falling back to vision.")
                extracted_data = {}
        else:
            print("'Text-First' found no significant text. Switching to 'Adaptive Vision'.")

        if not extracted_data:
            print("Adaptive Vision: Attempting medium resolution (150 DPI)...")
            med_res_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=150).tobytes("png"))) for page in pdf_doc]
            extracted_data = _extract_brochure_data_with_vision(med_res_images, user_api_key)
            
            is_poor_quality = "error" in extracted_data or \
                              (not extracted_data.get("contacts") and len(extracted_data.get("raw_text", "")) < 50)
            
            if is_poor_quality:
                print("Medium resolution failed. Retrying with high resolution (300 DPI)...")
                high_res_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for page in pdf_doc]
                extracted_data = _extract_brochure_data_with_vision(high_res_images, user_api_key)

        if "error" in extracted_data:
            return jsonify(extracted_data), 500

        final_brochure_object = {
            "id": brochure_id,
            "company_name": extracted_data.get("company_name", "Unknown Company"),
            "contacts": extracted_data.get("contacts", []),
            "raw_text": extracted_data.get("raw_text", ""),
            "image_filename": pdf_filename
        }

        for contact in final_brochure_object["contacts"]:
            contact["id"] = os.urandom(8).hex()
        
        user_brochures = _load_user_data(user_api_key, 'brochures')
        user_brochures.insert(0, final_brochure_object)
        _save_user_data(user_api_key, 'brochures', user_brochures)
        
        # IMPROVED RAG INDEXING - SEPARATE AND CLEAN CONTENT
        print("Indexing separated and cleaned content for high-quality RAG...")
        
        # Document 1: Clean contacts-only document
        contacts = final_brochure_object.get("contacts", [])
        if contacts:
            contact_text_parts = [f"Contact information for {final_brochure_object.get('company_name', 'this company')}:"]
            for contact in contacts:
                name = contact.get("Owner Name")
                email = contact.get("Email") 
                number = contact.get("Number")
                
                contact_info = []
                if name: contact_info.append(f"Name: {name}")
                if email: contact_info.append(f"Email: {email}")
                if number: contact_info.append(f"Phone: {number}")
                
                if contact_info:
                    contact_text_parts.append("- " + ", ".join(contact_info))
            
            contacts_document_text = "\n".join(contact_text_parts)
            rag_core.add_document_to_knowledge_base(
                user_api_key, 
                contacts_document_text, 
                f"{brochure_id}_contacts",
                'brochures'
            )

        # Document 2: Clean company information WITHOUT contact details
        clean_info_text = _create_clean_info_text(final_brochure_object)
        if clean_info_text and clean_info_text.strip():
            rag_core.add_document_to_knowledge_base(
                user_api_key, 
                clean_info_text, 
                f"{brochure_id}_info",
                'brochures'
            )
        
        print("RAG indexing completed successfully!")
        return jsonify(final_brochure_object)

    except Exception as e:
        print(f"An error occurred in process_brochure endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server processing failed: {e}'}), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
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
        
        print(f"Intent detected: {intent}")

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
        print(f"Error in /chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An internal error occurred.'}), 500

@app.route('/load_data/<mode>', methods=['POST'])
def load_data_endpoint(mode):
    user_api_key = request.json.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if mode not in ['cards', 'brochures']: return jsonify({'error': 'Invalid mode specified'}), 400
    user_data = _load_user_data(user_api_key, mode)
    return jsonify(user_data)

@app.route('/update_card/<mode>/<item_id>', methods=['POST'])
def update_card_endpoint(mode, item_id):
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
                    print(f"Warning: Failed to update document {item_id} in RAG index. Error: {e}")
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
                        # Re-index both parts with updated data
                        contacts = brochure.get("contacts", [])
                        if contacts:
                            contact_text_parts = [f"Contact information for {brochure.get('company_name', 'this company')}:"]
                            for contact in contacts:
                                name = contact.get("Owner Name")
                                email = contact.get("Email") 
                                number = contact.get("Number")
                                
                                contact_info = []
                                if name: contact_info.append(f"Name: {name}")
                                if email: contact_info.append(f"Email: {email}")
                                if number: contact_info.append(f"Phone: {number}")
                                
                                if contact_info:
                                    contact_text_parts.append("- " + ", ".join(contact_info))
                            
                            contacts_document_text = "\n".join(contact_text_parts)
                            rag_core.remove_document_from_knowledge_base(user_api_key, f"{item_id}_contacts", mode)
                            rag_core.add_document_to_knowledge_base(user_api_key, contacts_document_text, f"{item_id}_contacts", mode)
                    except Exception as e:
                        print(f"Warning: Failed to update document {item_id} in RAG index. Error: {e}")
                    return jsonify({"success": True})

    return jsonify({"success": False, "message": "Item not found"}), 404

@app.route('/delete_card/<mode>/<item_id>', methods=['DELETE'])
def delete_card_endpoint(mode, item_id):
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
                print(f"Warning: Failed to remove document {item_id} from RAG index. Error: {e}")
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
                    rag_core.remove_document_from_knowledge_base(user_api_key, f"{item_id}_info", mode)
                    rag_core.remove_document_from_knowledge_base(user_api_key, f"{item_id}_contacts", mode)
                except Exception as e:
                    print(f"Warning: Failed to remove document parts for {item_id} from RAG index. Error: {e}")
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