import os
os.environ['FAISS_NO_AVX2'] = '1'
import resource
resource.setrlimit(resource.RLIMIT_AS, (300 * 1024 * 1024, 300 * 1024 * 1024))

import io
import json
import hashlib
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from PIL import Image
import fitz  # PyMuPDF

app = Flask(__name__)
CORS(app)

# Initialize RAG system (lazy-loaded)
def get_rag_core():
    from rag_core import initialize_rag_system
    initialize_rag_system()
    import rag_core
    return rag_core

# --- Supabase Configuration ---
SUPABASE_URL = "https://nwcyfrvkfozlzwjimhmb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im53Y3lmcnZrZm96bHp3amltaG1iIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODE0NDAsImV4cCI6MjA3MDY1NzQ0MH0.51FFi8Tk51weqnUTC5fvKLldBWcNP_eYAzJzo6sDt88"
supabase = None  # Will be initialized on first use

def get_supabase():
    global supabase
    if supabase is None:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

# --- Helper Functions ---
def _get_user_data_filename(user_api_key, mode):
    user_hash = hashlib.sha256(user_api_key.encode()).hexdigest()[:16]
    return f'{user_hash}_{mode}_data.json'

def _load_user_data(user_api_key, mode):
    filename = _get_user_data_filename(user_api_key, mode)
    try:
        response = get_supabase().storage.from_("user_data").download(filename)
        return json.loads(response.decode('utf-8'))
    except Exception as e:
        print(f"ℹ️ Could not load data: {e}")
        return []

def _save_user_data(user_api_key, mode, data):
    filename = _get_user_data_filename(user_api_key, mode)
    try:
        get_supabase().storage.from_("user_data").upload(
            file=json.dumps(data).encode('utf-8'),
            path=filename,
            file_options={"content-type": "application/json", "upsert": "true"}
        )
        return True
    except Exception as e:
        print(f"❌ Failed to save data: {e}")
        return False

# --- Business Card Processing ---
def extract_card_data(image_bytes, user_api_key):
    print("🤖 Processing business card...")
    if not user_api_key: 
        return {"error": "API Key required"}

    try:
        import google.generativeai as genai
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash',
                                   generation_config={
                                       "max_output_tokens": 1000,
                                       "temperature": 0
                                   })
        img = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content([
            "Extract business card details as JSON with keys: Owner Name, Company Name, Email, Number, Address",
            img
        ])
        return json.loads(response.text.replace('```json', '').replace('```', ''))
    except Exception as e:
        print(f"❌ Card processing failed: {e}")
        return {"error": str(e)}

def extract_brochure_contacts_and_company(image_list, user_api_key):
    print("🤖 Brochure Step 1: Extracting contacts and company...")
    if not user_api_key: return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash',
                                    generation_config={
                                        "max_output_tokens": 1000,
                                        "temperature": 0
                                    })
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        prompt = """
        You are a world-class data extraction expert. Your ONLY task is to find two pieces of information from the provided brochure images:
        1. The primary **Company Name**.
        2. A list of **every individual person** mentioned. For each person, you MUST extract their "Owner Name", "Email", and "Number". It is CRITICAL that you find the phone number if it exists.

        Return ONLY a single JSON object with two keys: "company_name" and "contacts". The "contacts" key must be a LIST of JSON objects.
        If any field for a person is not found, use `null`. If no people are found, return an empty list for "contacts".
        Do NOT include any other information or text.
        """
        
        response = model.generate_content([prompt] + image_list)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)
    except Exception as e:
        print(f"❌ Error during brochure contact extraction: {e}")
        return {"error": f"Failed to parse contacts from brochure: {e}"}

def extract_brochure_raw_text(image_list, contacts_to_exclude, user_api_key):
    print("🤖 Brochure Step 2: Extracting remaining raw text...")
    if not user_api_key: return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash',
                                    generation_config={
                                        "max_output_tokens": 1000,
                                        "temperature": 0
                                    })
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

@app.route('/process_card', methods=['POST'])
def process_card_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_api_key = request.form.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    
    # File size check (2MB max for free tier)
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    if file_size > 2 * 1024 * 1024:
        return jsonify({'error': 'File too large (max 2MB)'}), 400

    try:
        image_bytes = file.read()
        extracted_info = extract_card_data(image_bytes, user_api_key)
        if "error" in extracted_info: return jsonify(extracted_info), 500

        file_id = os.urandom(8).hex()
        _, f_ext = os.path.splitext(file.filename)
        safe_ext = f_ext if f_ext.lower() in ['.png', '.jpg', '.jpeg', '.webp'] else '.png'
        image_filename = f"{file_id}{safe_ext}"
        
        supabase.storage.from_("uploads").upload(file=image_bytes, path=image_filename, file_options={"content-type": file.mimetype})
        
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
        return jsonify({'error': 'Server processing failed'}), 500

@app.route('/process_brochure', methods=['POST'])
def process_brochure_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_api_key = request.form.get('apiKey')
    if not user_api_key: return jsonify({'error': 'API Key is required'}), 400
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400

    # File size check (2MB max for free tier)
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    if file_size > 2 * 1024 * 1024:
        return jsonify({'error': 'File too large (max 2MB)'}), 400

    try:
        pdf_bytes = file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        brochure_id = os.urandom(8).hex()
        pdf_filename = f"{brochure_id}.pdf"
        
        supabase.storage.from_("uploads").upload(file=pdf_bytes, path=pdf_filename, file_options={"content-type": "application/pdf"})

        image_list = [Image.open(io.BytesIO(page.get_pixmap().tobytes("png"))) for page in pdf_doc]
        
        if not image_list:
            return jsonify({'error': 'No images found in the PDF brochure.'}), 400
            
        contact_data = extract_brochure_contacts_and_company(image_list, user_api_key)
        if "error" in contact_data: return jsonify(contact_data), 500
        
        raw_text_data = extract_brochure_raw_text(image_list, contact_data.get("contacts", []), user_api_key)
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
        
        raw_text_for_rag = final_brochure_object.get("raw_text", "")
        contacts_for_rag = final_brochure_object.get("contacts", [])
        if contacts_for_rag:
            raw_text_for_rag += "\n\n--- Extracted Contacts ---\n"
            for contact in contacts_for_rag:
                name = contact.get("Owner Name", "N/A")
                email = contact.get("Email", "N/A")
                number = contact.get("Number", "N/A")
                raw_text_for_rag += f"Name: {name}, Email: {email}, Number: {number}\n"
        
        rag_core.add_document_to_knowledge_base(user_api_key, raw_text_for_rag, brochure_id, 'brochures')
        
        return jsonify(final_brochure_object)
    except Exception as e:
        print(f"An error occurred in process_brochure endpoint: {e}")
        return jsonify({'error': f'Server processing failed: {e}'}), 500

# [Rest of your existing endpoints remain exactly the same...]
# Including: /chat, /load_data, /update_card, /delete_card, etc.

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)





