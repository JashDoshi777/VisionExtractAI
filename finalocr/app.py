# app.py

import os
import io
import json
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import rag_core # Import our RAG system

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
DATA_FILE = 'data.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

contacts_data = []

def load_data_from_disk():
    """Loads contact data from the JSON file into memory when the server starts."""
    global contacts_data
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                contacts_data = json.load(f)
                print(f"✅ Loaded {len(contacts_data)} contacts from {DATA_FILE}")
    except (IOError, json.JSONDecodeError) as e:
        print(f"⚠️ Could not load data file. Starting fresh. Error: {e}")
        contacts_data = []

def save_data_to_disk():
    """Saves the current contact data from memory to the JSON file."""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(contacts_data, f, indent=4)
    except IOError as e:
        print(f"❌ Could not save data to disk! Error: {e}")

def extract_data_with_vision(image_bytes, user_api_key, extract_raw_text=False):
    """
    Extracts data from an image using Gemini. Can extract structured JSON or raw text.
    """
    print(f"🤖 Processing image with Gemini Vision API (Mode: {'Raw Text' if extract_raw_text else 'JSON'})...")
    if not user_api_key or "YOUR_GOOGLE_AI_API_KEY" in user_api_key:
        return {"error": "A valid Google AI API Key was not provided."}
    
    try:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return {"error": f"Invalid API Key or configuration error: {e}"}

    try:
        img = Image.open(io.BytesIO(image_bytes))

        if extract_raw_text:
            prompt_instructions = "Extract all visible text from this image. Transcribe it exactly as you see it."
            response = model.generate_content([prompt_instructions, img])
            print("✅ Gemini Vision raw text extraction complete.")
            return {"raw_text": response.text}
        else:
            prompt_instructions = """
            You are a world-class expert at reading business cards from images. Your task is to analyze the provided image and extract the key information into a structured JSON format.
            CRITICAL INSTRUCTIONS:
            1.  Your entire response MUST be a single, valid JSON object. Do not include any text, explanations, or markdown like ```json before or after the JSON.
            2.  The JSON object must use these exact keys: "Owner Name", "Company Name", "Email", "Number", "Address".
            3.  If a piece of information is not present in the image, its value in the JSON must be `null`. Do not invent or guess information.
            4.  Combine multiple phone numbers or address lines into a single string, separated by ", ".
            5.  Clean the extracted text. Ensure phone numbers are clear and company names are properly formatted.
            """
            response = model.generate_content([prompt_instructions, img])
            json_text = response.text.strip()
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            parsed_info = json.loads(json_text)
            print("✅ Gemini Vision structured JSON extraction complete.")
            return {
                "Owner Name": parsed_info.get("Owner Name"),
                "Company Name": parsed_info.get("Company Name"),
                "Email": parsed_info.get("Email"),
                "Number": parsed_info.get("Number"),
                "Address": parsed_info.get("Address"),
            }
    except Exception as e:
        print(f"❌ Error during Gemini API call or JSON parsing: {e}")
        return {"error": f"Failed to parse AI response: {e}"}

@app.route('/process_card', methods=['POST'])
def process_card_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_api_key = request.form.get('apiKey')
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        extracted_info = extract_data_with_vision(image_bytes, user_api_key, extract_raw_text=False)
        if "error" in extracted_info: return jsonify(extracted_info), 500

        file_id = os.urandom(8).hex()
        _, f_ext = os.path.splitext(file.filename)
        safe_ext = f_ext if f_ext.lower() in ['.png', '.jpg', '.jpeg', '.webp'] else '.png'
        image_filename = f"{file_id}{safe_ext}"
        save_path = os.path.join(UPLOAD_FOLDER, image_filename)
        with open(save_path, 'wb') as f: f.write(image_bytes)
        print(f"✅ Image saved to: {save_path}")

        extracted_info['id'] = file_id
        extracted_info['image_filename'] = image_filename
        
        contacts_data.insert(0, extracted_info)
        save_data_to_disk()
        
        print("🧠 Starting RAG processing for the uploaded file...")
        raw_text_result = extract_data_with_vision(image_bytes, user_api_key, extract_raw_text=True)
        if "error" not in raw_text_result:
            document_text = raw_text_result.get("raw_text")
            rag_core.add_document_to_knowledge_base(user_api_key, document_text, file_id)
        else:
            print(f"⚠️ Could not extract raw text for RAG: {raw_text_result.get('error')}")

        print(f"--- Final Parsed Info ---\n{json.dumps(extracted_info, indent=2)}\n-------------------")
        return jsonify(extracted_info)
        
    except Exception as e:
        print(f"An error occurred in the endpoint: {e}")
        return jsonify({'error': 'Server processing failed'}), 500

# --- UPDATED CHAT ENDPOINT WITH "TWO-MODE BRAIN" ---
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    user_api_key = data.get('apiKey')
    query_text = data.get('query')

    if not user_api_key or not query_text:
        return jsonify({'error': 'API key and query are required.'}), 400

    try:
        # Step 1: Use Gemini to classify the user's intent.
        genai.configure(api_key=user_api_key)
        intent_model = genai.GenerativeModel('gemini-1.5-flash')
        
        intent_prompt = f"""
        Analyze the user's query and classify its intent. Respond with only a single word: 'research' or 'synthesis'.
        
        - 'research': The user is asking a specific question that can likely be answered by looking at a piece of a document.
          Examples: "What are the opening hours?", "What is the phone number for ACME Corp?", "What services does the brochure mention?"

        - 'synthesis': The user wants you to gather, combine, or reformat data from multiple sources. This often involves words like "list", "summarize", "table", "all contacts", "every email".
          Examples: "Give me a table of all contacts", "List all the email addresses you have", "Summarize the companies in my documents"
          
        User Query: "{query_text}"
        Intent:
        """
        
        intent_response = intent_model.generate_content(intent_prompt)
        intent = intent_response.text.strip().lower()
        print(f"🤖 Intent detected: {intent}")

        # Step 2: Choose the right "brain" based on the intent.
        if 'synthesis' in intent:
            # --- The "Analyst" Brain: Use the structured JSON data ---
            print("🧠 Using Analyst Brain (Structured Data)...")
            
            # Use the in-memory 'contacts_data' which is loaded from data.json
            if not contacts_data:
                return jsonify({'answer': "There is no contact data to synthesize. Please upload some business cards first."})

            synthesis_prompt = f"""
            You are a data analysis assistant. Based on the following JSON list of contacts, please fulfill the user's request.
            If the user asks for a table, format your response as a clean Markdown table.

            JSON Data:
            {json.dumps(contacts_data, indent=2)}

            User Request:
            {query_text}

            Answer:
            """
            
            synthesis_model = genai.GenerativeModel('gemini-1.5-flash')
            answer_response = synthesis_model.generate_content(synthesis_prompt)
            answer = answer_response.text
            
        else: # Default to 'research'
            # --- The "Researcher" Brain: Use the RAG system (as before) ---
            print("🧠 Using Researcher Brain (RAG)...")
            answer = rag_core.query_knowledge_base(user_api_key, query_text)

        return jsonify({'answer': answer})

    except Exception as e:
        print(f"❌ Error in /chat endpoint: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500


@app.route('/load_data', methods=['GET'])
def load_data_endpoint():
    """Endpoint for the frontend to fetch all contacts on page load."""
    return jsonify(contacts_data)

@app.route('/update_card/<card_id>', methods=['POST'])
def update_card_endpoint(card_id):
    """Endpoint to handle saving edits to a card."""
    update_data = request.get_json()
    field = update_data.get('field')
    value = update_data.get('value')
    
    for contact in contacts_data:
        if contact.get('id') == card_id:
            contact[field] = value
            save_data_to_disk()
            print(f"✅ Updated card {card_id}: set {field} to '{value}'")
            return jsonify({"success": True, "message": "Card updated"})
            
    return jsonify({"success": False, "message": "Card not found"}), 404

@app.route('/delete_card/<card_id>', methods=['DELETE'])
def delete_card_endpoint(card_id):
    """Endpoint to handle deleting a card."""
    global contacts_data
    original_len = len(contacts_data)
    contacts_data = [c for c in contacts_data if c.get('id') != card_id]
    
    if len(contacts_data) < original_len:
        save_data_to_disk()
        print(f"✅ Deleted card {card_id}")
        return jsonify({"success": True, "message": "Card deleted"})
        
    return jsonify({"success": False, "message": "Card not found"}), 404

@app.route('/')
def serve_dashboard():
    return send_from_directory('.', 'index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    load_data_from_disk()
    rag_core.initialize_rag_system()
    print("--- Server is starting! ---")
    print(f"Data will be saved to '{os.path.abspath(DATA_FILE)}'")
    print("To use the dashboard, open your web browser and go to: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)