import os
import json
import base64
import time
from io import BytesIO

# Third-party libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# --- Configuration ---
# WARNING: Keep your API Key secret. Do not share this file publicly.
API_KEY = "AIzaSyBfJhhyUp5cwn8jWdCCt_sbpHnDqwJWW70" 
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

app = Flask(__name__)
# Enable CORS so your frontend can communicate with this backend
CORS(app) 

# --- Gemini API Structured Output Schema ---
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "match_percentage": {"type": "STRING", "description": "The overall match score as a percentage string, e.g., '75%'."},
        "strengths": {"type": "STRING", "description": "A detailed, short paragraph summarizing the key strengths of the resume relative to the JD."},
        "improvement_suggestions": {"type": "STRING", "description": "A detailed, short paragraph providing specific suggestions for improving the resume's match."},
        "matching_skills": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "A list of relevant technical and soft skills found in the resume that match the JD."},
        "missing_skills": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "A list of critical skills mentioned in the JD that are missing or weakly represented in the resume."},
    },
    "required": ["match_percentage", "strengths", "improvement_suggestions", "matching_skills", "missing_skills"]
}

# --- Utility Functions ---

def file_to_base64_part(file_stream, mime_type):
    """Encodes a file stream to a base64 string for the Gemini API payload."""
    file_bytes = file_stream.read()
    encoded_string = base64.b64encode(file_bytes).decode('utf-8')
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": encoded_string
        }
    }

def call_gemini_api_with_backoff(payload, max_retries=5):
    """Calls the Gemini API with exponential backoff for handling transient errors."""
    for attempt in range(max_retries):
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() 

            result = response.json()
            if 'error' in result:
                error_message = result.get('error', {}).get('message', 'Unknown API Error')
                raise Exception(f"Gemini API error: {error_message}")
            
            return result

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API call failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Gemini API call failed after {max_retries} attempts: {e}")

# --- API Routes ---

@app.route('/', methods=['GET'])
def home():
    """Health check route to verify the server is running."""
    return jsonify({
        "status": "online",
        "message": "ATS Analysis Backend is running successfully.",
        "endpoints": {
            "analyze": "/analyze [POST]"
        }
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        # 1. Input Parsing and Validation
        jd = request.form.get('job_description')
        resume_file = request.files.get('resume_pdf')

        if not jd or not resume_file:
            return jsonify({"error": "Missing job description or resume file."}), 400

        if resume_file.mimetype != 'application/pdf':
            return jsonify({"error": "Only PDF files are supported."}), 400

        # 2. Prepare API Request Parts
        resume_part = file_to_base64_part(resume_file.stream, 'application/pdf')
        
        system_instruction = (
            "You are an expert ATS (Applicant Tracking System) and Career Coach. "
            "Analyze the provided resume (PDF) against the job description (text). "
            "Provide an extremely accurate and strict assessment. "
            "Return the feedback strictly in the requested JSON format."
        )

        user_prompt = (
            f"Analyze the resume against the following Job Description (JD) and provide a structured ATS report. "
            f"JD: {jd}"
        )

        # 3. Construct the Payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt},
                        resume_part
                    ]
                }
            ],
            "generationConfig": {  # Note: generationConfig is the standard field name
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA
            },
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            }
        }

        # 4. Call the Gemini API
        api_result = call_gemini_api_with_backoff(payload)

        # 5. Extract and Validate Result
        try:
            json_text = api_result['candidates'][0]['content']['parts'][0]['text']
            report_data = json.loads(json_text)
            
            if all(key in report_data for key in RESPONSE_SCHEMA['required']):
                return jsonify(report_data), 200
            else:
                return jsonify({"error": "AI response was missing required fields."}), 500

        except (json.JSONDecodeError, KeyError) as e:
            return jsonify({"error": "Failed to parse AI response."}), 500

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on localhost:5000
    app.run(host='127.0.0.1', port=5000, debug=True)