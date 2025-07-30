import os
import asyncio
import socketio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables from .env file
load_dotenv()

# Get Gemini API Key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")

# Initialize Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    raise

# Choose the Gemini model for evaluation.
GEMINI_MODEL_NAME = "gemini-1.5-flash"
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
print(f"Using Gemini model: {GEMINI_MODEL_NAME}")

# --- FastAPI and Socket.IO Setup ---
app = FastAPI()

# Define the allowed origin for client connections to prevent CORS issues.
# This should match the URL where your client.html is being served (e.g., VS Code Live Server).
# CLIENT_ORIGIN = "http://127.0.0.1:5500"

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=CLIENT_ORIGIN
)

# Mount the Socket.IO ASGI app onto the FastAPI app at the root path "/"
app.mount("/", socketio.ASGIApp(sio, app))

print("FastAPI and Socket.IO app initialized.")

# --- Gemini Prompt and Schema Definition ---
EVALUATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER"},
        "feedback": {"type": "STRING"},
        "sentiment": {"type": "STRING"},
        "moveNextSuggested": {"type": "BOOLEAN"},
        "followupQuestion": {"type": "STRING"},
        "isOffTopic": {"type": "BOOLEAN"},
        "isDontKnow": {"type": "BOOLEAN"},
    },
    "required": ["score", "feedback", "sentiment", "moveNextSuggested", "followupQuestion", "isOffTopic", "isDontKnow"],
}
print("Evaluation schema defined.")

EVALUATION_PROMPT_TEMPLATE = """
You are an AI tutor evaluating a student's answer to a question.
Your task is to analyze the provided question and the student's answer, and then provide a structured evaluation.

Here's how you should evaluate:
- **Score**: Assign a score from 0.0 to 1.0, where 1.0 is a perfect answer. Consider correctness, completeness, and clarity.
- **Feedback**: Provide concise, constructive feedback for improvement. If the answer is perfect, state that.
- **Sentiment**: Describe the sentiment of the student's answer. Use terms like "confident", "uncertain", "confused", "clear".
- **moveNextSuggested**: A boolean indicating if the student is ready to move to the next topic (True) or needs more practice/clarification on this topic (False).
- **followupQuestion**: Suggest a relevant follow-up question to deepen understanding or address gaps. If no follow-up is needed, leave it empty.
- **isOffTopic**: A boolean indicating if the answer is completely irrelevant to the question.
- **isDontKnow**: A boolean indicating if the answer explicitly states "I don't know" or clearly implies a lack of knowledge.

Ensure your output strictly adheres to the JSON format defined by the schema. Do not include any other text outside the JSON.

Question: {question}
Answer: {answer}
"""
print("Evaluation prompt template defined.")

# --- Socket.IO Event Handlers ---

@sio.event
async def connect(sid, environ):
    """Handles new client connections."""
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    """Handles client disconnections."""
    print(f"Client disconnected: {sid}")

@sio.event
async def evaluate_qa(sid, data):
    """
    Handles the 'evaluate_qa' event from the client.
    Receives question and answer, processes with Gemini, and emits feedback.
    """
    print(f"Received 'evaluate_qa' from {sid}: {data}")

    question = data.get("question")
    answer = data.get("answer")

    if not question or not answer:
        error_message = "Invalid input: 'question' and 'answer' are required."
        print(f"Error for {sid}: {error_message}")
        await sio.emit("evaluation_error", {"error": error_message}, room=sid)
        return

    try:
        print(f"Constructing prompt for SID {sid}...")
        full_prompt = EVALUATION_PROMPT_TEMPLATE.format(question=question, answer=answer)
        print(f"Prompt constructed. Calling Gemini API for SID {sid}...")

        response = await gemini_model.generate_content_async(
            contents=[{"text": full_prompt}],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=EVALUATION_SCHEMA,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
            )
        )
        print(f"Gemini API call completed for SID {sid}. Processing response...")

        evaluation_result_str = response.text
        print(f"Gemini raw response for {sid}: {evaluation_result_str}")

        try:
            parsed_evaluation = json.loads(evaluation_result_str)
            print(f"Parsed evaluation for {sid}: {parsed_evaluation}")

            # --- NEW FEATURE LOGIC ---
            # If score is greater than 0.65, set followupQuestion to empty string
            if parsed_evaluation.get("score", 0.0) > 0.65:
                print(f"Score {parsed_evaluation['score']} > 0.65. Clearing followupQuestion for SID {sid}.")
                parsed_evaluation["followupQuestion"] = ""
            # --- END NEW FEATURE LOGIC ---

            # Convert the potentially modified dictionary back to a JSON string
            final_evaluation_result = json.dumps(parsed_evaluation)
            print(f"Final evaluation result to emit for {sid}: {final_evaluation_result}")

        except json.JSONDecodeError as json_e:
            error_message = f"Gemini returned invalid JSON. Raw response: {evaluation_result_str}. Error: {json_e}"
            print(f"JSON Parsing Error for {sid}: {error_message}")
            await sio.emit("evaluation_error", {"error": error_message}, room=sid)
            return

        await sio.emit("evaluation_result", final_evaluation_result, room=sid)
        print(f"Emitted evaluation_result to {sid}")

    except Exception as e:
        error_message = f"An unexpected error occurred during Gemini API call or processing: {type(e).__name__}: {e}"
        print(f"Caught an exception for SID {sid}: {error_message}")
        await sio.emit("evaluation_error", {"error": error_message}, room=sid)

# --- FastAPI REST Endpoint (Optional, for basic health check or direct testing) ---
@app.get("/")
async def read_root():
    return {"message": "Gemini AI Tutor FastAPI Server is running. Connect via Socket.IO."}

# To run app:
# 1. Save the code as `main.py`.
# 2. Create a `.env` file in the same directory and add your Gemini API key:
#    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# 3. Install necessary libraries:
#    pip install fastapi uvicorn python-socketio python-dotenv google-generativeai
# 4. Run the server:
#    uvicorn main:app --host 0.0.0.0 --port 8000 --reload