import os
import asyncio
import socketio
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables from .env file for API keys and other configurations.
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Critical error: application cannot proceed without the API key.
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")

try:
    # Initialize the Gemini API client with the provided key.
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    # Log and re-raise if API configuration fails, indicating a setup issue.
    print(f"Error configuring Gemini API: {e}")
    raise

# Define the Gemini model to be used for content generation.
# 'gemini-1.5-flash' is chosen for its balance of speed and capability in evaluation tasks.
GEMINI_MODEL_NAME = "gemini-1.5-flash"
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
print(f"Using Gemini model: {GEMINI_MODEL_NAME}")

# --- FastAPI and Socket.IO Setup ---
app = FastAPI()

# Retrieve CLIENT_ORIGIN from environment variables.
# This value will be set in Render's dashboard during deployment.
# For local testing, ensure your .env has CLIENT_ORIGIN set (e.g., CLIENT_ORIGIN="*").
client_origin_env = os.getenv("CLIENT_ORIGIN")

# Process client_origin_env for cors_allowed_origins.
# If client_origin_env is a comma-separated string, split it into a list.
# If client_origin_env is None or empty, default to an empty list (no origins allowed by default, safer for production).
if client_origin_env:
    # If multiple origins are specified (comma-separated), split them into a list.
    # Otherwise, treat it as a single string.
    cors_allowed_origins_list = client_origin_env.split(',') if ',' in client_origin_env else client_origin_env
else:
    # Default to an empty list if the environment variable is not set.
    # This is a safe default, meaning no CORS origins are allowed unless explicitly configured.
    cors_allowed_origins_list = []

# Initialize Socket.IO server in ASGI mode, allowing specified origins.
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=cors_allowed_origins_list # Use the processed list/string
)

# Mount the Socket.IO ASGI application onto the FastAPI application at the root path.
# This allows both Socket.IO events and standard HTTP requests to be handled by the same server.
app.mount("/", socketio.ASGIApp(sio, app))
print("FastAPI and Socket.IO app initialized.")

# --- Gemini Prompt and Structured Output Schema Definition ---
# This dictionary defines the expected JSON structure for Gemini's evaluation output.
# It ensures consistent and parseable results for the client.
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

# This template guides Gemini on how to evaluate the student's answer
# and format its response according to the defined schema.
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
async def connect(sid: str, environ: dict):
    """
    Handles new client connections to the Socket.IO server.
    :param sid: The session ID of the connected client.
    :param environ: A dictionary containing environment variables for the connection.
    """
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid: str):
    """
    Handles client disconnections from the Socket.IO server.
    :param sid: The session ID of the disconnected client.
    """
    print(f"Client disconnected: {sid}")

@sio.event
async def evaluate_qa(sid: str, data: dict):
    """
    Receives a question and answer from the client via a Socket.IO event,
    processes them using the Gemini API, and emits the structured evaluation back.
    :param sid: The session ID of the client sending the request.
    :param data: A dictionary containing 'question' and 'answer' strings.
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
        # Format the prompt with the received question and answer.
        full_prompt = EVALUATION_PROMPT_TEMPLATE.format(question=question, answer=answer)
        print(f"Calling Gemini API for SID {sid} with prompt...")

        # Call the Gemini API to generate the evaluation.
        # The generation_config ensures the response is JSON and follows the defined schema.
        response = await genai.GenerativeModel(GEMINI_MODEL_NAME).generate_content_async(
            contents=[{"text": full_prompt}],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=EVALUATION_SCHEMA,
                temperature=0.2, # Lower temperature for more deterministic and factual output.
                top_p=0.9,
                top_k=40,
            )
        )
        print(f"Gemini API call completed for SID {sid}. Processing response...")

        # Extract the raw JSON string from Gemini's response.
        evaluation_result_str = response.text
        print(f"Gemini raw response for {sid}: {evaluation_result_str}")

        try:
            # Parse the JSON string into a Python dictionary.
            parsed_evaluation = json.loads(evaluation_result_str)
            print(f"Parsed evaluation for {sid}: {parsed_evaluation}")

            # Implement the feature: if score > 0.65, clear the followupQuestion.
            if parsed_evaluation.get("score", 0.0) > 0.65:
                print(f"Score {parsed_evaluation['score']} > 0.65. Clearing followupQuestion for SID {sid}.")
                parsed_evaluation["followupQuestion"] = ""

            # Convert the potentially modified dictionary back to a JSON string for emission.
            final_evaluation_result = json.dumps(parsed_evaluation)
            print(f"Final evaluation result to emit for {sid}: {final_evaluation_result}")

        except json.JSONDecodeError as json_e:
            # Handle cases where Gemini's response is not valid JSON.
            error_message = f"Gemini returned invalid JSON. Raw response: {evaluation_result_str}. Error: {json_e}"
            print(f"JSON Parsing Error for {sid}: {error_message}")
            await sio.emit("evaluation_error", {"error": error_message}, room=sid)
            return

        # Emit the final structured evaluation result back to the client.
        await sio.emit("evaluation_result", final_evaluation_result, room=sid)
        print(f"Emitted evaluation_result to {sid}")

    except Exception as e:
        # Catch any unexpected errors during the API call or processing.
        error_message = f"An unexpected error occurred during Gemini API call or processing: {type(e).__name__}: {e}"
        print(f"Caught an exception for SID {sid}: {error_message}")
        await sio.emit("evaluation_error", {"error": error_message}, room=sid)

# --- FastAPI REST Endpoint (for health check or direct access) ---
@app.get("/")
async def read_root():
    """
    Root endpoint for the FastAPI application.
    Returns a simple message indicating the server is running.
    """
    return {"message": "Gemini AI Tutor FastAPI Server is running. Connect via Socket.IO."}

# --- Execution Instructions ---
# To run this application:
# 1. Save the code as `main.py`.
# 2. Create a `.env` file in the same directory and add your Gemini API key:
#    GEMINI_API_KEY="YOUR_API_KEY_HERE"
#    CLIENT_ORIGIN="http://127.0.0.1:5500" (for local testing with Live Server)
#    OR CLIENT_ORIGIN="*" (for broader local testing)
# 3. Install necessary libraries (if not already installed in your virtual environment):
#    pip install fastapi uvicorn python-socketio python-dotenv google-generativeai gunicorn
# 4. Run the server from your terminal:
#    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
