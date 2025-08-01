import os
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# Schema for the POST payload
class QAInput(BaseModel):
    question: str
    answer: str

# JSON schema to ensure consistent model response
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
    "required": [
        "score", "feedback", "sentiment",
        "moveNextSuggested", "followupQuestion",
        "isOffTopic", "isDontKnow"
    ]
}

# Prompt Template
PROMPT_TEMPLATE = """
You are an AI tutor evaluating a student's answer to a question.
Your task is to analyze the provided question and the student's answer, and then provide a structured evaluation.

Here's how you should evaluate:
- **Score**: Assign a score from 0.0 to 1.0, where 1.0 is a perfect answer.
- **Feedback**: Provide concise, constructive feedback.
- **Sentiment**: Describe the student's tone, e.g., "confident", "uncertain".
- **moveNextSuggested**: True if ready to move on next topic, else False.
- **followupQuestion**: Suggest one only if needed of Score less than 0.6, else empty.
- **isOffTopic**: True if irrelevant.
- **isDontKnow**: True if the student said "I don't know".

Output must be valid JSON with this structure.

Question: {question}
Answer: {answer}
"""

@app.post("/evaluate")
async def evaluate_qa(payload: QAInput):
    try:
        prompt = PROMPT_TEMPLATE.format(question=payload.question, answer=payload.answer)

        response = await model.generate_content_async(
            contents=[{"text": prompt}],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=EVALUATION_SCHEMA,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
            )
        )

        response_text = response.text
        parsed = json.loads(response_text)

        # Business rule: clear followupQuestion if score > 0.65
        if parsed.get("score", 0.0) > 0.65:
            parsed["followupQuestion"] = ""

        return parsed

    except json.JSONDecodeError as je:
        raise HTTPException(status_code=500, detail=f"Invalid JSON returned from Gemini: {je}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Gemini QA Evaluator API is running"}
