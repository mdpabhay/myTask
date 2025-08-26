import os
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, List
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for user data (until server restarts)
USER_PROFILES: Dict[str, Dict[str, Any]] = {}
CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = {}

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

# JSON schema for evaluation response
EVALUATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "INTEGER"},
        "feedback": {"type": "STRING"},
        "sentiment": {"type": "STRING"},
        "moveNextSuggested": {"type": "BOOLEAN"},
        "followupQuestion": {"type": "STRING"},
        "isOffTopic": {"type": "BOOLEAN"},
        "isDontKnow": {"type": "BOOLEAN"},
        "extractedUserInfo": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "college": {"type": "STRING"},
                "degree": {"type": "STRING"},
                "address": {"type": "STRING"},
                "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                "experience": {"type": "STRING"},
                "previousCompanies": {"type": "ARRAY", "items": {"type": "STRING"}},
                "specialization": {"type": "STRING"},
                "achievements": {"type": "ARRAY", "items": {"type": "STRING"}},
                "careerGoals": {"type": "STRING"},
                "projects": {"type": "ARRAY", "items": {"type": "STRING"}},
                "certifications": {"type": "ARRAY", "items": {"type": "STRING"}}
            }
        }
    },
    "required": [
        "score", "feedback", "sentiment", "moveNextSuggested", 
        "followupQuestion", "isOffTopic", "isDontKnow", "extractedUserInfo"
    ]
}

def get_session_id(request) -> str:
    """Generate session ID from client info (simplified for demo)"""
    # In production, use proper session management
    try:
        client_ip = request.client.host if request.client else 'default'
    except:
        client_ip = 'default'
    return f"session_{hash(client_ip) % 1000000}"

def get_user_context(session_id: str) -> str:
    """Build context string from stored user information"""
    if session_id not in USER_PROFILES:
        return "CONTEXT: This is the first interaction with this user. Extract and remember their personal/professional information."
    
    profile = USER_PROFILES[session_id]
    context_parts = ["KNOWN USER INFORMATION:"]
    
    if profile.get("name"):
        context_parts.append(f"Name: {profile['name']}")
    if profile.get("college"):
        context_parts.append(f"College: {profile['college']}")
    if profile.get("degree"):
        context_parts.append(f"Degree: {profile['degree']}")
    if profile.get("address"):
        context_parts.append(f"Location: {profile['address']}")
    if profile.get("skills"):
        context_parts.append(f"Skills: {', '.join(profile['skills'])}")
    if profile.get("experience"):
        context_parts.append(f"Experience: {profile['experience']}")
    if profile.get("previousCompanies"):
        context_parts.append(f"Companies: {', '.join(profile['previousCompanies'])}")
    if profile.get("specialization"):
        context_parts.append(f"Specialization: {profile['specialization']}")
    if profile.get("projects"):
        context_parts.append(f"Projects: {', '.join(profile['projects'])}")
    if profile.get("careerGoals"):
        context_parts.append(f"Career Goals: {profile['careerGoals']}")
    
    context_parts.append("\nUSE this information to personalize your feedback and questions. Address them by name and reference their background.")
    return "\n".join(context_parts)

def update_user_profile(session_id: str, extracted_info: Dict[str, Any]):
    """Update user profile with newly extracted information"""
    if session_id not in USER_PROFILES:
        USER_PROFILES[session_id] = {}
    
    profile = USER_PROFILES[session_id]
    
    # Update with new information, merging arrays
    for key, value in extracted_info.items():
        if not value:  # Skip empty values
            continue
            
        if key in ["skills", "previousCompanies", "achievements", "projects", "certifications"]:
            # Merge arrays, avoid duplicates
            if key not in profile:
                profile[key] = []
            for item in value:
                if item and item not in profile[key]:
                    profile[key].append(item)
        else:
            # Update single values only if new value is more detailed
            if key not in profile or len(str(value)) > len(str(profile.get(key, ""))):
                profile[key] = value

# Optimized Prompt Template
PROMPT_TEMPLATE = """
You are conducting a professional interview. Evaluate the candidate's response and provide personalized feedback.

{user_context}

CURRENT QUESTION: {question}
CANDIDATE'S ANSWER: {answer}

INSTRUCTIONS:
1. **Score (0-100)**: Rate the answer quality, completeness, and professionalism
2. **Feedback**: Be specific and personal. Use their name if known. Reference their background when relevant
3. **Extract Information**: From their answer, extract any personal/professional details like name, college, skills, experience, etc.
4. **Follow-up Logic**: 
   - If score ≥ 65: Leave followupQuestion empty
   - If score < 65: Create a personalized follow-up question using their information
5. **Personalization**: Always reference known information about them in your feedback

Be professional but personable. Make them feel like you remember their background.

Respond in valid JSON format.
"""

@app.post("/evaluate")
async def evaluate_qa(payload: QAInput, request: Request):
    try:
        # Get or create session
        session_id = get_session_id(request)
        
        # Build user context
        user_context = get_user_context(session_id)
        
        # Store conversation history
        if session_id not in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[session_id] = []
        CONVERSATION_HISTORY[session_id].append({
            "question": payload.question,
            "answer": payload.answer
        })
        
        # Format prompt
        prompt = PROMPT_TEMPLATE.format(
            user_context=user_context,
            question=payload.question,
            answer=payload.answer
        )

        # Get AI response
        response = await model.generate_content_async(
            contents=[{"text": prompt}],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=EVALUATION_SCHEMA,
                temperature=0.3,
                top_p=0.9,
                top_k=40,
            )
        )

        response_text = response.text
        parsed = json.loads(response_text)

        # Update user profile with extracted information
        if "extractedUserInfo" in parsed:
            extracted_info = parsed["extractedUserInfo"]
            # Clean empty values
            cleaned_info = {k: v for k, v in extracted_info.items() if v}
            if cleaned_info:
                update_user_profile(session_id, cleaned_info)

        # Apply business rules
        if parsed.get("score", 0) >= 65:
            parsed["followupQuestion"] = ""

        # Return only the required fields
        result = {
            "feedback": parsed.get("feedback", ""),
            "followupQuestion": parsed.get("followupQuestion", ""),
            "isDontKnow": parsed.get("isDontKnow", False),
            "isOffTopic": parsed.get("isOffTopic", False),
            "moveNextSuggested": parsed.get("moveNextSuggested", False),
            "score": parsed.get("score", 0),
            "sentiment": parsed.get("sentiment", "Neutral")
        }

        return result

    except json.JSONDecodeError as je:
        raise HTTPException(status_code=500, detail=f"Invalid JSON returned from AI: {je}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Smart Interview Evaluator API is running"}

@app.get("/debug/profile/{session_id}")
async def get_user_profile(session_id: str):
    """Debug endpoint to see stored user profile"""
    return {
        "profile": USER_PROFILES.get(session_id, {}),
        "conversation_count": len(CONVERSATION_HISTORY.get(session_id, []))
    }

@app.delete("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset user session data"""
    if session_id in USER_PROFILES:
        del USER_PROFILES[session_id]
    if session_id in CONVERSATION_HISTORY:
        del CONVERSATION_HISTORY[session_id]
    return {"message": f"Session {session_id} reset successfully"}
