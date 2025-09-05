import os
import json
import uuid
import hashlib
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import threading

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

# Thread-safe storage for user data with locks
USER_PROFILES: Dict[str, Dict[str, Any]] = {}
CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = {}
_profile_locks: Dict[str, threading.Lock] = {}
_main_lock = threading.Lock()

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

def get_session_id(request: Request, user_session: Optional[str] = None) -> str:
    """Generate unique session ID with multiple fallbacks for isolation"""
    if user_session:
        # Use provided session ID (recommended)
        return f"usr_{hashlib.md5(user_session.encode()).hexdigest()[:12]}"
    
    # Fallback: Generate from multiple request attributes
    try:
        client_ip = request.client.host if request.client else 'unknown'
        user_agent = request.headers.get("user-agent", "unknown")
        x_forwarded = request.headers.get("x-forwarded-for", "")
        
        # Create unique identifier
        unique_string = f"{client_ip}_{user_agent}_{x_forwarded}_{id(request)}"
        session_hash = hashlib.md5(unique_string.encode()).hexdigest()[:12]
        return f"auto_{session_hash}"
    except:
        # Ultimate fallback
        return f"fallback_{uuid.uuid4().hex[:12]}"

def get_user_lock(session_id: str) -> threading.Lock:
    """Get thread-safe lock for specific user session"""
    with _main_lock:
        if session_id not in _profile_locks:
            _profile_locks[session_id] = threading.Lock()
        return _profile_locks[session_id]

def get_user_context(session_id: str) -> str:
    """Thread-safe context retrieval"""
    user_lock = get_user_lock(session_id)
    with user_lock:
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
    """Thread-safe profile update"""
    user_lock = get_user_lock(session_id)
    with user_lock:
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

def store_conversation(session_id: str, question: str, answer: str):
    """Thread-safe conversation storage"""
    user_lock = get_user_lock(session_id)
    with user_lock:
        if session_id not in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[session_id] = []
        CONVERSATION_HISTORY[session_id].append({
            "question": question,
            "answer": answer
        })

# Optimized Prompt Template
PROMPT_TEMPLATE = """
You are conducting a professional interview for a Software/IT role. 
Evaluate the candidate's response and follow the exact instructions below.

{user_context}

CURRENT QUESTION: {question}  
CANDIDATE'S ANSWER: {answer}  

INSTRUCTIONS:  
1. **Detection of "Don't Know / Skip" Responses**:  
   - If the candidate's answer matches or is similar to ANY of these phrases:  
     "I don't know", "No idea", "Not sure", "I have no clue", "Leave this question",  
     "Skip this question", "Next question", "Pass", "I can't answer this",  
     "No knowledge of this concept", "I haven't studied this", "This is out of my syllabus",  
     "I don't remember", "I don't understand this", "I'm blank on this one", "Let's skip",  
     "I'm not confident about this one", "I think I'll skip this question",  
     "I'm unsure, maybe we can move on", "I'm afraid I don't know this",  
     "This isn't clear to me right now", "I would prefer to skip this question",  
     "Sorry, I cannot answer this correctly", "Sorry, I don't know the exact answer",  
     "I'm not certain about this concept", "I haven't practiced this yet",  
     "I don't think I can answer this properly", "Sorry, I am not aware about this question",  
     "Sorry, I don't have knowledge about this topic", "Sorry, I haven't learned this concept yet",  
     "Sorry, I'm not familiar with this", "Sorry, I can't explain this right now",  
     "Sorry, this is new to me", "Apologies, I don't know this one",  
     "Sorry, I haven't studied this area", "Sorry, I'm not updated on this topic",  
     "Apologies, I can't answer this question", "I don't have the right knowledge to answer this properly",  
     "This is outside my current understanding", "I would need to study more to answer this",  
     "I'm still learning, so I cannot answer this now", "I apologize, but I cannot attempt this question",  
     "This is beyond my current preparation", "I respect the question, but I can't answer it right now",  
     "I need more preparation to answer this question", "Currently, I don't have clarity on this",  
     "This topic is not in my current knowledge base", "Nope, don't know", "Totally blank",  
     "Brain freeze", "Out of my league", "Pass on this one", "Not in my head right now",  
     "Clueless here", "I got nothing", "My mind is blank", "This went over my head".  

     → In this case:  
        - Set isDontKnow: true
        - Set score: 0, feedback: "", sentiment: "", isOffTopic: false, moveNextSuggested: false
        - ONLY return a **new crisp follow-up question(max 14 words)** that is **different from the current topic**,  
          but still IT/software-related.  
        - Follow-up must be short (max 15 words), knowledge-based, less difficult, always start with the definition and basic question and cover topics such as:  
          - Programming (Python, Java, C++ and other Programming languages)  
          - Databases (SQL, indexing, transactions)  
          - Operating Systems (OS related definitions, threads, scheduling, memory)  
          - Networking (OSI model, Protocol, HTTP, TCP/IP, APIs)  
          - Cloud & DevOps (CI/CD, containers, scaling)  
          - Cybersecurity (encryption, authentication, firewalls)  
          - Data Structures & Algorithms  
          - Software Engineering (Agile, Git/GitHub, testing)  

2. **Normal Evaluation (Non-Skip Answers)**:  
        - **Score (0-100)**: Rate the answer quality, completeness, and professionalism
        - **Feedback**: Provide a concise, personal one-line feedback and must be short (max 15 words). Use the student's name and ohter info if known, or 'student' otherwise.
        - **Extract Information**: From their answer, extract any personal/professional details like name, college, skills, experience, etc.
        - **Follow-up Logic**: 
            - If score ≥ 65: Leave followupQuestion empty
            - If score < 65: Create a personalized follow-up question using their information and question must be crisp, knowledge-based IT/software-related question (max 15 words). Use student's name if available. Questions must vary across topics like:
                - The student's given answer or highlight a missing concept.
                - Programming (Python, Java, C++)
                - Databases (SQL, transactions, indexing)
                - Operating Systems (threads, memory, scheduling)
                - Networking (TCP/IP, HTTP, APIs)
                - Cloud & DevOps (containers, CI/CD)
                - Cybersecurity (encryption, authentication)
                - Data Structures & Algorithms
                - Software Engineering (Agile, version control, testing)

3. **Personalization**: Always use extracted personal details when giving feedback or forming follow-up questions in such that follow-up question should not repeat with the student and always make sure it is an interview.  

OUTPUT: Return only valid JSON with required fields.  
"""

@app.post("/evaluate")
async def evaluate_qa(
    payload: QAInput, 
    request: Request,
    x_user_session: Optional[str] = Header(None, alias="X-User-Session")
):
    try:
        # Get unique session ID for this user
        session_id = get_session_id(request, x_user_session)
        
        # Build user context (thread-safe)
        user_context = get_user_context(session_id)
        
        # Store conversation history (thread-safe)
        store_conversation(session_id, payload.question, payload.answer)
        
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

        # Update user profile with extracted information (thread-safe)
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
