"""
FastAPI service for handling chat responses (Updated to Gemini)
"""
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from utils.llm_client import generate_chat_response, validate_gemini_api_key

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AGI-119 Chat API (Gemini Powered)", version="1.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---
class Message(BaseModel):
    role: str  # "user" or "model" (Gemini uses 'model' instead of 'assistant')
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None

class ChatResponse(BaseModel):
    status: str  # "success" or "error"
    response: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0

class APIKeyValidation(BaseModel):
    api_key: str

class APIKeyValidationResponse(BaseModel):
    valid: bool
    message: str
    error: Optional[str] = None

# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests using Gemini"""
    try:
        # Gemini wrapper expects list of dicts: [{"role": "user", "content": "..."}]
        # We normalize 'assistant' to 'model' for Gemini compatibility
        messages = []
        for m in request.messages:
            role = "model" if m.role in ["assistant", "bot", "model"] else "user"
            messages.append({"role": role, "content": m.content})
        
        logger.info(f"Processing chat request for Gemini with {len(messages)} messages")
        
        # Use the stable Gemini client we fixed earlier
        result = generate_chat_response(
            messages=messages,
            api_key=request.api_key,
            max_tokens=request.max_tokens
        )
        
        # Mapping result to ChatResponse model
        return ChatResponse(
            status=str(result.get("status", "success")),
            response=str(result.get("response", "")),
            error=result.get("error"),
            tokens_used=int(result.get("tokens_used", 0))
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return ChatResponse(
            status="error",
            response=None,
            error=f"System Error: {str(e)}",
            tokens_used=0
        )

@app.post("/validate-api-key", response_model=APIKeyValidationResponse)
async def validate_api_key_endpoint(request: APIKeyValidation) -> APIKeyValidationResponse:
    """Validate a Gemini API key"""
    try:
        result = validate_gemini_api_key(request.api_key)
        return APIKeyValidationResponse(
            valid=bool(result.get("valid")),
            message=str(result.get("message", "Validation complete")),
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        return APIKeyValidationResponse(
            valid=False,
            message="Validation engine error",
            error="SYSTEM_ERROR"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "Gemini-1.5-Flash",
        "service": "AGI-119 Chat API"
    }

@app.post("/chat-simple")
async def chat_simple(
    prompt: str,
    api_key: Optional[str] = None,
    max_tokens: int = 1000
):
    """Simplified chat endpoint for single prompt requests"""
    try:
        messages = [{"role": "user", "content": prompt}]
        
        result = generate_chat_response(
            messages=messages,
            api_key=api_key,
            max_tokens=max_tokens
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error in simple chat endpoint: {str(e)}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Note: Using port 8000 as per your original configuration
    uvicorn.run(app, host="0.0.0.0", port=8000)