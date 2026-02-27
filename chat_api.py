"""
FastAPI service for handling chat responses (Groq only)
"""
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import logging
from utils.groq_client import generate_groq_response, validate_groq_token

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AGI-119 Chat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 500
    api_key: Optional[str] = None  # Allow passing custom API key


class ChatResponse(BaseModel):
    status: str  # "success" or "error"
    response: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    tokens_used: int = 0


class APIKeyValidation(BaseModel):
    api_key: str


class APIKeyValidationResponse(BaseModel):
    valid: bool
    message: str
    error: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat requests
    
    Args:
        request: ChatRequest containing messages and optional API key
        
    Returns:
        ChatResponse with generated response or error
    """
    try:
        # Convert Pydantic models to dicts
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        logger.info(f"Processing chat request with {len(messages)} messages")
        
        # Generate response via Groq client
        result = generate_groq_response(
            messages=messages,
            api_key=request.api_key,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return ChatResponse(
            status="error",
            response=None,
            error=f"An error occurred: {str(e)}",
            error_type="UNKNOWN_ERROR",
            tokens_used=0
        )


@app.post("/validate-api-key", response_model=APIKeyValidationResponse)
async def validate_api_key(request: APIKeyValidation) -> APIKeyValidationResponse:
    """
    Validate an LLM API key (Groq/HF compatible)
    
    Args:
        request: API key to validate
        
    Returns:
        Validation result
    """
    try:
        result = validate_groq_token(request.api_key)
        return APIKeyValidationResponse(**result)
    
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        return APIKeyValidationResponse(
            valid=False,
            message="Validation error",
            error="VALIDATION_ERROR"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AGI-119 Chat API",
        "version": "1.0.0"
    }


@app.post("/chat-simple")
async def chat_simple(
    prompt: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500
):
    """
    Simplified chat endpoint for single prompt requests
    
    Args:
        prompt: Single message from user
        api_key: Optional custom API key
        temperature: Creativity level
        max_tokens: Max response tokens
        
    Returns:
        Chat response
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        
        result = generate_groq_response(
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in simple chat endpoint: {str(e)}")
        return ChatResponse(
            status="error",
            response=None,
            error=str(e),
            error_type="UNKNOWN_ERROR",
            tokens_used=0
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
