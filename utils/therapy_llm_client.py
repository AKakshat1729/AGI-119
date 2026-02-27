"""
Enhanced LLM Client with Clinical Resources Integration
"""
import os
import requests
from flask import session
from utils.llm_client import generate_chat_response
from typing import List, Dict, Tuple
from clinical_resources import (
    get_coping_strategies, get_therapeutic_approach,
    get_condition_info, get_crisis_resources, embed_clinical_context
)

MAX_RETRIES = 2
REQUEST_TIMEOUT = 30


class TherapyLLMClient:
    """Enhanced LLM client with clinical resources"""
    
    def __init__(self):
        self.api_key = None
        # Default model - use environment or gemini recommended
        self.model = os.environ.get('LLM_MODEL', 'gemini-1.5-flash')
        
    def get_api_key(self, custom_key=None):
        """Get API key from various sources (Gemini)"""
        api_key = (
            custom_key or 
            session.get('gemini_api_key') or 
            os.environ.get("GEMINI_API_KEY")
        )
        
        if not api_key:
            return None
            
        return api_key.strip().replace('"', '').replace("'", "")
    
    def build_clinical_prompt(self, user_message: str, message_history: List[Dict]) -> Tuple[str, str]:
        """Build enhanced prompt with clinical resources"""
        
        # Detect potential topics
        keywords = {
            "anxiety": ["anxious", "worried", "nervous", "fear", "panic"],
            "depression": ["sad", "hopeless", "depressed", "empty", "numb"],
            "stress": ["stressed", "overwhelmed", "pressure", "busy", "work"],
            "sleep": ["sleep", "insomnia", "tired", "exhausted", "rest"],
            "trauma": ["trauma", "flashback", "nightmare", "triggered", "abuse"]
        }
        
        detected_condition = None
        user_msg_lower = user_message.lower()
        
        for condition, phrases in keywords.items():
            if any(phrase in user_msg_lower for phrase in phrases):
                detected_condition = condition
                break
        
        # Build system prompt with clinical context
        system_prompt = embed_clinical_context(user_message, detected_condition)
        
        # Add clinical resources if condition detected
        if detected_condition:
            strategies = get_coping_strategies(detected_condition)
            if strategies:
                system_prompt += f"\n\nRelevant Evidence-Based Strategies for {detected_condition.title()}:\n"
                for i, strategy in enumerate(strategies[:3], 1):
                    system_prompt += f"{i}. {strategy}\n"
        
        # Add crisis resources context if warning signs detected
        warning_phrases = ["kill", "suicide", "die", "harm", "hurt myself"]
        if any(phrase in user_msg_lower for phrase in warning_phrases):
            system_prompt += "\n\nCRISIS RESOURCES: This person may be in crisis.\n"
            system_prompt += "Resources: Call 988 (US), Text HOME to 741741, or go to emergency room.\n"
        
        # Build conversation
        conversation = system_prompt + "\n\nConversation History:\n"
        
        for msg in message_history[-4:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                conversation += f"\nUser: {content}"
            elif role in ['assistant', 'bot']:
                conversation += f"\nTherapist: {content}"
        
        conversation += f"\n\nUser: {user_message}\nTherapist:"
        
        return conversation, detected_condition
    
    def call_gemini(self, messages: List[Dict], custom_api_key: str = None) -> Dict:
        """Call Gemini via `generate_chat_response` wrapper"""
        api_key = custom_api_key or self.get_api_key()
        if not api_key:
            return {"error": "No API key configured", "message": "Please configure your Gemini API key in settings"}

        try:
            result = generate_chat_response(messages=messages, model=self.model, api_key=api_key)
            if isinstance(result, dict) and result.get('status') == 'error':
                return {"error": result.get('error', 'LLM error'), "message": result.get('response')}
            # normalize
            return {"success": True, "response": result.get('response') if isinstance(result, dict) else str(result), "tokens_used": result.get('tokens_used', 0) if isinstance(result, dict) else 0}
        except Exception as e:
            return {"error": "LLM call failed", "message": str(e)}
    
    def generate_therapy_response(self, user_message: str, message_history: List[Dict], 
                                  custom_api_key: str = None) -> Dict:
        """Generate therapy response with clinical context"""
        
        try:
            # Build clinical prompt
            prompt, detected_condition = self.build_clinical_prompt(user_message, message_history)
            
            # Prepare messages for API
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a highly empathetic, ethically-grounded AI therapy assistant. "
                        "You use evidence-based therapeutic techniques while maintaining professional boundaries. "
                        "Always remind users that AI is not a substitute for professional mental health care."
                    )
                }
            ]
            
            # Add history
            for msg in message_history[-6:]:
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        "role": msg.get('role'),
                        "content": msg.get('content', '')
                    })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Get response from LLM
            result = self.call_gemini(messages, custom_api_key)
            
            if "error" in result:
                return result
            
            # Enhance response if clinical condition detected
            response_text = result.get('response', '')
            
            if detected_condition:
                response_text += f"\n\n*Note: Based on what you shared, here are some evidence-based approaches related to {detected_condition}:*"
                strategies = get_coping_strategies(detected_condition)
                for strategy in strategies[:2]:
                    response_text += f"\n• {strategy}"
            
            # Add safety disclaimer
            response_text += "\n\n*Remember: I'm an AI assistant. For serious mental health concerns, please speak with a licensed therapist or counselor.*"
            
            return {
                "success": True,
                "response": response_text,
                "detected_condition": detected_condition,
                "tokens_used": result.get('tokens_used', 0)
            }
            
        except Exception as e:
            return {
                "error": "Response Generation Error",
                "message": f"Failed to generate response: {str(e)}"
            }


def get_llm_response(user_message: str, message_history: List[Dict] = None, 
                     custom_api_key: str = None) -> Dict:
    """Main function to get LLM response with clinical integration"""
    
    if message_history is None:
        message_history = []
    
    client = TherapyLLMClient()
    return client.generate_therapy_response(user_message, message_history, custom_api_key)


def validate_api_key(api_key: str) -> Dict:
    """Validate API key by making a test request"""
    try:
        test_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Say 'Hello' if you receive this message."
            }
        ]
        
        # Validate by calling generate_chat_response with a tiny prompt
        try:
            resp = generate_chat_response(test_messages, model=os.environ.get('LLM_MODEL', 'gemini-1.5-flash'), api_key=api_key)
            if isinstance(resp, dict) and resp.get('status') == 'error':
                return {"valid": False, "message": str(resp.get('error') or resp.get('response')), "error": "INVALID_KEY"}
            return {"valid": True, "message": "API key is valid!", "error": None}
        except Exception as e:
            return {"valid": False, "message": f"Validation error: {e}", "error": "VALIDATION_ERROR"}
            
    except Exception as e:
        return {
            "valid": False,
            "message": f"Validation error: {str(e)}",
            "error": "VALIDATION_ERROR"
        }
