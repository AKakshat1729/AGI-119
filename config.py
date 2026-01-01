import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Get key from environment, default to None if missing
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

def validate_api_keys():
    """Validate that all required API keys are properly configured."""
    errors = []
    
    if not API_KEY or API_KEY == "your_assemblyai_api_key_here":
        errors.append("AssemblyAI API key is missing. Please set ASSEMBLYAI_API_KEY in your .env file.")
        
    return errors