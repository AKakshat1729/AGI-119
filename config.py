API_KEY = "d39d6cb8059442788bb7af73239f1928"

# API Key Validation
def validate_api_keys():
    """Validate that all required API keys are properly configured."""
    errors = []

    if not API_KEY or API_KEY == "your_assemblyai_api_key_here":
        errors.append("AssemblyAI API key is not configured. Please set API_KEY in config.py")

    return errors
