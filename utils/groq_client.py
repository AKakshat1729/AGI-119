import os
import requests
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()


class GroqAPIError(Exception):
    pass


class GroqClient:
    """Minimal Groq API client with compatible interface for this project."""

    DEFAULT_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/v1")
    DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq-1")

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = api_url or self.DEFAULT_API_URL
        # Use Groq model only (ignore any passed model parameter)
        self.model = os.getenv("GROQ_MODEL") or self.DEFAULT_MODEL

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Please set it in your .env or pass it explicitly.")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _make_url(self, path: str) -> str:
        return f"{self.api_url.rstrip('/')}/{path.lstrip('/')}"

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Dict[str, any]:
        """Send a generation request to the Groq API.

        NOTE: Different Groq deployments may use different paths. This implementation
        uses a conservative path of `/models/{model}/predict`. If your Groq host
        expects a different endpoint, set `GROQ_API_URL` accordingly in your environment.
        """
        try:
            url = self._make_url(f"models/{self.model}/predict")
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            resp = self.session.post(url, json=payload, timeout=20)
            if resp.status_code == 401:
                return {"status": "error", "response": None, "error": "Invalid API key", "error_type": "INVALID_API_KEY", "tokens_used": 0}
            if resp.status_code == 429:
                return {"status": "error", "response": None, "error": "Rate limit or quota exceeded", "error_type": "RATE_LIMIT", "tokens_used": 0}

            resp.raise_for_status()
            data = resp.json()

            # Try common response shapes
            text = None
            if isinstance(data, dict):
                # common fields: 'text', 'output', 'result', or nested
                text = data.get("text") or data.get("output") or data.get("result")
                if isinstance(text, list):
                    text = "\n".join([str(t) for t in text])
                if isinstance(text, dict) and "text" in text:
                    text = text.get("text")

            if not text:
                # Fallback: stringify JSON
                text = str(data)

            text = (text or "").strip()

            return {"status": "success", "response": text, "error": None, "error_type": None, "tokens_used": len(text.split())}

        except requests.exceptions.RequestException as e:
            em = str(e)
            if "timeout" in em.lower():
                return {"status": "error", "response": None, "error": "Request timed out", "error_type": "TIMEOUT", "tokens_used": 0}
            return {"status": "error", "response": None, "error": em, "error_type": "REQUEST_EXCEPTION", "tokens_used": 0}

    @staticmethod
    def validate_api_key(api_key: str, api_url: Optional[str] = None) -> Dict[str, any]:
        base = (api_url or GroqClient.DEFAULT_API_URL).rstrip('/')
        list_url = base + "/models"
        try:
            resp = requests.get(list_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            if resp.status_code == 200:
                return {"valid": True, "message": "✓ API key is valid (models list)", "error": None}
            if resp.status_code == 401:
                return {"valid": False, "message": "❌ Invalid API Key - Unauthorized", "error": "INVALID_API_KEY"}
            if resp.status_code == 429:
                return {"valid": False, "message": "⚠️ Rate limited / Quota exceeded", "error": "QUOTA_EXCEEDED"}

            # Some Groq deployments return 403 for list access but allow predict.
            # Try a minimal predict call to confirm the token's validity.
            try:
                predict_url = base + f"/models/{GroqClient.DEFAULT_MODEL}/predict"
                payload = {"prompt": "Hello", "max_tokens": 1}
                p = requests.post(predict_url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=10)
                if p.status_code == 200:
                    return {"valid": True, "message": "✓ API key is valid (predict)", "error": None}
                if p.status_code == 401:
                    return {"valid": False, "message": "❌ Invalid API Key - Unauthorized", "error": "INVALID_API_KEY"}
                if p.status_code == 429:
                    return {"valid": False, "message": "⚠️ Rate limited / Quota exceeded", "error": "QUOTA_EXCEEDED"}
                # Other non-200 indicates validation failure but include HTTP code
                return {"valid": False, "message": f"❌ Validation failed: HTTP {p.status_code}", "error": "VALIDATION_ERROR"}
            except requests.exceptions.RequestException as e:
                return {"valid": False, "message": f"❌ Validation request failed: {str(e)}", "error": "REQUEST_FAILED"}

        except requests.exceptions.RequestException as e:
            # Network error (can't reach Groq). Allow user to save token anyway and test on chat.
            return {"valid": True, "message": f"⚠️ Could not validate online (network unavailable), token saved. Will validate on first chat use.", "error": None}


def get_groq_client(api_key: Optional[str] = None, model: Optional[str] = None, api_url: Optional[str] = None) -> GroqClient:
    return GroqClient(api_key=api_key, model=model, api_url=api_url)


def generate_groq_response(messages: List[Dict[str, str]], api_key: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 500) -> Dict[str, any]:
    try:
        client = get_groq_client(api_key=api_key)
        # Convert messages to a single prompt
        text_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_messages.append(f"{role}: {content}")
        prompt = "\n".join(text_messages)
        return client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return {"status": "error", "response": None, "error": str(e), "error_type": "INITIALIZATION_ERROR", "tokens_used": 0}


def validate_groq_token(api_key: str, api_url: Optional[str] = None) -> Dict[str, any]:
    return GroqClient.validate_api_key(api_key, api_url=api_url)
