from utils.groq_client import validate_groq_token
import json
import os
import sys

token = os.environ.get('TEST_GROQ_TOKEN') or (sys.argv[1] if len(sys.argv) > 1 else '')
api_url = os.environ.get('TEST_GROQ_API_URL') or (sys.argv[2] if len(sys.argv) > 2 else None)

if not token:
	print('Usage: python scripts/validate_token.py <TOKEN> [API_URL]')
	sys.exit(2)

res = validate_groq_token(token, api_url=api_url)
print(json.dumps(res, indent=2))
