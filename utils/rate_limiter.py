import json
import os
from datetime import datetime

USAGE_FILE = "daily_usage.json"
DAILY_TOKEN_LIMIT = 50000 # Example limit, can be adjusted

def get_today_str():
    return datetime.now().strftime("%Y-%m-%d")

def load_usage():
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_usage(usage):
    with open(USAGE_FILE, 'w') as f:
        json.dump(usage, f, indent=2)

def check_quota(tokens_to_add: int = 0) -> bool:
    """
    Checks if adding tokens_to_add would exceed the daily limit.
    Returns True if within quota, False otherwise.
    """
    usage = load_usage()
    today = get_today_str()
    
    current_tokens = usage.get(today, 0)
    if current_tokens + tokens_to_add > DAILY_TOKEN_LIMIT:
        return False
    return True

def update_usage(tokens: int):
    usage = load_usage()
    today = get_today_str()
    
    usage[today] = usage.get(today, 0) + tokens
    save_usage(usage)

def get_current_usage():
    usage = load_usage()
    today = get_today_str()
    return usage.get(today, 0)
