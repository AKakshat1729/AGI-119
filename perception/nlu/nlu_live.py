import os
from dotenv import load_dotenv
import json
import re
import nltk
from google.generativeai import GenerativeModel
import google.generativeai as genai

try:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("maxent_ne_chunker_tab", quiet=True)
    nltk.download("words", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    print(f"NLTK download failed: {e}")
    pass

def has_non_ascii(text):
    return bool(re.search(r'[^\x00-\x7F]', text))

def llm_nlu_fallback(text: str) -> dict:
    """
    Fallback LLM-based NLU for multilingual/complex text.
    """
    try:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {}
            
        genai.configure(api_key=api_key)
        model = GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""Perform Natural Language Understanding (NLU) on this text (could be English, Hindi, or Hinglish).
        Extract:
        1. Entities (names, places, dates, specific things)
        2. Semantic Roles (who did what, actions)
        
        Return ONLY a JSON object with:
        {{
            "entities": [{"entity": "string", "type": "string"}],
            "semantic_roles": [{"word": "string", "role": "string"}]
        }}
        
        Text: {text}
        """
        
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"[NLU FALLBACK ERROR] {e}")
        return {}

def get_entities(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tags)
    entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity = " ".join([word for word, tag in subtree.leaves()])
            label = subtree.label()
            entities.append({"entity": entity, "type": label})
    return entities

def get_roles(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    roles = []
    for w, t in tags:
        if t.startswith("NN"): roles.append({"word": w, "role": "entity"})
        elif t.startswith("VB"): roles.append({"word": w, "role": "action"})
    return roles

def nlu_process(text, tone_obj):
    # Detect if non-English
    if has_non_ascii(text) or tone_obj.get("multilingual", False):
        llm_nlu = llm_nlu_fallback(text)
        if llm_nlu:
            return {
                "transcript": text,
                "sentiment": tone_obj["sentiment"],
                "emotions": tone_obj["emotions"],
                "entities": llm_nlu.get("entities", []),
                "semantic_roles": llm_nlu.get("semantic_roles", []),
                "multilingual": True
            }
            
    return {
        "transcript": text,
        "sentiment": tone_obj["sentiment"],
        "emotions": tone_obj["emotions"],
        "entities": get_entities(text),
        "semantic_roles": get_roles(text),
        "multilingual": False
    }
