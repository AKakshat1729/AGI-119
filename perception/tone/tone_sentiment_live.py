import os
import json
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import nltk 
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import google.generativeai as genai

# --- Pylance Pacifier ---
# Forces Pylance to stop complaining about "configure" or "GenerativeModel" exports
genai_client: Any = genai

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def has_non_ascii(text: str) -> bool:
    """Detect characters outside the standard ASCII range (e.g., Devanagari)."""
    return bool(re.search(r'[^\x00-\x7F]', text))

def llm_sentiment_analyzer(text: str) -> dict:
    """
    Fallback LLM-based sentiment analysis for multilingual/complex text.
    """
    try:
        load_dotenv()
        # Use str fallback to ensure type safety
        api_key = str(os.environ.get("GEMINI_API_KEY") or "")
        if not api_key:
            return {}
            
        genai_client.configure(api_key=api_key)
        
        # Switched to 1.5-flash for better stability and demo quota
        model = genai_client.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""Analyze the sentiment of this text (could be English, Hindi, or Hinglish).
        Return ONLY a JSON object with:
        {{
            "polarity": float (-1.0 to 1.0),
            "emotions": list,
            "overall_mood": string ("positive", "negative", "neutral")
        }}
        
        Text: {text}
        """
        
        response = model.generate_content(prompt)
        # Clean response text if it has markdown formatting
        raw_text = response.text
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"[SENTIMENT FALLBACK ERROR] {e}")
        return {}

# Expanded emotion lexicon
emotion_lexicon: Dict[str, List[str]] = {
    "happy": [
        "happy", "joy", "joyful", "excited", "delighted", "pleased", "cheerful", "glad", "love", "like", "enjoy",
        "content", "satisfied", "thrilled", "blissful", "grateful", "ecstatic", "optimistic", "hopeful", "proud",
        "radiant", "smiling", "enthusiastic", "elated", "overjoyed", "khush", "aanand", "sukhi", "mazza",
        "अच्चा", "खुश", "आनंद", "सुखी", "मज़ा", "प्यार"
    ],
    "sad": [
        "sad", "unhappy", "depressed", "sorrow", "grief", "miserable", "down", "dislike", "disappointed", "regret",
        "lonely", "heartbroken", "hopeless", "tearful", "gloomy", "melancholy", "blue", "discouraged", "hurt",
        "weary", "lost", "empty", "abandoned", "devastated", "pained", "udaas", "dukh", "duki", "pareshan", "dard",
        "उदास", "दुःख", "दुखी", "परेशान", "दर्द", "अकेला", "रोया"
    ],
    "angry": [
        "angry", "mad", "furious", "irritated", "annoyed", "rage", "frustrated", "hate", "resentful", "disgusted",
        "offended", "outraged", "hostile", "agitated", "bitter", "enraged", "cross", "snappy", "fuming", "provoked",
        "irate", "vengeful", "exasperated", "livid", "infuriated", "gussa", "krodh", "nafrat",
        "गुस्सा", "क्रोध", "नफरत", "पागल", "चिड़चिड़ा"
    ],
    "fear": [
        "fear", "scared", "afraid", "terrified", "anxious", "worried", "nervous", "panic", "frightened", "alarmed",
        "uneasy", "insecure", "shaky", "tense", "apprehensive", "paranoid", "timid", "dread", "phobic", "startled",
        "distressed", "hesitant", "shocked", "petrified", "restless", "darr", "darra", "bhaya", "chinta",
        "डर", "डरा", "भय", "चिंता", "घबराहट"
    ],
    "surprise": [
        "surprise", "shocked", "amazed", "astonished", "startled", "stunned", "speechless", "bewildered", "flabbergasted",
        "impressed", "baffled", "taken aback", "unexpected", "staggered", "dumbfounded", "incredulous", "perplexed",
        "astounded", "wondered", "wow", "hairat", "ajeeb",
        "आश्चर्य", "अजीब", "वौव", "हैरान"
    ],
    "disgust": [
        "disgust", "repulsed", "rubbish", "gross", "nauseated", "disgusted", "sickened", "revolted", "offensive",
        "detestable", "abhorrent", "loathsome", "repellent", "vile", "nasty", "filthy", "repugnant", "distasteful",
        "yuck", "horrid", "odious", "nauseous", "appalled", "unclean", "unpleasant", "ghinn", "bekaar",
        "घृणा", "बेकार", "गंदा", "छी"
    ]
}

question_words = ["who", "what", "when", "where", "why", "how", "is", "are", "do", "does", "did", "can", "could", "will", "would", "should"]

def detect_emotions(text: str) -> list:
    tokens = nltk.word_tokenize(text.lower())
    detected = set()
    negations = {"not", "no", "never", "n't", "dont", "don't", "didn't", "doesn't", "isn't", "wasn't", "aren't", "cannot"}

    for emotion, keywords in emotion_lexicon.items():
        for word in keywords:
            if word in tokens:
                word_index = tokens.index(word)
                window_start = max(0, word_index - 3)
                window = tokens[window_start:word_index]
                if any(neg in window for neg in negations):
                    if emotion == "happy": detected.add("sad")
                    elif emotion == "sad": detected.add("happy")
                    else: detected.add(emotion)
                else:
                    detected.add(emotion)

    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity) # type: ignore

    if not detected:
        if polarity > 0.2: detected.add("happy")
        elif polarity < -0.2: detected.add("sad")
        else: detected.add("neutral")
    else:
        if polarity > 0.5: detected.add("happy")
        elif polarity < -0.5: detected.add("sad")

    return list(detected) if detected else ["neutral"]

def is_questioning(text: str) -> bool:
    if '?' in text: return True
    tokens = nltk.word_tokenize(text.lower())
    return any(word in tokens for word in question_words)

def analyze_tone(text: str, pitch: Optional[float] = None) -> dict:
    blob = TextBlob(text)
    sentiment = blob.sentiment
    polarity = float(sentiment.polarity) # type: ignore
    subjectivity = float(sentiment.subjectivity) # type: ignore

    vader_scores = sia.polarity_scores(text)
    compound = vader_scores['compound']
    emotions = detect_emotions(text)
    questioning = is_questioning(text)

    if has_non_ascii(text) or (polarity == 0 and len(text.split()) > 3):
        llm_data = llm_sentiment_analyzer(text)
        if llm_data:
            return {
                "sentiment": {
                    "polarity": llm_data.get("polarity", polarity),
                    "subjectivity": subjectivity,
                    "compound_score": vader_scores['compound']
                },
                "emotions": list(set(emotions + llm_data.get("emotions", []))),
                "overall_mood": llm_data.get("overall_mood", "neutral"),
                "is_questioning": questioning,
                "pitch": pitch,
                "multilingual": True
            }

    overall_mood = "neutral"
    if polarity > 0.1: overall_mood = "positive"
    elif polarity < -0.1: overall_mood = "negative"

    if pitch is not None:
        if pitch < 100.0:
            overall_mood = "negative"
            if "sad" not in emotions: emotions.append("sad")
        elif pitch > 200.0:
            overall_mood = "positive"
            if "happy" not in emotions: emotions.append("happy")

    return {
        "sentiment": {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "compound_score": compound
        },
        "emotions": emotions,
        "overall_mood": overall_mood,
        "is_questioning": questioning,
        "pitch": pitch,
        "multilingual": False
    }