from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk 

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Expanded emotion lexicon for keyword-based detection including negations and dislike-related words

emotion_lexicon = {
    "happy": [
        "happy", "joy", "joyful", "excited", "delighted", "pleased", "cheerful", "glad", "love", "like", "enjoy",
        "content", "satisfied", "thrilled", "blissful", "grateful", "ecstatic", "optimistic", "hopeful", "proud",
        "radiant", "smiling", "enthusiastic", "elated", "overjoyed"
    ],
    "sad": [
        "sad", "unhappy", "depressed", "sorrow", "grief", "miserable", "down", "dislike", "disappointed", "regret",
        "lonely", "heartbroken", "hopeless", "tearful", "gloomy", "melancholy", "blue", "discouraged", "hurt",
        "weary", "lost", "empty", "abandoned", "devastated", "pained"
    ],
    "angry": [
        "angry", "mad", "furious", "irritated", "annoyed", "rage", "frustrated", "hate", "resentful", "disgusted",
        "offended", "outraged", "hostile", "agitated", "bitter", "enraged", "cross", "snappy", "fuming", "provoked",
        "irate", "vengeful", "exasperated", "livid", "infuriated"
    ],
    "fear": [
        "fear", "scared", "afraid", "terrified", "anxious", "worried", "nervous", "panic", "frightened", "alarmed",
        "uneasy", "insecure", "shaky", "tense", "apprehensive", "paranoid", "timid", "dread", "phobic", "startled",
        "distressed", "hesitant", "shocked", "petrified", "restless"
    ],
    "surprise": [
        "surprise", "shocked", "amazed", "astonished", "startled", "stunned", "speechless", "bewildered", "flabbergasted",
        "impressed", "baffled", "taken aback", "unexpected", "staggered", "dumbfounded", "incredulous", "perplexed",
        "astounded", "wondered", "wow"
    ],
    "disgust": [
        "disgust", "repulsed", "rubbish", "gross", "nauseated", "disgusted", "sickened", "revolted", "offensive",
        "detestable", "abhorrent", "loathsome", "repellent", "vile", "nasty", "filthy", "repugnant", "distasteful",
        "yuck", "horrid", "odious", "nauseous", "appalled", "unclean", "unpleasant"
    ]
}


question_words = ["who", "what", "when", "where", "why", "how", "is", "are", "do", "does", "did", "can", "could", "will", "would", "should"]

def detect_emotions(text: str) -> list:
    """
    Detect emotions based on keyword matching, negation handling, and polarity score.
    More sensitive to negative and positive emotions using polarity thresholds.
    """
    tokens = nltk.word_tokenize(text.lower())
    detected = set()

    # Simple negation words
    negations = {"not", "no", "never", "n't", "dont", "don't", "didn't", "doesn't", "isn't", "wasn't", "aren't", "cannot"}

    # Keyword-based detection with negation handling
    for emotion, keywords in emotion_lexicon.items():
        for word in keywords:
            if word in tokens:
                # Check for negation within 3 words before the keyword
                word_index = tokens.index(word)
                window_start = max(0, word_index - 3)
                window = tokens[window_start:word_index]
                if any(neg in window for neg in negations):
                    # If negation found, invert emotion if applicable
                    if emotion == "happy":
                        detected.add("sad")
                    elif emotion == "sad":
                        detected.add("happy")
                    else:
                        detected.add(emotion)
                else:
                    detected.add(emotion)

    # Polarity-based detection enhancement
    blob = TextBlob(text)
    sentiment = blob.sentiment
    polarity = sentiment.polarity  # type: ignore

    # If no keywords detected, use polarity to infer emotion
    if not detected:
        if polarity > 0.2:
            detected.add("happy")
        elif polarity < -0.2:
            detected.add("sad")
        else:
            detected.add("neutral")
    else:
        # If keywords detected but polarity is strongly negative or positive, add corresponding emotion
        if polarity > 0.5:
            detected.add("happy")
        elif polarity < -0.5:
            detected.add("sad")

    return list(detected) if detected else ["neutral"]

def is_questioning(text: str) -> bool:
    """
    Detect if the text is a question.
    """
    if '?' in text:
        return True
    tokens = nltk.word_tokenize(text.lower())
    return any(word in tokens for word in question_words)

def analyze_tone(text: str, pitch: float | None  = None) -> dict:
    """
    Enhanced tone & sentiment analysis for therapeutic context.
    Accepts optional pitch (Hz) to enhance tone sensitivity.
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment
    polarity = sentiment.polarity  # type: ignore
    subjectivity = sentiment.subjectivity  # type: ignore

    # VADER compound score
    vader_scores = sia.polarity_scores(text)
    compound = vader_scores['compound']

    # Detect emotions
    emotions = detect_emotions(text)

    # Overall mood based on polarity
    if polarity > 0:
        overall_mood = "positive"
    elif polarity < 0:
        overall_mood = "negative"
    else:
        overall_mood = "neutral"

    # Questioning detection
    questioning = is_questioning(text)

    # Enhance mood and emotions based on pitch if provided
    if pitch is not None:
        # Define pitch thresholds (Hz) for low and high pitch - these can be tuned
        low_pitch_threshold = 100.0
        high_pitch_threshold = 200.0

        if pitch < low_pitch_threshold:
            # Low pitch may indicate sadness or calmness
            overall_mood = "negative"
            if "sad" not in emotions:
                emotions.append("sad")
        elif pitch > high_pitch_threshold:
            # High pitch may indicate excitement or happiness
            overall_mood = "positive"
            if "happy" not in emotions:
                emotions.append("happy")

    return {
        "sentiment": {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "compound_score": compound
        },
        "emotions": emotions,
        "overall_mood": overall_mood,
        "is_questioning": questioning,
        "pitch": pitch
    }
