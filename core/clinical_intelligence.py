"""
core/clinical_intelligence.py
──────────────────────────────────────────────────────────────────────────────
Lightweight Clinical Intelligence Layer for AGI-119 Therapist
Uses numpy + pandas for real statistical analytics.
Runs entirely locally — NO external API calls.

Modules:
  1.  EmotionClassifier          - Keyword + weight emotion scoring
  2.  StressorThemeExtractor     - Detects 10 stressor categories
  3.  MedicalProfileExtractor    - Regex/keyword medical data mining (no LLM)
  4.  SafetyModule               - Risk detection + emergency guidance
  5.  SessionAnalyticsStore      - SQLite persistence layer
  6.  TherapyAnalyticsEngine     - numpy/pandas statistical computations
  7.  DashboardDataGenerator     - Facade; main entry point for app.py
──────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import uuid
import sqlite3
import threading
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    print("[CLINICAL] pandas not found – falling back to pure numpy for trends.")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  EMOTION CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_PATTERNS: Dict[str, Dict] = {
    "anxiety": {
        "keywords": [
            "anxious","anxiety","worried","worry","nervous","panic","fear",
            "scared","dread","apprehensive","restless","tense","overwhelmed",
            "phobia","uneasy","on edge","heart racing","can't breathe","hyperventilate"
        ],
        "weight": 1.0
    },
    "depression": {
        "keywords": [
            "depressed","depression","hopeless","worthless","empty","sad","sadness",
            "numb","low","grief","melancholy","crying","tears","dark thoughts",
            "no point","meaningless","can't feel","feel nothing","no energy",
            "lost interest","don't enjoy anything","no motivation"
        ],
        "weight": 1.0
    },
    "stress": {
        "keywords": [
            "stressed","stress","pressure","burden","loaded","exhausted","drained",
            "too much","can't cope","burned out","burnout","deadline","workload",
            "no time","falling behind","overwhelmed","responsibilities"
        ],
        "weight": 0.9
    },
    "loneliness": {
        "keywords": [
            "lonely","alone","isolated","no one","disconnected","left out",
            "invisible","nobody cares","no friends","abandoned","rejected",
            "socially isolated","don't belong","excluded"
        ],
        "weight": 0.9
    },
    "anger": {
        "keywords": [
            "angry","anger","furious","rage","irritated","frustrated","annoyed",
            "mad","resentment","bitter","hatred","hostile","aggressive","outraged",
            "livid","can't stand"
        ],
        "weight": 0.8
    },
    "trauma": {
        "keywords": [
            "trauma","traumatic","flashback","nightmare","ptsd","abuse","violated",
            "assault","attacked","harassed","victim","can't forget","triggers",
            "haunted","intrusive thoughts","reliving"
        ],
        "weight": 1.0
    },
    "neutral": {
        "keywords": [
            "okay","fine","alright","normal","average","so-so","not bad",
            "managing","getting by","okay I guess","just existing"
        ],
        "weight": 0.4
    },
    "positive": {
        "keywords": [
            "happy","joy","excited","grateful","hopeful","better","improving",
            "good","great","wonderful","positive","motivated","content","peaceful",
            "calm","relieved","proud","accomplished","energized","thriving"
        ],
        "weight": 0.7
    },
}

NEGATIVE_EMOTIONS = {"anxiety","depression","stress","loneliness","anger","trauma"}
POSITIVE_EMOTIONS = {"positive","neutral"}


class EmotionClassifier:
    def classify(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for emotion, cfg in EMOTION_PATTERNS.items():
            hits = sum(1 for kw in cfg["keywords"] if kw in text_lower)
            if hits:
                raw = min(hits, 6) / 6.0
                scores[emotion] = round(raw * cfg["weight"], 4)

        if not scores:
            return {"emotion_label": "neutral", "confidence_score": 0.30,
                    "all_scores": {"neutral": 0.30}}

        dominant = max(scores, key=scores.get)
        confidence = min(scores[dominant] + 0.28, 1.0)
        return {
            "emotion_label": dominant,
            "confidence_score": round(confidence, 4),
            "all_scores": scores
        }

    def is_negative(self, label: str) -> bool:
        return label in NEGATIVE_EMOTIONS


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STRESSOR THEME EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

STRESSOR_THEMES: Dict[str, List[str]] = {
    "academic_pressure": [
        "exam","study","university","college","grades","assignment","thesis",
        "deadline","professor","fail","test","homework","school","semester",
        "gpa","marks","academic","lecture","classes"
    ],
    "financial_stress": [
        "money","debt","broke","afford","bills","rent","loan","financial",
        "poverty","unemployed","job loss","can't pay","salary","income",
        "expenses","budget","savings","bankrupt"
    ],
    "relationship_issues": [
        "breakup","divorce","cheating","partner","boyfriend","girlfriend",
        "husband","wife","relationship","fight","argument","toxic","heartbreak",
        "separated","trust issues","manipulation","jealousy","infidelity"
    ],
    "insomnia": [
        "sleep","insomnia","can't sleep","awake at night","tired","fatigue",
        "exhausted","sleeping pills","nightmares","lying awake","sleep schedule",
        "oversleeping","no rest","restless night","3am","4am"
    ],
    "burnout": [
        "burnout","burned out","no energy","drained","overworked","productivity",
        "no motivation","lost interest","tired of everything","numb to work",
        "can't focus","distracted","procrastinating","unmotivated"
    ],
    "grief_loss": [
        "death","died","passed away","funeral","loss","grief","mourning",
        "miss them","gone","bereaved","parent died","friend died","bereavement",
        "widow","orphan","memorial"
    ],
    "social_anxiety": [
        "social anxiety","crowd","people","judgment","embarrassed","shy",
        "nervous around people","avoid social","public speaking","introvert",
        "social situations","social media","comparison","fear of judgment"
    ],
    "trauma_abuse": [
        "trauma","abuse","assault","violated","ptsd","childhood","victim",
        "flashback","nightmare","survivor","domestic violence","harassment","rape"
    ],
    "self_esteem": [
        "worthless","useless","ugly","hate myself","not good enough",
        "inadequate","inferior","failure","nobody likes","rejected","imposter",
        "confidence","self-doubt","insecure","body image","low self-worth"
    ],
    "work_career": [
        "work","career","boss","coworker","office","promotion","fired","quit",
        "job","workplace","toxic boss","overtime","work-life balance","remote work",
        "layoff","interview","rejection"
    ],
}


class StressorThemeExtractor:
    def extract_themes(self, text: str) -> List[str]:
        t = text.lower()
        return [theme for theme, kws in STRESSOR_THEMES.items()
                if any(kw in t for kw in kws)]

    def theme_frequency(self, texts: List[str]) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for text in texts:
            for theme in self.extract_themes(text):
                freq[theme] = freq.get(theme, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    def dominant_theme(self, texts: List[str]) -> Optional[str]:
        freq = self.theme_frequency(texts)
        return max(freq, key=freq.get) if freq else None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MEDICAL PROFILE EXTRACTOR  (regex/keyword — no LLM needed)
#     Parses raw transcript text and extracts structured medical/personal data
# ─────────────────────────────────────────────────────────────────────────────

_MEDICAL_PATTERNS = {
    "diagnosed_conditions": [
        r"\b(depression|anxiety disorder|bipolar|schizophrenia|ptsd|ocd|adhd|"
        r"autism|eating disorder|borderline personality|panic disorder|"
        r"social anxiety disorder|generalized anxiety|major depressive disorder"
        r"|cyclothymia|dysthymia)\b"
    ],
    "medications": [
        r"\b(sertraline|fluoxetine|lexapro|zoloft|prozac|xanax|valium|"
        r"lorazepam|clonazepam|lithium|risperdal|abilify|wellbutrin|effexor|"
        r"citalopram|escitalopram|paroxetine|venlafaxine|bupropion|quetiapine|"
        r"olanzapine|antidepressant|antipsychotic|sleeping pill|melatonin|"
        r"medication|prescription|mg|pill|tablet)\b"
    ],
    "sleep_issues": [
        r"\b(insomnia|can'?t sleep|sleep disorder|sleep apnea|nightmare|"
        r"sleep deprivation|hypersomnia|oversleeping|restless sleep|"
        r"disrupted sleep|sleep schedule|up all night|awake at \d)\b"
    ],
    "anxiety_patterns": [
        r"\b(panic attack|racing heart|shortness of breath|chest tightness|"
        r"hyperventilat|phobia|trigger|avoidance|compulsion|intrusive thought)\b"
    ],
    "substance_use": [
        r"\b(alcohol|drinking|drunk|wine|beer|vodka|whiskey|cannabis|weed|"
        r"marijuana|cocaine|heroin|opioid|drugs|substance|addicted|addiction|"
        r"smoking|cigarette|vaping|relapse|sobriety)\b"
    ],
}

_IDENTITY_PATTERNS = {
    "name": [r"(?:my name is|i'?m called|call me)\s+([A-Z][a-z]+)"],
    "age": [r"(?:i'?m|i am)\s+(\d{1,2})\s+years?\s+old", r"(?:age|aged)\s+(\d{1,2})"],
    "profession": [
        r"(?:i'?m a|i work as a|i am a|my job is|working as)\s+"
        r"(student|doctor|engineer|teacher|lawyer|nurse|designer|developer|"
        r"manager|consultant|professor|entrepreneur|writer|artist|chef|pilot|"
        r"pharmacist|accountant|analyst|researcher)\b"
    ],
    "family": [
        r"\b(?:my (mother|father|mom|dad|sister|brother|wife|husband|son|daughter|"
        r"parents|children|family|boyfriend|girlfriend|partner))\b"
    ],
}

_LIFE_EVENT_PATTERNS = [
    r"\b(breakup|broke up|divorce|separated|lost my job|fired|laid off|"
    r"death|died|passed away|failed|accident|diagnosed|hospitalized|"
    r"moved|relocated|graduated|dropped out|fight with|argument with)\b"
]

_RISK_PATTERNS = {
    "suicidal_ideation": [
        r"\b(suicide|suicidal|kill myself|end my life|take my life|"
        r"want to die|don'?t want to live|life is not worth|rather be dead|"
        r"end it all|no reason to live|die by suicide)\b"
    ],
    "self_harm": [
        r"\b(self[- ]?harm|cut myself|hurt myself|harm myself|self[- ]?injur|"
        r"scratch|burn myself|hit myself|blade|razor|cut my arm)\b"
    ],
    "intent_to_harm_others": [
        r"\b(hurt someone|kill someone|harm others|threaten|murder|attack|"
        r"plan to hurt|violence against)\b"
    ],
    "substance_crisis": [
        r"\b(overdose|opioid crisis|drug overdose|alcohol poisoning|"
        r"took too many pills|swallowed pills)\b"
    ],
}


class MedicalProfileExtractor:
    """
    Extracts structured medical + personal data from conversation transcripts
    using regex patterns. No LLM required — runs instantly, fully offline.
    """

    def extract_medical(self, texts: List[str]) -> Dict[str, List[str]]:
        full_text = " ".join(texts).lower()
        results: Dict[str, List[str]] = {}

        for category, patterns in _MEDICAL_PATTERNS.items():
            found = set()
            for pat in patterns:
                for m in re.finditer(pat, full_text, re.IGNORECASE):
                    found.add(m.group(0).strip().title())
            if found:
                results[category] = sorted(found)

        return results

    def extract_identity(self, texts: List[str]) -> Dict[str, Optional[str]]:
        full_text = " ".join(texts)
        result = {}
        for field, patterns in _IDENTITY_PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, full_text, re.IGNORECASE)
                if m:
                    result[field] = m.group(1).strip().title()
                    break
        return result

    def extract_life_events(self, texts: List[str]) -> List[str]:
        full_text = " ".join(texts).lower()
        events = set()
        for pat in _LIFE_EVENT_PATTERNS:
            for m in re.finditer(pat, full_text, re.IGNORECASE):
                events.add(m.group(0).strip().title())
        return sorted(events)

    def extract_risk_indicators(self, texts: List[str]) -> Dict[str, List[str]]:
        full_text = " ".join(texts).lower()
        risks: Dict[str, List[str]] = {}
        for cat, patterns in _RISK_PATTERNS.items():
            found = set()
            for pat in patterns:
                for m in re.finditer(pat, full_text, re.IGNORECASE):
                    found.add(m.group(0).strip().lower())
            if found:
                risks[cat] = sorted(found)
        return risks

    def build_full_report(self, user_id: str, transcripts: List[str]) -> Dict[str, Any]:
        """
        Returns a structured medical + personal report dict ready for the dashboard.
        """
        medical = self.extract_medical(transcripts)
        identity = self.extract_identity(transcripts)
        events = self.extract_life_events(transcripts)
        risks = self.extract_risk_indicators(transcripts)

        medical_history = []
        if identity.get("name"):
            medical_history.append(f"👤 Name: {identity['name']}")
        if identity.get("age"):
            medical_history.append(f"🎂 Age: {identity['age']}")
        if identity.get("profession"):
            medical_history.append(f"💼 Profession: {identity['profession']}")

        for cond in medical.get("diagnosed_conditions", []):
            medical_history.append(f"🏥 Diagnosed: {cond}")
        for med in medical.get("medications", []):
            medical_history.append(f"💊 Medication: {med}")
        for si in medical.get("sleep_issues", []):
            medical_history.append(f"😴 Sleep Issue: {si}")
        for ap in medical.get("anxiety_patterns", []):
            medical_history.append(f"⚡ Anxiety Pattern: {ap}")
        for su in medical.get("substance_use", []):
            medical_history.append(f"🔴 Substance Mention: {su}")

        personal_profile = []
        for fam in medical.get("family", []) if "family" in medical else []:
            personal_profile.append(f"👨‍👩‍👧 Family: {fam}")
        for ev in events:
            personal_profile.append(f"📌 Life Event: {ev}")

        risk_flags = []
        for cat, instances in risks.items():
            label = cat.replace("_", " ").title()
            risk_flags.append(f"⚠️ {label}: {', '.join(instances[:3])}")

        return {
            "user_id": user_id,
            "identity": identity,
            "medical_history": medical_history or ["No medical records detected yet."],
            "personal_profile": personal_profile or ["No personal profile data detected yet."],
            "risk_flags": risk_flags,
            "raw": {
                "medical": medical,
                "events": events,
                "risks": risks,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAFETY MODULE
# ─────────────────────────────────────────────────────────────────────────────

EMERGENCY_RESOURCES = {
    "🆘 Suicide & Crisis Lifeline (US)": "Call or text **988**",
    "💬 Crisis Text Line": "Text HOME to **741741**",
    "🇮🇳 iCall (India)": "**9152987821**",
    "🇮🇳 Vandrevala Foundation (India)": "**1860-2662-345** (24/7)",
    "🌍 International Crisis Centres": "https://www.iasp.info/resources/Crisis_Centres/",
    "🚨 Emergency Services": "Call **911** (US) / **112** (EU) / **100** (India)",
    "🏥 NIMHANS Helpline (India)": "**080-46110007**",
    "💙 Fortis Stress Helpline (India)": "**8376804102**",
}

SAFETY_TIPS = [
    "Reach out to a trusted friend or family member right now.",
    "Move to a safe, public space if you feel in danger.",
    "Remove access to anything that could be used for self-harm.",
    "Call a crisis line — trained counsellors are available 24/7.",
    "Go to the nearest emergency room if the urge becomes overwhelming.",
    "Practice grounding: name 5 things you can see, 4 you can touch.",
]


class SafetyModule:
    def analyze(self, text: str) -> Dict[str, Any]:
        t = text.lower()
        detected: Dict[str, List[str]] = {}

        for cat, patterns in _RISK_PATTERNS.items():
            hits = set()
            for pat in patterns:
                for m in re.finditer(pat, t, re.IGNORECASE):
                    hits.add(m.group(0))
            if hits:
                detected[cat] = sorted(hits)

        risk_flag = bool(detected)
        severity = "NONE"
        if "suicidal_ideation" in detected:
            severity = "HIGH"
        elif detected:
            severity = "MODERATE"

        result: Dict[str, Any] = {
            "risk_flag": risk_flag,
            "risk_categories": list(detected.keys()),
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }

        if risk_flag:
            result["emergency_resources"] = EMERGENCY_RESOURCES
            result["safety_tips"] = SAFETY_TIPS
            result["safety_message"] = (
                "You are not alone — your feelings are valid and help is available. "
                "Please reach out to a crisis professional immediately."
            )

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SESSION ANALYTICS STORE  (SQLite)
# ─────────────────────────────────────────────────────────────────────────────

class SessionAnalyticsStore:
    def __init__(self, db_path: str = "session_analytics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS session_analytics (
                session_id    TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                emotion       TEXT,
                confidence    REAL,
                themes        TEXT,
                risk_flag     INTEGER DEFAULT 0,
                mood_score    REAL DEFAULT 0.0,
                message_count INTEGER DEFAULT 0,
                transcript    TEXT,
                timestamp     TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id         TEXT PRIMARY KEY,
                session_id TEXT,
                user_id    TEXT NOT NULL,
                categories TEXT,
                severity   TEXT,
                timestamp  TEXT
            )
        """)
        conn.commit()
        self._migrate_db(conn)
        conn.close()

    def _migrate_db(self, conn):
        """Safely add columns that may be missing from older DB versions."""
        c = conn.cursor()
        try:
            c.execute("SELECT transcript FROM session_analytics LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE session_analytics ADD COLUMN transcript TEXT")
            conn.commit()
        try:
            c.execute("SELECT mood_score FROM session_analytics LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE session_analytics ADD COLUMN mood_score REAL DEFAULT 0.0")
            conn.commit()
        try:
            c.execute("SELECT message_count FROM session_analytics LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE session_analytics ADD COLUMN message_count INTEGER DEFAULT 0")
            conn.commit()

    def upsert_session(self, session_id: str, user_id: str, emotion: str,
                       confidence: float, themes: List[str], risk_flag: bool,
                       mood_score: float = 0.0, message_count: int = 0,
                       transcript: str = ""):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO session_analytics
            (session_id, user_id, emotion, confidence, themes,
             risk_flag, mood_score, message_count, transcript, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (session_id, user_id, emotion, confidence,
              json.dumps(themes), int(risk_flag), mood_score,
              message_count, transcript[:2000],
              datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def store_risk_alert(self, session_id: str, user_id: str,
                         categories: List[str], severity: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO risk_alerts (id, session_id, user_id, categories, severity, timestamp)
            VALUES (?,?,?,?,?,?)
        """, (str(uuid.uuid4()), session_id, user_id,
              json.dumps(categories), severity, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_user_sessions(self, user_id: str, limit: int = 200) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT * FROM session_analytics
            WHERE user_id=? ORDER BY timestamp DESC LIMIT ?
        """, (user_id, limit))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_user_transcripts(self, user_id: str, limit: int = 50) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT transcript FROM session_analytics
            WHERE user_id=? AND transcript IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (user_id, limit))
        rows = [r[0] for r in c.fetchall() if r[0]]
        conn.close()
        return rows

    def get_risk_alerts(self, user_id: str, limit: int = 50) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT * FROM risk_alerts
            WHERE user_id=? ORDER BY timestamp DESC LIMIT ?
        """, (user_id, limit))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# 6.  THERAPY ANALYTICS ENGINE  (numpy + pandas)
# ─────────────────────────────────────────────────────────────────────────────

class TherapyAnalyticsEngine:
    """Pure statistical analytics — numpy for all number crunching."""

    # ── helpers ────────────────────────────────────────────────────────────
    def _df(self, sessions: List[Dict]):
        """Convert sessions list → pandas DataFrame (or lightweight dict)."""
        if not sessions:
            return None
        if _HAS_PANDAS:
            df = pd.DataFrame(sessions)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["date"] = df["timestamp"].dt.date.astype(str)
            return df
        return sessions  # fallback

    def _confidences(self, sessions: List[Dict]) -> np.ndarray:
        return np.array([s.get("confidence", 0.5) for s in sessions], dtype=float)

    def _is_negative(self, emotion: str) -> bool:
        return emotion in NEGATIVE_EMOTIONS

    def _is_positive(self, emotion: str) -> bool:
        return emotion in POSITIVE_EMOTIONS

    # ── KPI computations ───────────────────────────────────────────────────
    def compute_therapy_progress_score(self, sessions: List[Dict]) -> float:
        """TPS = (positive / total) - (high_anxiety / total)  →  [-1, +1]"""
        if not sessions:
            return 0.0
        n = len(sessions)
        confs = self._confidences(sessions)
        emotions = [s.get("emotion", "neutral") for s in sessions]

        pos_mask   = np.array([self._is_positive(e) for e in emotions])
        neg_mask   = np.array([self._is_negative(e) for e in emotions])
        high_anx   = neg_mask & (confs > 0.60)

        tps = (pos_mask.sum() / n) - (high_anx.sum() / n)
        return float(np.clip(tps, -1.0, 1.0).round(4))

    def compute_emotional_volatility_index(self, sessions: List[Dict]) -> float:
        """EVI = variance of confidence scores over last 7 sessions."""
        recent = sorted(sessions, key=lambda x: x.get("timestamp",""), reverse=True)[:7]
        if len(recent) < 2:
            return 0.0
        confs = self._confidences(recent)
        return float(round(float(np.var(confs)), 4))

    def compute_mood_stability_index(self, sessions: List[Dict]) -> float:
        """MSI = 1 - 4*EVI  clamped to [0,1]. Higher = more stable."""
        evi = self.compute_emotional_volatility_index(sessions)
        return float(round(max(0.0, 1.0 - evi * 4), 4))

    def compute_dominant_negative_pct(self, sessions: List[Dict]) -> float:
        if not sessions:
            return 0.0
        neg = sum(1 for s in sessions if self._is_negative(s.get("emotion","neutral")))
        return round(neg / len(sessions), 4)

    # ── Time-series ────────────────────────────────────────────────────────
    def anxiety_trend(self, sessions: List[Dict]) -> List[Dict]:
        ordered = sorted(sessions, key=lambda x: x.get("timestamp",""))
        trend = []
        for s in ordered:
            score = 0.0
            if s.get("emotion") in {"anxiety","depression","stress","trauma"}:
                score = s.get("confidence", 0.0)
            trend.append({"date": s.get("timestamp","")[:10], "score": round(score, 3)})
        return trend

    def improvement_trend(self, sessions: List[Dict]) -> List[Dict]:
        """Daily rolling positive-session ratio."""
        grouped: Dict[str, List] = {}
        for s in sessions:
            day = s.get("timestamp","")[:10]
            grouped.setdefault(day, []).append(s)

        trend = []
        for day in sorted(grouped):
            day_s = grouped[day]
            pos = sum(1 for s in day_s if self._is_positive(s.get("emotion","neutral")))
            trend.append({"date": day, "score": round(pos / len(day_s), 3)})
        return trend

    def emotion_distribution_over_time(self, sessions: List[Dict]) -> List[Dict]:
        """Returns per-session {date, emotion, confidence} for heatmap."""
        ordered = sorted(sessions, key=lambda x: x.get("timestamp",""))
        return [{"date": s.get("timestamp","")[:10],
                 "emotion": s.get("emotion","neutral"),
                 "score": s.get("confidence", 0.5)}
                for s in ordered]

    def moving_average_anxiety(self, sessions: List[Dict], window: int = 3) -> List[Dict]:
        """Smoothed anxiety scores using numpy convolve."""
        trend = self.anxiety_trend(sessions)
        if len(trend) < window:
            return trend
        scores = np.array([t["score"] for t in trend], dtype=float)
        kernel = np.ones(window) / window
        smoothed = np.convolve(scores, kernel, mode="same")
        return [{"date": trend[i]["date"], "score": round(float(smoothed[i]), 3)}
                for i in range(len(trend))]

    # ── Topic utilities ────────────────────────────────────────────────────
    def topic_frequency(self, sessions: List[Dict]) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for s in sessions:
            try:
                themes_list = json.loads(s.get("themes","[]"))
            except Exception:
                themes_list = []
            for t in themes_list:
                freq[t] = freq.get(t, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    def dominant_stressor(self, sessions: List[Dict]) -> str:
        freq = self.topic_frequency(sessions)
        return max(freq, key=freq.get) if freq else "general_stress"

    def categorize_session(self, emotion: str, confidence: float) -> str:
        if emotion in {"anxiety","depression","trauma"} and confidence > 0.50:
            return "anxiety"
        if self._is_positive(emotion):
            return "positive"
        return "neutral"

    # ── Statistical summary ────────────────────────────────────────────────
    def statistical_summary(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Returns mean/std/min/max of confidence scores using numpy."""
        if not sessions:
            return {}
        confs = self._confidences(sessions)
        return {
            "mean_confidence": float(round(np.mean(confs), 4)),
            "std_confidence":  float(round(np.std(confs), 4)),
            "min_confidence":  float(round(np.min(confs), 4)),
            "max_confidence":  float(round(np.max(confs), 4)),
            "median_confidence": float(round(np.median(confs), 4)),
            "session_count":   len(sessions),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DASHBOARD DATA GENERATOR  (Main Facade)
# ─────────────────────────────────────────────────────────────────────────────

class DashboardDataGenerator:
    def __init__(self, db_path: str = "session_analytics.db"):
        self.emotion_clf   = EmotionClassifier()
        self.theme_ext     = StressorThemeExtractor()
        self.medical_ext   = MedicalProfileExtractor()
        self.safety_mod    = SafetyModule()
        self.store         = SessionAnalyticsStore(db_path=db_path)
        self.engine        = TherapyAnalyticsEngine()

    # ── Process & persist one session ─────────────────────────────────────
    def process_session(self, user_id: str, session_id: str,
                        transcript: str, message_count: int = 0):
        try:
            result    = self.emotion_clf.classify(transcript)
            emotion   = result["emotion_label"]
            confidence = result["confidence_score"]
            themes    = self.theme_ext.extract_themes(transcript)
            safety    = self.safety_mod.analyze(transcript)
            risk_flag = safety["risk_flag"]
            mood_score = confidence if emotion in POSITIVE_EMOTIONS else -confidence

            self.store.upsert_session(
                session_id=session_id,
                user_id=user_id,
                emotion=emotion,
                confidence=confidence,
                themes=themes,
                risk_flag=risk_flag,
                mood_score=mood_score,
                message_count=message_count,
                transcript=transcript,
            )

            if risk_flag:
                self.store.store_risk_alert(
                    session_id=session_id,
                    user_id=user_id,
                    categories=safety["risk_categories"],
                    severity=safety.get("severity","MODERATE"),
                )

            print(f"[CLINICAL] ✓ {session_id} | emotion={emotion} "
                  f"conf={confidence:.2f} themes={themes} risk={risk_flag}")
        except Exception as e:
            print(f"[CLINICAL ERROR] process_session: {e}")

    def process_session_async(self, user_id: str, session_id: str,
                              transcript: str, message_count: int = 0):
        threading.Thread(
            target=self.process_session,
            args=(user_id, session_id, transcript, message_count),
            daemon=True
        ).start()

    # ── Real-time safety check ─────────────────────────────────────────────
    def check_safety(self, text: str) -> Dict[str, Any]:
        return self.safety_mod.analyze(text)

    # ── Dashboard JSON ─────────────────────────────────────────────────────
    def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        try:
            sessions   = self.store.get_user_sessions(user_id)
            alerts_raw = self.store.get_risk_alerts(user_id)

            # numpy/pandas analytics
            tps  = self.engine.compute_therapy_progress_score(sessions)
            evi  = self.engine.compute_emotional_volatility_index(sessions)
            msi  = self.engine.compute_mood_stability_index(sessions)
            stats = self.engine.statistical_summary(sessions)

            anxiety_trend     = self.engine.anxiety_trend(sessions)
            smoothed_anxiety  = self.engine.moving_average_anxiety(sessions, window=3)
            imp_trend         = self.engine.improvement_trend(sessions)
            topic_freq        = self.engine.topic_frequency(sessions)
            dom_stressor      = self.engine.dominant_stressor(sessions)
            most_frequent     = next(iter(topic_freq), "none")
            heatmap_data      = self.engine.emotion_distribution_over_time(sessions)

            categories = {"anxiety": 0, "positive": 0, "neutral": 0}
            for s in sessions:
                cat = self.engine.categorize_session(
                    s.get("emotion","neutral"), s.get("confidence",0.5)
                )
                categories[cat] = categories.get(cat, 0) + 1

            risk_alerts = []
            for a in alerts_raw[:15]:
                try:
                    cats = json.loads(a.get("categories","[]"))
                except Exception:
                    cats = []
                risk_alerts.append({
                    "session_id": a.get("session_id",""),
                    "severity":   a.get("severity","MODERATE"),
                    "categories": cats,
                    "timestamp":  a.get("timestamp",""),
                })

            return {
                "success": True,
                "user_id": user_id,
                "total_sessions": len(sessions),
                "anxiety_trend": anxiety_trend,
                "smoothed_anxiety_trend": smoothed_anxiety,
                "mood_stability_index": msi,
                "dominant_stressor": dom_stressor,
                "therapy_progress_score": tps,
                "emotional_volatility": evi,
                "most_frequent_topic": most_frequent,
                "improvement_trend": imp_trend,
                "risk_alerts": risk_alerts,
                "topic_frequency": topic_freq,
                "session_category_breakdown": categories,
                "heatmap_data": heatmap_data,
                "statistical_summary": stats,
                "emergency_resources": EMERGENCY_RESOURCES,
                "safety_tips": SAFETY_TIPS,
                "generated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "user_id": user_id}

    # ── Medical report (uses transcript history) ───────────────────────────
    def get_medical_report(self, user_id: str) -> Dict[str, Any]:
        """
        Pulls all stored transcripts for a user and runs MedicalProfileExtractor
        to produce a fully-populated medical + personal profile.
        """
        try:
            transcripts = self.store.get_user_transcripts(user_id)
            report = self.medical_ext.build_full_report(user_id, transcripts)
            report["success"] = True
            return report
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "medical_history": ["Error loading medical records."],
                "personal_profile": ["Error loading personal profile."],
                "risk_flags": []
            }

    # ── Memory context ─────────────────────────────────────────────────────
    def get_user_memory_context(self, user_id: str) -> Dict[str, Any]:
        try:
            sessions   = self.store.get_user_sessions(user_id, limit=50)
            topic_freq = self.engine.topic_frequency(sessions)
            emotion_counts: Dict[str, int] = {}
            for s in sessions:
                e = s.get("emotion","neutral")
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            return {
                "success": True,
                "user_id": user_id,
                "session_count": len(sessions),
                "dominant_emotions": emotion_counts,
                "topic_frequency": topic_freq,
                "dominant_stressor": self.engine.dominant_stressor(sessions),
                "therapy_progress_score": self.engine.compute_therapy_progress_score(sessions),
                "statistical_summary": self.engine.statistical_summary(sessions),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Risk alerts ────────────────────────────────────────────────────────
    def get_risk_alerts(self, user_id: str) -> Dict[str, Any]:
        try:
            alerts = self.store.get_risk_alerts(user_id)
            formatted = []
            for a in alerts:
                try:
                    cats = json.loads(a.get("categories","[]"))
                except Exception:
                    cats = []
                formatted.append({
                    "session_id": a.get("session_id"),
                    "severity":   a.get("severity"),
                    "categories": cats,
                    "timestamp":  a.get("timestamp"),
                })
            return {
                "success": True,
                "risk_alerts": formatted,
                "count": len(formatted),
                "emergency_resources": EMERGENCY_RESOURCES,
                "safety_tips": SAFETY_TIPS,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "risk_alerts": []}


# ── Singleton ──────────────────────────────────────────────────────────────
_engine: Optional[DashboardDataGenerator] = None

def get_clinical_engine() -> DashboardDataGenerator:
    global _engine
    if _engine is None:
        _engine = DashboardDataGenerator()
    return _engine
