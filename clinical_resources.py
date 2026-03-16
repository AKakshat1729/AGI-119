"""
Clinical Resources Database
Evidence-based therapeutic techniques, coping strategies, and mental health information
"""

CLINICAL_RESOURCES = {
    "coping_strategies": {
        "anxiety": [
            "Deep Breathing (4-7-8 technique): Inhale for 4, hold for 7, exhale for 8",
            "Progressive Muscle Relaxation: Tense and relax muscle groups systematically",
            "Grounding Technique (5-4-3-2-1): Identify 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste",
            "Box Breathing: Inhale for 4, hold for 4, exhale for 4, hold for 4",
            "Mindfulness Meditation: Focus on present moment without judgment",
            "Cognitive Reframing: Challenge anxious thoughts with evidence-based thinking"
        ],
        "depression": [
            "Behavioral Activation: Schedule pleasurable and meaningful activities",
            "Positive Self-Talk: Replace negative thoughts with realistic, compassionate ones",
            "Physical Exercise: 30 minutes of moderate activity 5x/week improves symptoms",
            "Sleep Hygiene: Maintain consistent sleep schedule and dark, cool bedroom",
            "Social Connection: Reach out to friends, family, or support groups",
            "Journaling: Write thoughts and feelings to process emotions"
        ],
        "stress": [
            "Time Management: Prioritize tasks using Eisenhower Matrix",
            "Yoga & Stretching: Reduces cortisol levels and muscle tension",
            "Journaling: Process stressors and identify patterns",
            "Boundary Setting: Learn to say no to unreasonable demands",
            "Nature Exposure: Spend 20+ minutes in natural settings daily",
            "Meditation: Practice 10-20 minutes daily for stress reduction"
        ],
        "insomnia": [
            "Sleep Restriction Therapy: Limit bed time to actual sleep time",
            "Stimulus Control: Use bed only for sleep and intimacy",
            "Muscle Relaxation: Progressive muscle relaxation before bed",
            "Cognitive Techniques: Challenge catastrophic thoughts about sleep",
            "Consistent Schedule: Same bedtime and wake time daily",
            "Limit Screens: No screens 1 hour before bed (blue light interferes)"
        ]
    },
    
    "therapeutic_approaches": {
        "CBT": {
            "full_name": "Cognitive Behavioral Therapy",
            "description": "Focuses on identifying and changing negative thought patterns and behaviors",
            "techniques": [
                "Thought Records: Document thoughts, feelings, situations, and evidence",
                "Behavioral Experiments: Test beliefs through real-world experiments",
                "Problem-Solving: Define problem, generate solutions, evaluate, implement",
                "Exposure: Gradual confrontation of feared situations or thoughts"
            ]
        },
        "DBT": {
            "full_name": "Dialectical Behavior Therapy",
            "description": "Combines CBT with mindfulness and acceptance",
            "techniques": [
                "Mindfulness: Observe thoughts without judgment",
                "Distress Tolerance: Survive crisis without making it worse",
                "Emotion Regulation: Understand and manage intense emotions",
                "Interpersonal Effectiveness: Communication and relationship skills"
            ]
        },
        "ACT": {
            "full_name": "Acceptance and Commitment Therapy",
            "description": "Accept difficult thoughts/feelings while committed to meaningful living",
            "techniques": [
                "Acceptance: Allow uncomfortable feelings without fighting",
                "Mindfulness: Present-moment awareness",
                "Values Clarification: Identify what matters most",
                "Committed Action: Take steps aligned with values"
            ]
        },
        "Psychodynamic": {
            "full_name": "Psychodynamic Therapy",
            "description": "Explores unconscious patterns and past experiences",
            "techniques": [
                "Free Association: Express thoughts freely",
                "Dream Analysis: Explore unconscious material",
                "Attachment Exploration: Examine relationship patterns",
                "Insight Development: Understand root causes of issues"
            ]
        }
    },
    
    "mental_health_conditions": {
        "GAD": {
            "name": "Generalized Anxiety Disorder",
            "symptoms": "Excessive worry about multiple areas, restlessness, fatigue, difficulty concentrating",
            "red_flags": "Interferes with work/school/relationships, persists >6 months",
            "self_help": [
                "Regular exercise (especially cardio)",
                "Meditation and mindfulness practice",
                "Limit caffeine and alcohol",
                "Establish daily routine",
                "Challenge worry thoughts"
            ],
            "when_to_seek_help": "If symptoms persist >2 weeks or significantly impact functioning"
        },
        "MDD": {
            "name": "Major Depressive Disorder",
            "symptoms": "Persistent sadness, loss of interest, sleep changes, fatigue, worthlessness",
            "red_flags": "Suicidal thoughts, severe functional impairment, lasting >2 weeks",
            "self_help": [
                "Maintain regular sleep schedule",
                "Exercise 5x/week",
                "Maintain social connections",
                "Practice self-compassion",
                "Engage in valued activities"
            ],
            "when_to_seek_help": "With suicidal thoughts, loss of functioning, or symptoms >2 weeks"
        },
        "PTSD": {
            "name": "Post-Traumatic Stress Disorder",
            "symptoms": "Intrusive memories, avoidance, negative mood, hyperarousal",
            "red_flags": "Following specific trauma, causes functional impairment",
            "self_help": [
                "Grounding techniques for flashbacks",
                "Gradual exposure to reminders",
                "Physical exercise",
                "Social support",
                "Consistent routine"
            ],
            "when_to_seek_help": "PTSD should be treated by trauma-informed professional"
        },
        "Panic": {
            "name": "Panic Disorder",
            "symptoms": "Sudden panic attacks, fear of recurrence, avoidance behavior",
            "red_flags": "Attacks interfere with daily life, agoraphobia developing",
            "self_help": [
                "Deep breathing exercises",
                "Progressive muscle relaxation",
                "Regular exercise",
                "Limit caffeine",
                "Exposure to avoided situations"
            ],
            "when_to_seek_help": "If panic attacks are frequent or avoidance is increasing"
        }
    },
    
    "crisis_resources": {
        "crisis_lines": {
            "National Suicide Prevention Lifeline": "988 (call or text)",
            "Crisis Text Line": "Text HOME to 741741",
            "International Association for Suicide Prevention": "https://www.iasp.info/resources/Crisis_Centres/",
            "SAMHSA National Helpline": "1-800-662-4357 (free, confidential, 24/7)"
        },
        "immediate_actions": [
            "If in immediate danger: Call 911 or go to emergency room",
            "Tell someone trusted what you're experiencing",
            "Remove access to means of self-harm if possible",
            "Stay with someone safe until crisis passes",
            "Call a crisis line: 988"
        ]
    },
    
    "healthy_habits": {
        "sleep": [
            "7-9 hours per night for adults",
            "Consistent sleep/wake times daily",
            "Cool (60-67°F), dark, quiet bedroom",
            "No screens 1 hour before bed",
            "Avoid caffeine after 2 PM"
        ],
        "exercise": [
            "150 minutes moderate aerobic activity/week",
            "Strength training 2x/week",
            "Flexibility work daily",
            "Start slowly and build gradually",
            "Find activities you enjoy for adherence"
        ],
        "nutrition": [
            "Eat balanced meals with protein, whole grains, fruits, vegetables",
            "Stay hydrated (8 glasses water daily)",
            "Limit sugar and processed foods",
            "Eat regularly to maintain blood sugar",
            "Limit alcohol to moderate levels"
        ],
        "social": [
            "Maintain regular contact with loved ones",
            "Join groups aligned with interests",
            "Volunteer for meaningful causes",
            "Invest in reciprocal relationships",
            "Set healthy boundaries"
        ]
    },
    
    "emergency_warning_signs": [
        "Suicidal thoughts or plans",
        "Self-harm urges or behaviors",
        "Severe hallucinations or delusions",
        "Complete inability to function (eat, hygiene, etc.)",
        "Severe substance use affecting functioning",
        "Thoughts of harming others",
        "Complete loss of reality orientation"
    ],
    
    "evidence_based_facts": {
        "mental_health": [
            "1 in 5 adults experience mental illness annually",
            "Mental health conditions are highly treatable",
            "80% of people with depression improve with treatment",
            "Exercise is as effective as medication for mild-moderate depression",
            "Early intervention significantly improves outcomes"
        ],
        "therapy_effectiveness": [
            "CBT has 60-70% remission rate for anxiety",
            "DBT reduces suicidal behavior by 50%",
            "Medication + therapy is most effective for depression",
            "Family therapy improves relationship patterns",
            "Regular therapy attendance predicts better outcomes"
        ]
    }
}




def get_coping_strategies(condition):
    """Get coping strategies for a specific condition"""
    return CLINICAL_RESOURCES["coping_strategies"].get(condition.lower(), [])

def get_therapeutic_approach(approach):
    """Get information about a specific therapeutic approach"""
    return CLINICAL_RESOURCES["therapeutic_approaches"].get(approach.upper(), {})

def get_condition_info(condition):
    """Get information about a mental health condition"""
    return CLINICAL_RESOURCES["mental_health_conditions"].get(condition.upper(), {})

def get_crisis_resources():
    """Get crisis hotlines and resources"""
    return CLINICAL_RESOURCES["crisis_resources"]

def get_all_resources():
    """Get all clinical resources"""
    return CLINICAL_RESOURCES

def get_evidence_facts(category="mental_health"):
    """Get evidence-based facts"""
    return CLINICAL_RESOURCES["evidence_based_facts"].get(category, [])

def embed_clinical_context(user_message, symptoms_detected=None):
    """
    Embed clinical context into LLM prompt
    """
    context = """
    You are an empathetic, evidence-based AI therapist assistant. 
    When responding:
    1. Validate their feelings and experiences
    2. Provide evidence-based coping strategies
    3. Suggest therapeutic approaches that have strong research support
    4. Never diagnose - only suggest they speak with professionals
    5. Recognize warning signs and recommend professional help
    6. Use this clinical knowledge to inform responses
    
    Clinical Context Available:
    - Evidence-based coping strategies for common conditions
    - Psychotherapy approaches (CBT, DBT, ACT, Psychodynamic)
    - Mental health condition information
    - Crisis resources
    - Healthy lifestyle recommendations
    
    Always maintain boundaries: you support but don't replace professional care.
    """
    return context
