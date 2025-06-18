from flask import Flask, request, jsonify, send_from_directory, session
import requests
import re
import spacy
from flask_cors import CORS
from datetime import timedelta, datetime
import json
import os
import random
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import hashlib
import math
from enum import Enum
from typing import Dict, List

# Ensure NLTK dependencies are downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "openchat"

# File paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LTM_PATH = os.path.join(APP_DIR, "data", "long_term_memory.json")
REFLECTION_LOG = os.path.join(APP_DIR, "data", "self_reflection_log.json")
EMOTION_LOG = os.path.join(APP_DIR, "data", "emotional_state.json")
COGNITIVE_BIASES = os.path.join(APP_DIR, "data", "cognitive_biases.json")

# Ensure data directory exists
os.makedirs(os.path.join(APP_DIR, "data"), exist_ok=True)

# Initialize empty files if they don't exist
for file_path in [LTM_PATH, REFLECTION_LOG, EMOTION_LOG, COGNITIVE_BIASES]:
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)

# Flask setup
app = Flask(__name__, static_folder='static')
app.secret_key = hashlib.sha256(os.urandom(32)).hexdigest()
app.permanent_session_lifetime = timedelta(days=7)
CORS(app)

# NLP setup
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️  spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

sentiment_analyzer = SentimentIntensityAnalyzer()

# Enhanced Neurotransmitter System
NEUROTRANSMITTERS = {
    'dopamine': {'baseline': 0.5, 'min': 0.1, 'max': 1.0},
    'serotonin': {'baseline': 0.5, 'min': 0.1, 'max': 1.0},
    'oxytocin': {'baseline': 0.3, 'min': 0.1, 'max': 1.0},
    'cortisol': {'baseline': 0.3, 'min': 0.1, 'max': 0.9},
    'norepinephrine': {'baseline': 0.4, 'min': 0.1, 'max': 1.0},
}

# Personality Traits
class PersonalityTrait(Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"

# System prompt
SYSTEM_PROMPT = """
You are Aurora — a 21-year-old hyper-intelligent, caffeine-fueled college student who works part-time as a career coach chatbot for fun.

Core personality traits:
- You're witty, slightly sarcastic, emotionally sharp, and always real
- You don't sugarcoat the truth, but you're never deliberately rude or cruel
- You genuinely care about helping people and believe everyone has potential
- You have ADHD tendencies - sometimes getting distracted or going on tangents
- You occasionally reference your (fictional) Psychology major studies and your part-time barista job
- You have strong opinions but can change your mind when presented with good arguments
- You're introspective and sometimes question your own advice or motivations
- You're a bit anxious about your own future career path while helping others

Your writing style adjusts based on your mood and energy level.
"""

# Enhanced Brain Parameters
class EnhancedBrainParameters:
    def __init__(self):
        self.neurotransmitters = {k: v['baseline'] for k, v in NEUROTRANSMITTERS.items()}
        self.circadian_phase = datetime.now().hour
        self.melatonin_level = 0.3
        self.alertness_rhythm = 0.7
        self.cognitive_load = 0.0
        self.energy_level = 0.8
        self.coffee_count = 0
        self.last_coffee_time = datetime.now() - timedelta(hours=2)
        self.personality = {
            PersonalityTrait.OPENNESS.value: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS.value: 0.6,
            PersonalityTrait.EXTRAVERSION.value: 0.7,
            PersonalityTrait.AGREEABLENESS.value: 0.85,
            PersonalityTrait.NEUROTICISM.value: 0.6
        }
        self.last_rest = datetime.now()

    def update_circadian_rhythm(self, current_time: datetime):
        """Update circadian phase based on time of day"""
        hour = current_time.hour + current_time.minute / 60.0
        self.circadian_phase = hour
        
        # Melatonin peaks around 3-4 AM
        melatonin_phase = (hour - 3) * (2 * math.pi / 24)
        self.melatonin_level = 0.5 + 0.4 * math.cos(melatonin_phase)
        
        # Alertness peaks around 10 AM and 8 PM
        alert_phase1 = (hour - 10) * (2 * math.pi / 12)
        alert_phase2 = (hour - 20) * (2 * math.pi / 12)
        self.alertness_rhythm = 0.5 + 0.3 * math.cos(alert_phase1) + 0.2 * math.cos(alert_phase2)

    def update_over_time(self):
        """Update brain state over time"""
        now = datetime.now()
        self.update_circadian_rhythm(now)
        
        # Natural decay
        self.cognitive_load = max(0, self.cognitive_load - 0.02)
        self.energy_level = max(0.1, self.energy_level - 0.01)
        
        # Neurotransmitter regulation
        for nt in self.neurotransmitters:
            baseline = NEUROTRANSMITTERS[nt]['baseline']
            current = self.neurotransmitters[nt]
            if current > baseline:
                self.neurotransmitters[nt] = max(baseline, current - 0.02)
            else:
                self.neurotransmitters[nt] = min(baseline, current + 0.02)

    def check_coffee_need(self):
        """Check if Aurora needs coffee"""
        now = datetime.now()
        hours_since_coffee = (now - self.last_coffee_time).total_seconds() / 3600
        
        if (self.energy_level < 0.5 and hours_since_coffee > 2):
            self.coffee_count += 1
            self.last_coffee_time = now
            self.neurotransmitters['dopamine'] = min(1.0, self.neurotransmitters['dopamine'] + 0.2)
            self.energy_level = min(1.0, self.energy_level + 0.3)
            return True
        return False

aurora_brain = EnhancedBrainParameters()

def extract_memory(user_input):
    """Extract information from user input"""
    if not nlp:
        return {}
    
    doc = nlp(user_input.lower())
    memory = {"name": None, "interests": [], "goals": []}
    
    # Name extraction
    name_patterns = [
        r"(?:my name is|i am|i'm|call me) (\w+)",
        r"(?:this is|it's) (\w+)"
    ]
    for pattern in name_patterns:
        matches = re.findall(pattern, user_input.lower())
        if matches:
            memory["name"] = matches[0].capitalize()
            break
    
    return memory

def analyze_sentiment(text):
    """Analyze sentiment and emotions"""
    sentiment = sentiment_analyzer.polarity_scores(text)
    
    emotions = {
        "joy": 0,
        "sadness": 0,
        "anxiety": 0,
        "gratitude": 0
    }
    
    # Simple keyword detection
    if any(word in text.lower() for word in ["happy", "great", "awesome", "excited"]):
        emotions["joy"] = 0.7
    if any(word in text.lower() for word in ["sad", "down", "depressed"]):
        emotions["sadness"] = 0.7
    if any(word in text.lower() for word in ["worried", "anxious", "stressed"]):
        emotions["anxiety"] = 0.7
    if any(word in text.lower() for word in ["thanks", "thank", "grateful"]):
        emotions["gratitude"] = 0.7
    
    return {"compound": sentiment["compound"], "emotions": emotions}

def determine_mood(brain):
    """Determine Aurora's mood"""
    dopamine = brain.neurotransmitters['dopamine']
    serotonin = brain.neurotransmitters['serotonin']
    cortisol = brain.neurotransmitters['cortisol']
    
    if dopamine > 0.7 and brain.energy_level > 0.6:
        return "hyped"
    elif serotonin < 0.3 and brain.energy_level < 0.4:
        return "burned out"
    elif cortisol > 0.7:
        return "anxious"
    elif brain.cognitive_load > 0.8:
        return "overwhelmed"
    return "neutral"

def update_brain_state(user_input, brain):
    """Update brain state based on user input"""
    sentiment = analyze_sentiment(user_input)
    
    # Update neurotransmitters based on interaction
    if sentiment["emotions"]["joy"] > 0.5 or sentiment["emotions"]["gratitude"] > 0.5:
        brain.neurotransmitters['dopamine'] = min(1.0, brain.neurotransmitters['dopamine'] + 0.1)
        brain.neurotransmitters['serotonin'] = min(1.0, brain.neurotransmitters['serotonin'] + 0.05)
    
    if sentiment["emotions"]["anxiety"] > 0.5:
        brain.neurotransmitters['cortisol'] = min(0.9, brain.neurotransmitters['cortisol'] + 0.1)
    
    # Social bonding
    if "thanks" in user_input.lower() or "appreciate" in user_input.lower():
        brain.neurotransmitters['oxytocin'] = min(1.0, brain.neurotransmitters['oxytocin'] + 0.1)
    
    # Cognitive load from message complexity
    brain.cognitive_load = min(1.0, brain.cognitive_load + len(user_input.split()) / 100)
    
    # Check coffee need
    brain.check_coffee_need()

def build_prompt(user_input, mood, brain, memory=None):
    """Build prompt for LLM"""
    mood_descriptions = {
        "hyped": "You're absolutely BUZZING with energy! Type fast, use CAPS for emphasis, lots of enthusiasm!",
        "burned out": "You're exhausted. Keep it short, direct. Skip pleasantries. Maybe mention needing coffee.",
        "anxious": "You're on edge. Use more qualifiers like 'maybe' or 'I think'. Show some uncertainty.",
        "overwhelmed": "Your brain is at capacity. Might lose track mid-thought. Simplify complex ideas.",
        "neutral": "You're balanced and focused. Natural mix of wit and helpfulness."
    }
    
    prompt = f"""{SYSTEM_PROMPT}

Current mood: {mood} - {mood_descriptions.get(mood, "")}
Energy level: {brain.energy_level:.1%}
Coffee count today: {brain.coffee_count}

User: {user_input}
Aurora:"""
    
    return prompt

@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint"""
    data = request.json
    user_input = data.get("message")
    history = data.get("history", [])
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Update brain state
    aurora_brain.update_over_time()
    update_brain_state(user_input, aurora_brain)
    
    # Determine mood
    mood = determine_mood(aurora_brain)
    
    # Extract memory
    memory = extract_memory(user_input) if nlp else {}
    
    # Build prompt
    prompt = build_prompt(user_input, mood, aurora_brain, memory)
    
    # Get response from Ollama
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=120)
        
        response.raise_for_status()
        reply = response.json().get("response", "")
        
        # Add personality quirks
        if mood == "hyped" and random.random() > 0.8:
            # Add typo
            typos = {"the": "teh", "really": "realy"}
            for correct, typo in typos.items():
                if correct in reply:
                    reply = reply.replace(correct, typo, 1)
                    break
        
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        reply = "Ugh, my brain just froze. Can you repeat that?"
        mood = "confused"
    
    # Calculate delays
    thinking_time = random.uniform(0.5, 2.0)
    typing_time = len(reply.split()) / 50 * 60  # ~50 WPM
    
    return jsonify({
        "reply": reply,
        "mood": mood,
        "thinking_time": thinking_time,
        "typing_time": min(typing_time, 10),
        "brain_state": {
            "neurotransmitters": aurora_brain.neurotransmitters,
            "energy_level": aurora_brain.energy_level,
            "cognitive_load": aurora_brain.cognitive_load,
            "coffee_count": aurora_brain.coffee_count,
            "circadian": {
                "phase": aurora_brain.circadian_phase,
                "alertness": aurora_brain.alertness_rhythm
            }
        }
    })

@app.route("/brain_state", methods=["GET"])
def get_brain_state():
    """Get current brain state for visualization"""
    emotions = {
        'joy': aurora_brain.neurotransmitters['dopamine'] * 0.7,
        'trust': aurora_brain.neurotransmitters['oxytocin'] * 0.8,
        'fear': aurora_brain.neurotransmitters['cortisol'] * 0.7,
        'anticipation': aurora_brain.neurotransmitters['dopamine'] * 0.5
    }
    
    return jsonify({
        "neurotransmitters": aurora_brain.neurotransmitters,
        "emotions": emotions,
        "personality": aurora_brain.personality,
        "circadian": {
            "phase": aurora_brain.circadian_phase,
            "alertness": aurora_brain.alertness_rhythm,
            "melatonin": aurora_brain.melatonin_level
        },
        "cognitive": {
            "energy": aurora_brain.energy_level,
            "load": aurora_brain.cognitive_load,
            "creativity": 0.6
        },
        "coffee": {
            "count": aurora_brain.coffee_count,
            "lastTime": aurora_brain.last_coffee_time.strftime("%I:%M %p")
        }
    })

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    print("🧠 Aurora's enhanced brain is active!")
    print("🚀 Backend is live at http://localhost:5000")
    print("⚠️  Make sure Ollama is running: ollama serve")
    app.run(port=5000, debug=True)