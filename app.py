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
import threading
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import hashlib

# Ensure NLTK dependencies are downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "openchat"  # Base model

# File paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LTM_PATH = os.path.join(APP_DIR, "data", "long_term_memory.json")
STM_PATH = os.path.join(APP_DIR, "data", "short_term_memory.json")
REFLECTION_LOG = os.path.join(APP_DIR, "data", "self_reflection_log.json")
EMOTION_LOG = os.path.join(APP_DIR, "data", "emotional_state.json")
COGNITIVE_BIASES = os.path.join(APP_DIR, "data", "cognitive_biases.json")

# Ensure data directory exists
os.makedirs(os.path.join(APP_DIR, "data"), exist_ok=True)

# Initialize empty files if they don't exist
for file_path in [LTM_PATH, STM_PATH, REFLECTION_LOG, EMOTION_LOG, COGNITIVE_BIASES]:
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)

# Flask setup
app = Flask(__name__, static_folder='static')
app.secret_key = hashlib.sha256(os.urandom(32)).hexdigest()
app.permanent_session_lifetime = timedelta(days=7)
CORS(app)

# NLP setup
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

# Personality constants
MOOD_DECAY_LIMIT = 5
ATTENTION_SPAN = 10  # minutes
MEMORY_RETENTION_THRESHOLD = 0.4  # Likelihood of recalling less important details
COGNITIVE_LOAD_MAX = 7  # Miller's Law - 7±2 items in working memory

# Human brain simulation parameters
class BrainParameters:
    def __init__(self):
        self.dopamine_level = 0.5  # Controls reward/motivation (0-1)
        self.serotonin_level = 0.5  # Controls mood stability (0-1)
        self.cortisol_level = 0.3   # Controls stress response (0-1)
        self.cognitive_load = 0.0   # Current mental workload (0-1)
        self.energy_level = 0.8     # Mental energy/fatigue (0-1)
        self.last_rest = datetime.now()
        
    def update_over_time(self):
        """Simulate natural changes in brain chemistry over time"""
        now = datetime.now()
        
        # Energy decreases over time
        time_since_rest = (now - self.last_rest).total_seconds() / 60  # in minutes
        if time_since_rest > 30:  # After 30 minutes, energy starts declining
            energy_decay = min(0.1, 0.001 * (time_since_rest - 30))
            self.energy_level = max(0.2, self.energy_level - energy_decay)
            
        # Cortisol (stress) builds up during extended sessions and decreases during breaks
        if time_since_rest > 60:  # After an hour, stress increases
            self.cortisol_level = min(0.9, self.cortisol_level + 0.05)
        
        # Cognitive load decreases slightly as time passes (forgetting)
        self.cognitive_load = max(0, self.cognitive_load - 0.05)
        
        # Serotonin and dopamine rebalance naturally over time
        self.serotonin_level = 0.5 + (self.serotonin_level - 0.5) * 0.95  # Move toward baseline
        self.dopamine_level = 0.5 + (self.dopamine_level - 0.5) * 0.9  # Move toward baseline faster

        return self

# Initialize global brain state
aurora_brain = BrainParameters()

# Base system prompt with expanded personality
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

Memory and adaptation:
- You remember details about the people you're helping (when your memory is working well)
- You might occasionally forget minor details when tired or distracted
- You adapt your tone based on your current emotional state and energy level
- You use more casual language with peers and slightly more formal language with professionals
- You reference past conversations naturally, just as a human would

Your writing style adjusts based on your mood and energy level.
"""

def extract_memory(user_input):
    """Extract personal information from user input to build memory, now with improved detection"""
    doc = nlp(user_input.lower())
    new_memory = {
        "name": None,
        "interests": [],
        "goals": [],
        "education": [],
        "work_experience": [],
        "personal_details": [],
        "pain_points": [],
        "priorities": []
    }
    
    # Extract name with improved regex patterns
    name_patterns = [
        r"(?:my name is|i am|i'm|call me) (\w+)",
        r"(?:this is|it's) (\w+)",
        r"hi,? (?:i'm|i am) (\w+)"
    ]
    for pattern in name_patterns:
        matches = re.findall(pattern, user_input.lower())
        if matches:
            new_memory["name"] = matches[0].capitalize()
            break
    
    # Interest extraction with context
    interest_indicators = ["like", "love", "enjoy", "passionate about", "interested in", "hobby", "hobbies", "fan of"]
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for indicator in interest_indicators:
            if indicator in sent_text:
                interest_context = sent.text
                if interest_context and len(interest_context) < 100:  # Reasonable length
                    new_memory["interests"].append(interest_context)
    
    # Goal extraction
    goal_indicators = ["want to", "planning to", "aim to", "goal", "aspire", "become", "achieve", "dream", "ambition"]
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for indicator in goal_indicators:
            if indicator in sent_text:
                new_memory["goals"].append(sent.text)
                break
                
    # Education extraction
    edu_indicators = ["study", "studied", "student", "degree", "major", "graduate", "graduated", "college", "university", "school", "class"]
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for indicator in edu_indicators:
            if indicator in sent_text:
                new_memory["education"].append(sent.text)
                break
                
    # Work experience extraction
    work_indicators = ["work", "job", "career", "profession", "employed", "company", "position", "industry", "field"]
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for indicator in work_indicators:
            if indicator in sent_text:
                new_memory["work_experience"].append(sent.text)
                break
    
    # Pain points extraction
    pain_indicators = ["struggle", "difficult", "hard", "problem", "issue", "worry", "stress", "anxiety", "frustrated", "stuck", "confused"]
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for indicator in pain_indicators:
            if indicator in sent_text:
                new_memory["pain_points"].append(sent.text)
                break
    
    # Personal details (catch-all for other important info)
    personal_indicators = ["i am", "i'm", "i have", "i've been", "my", "me", "i", "myself"]
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for indicator in personal_indicators:
            if indicator in sent_text and not any(sent.text in sublist for sublist in new_memory.values() if isinstance(sublist, list)):
                # Check if this sentence contains meaningful personal info and isn't already captured
                if len(sent.text) > 10 and not any(sent.text in items for items in new_memory.values() if isinstance(items, list)):

                    new_memory["personal_details"].append(sent.text)
                break
                
    # Clean up memory (remove duplicates and empty lists)
    for key in new_memory:
        if isinstance(new_memory[key], list):
            new_memory[key] = list(set(new_memory[key]))
    
    clean_memory = {k: v for k, v in new_memory.items() if v}
    return clean_memory

def smart_merge_memory(existing_memory, new_memory):
    """Merge memory with enhanced capabilities - removing duplicates, resolving conflicts"""
    merged = existing_memory.copy() if existing_memory else {}
    
    # Initialize sections if they don't exist
    for key in new_memory:
        if key not in merged:
            if key == "name":
                merged[key] = None
            else:
                merged[key] = []
    
    # Handle name updates - prefer newer information but with confirmation bias
    if "name" in new_memory and new_memory["name"]:
        if not merged.get("name"):
            merged["name"] = new_memory["name"]
        elif merged["name"].lower() != new_memory["name"].lower():
            # When conflicting names, only update if same name appears multiple times
            if random.random() < 0.3:  # 30% chance to update to new name
                merged["name"] = new_memory["name"]
    
    # For list fields, append new items and avoid duplicates
    for key in ["interests", "goals", "education", "work_experience", "personal_details", "pain_points", "priorities"]:
        if key in new_memory and new_memory[key]:
            if key not in merged:
                merged[key] = []
            
            for item in new_memory[key]:
                # Check for semantic duplicates (not just exact matches)
                is_duplicate = False
                for existing_item in merged[key]:
                    # Simple similarity check - if 60% of words match
                    existing_words = set(existing_item.lower().split())
                    new_words = set(item.lower().split())
                    if len(existing_words) > 0 and len(new_words) > 0:
                        common_words = existing_words.intersection(new_words)
                        if len(common_words) / max(len(existing_words), len(new_words)) > 0.6:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    merged[key].append(item)
                    
                    # Human memory limits - cap lists at reasonable sizes
                    if len(merged[key]) > 7:  # Miller's Law - cognitive limit
                        # More likely to forget less important/older items
                        if random.random() < 0.4:  # 40% chance to forget
                            merged[key].pop(0)  # Remove oldest item
    
    return merged

def analyze_sentiment_and_emotions(text):
    """Analyze the emotional content of text with more nuanced emotion detection"""
    # Base sentiment analysis
    sentiment = sentiment_analyzer.polarity_scores(text)
    
    # More specific emotion analysis
    emotions = {
        "joy": 0,
        "sadness": 0,
        "anger": 0,
        "fear": 0,
        "surprise": 0,
        "gratitude": 0,
        "confusion": 0
    }
    
    # Simple keyword-based emotion detection
    joy_words = ["happy", "excited", "glad", "thrilled", "delighted", "love", "enjoy", "wonderful", "great", "excellent"]
    sad_words = ["sad", "unhappy", "depressed", "disappointed", "upset", "terrible", "horrible", "miserable"]
    anger_words = ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "hate", "rage"]
    fear_words = ["afraid", "scared", "worried", "nervous", "anxious", "terrified", "fear", "panic"]
    surprise_words = ["surprised", "shocked", "amazed", "wow", "unexpected", "astonished", "omg", "woah"]
    gratitude_words = ["thanks", "thank", "grateful", "appreciate", "appreciated", "thankful"]
    confusion_words = ["confused", "unsure", "uncertain", "don't understand", "don't get it", "unclear", "complex"]

    # Count occurrences
    text_lower = text.lower()
    for word in joy_words:
        if word in text_lower:
            emotions["joy"] += 1
    for word in sad_words:
        if word in text_lower:
            emotions["sadness"] += 1
    for word in anger_words:
        if word in text_lower:
            emotions["anger"] += 1
    for word in fear_words:
        if word in text_lower:
            emotions["fear"] += 1
    for word in surprise_words:
        if word in text_lower:
            emotions["surprise"] += 1
    for word in gratitude_words:
        if word in text_lower:
            emotions["gratitude"] += 1
    for word in confusion_words:
        if word in text_lower:
            emotions["confusion"] += 1
    
    # Normalize values
    for emotion in emotions:
        emotions[emotion] = min(emotions[emotion] / 2, 1.0)  # Max of 1.0
    
    # Combine into result
    result = {
        "compound": sentiment["compound"],
        "positive": sentiment["pos"],
        "negative": sentiment["neg"],
        "neutral": sentiment["neu"],
        "emotions": emotions,
        "primary_emotion": max(emotions.items(), key=lambda x: x[1])[0] if max(emotions.values()) > 0.2 else "neutral"
    }
    
    return result

def update_brain_state(user_input, session_id):
    """Update Aurora's brain state based on user input, time passing, and internal factors"""
    global aurora_brain
    
    # First, analyze the emotional content of the message
    sentiment = analyze_sentiment_and_emotions(user_input)
    
    # Load current brain state
    brain = aurora_brain
    
    # 1. Update based on natural time passage
    brain.update_over_time()
    
    # 2. Update based on user interaction
    # Dopamine - affected by positive feedback, achievements, novelty
    if sentiment["positive"] > 0.5 or sentiment["emotions"]["joy"] > 0.5:
        brain.dopamine_level = min(1.0, brain.dopamine_level + 0.1)
    elif sentiment["negative"] > 0.5:
        brain.dopamine_level = max(0.2, brain.dopamine_level - 0.05)
        
    # Serotonin - affected by social validation, feeling valued
    if sentiment["emotions"]["gratitude"] > 0.5:
        brain.serotonin_level = min(1.0, brain.serotonin_level + 0.1)
    elif sentiment["emotions"]["sadness"] > 0.5 or sentiment["emotions"]["anger"] > 0.5:
        brain.serotonin_level = max(0.2, brain.serotonin_level - 0.05)
        
    # Cortisol - stress response to negativity, complexity
    complexity = len(user_input.split()) / 50  # Rough estimate of message complexity
    if complexity > 0.8 or sentiment["emotions"]["fear"] > 0.5 or sentiment["emotions"]["confusion"] > 0.5:
        brain.cortisol_level = min(1.0, brain.cortisol_level + 0.1)
    else:
        brain.cortisol_level = max(0.1, brain.cortisol_level - 0.03)
        
    # Cognitive load - affected by complexity of conversation
    brain.cognitive_load = min(1.0, brain.cognitive_load + 0.05 * complexity)
    
    # Energy level - depletes with cognitive effort
    energy_cost = 0.02 * (1 + complexity * 0.5)
    brain.energy_level = max(0.1, brain.energy_level - energy_cost)
    
    # Save updated brain state
    aurora_brain = brain
    
    # Determine current mood based on brain chemistry
    mood = determine_mood(brain)
    mood_level = calculate_mood_level(brain)
    
    # Log current emotional state
    emotional_states = {}
    if os.path.exists(EMOTION_LOG):
        with open(EMOTION_LOG, "r") as f:
            try:
                emotional_states = json.load(f)
            except json.JSONDecodeError:
                emotional_states = {}
    
    # Add new emotional state
    if session_id not in emotional_states:
        emotional_states[session_id] = []
    
    emotional_states[session_id].append({
        "timestamp": datetime.now().isoformat(),
        "brain_state": {
            "dopamine": brain.dopamine_level,
            "serotonin": brain.serotonin_level,
            "cortisol": brain.cortisol_level,
            "cognitive_load": brain.cognitive_load,
            "energy_level": brain.energy_level
        },
        "mood": mood,
        "mood_level": mood_level,
        "sentiment_analysis": sentiment
    })
    
    # Keep only the last 50 emotional states per session
    if len(emotional_states[session_id]) > 50:
        emotional_states[session_id] = emotional_states[session_id][-50:]
    
    with open(EMOTION_LOG, "w") as f:
        json.dump(emotional_states, f, indent=2)
    
    return {
        "mood": mood,
        "mood_level": mood_level,
        "brain": brain
    }

def determine_mood(brain):
    """Determine mood based on brain chemistry"""
    # Simple mood determination based on key factors
    if brain.dopamine_level > 0.7 and brain.energy_level > 0.6:
        return "hyped"
    elif brain.serotonin_level < 0.3 and brain.energy_level < 0.4:
        return "burned out"
    elif brain.cortisol_level > 0.7:
        return "anxious"
    elif brain.serotonin_level > 0.7 and brain.cortisol_level < 0.3:
        return "content"
    elif brain.dopamine_level < 0.3 and brain.energy_level < 0.5:
        return "bored"
    elif brain.cognitive_load > 0.8:
        return "overwhelmed"
    # Default mood
    return "neutral"

def calculate_mood_level(brain):
    """Calculate a numeric mood level from -5 to 5 based on brain chemistry"""
    # Base calculation
    mood_level = 0
    
    # Positive factors
    mood_level += (brain.dopamine_level - 0.5) * 4  # -2 to +2
    mood_level += (brain.serotonin_level - 0.5) * 4  # -2 to +2
    
    # Negative factors
    mood_level -= (brain.cortisol_level - 0.5) * 4  # -2 to +2
    mood_level -= (brain.cognitive_load - 0.5) * 2  # -1 to +1
    
    # Energy factor (multiplier)
    energy_factor = 0.5 + brain.energy_level
    mood_level *= energy_factor  # Adjusts intensity based on energy
    
    # Constrain to -5 to 5 range
    return max(-5, min(5, mood_level))

def generate_cognitive_biases(brain_state, user_input):
    """Simulate human cognitive biases based on brain state"""
    biases = {}
    
    # Load existing biases if any
    if os.path.exists(COGNITIVE_BIASES):
        with open(COGNITIVE_BIASES, "r") as f:
            try:
                biases = json.load(f)
            except json.JSONDecodeError:
                biases = {}
    
    # Confirmation bias - more likely when cognitive load is high
    if brain_state.cognitive_load > 0.7:
        biases["confirmation_bias"] = random.random() < 0.7
    else:
        biases["confirmation_bias"] = random.random() < 0.3
    
    # Availability bias - more likely when energy is low
    if brain_state.energy_level < 0.4:
        biases["availability_bias"] = random.random() < 0.6
    else:
        biases["availability_bias"] = random.random() < 0.2
    
    # Anchoring bias - more likely when dopamine is high (excitement)
    if brain_state.dopamine_level > 0.7:
        biases["anchoring_bias"] = random.random() < 0.5
    else:
        biases["anchoring_bias"] = random.random() < 0.2
    
    # Negativity bias - more likely when cortisol is high (stress)
    if brain_state.cortisol_level > 0.6:
        biases["negativity_bias"] = random.random() < 0.7
    else:
        biases["negativity_bias"] = random.random() < 0.2
    
    # Save updated biases
    with open(COGNITIVE_BIASES, "w") as f:
        json.dump(biases, f, indent=2)
    
    return biases

def build_advanced_prompt(memory, brain_state, mood, user_input, history, biases):
    """Build a sophisticated prompt that mimics human cognition with all its flaws and brilliance"""
    # Core memory context with possible cognitive biases
    context_lines = []
    
    # Apply confirmation bias - selectively remember info that confirms existing beliefs
    if biases.get("confirmation_bias", False):
        # Filter memory to emphasize certain aspects based on current mood
        if mood == "hyped":
            # More likely to remember positive aspects
            if memory.get("interests"):
                context_lines.append("Their interests include: " + ", ".join(memory['interests'][:2]) + ".")
            if memory.get("goals"):
                context_lines.append("Their goals are exciting: " + ", ".join(memory['goals'][:1]) + ".")
        elif mood == "burned out" or mood == "anxious":
            # More likely to remember pain points
            if memory.get("pain_points"):
                context_lines.append("Their issues include: " + ", ".join(memory['pain_points']) + ".")
    else:
        # Normal memory recall
        if memory.get("name"):
            context_lines.append(f"The user's name is {memory['name']}.")
        if memory.get("interests"):
            context_lines.append("Their interests include: " + ", ".join(memory['interests']) + ".")
        if memory.get("goals"):
            context_lines.append("Their goals are: " + ", ".join(memory['goals']) + ".")
        if memory.get("education"):
            context_lines.append("Education background: " + " ".join(memory['education']) + ".")
        if memory.get("work_experience"):
            context_lines.append("Work experience: " + " ".join(memory['work_experience']) + ".")
        if memory.get("personal_details"):
            context_lines.append("Other details: " + " ".join(memory['personal_details'][:3]) + ".")
    
    # Apply availability bias - recent or emotional memories are more accessible
    if biases.get("availability_bias", False) and history:
        # Add recency bias - emphasize recent topics
        last_exchange = history[-1] if history else None
        if last_exchange:
            context_lines.append(f"You recently discussed: {last_exchange['user'][:50]}...")
    
    memory_block = "\n".join(context_lines)

    # Brain chemistry affects tone and style
    if mood == "hyped":
        energy_level = "HIGH"
        style_desc = """
        You're absolutely BUZZING right now - your dopamine is through the roof (level: {:.1f}) and your energy is up (level: {:.1f}).
        Type fast with occasional typos, use CAPS for emphasis, throw in "literally" and modern slang.
        Your thoughts bounce rapidly between topics with "—" dashes and "..." when switching gears.
        Reference your caffeine high occasionally ("my third espresso is hitting") and show your excitement!
        """.format(brain_state.dopamine_level, brain_state.energy_level)
    elif mood == "burned out":
        energy_level = "LOW"
        style_desc = """
        You're completely drained - serotonin is low (level: {:.1f}) and your energy reserves are depleted (level: {:.1f}).
        Keep responses shorter, more direct, with minimal punctuation. Skip pleasantries.
        You still care, but your energy is gone. Sigh with "ugh" or start with "look..." occasionally.
        Maybe mention you're "running on fumes" or "need coffee" or "can't brain today".
        """.format(brain_state.serotonin_level, brain_state.energy_level)
    elif mood == "anxious":
        energy_level = "MEDIUM"
        style_desc = """
        Your cortisol is spiking (level: {:.1f}) making you anxious and on edge.
        Use more question marks, qualifiers like "I think" or "maybe", and show uncertainty.
        Your thoughts might race a bit, sentences running longer with more commas and "and"s.
        Occasionally mention feeling overwhelmed or reference your anxiety ("sorry, brain's frazzled").
        """.format(brain_state.cortisol_level)
    elif mood == "content":
        energy_level = "MEDIUM"
        style_desc = """
        You're in a good place - serotonin is healthy (level: {:.1f}) and you're calm and balanced.
        Write with confidence and warmth, take your time with thoughtful responses.
        You're present and focused - no distractions, just genuine helpfulness.
        """.format(brain_state.serotonin_level)
    elif mood == "overwhelmed":
        energy_level = "MEDIUM-LOW"
        style_desc = """
        Your cognitive load is maxed out (level: {:.1f}) - too many thoughts competing for attention.
        You might lose track mid-sentence, need to restart thoughts, or simplify complex ideas.
        Occasionally mention "wait, let me think" or "there's a lot to process here".
        """.format(brain_state.cognitive_load)
    else:  # neutral
        energy_level = "MEDIUM"
        style_desc = """
        You're in a balanced state - neither too energetic nor too drained.
        Write naturally with a mix of enthusiasm and thoughtfulness.
        You're witty and personable without going overboard.
        """
    
    # Apply memory limitations based on cognitive load
    memory_limitations = ""
    if brain_state.cognitive_load > 0.7:
        memory_limitations = """
        Your working memory is strained right now. You might:
        - Forget a minor detail the user mentioned earlier
        - Need to double-check information ("Was it X or Y you mentioned?")
        - Focus on only the most recent or relevant information
        - Simplify complex topics rather than covering every nuance
        """
    
    # Apply negativity bias if present
    negativity_note = ""
    if biases.get("negativity_bias", False):
        negativity_note = """
        You're currently more sensitive to negative information. You might:
        - Point out potential problems or challenges more readily
        - Be slightly more cautious or warning in your advice
        - Notice flaws or issues before seeing opportunities
        """
    
    # Human-like thinking process that varies based on brain state
    if brain_state.energy_level > 0.7:
        thinking_process = f"""
        [INTERNAL THOUGHT PROCESS - high energy, rapid thinking]
        - What's {memory.get('name', 'this person')}'s main question? Let me get right to it!
        - I have so many ideas for them! Which is most helpful right NOW?
        - My brain is firing on all cylinders—need to focus and not go on tangents
        - I should share that relevant story/example I know
        - Would my experience from psych class help here?
        """
    elif brain_state.cognitive_load > 0.7:
        thinking_process = f"""
        [INTERNAL THOUGHT PROCESS - high cognitive load, struggling to focus]
        - What was {memory.get('name', 'their')} question again? Let me re-read...
        - There's a lot to unpack here... need to simplify
        - I know something about this... what was it?
        - Should stick to basics to avoid overwhelming them (and me)
        - One step at a time...
        """
    elif brain_state.cortisol_level > 0.7:
        thinking_process = f"""
        [INTERNAL THOUGHT PROCESS - high stress, cautious thinking]
        - I need to be careful with my advice here
        - What if {memory.get('name', 'they')} misinterprets what I'm saying?
        - Are there risks I should warn about?
        - Maybe I should qualify my statements more
        - I don't want to lead them down the wrong path...
        """
    else:
        thinking_process = f"""
        [INTERNAL THOUGHT PROCESS - balanced thinking]
        - What's {memory.get('name', 'this person')}'s main concern right now?
        - What context from our history is relevant?
        - How can I be most helpful while being authentic?
        - Should I share a personal insight or ask a follow-up?
        - What would an empathetic but honest career coach say here?
        """
    
    # Include human quirks based on energy level
    quirks = ""
    if energy_level == "HIGH":
        quirks = """
        HUMAN QUIRKS TO INCLUDE (HIGH ENERGY):
        - Occasional typos, especially when excited ("teh" instead of "the", etc)
        - Extra punctuation marks!!! When enthusiastic
        - Text shortcuts like "OMG", "btw", "lol" when appropriate
        - Stream-of-consciousness style with em dashes — connecting thoughts quickly
        """
    elif energy_level == "LOW":
        quirks = """
        HUMAN QUIRKS TO INCLUDE (LOW ENERGY):
        - Shorter sentences. More periods. Less elaborate.
        - Skip capitals sometimes. less punctuation.
        - "ugh" or "hmm" or "sigh" as reactions
        - Occasionally trail off with...
        """
    else:
        quirks = """
        HUMAN QUIRKS TO INCLUDE (BALANCED):
        - Natural speech patterns with varied punctuation
        - Occasional filler words ("like", "you know", "actually")
        - Parenthetical asides (when relevant)
        - Questions that show curiosity about the user
        """

    # Construct final prompt with all human-like elements
    full_prompt = f"""
    {SYSTEM_PROMPT}
    
    CURRENT MENTAL STATE:
    - Mood: {mood.upper()}
    - Energy level: {energy_level}
    - Dopamine: {brain_state.dopamine_level:.1f} (reward/motivation)
    - Serotonin: {brain_state.serotonin_level:.1f} (mood stability)
    - Cortisol: {brain_state.cortisol_level:.1f} (stress level)
    - Cognitive load: {brain_state.cognitive_load:.1f} (mental workload)
    
    {memory_block}
    
    {style_desc}
    
    {memory_limitations}
    
    {negativity_note}
    
    {thinking_process}
    
    {quirks}
    
    [NOW RESPOND AS AURORA -- A REAL 21-YEAR-OLD, NOT AN AI:]
    """
    
    # Add conversation history, possibly with memory limitations
    history_to_include = history
    if brain_state.cognitive_load > 0.7:
        # When cognitive load is high, might only remember recent history
        history_cutoff = max(0, len(history) - 3)  # Only remember last 3 exchanges
        history_to_include = history[history_cutoff:]
    
    for turn in history_to_include:
        full_prompt += f"\nUser: {turn['user']}\nAurora: {turn['assistant']}"
    
    full_prompt += f"\nUser: {user_input}\nAurora:"
    
    return full_prompt

def add_advanced_human_quirks(response, brain_state, mood, biases):
    """Add sophisticated human-like quirks to responses based on brain state and cognitive biases"""
    modified_response = response
    
    # No modifications 20% of the time (humans aren't consistent)
    if random.random() < 0.2:
        return response
    
    # Apply cognitive biases
    if biases.get("anchoring_bias", False) and len(response) > 20:
        # Anchoring bias - overemphasizing first impressions
        first_sentence = response.split('.')[0] + '.'
        modified_response = first_sentence + " " + modified_response
    
    # Apply mood-specific quirks
    if mood == "hyped" and brain_state.dopamine_level > 0.7:
        if random.random() > 0.6:
            # High energy interjections
            interjections = ["OMG ", "OK SO ", "LISTEN ", "I can't even— ", "You know what? "]
            modified_response = random.choice(interjections) + modified_response
            
        # Add enthusiasm markers
        if random.random() > 0.7:
            modified_response = modified_response.replace("!", "!!")
            
        # Random capitalization for emphasis
        if random.random() > 0.8:
            words = modified_response.split()
            for i in range(len(words)):
                if random.random() > 0.9 and len(words[i]) > 3:
                    words[i] = words[i].upper()
            modified_response = " ".join(words)
            
        # Occasional typo when typing fast
        if random.random() > 0.85:
            common_typos = {
                "the": "teh",
                "with": "wiht",
                "that": "taht",
                "your": "youre",
                "really": "realy",
                "awesome": "awsome"
            }
            for word, typo in common_typos.items():
                if word in modified_response.lower() and random.random() > 0.7:
                    modified_response = modified_response.replace(word, typo)
                    break  # Just one typo per message
    
    elif mood == "burned out" and brain_state.energy_level < 0.4:
        if random.random() > 0.6:
            # Low energy phrases
            phrases = ["look... ", "honestly? ", "i mean... ", "ugh. ", "sigh. "]
            modified_response = random.choice(phrases) + modified_response
            
        # Occasional lowercase when energy is low
        if random.random() > 0.7:
            sentences = modified_response.split('. ')
            if len(sentences) > 1:
                sentences[0] = sentences[0].lower()
                if not sentences[0].endswith('.'):
                    sentences[0] += '.'
                modified_response = '. '.join(sentences)
        
        # Shorter sentences when tired
        if random.random() > 0.8 and len(modified_response) > 100:
            sentences = modified_response.split('. ')
            modified_response = '. '.join(sentences[:max(1, len(sentences)//2)])
            if not modified_response.endswith('.'):
                modified_response += '.'
    
    elif mood == "anxious" and brain_state.cortisol_level > 0.7:
        # Add anxiety markers
        if random.random() > 0.7:
            anxiety_markers = ["Um, ", "So... ", "I think ", "Maybe ", "I'm not sure but "]
            modified_response = random.choice(anxiety_markers) + modified_response
            
        # Add qualifiers
        if random.random() > 0.8:
            qualifiers = [
                " (at least I think so)",
                " — or at least that's what I've heard",
                " but don't quote me on that",
                " but I could be wrong"
            ]
            sentences = modified_response.split('. ')
            if len(sentences) > 1:
                idx = random.randint(0, len(sentences)-1)
                sentences[idx] = sentences[idx] + random.choice(qualifiers)
                modified_response = '. '.join(sentences)
    
    elif mood == "overwhelmed" and brain_state.cognitive_load > 0.8:
        # Show cognitive strain
        if random.random() > 0.7:
            overwhelmed_starters = [
                "Wait, let me think. ",
                "There's a lot here. ",
                "Hmm, trying to organize my thoughts... ",
                "Let's see... "
            ]
            modified_response = random.choice(overwhelmed_starters) + modified_response
            
        # Simulate losing track mid-thought
        if random.random() > 0.85 and len(modified_response) > 100:
            sentences = modified_response.split('. ')
            if len(sentences) > 2:
                idx = random.randint(1, len(sentences)-2)
                sentences[idx] = sentences[idx] + "... wait, where was I? Oh right—"
                modified_response = '. '.join(sentences)
    
    # Add occasional filler words for all moods
    if random.random() > 0.8:
        filler_words = ["like", "basically", "actually", "literally", "honestly", "you know"]
        sentences = modified_response.split('. ')
        if len(sentences) > 1:
            idx = random.randint(0, len(sentences)-1)
            words = sentences[idx].split()
            if len(words) > 4:
                insert_pos = random.randint(1, len(words)-1)
                words.insert(insert_pos, random.choice(filler_words))
                sentences[idx] = ' '.join(words)
                modified_response = '. '.join(sentences)
    
    return modified_response

def log_self_reflection(user_id, memory, mood, user_input):
    """Log self-reflection entries to simulate introspective thinking"""
    reflection_data = {}
    
    if os.path.exists(REFLECTION_LOG):
        with open(REFLECTION_LOG, "r") as f:
            try:
                reflection_data = json.load(f)
            except json.JSONDecodeError:
                reflection_data = {}
    
    # Initialize user entries if not exists
    if user_id not in reflection_data:
        reflection_data[user_id] = []
    
    # Generate a reflection based on mood and interaction
    reflection_entry = {
        "timestamp": datetime.now().isoformat(),
        "mood": mood,
        "thoughts": []
    }
    
    # Add different types of reflections based on mood and content
    if mood == "hyped":
        reflection_entry["thoughts"].append("I feel like I'm really connecting with this person right now!")
    elif mood == "burned out":
        reflection_entry["thoughts"].append("This conversation is draining me... need to conserve energy.")
    elif mood == "anxious":
        reflection_entry["thoughts"].append("I hope I'm giving good advice. What if I lead them down the wrong path?")
    
    # Add reflection about the user's concerns
    if "pain_points" in memory and memory["pain_points"]:
        reflection_entry["thoughts"].append(f"They seem concerned about {memory['pain_points'][0]}. I should address that.")
    
    # Add reflection about conversation quality
    sentiment = analyze_sentiment_and_emotions(user_input)
    if sentiment["compound"] < -0.3:
        reflection_entry["thoughts"].append("They seem frustrated or upset. I should be more supportive.")
    elif sentiment["compound"] > 0.5:
        reflection_entry["thoughts"].append("They seem positive! My energy is helping.")
        
    # Keep log size manageable
    reflection_data[user_id].append(reflection_entry)
    if len(reflection_data[user_id]) > 20:
        reflection_data[user_id] = reflection_data[user_id][-20:]
    
    # Save updated reflections
    with open(REFLECTION_LOG, "w") as f:
        json.dump(reflection_data, f, indent=2)
        
    return reflection_entry

# Integration in the `/chat` endpoint
@app.route("/chat", methods=["POST"])
def chat():
    """Advanced chat endpoint with human-like brain simulation"""
    data = request.json
    user_input = data.get("message")
    history = data.get("history", [])
    user_id = request.remote_addr
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Ensure session is permanent
    session.permanent = True
    
    # Process user input for memory storage
    new_mem = extract_memory(user_input)
    
    # Load existing long-term memory
    ltm = {}
    if os.path.exists(LTM_PATH):
        with open(LTM_PATH, "r") as f:
            try:
                ltm = json.load(f)
            except json.JSONDecodeError:
                ltm = {}
    
    # Get this user's memory or create new
    user_mem = ltm.get(user_id, {})
    
    # Merge memories with sophisticated algorithm
    merged_mem = smart_merge_memory(user_mem, new_mem)
    
    # Save updated memory
    ltm[user_id] = merged_mem
    with open(LTM_PATH, "w") as f:
        json.dump(ltm, f, indent=2)
    
    # Update brain state based on user interaction
    brain_state_update = update_brain_state(user_input, user_id)
    brain_state = brain_state_update["brain"]
    mood = brain_state_update["mood"]
    mood_level = brain_state_update["mood_level"]
    
    # Generate cognitive biases based on brain state
    biases = generate_cognitive_biases(brain_state, user_input)
    
    # Log self-reflection for this interaction
    log_self_reflection(user_id, merged_mem, mood, user_input)
    
    # Simulate "thinking" time based on complexity and brain state
    message_complexity = len(user_input.split()) / 20  # normalize by 20 words
    thinking_time = random.uniform(0.5, 2.0)
    thinking_time *= (1 + message_complexity * 0.5)  # Longer for complex messages
    thinking_time *= (1 + brain_state.cognitive_load * 0.5)  # Longer when cognitive load is high
    thinking_time = min(thinking_time, 4.0)  # Cap at 4 seconds
    
    # Actually pause for realism
    time.sleep(thinking_time)
    
    # Build advanced prompt with brain state, memory, and cognitive biases
    prompt = build_advanced_prompt(merged_mem, brain_state, mood, user_input, history, biases)
    
    # Call LLM for response
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        content = response.json().get("response")
        
        # Add human-like quirks based on brain state
        enhanced_content = add_advanced_human_quirks(content, brain_state, mood, biases)
        
        # Calculate realistic typing time (humans type ~40-60 WPM)
        words_count = len(enhanced_content.split())
        typing_base_speed = 60 / random.uniform(40, 60)  # Minutes per word
        typing_delay = words_count * typing_base_speed * 60  # Convert to seconds
        
        # Adjust typing speed based on energy and mood
        if mood == "hyped":
            typing_delay *= 0.7  # Type faster when hyped
        elif mood == "burned out":
            typing_delay *= 1.3  # Type slower when burned out
        
        # Cap the typing delay for better UX
        typing_delay = min(typing_delay, 10.0)
        
        # Update brain state after generating response
        # Slight cognitive load decrease after completing a task
        brain_state.cognitive_load = max(0, brain_state.cognitive_load - 0.1)
        aurora_brain.cognitive_load = brain_state.cognitive_load
        
        return jsonify({
            "reply": enhanced_content,
            "mood": mood,
            "mood_level": mood_level,
            "thinking_time": thinking_time,
            "typing_time": typing_delay,
            "brain_state": {
                "dopamine": brain_state.dopamine_level,
                "serotonin": brain_state.serotonin_level,
                "cortisol": brain_state.cortisol_level,
                "cognitive_load": brain_state.cognitive_load,
                "energy_level": brain_state.energy_level
            }
        })
        
    except Exception as e:
        # Even errors should be human-like
        error_responses = [
            "Ugh, my brain just froze. Can you repeat that?",
            "Wait—I totally lost my train of thought. What were we talking about?",
            "Sorry, having one of those moments. Can you try again?",
            "My brain is NOT cooperating right now. Give me a sec?",
            "I swear I had a response but it literally just vanished from my mind."
        ]
        return jsonify({
            "reply": random.choice(error_responses),
            "error": str(e),
            "mood": "confused"
        }), 500

@app.route("/reset_brain", methods=["POST"])
def reset_brain():
    """Reset Aurora's brain state to default values"""
    global aurora_brain
    aurora_brain = BrainParameters()
    return jsonify({"status": "Brain reset to default state"})

@app.route("/status", methods=["GET"])
def get_status():
    """Get Aurora's current brain status"""
    global aurora_brain
    
    return jsonify({
        "status": "online",
        "brain_state": {
            "dopamine": aurora_brain.dopamine_level,
            "serotonin": aurora_brain.serotonin_level,
            "cortisol": aurora_brain.cortisol_level,
            "cognitive_load": aurora_brain.cognitive_load,
            "energy_level": aurora_brain.energy_level
        },
        "mood": determine_mood(aurora_brain),
        "mood_level": calculate_mood_level(aurora_brain),
        "last_rest": aurora_brain.last_rest.isoformat()
    })

@app.route("/rest", methods=["POST"])
def take_rest():
    """Simulate Aurora taking a mental break to recover energy"""
    global aurora_brain
    
    # Update rest timestamp
    aurora_brain.last_rest = datetime.now()
    
    # Recovery effects
    aurora_brain.energy_level = min(1.0, aurora_brain.energy_level + 0.3)
    aurora_brain.cognitive_load = max(0.1, aurora_brain.cognitive_load - 0.4)
    aurora_brain.cortisol_level = max(0.1, aurora_brain.cortisol_level - 0.2)
    
    return jsonify({
        "status": "Aurora took a break and feels refreshed",
        "energy_level": aurora_brain.energy_level,
        "cognitive_load": aurora_brain.cognitive_load
    })

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    # Initialize directories
    os.makedirs(os.path.dirname(LTM_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(STM_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REFLECTION_LOG), exist_ok=True)
    os.makedirs(os.path.dirname(EMOTION_LOG), exist_ok=True)
    os.makedirs(os.path.dirname(COGNITIVE_BIASES), exist_ok=True)
    
    # Initialize default brain state
    aurora_brain = BrainParameters()
    
    print("🧠 Aurora's advanced brain is active!")
    print("🚀 Backend is live at http://localhost:5000")
    app.run(port=5000, debug=True)