# Integration module for Enhanced Aurora Brain with existing Flask app
# This integrates the new brain model with Aurora's personality and chat system

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import re
from enhanced_brain_model import (
    EnhancedBrainModel, PersonalityTrait, 
    EmotionalState, TheoryOfMind, WorkingMemory
)

class AuroraIntegration:
    """Integrates enhanced brain model with Aurora's personality system"""
    
    def __init__(self):
        self.brain = EnhancedBrainModel()
        
        # Set Aurora's personality traits
        self.brain.personality.traits = {
            PersonalityTrait.OPENNESS.value: 0.8,  # Creative, curious
            PersonalityTrait.CONSCIENTIOUSNESS.value: 0.6,  # Somewhat organized but ADHD
            PersonalityTrait.EXTRAVERSION.value: 0.7,  # Social, energetic
            PersonalityTrait.AGREEABLENESS.value: 0.85,  # Very agreeable (her "problem")
            PersonalityTrait.NEUROTICISM.value: 0.6  # Some anxiety about future
        }
        
        # Aurora-specific attributes
        self.coffee_count = 0
        self.last_coffee_time = datetime.now() - timedelta(hours=2)
        self.current_mood_description = ""
        
    def process_user_input(self, user_input: str, user_memory: Dict) -> Dict:
        """Process user input and return brain state updates"""
        current_time = datetime.now()
        
        # Extract stimuli from user input
        stimuli = self._extract_stimuli(user_input, user_memory)
        
        # Update brain state
        self.brain.update(current_time, stimuli)
        
        # Check if Aurora needs coffee
        self._check_coffee_need(current_time)
        
        # Generate mood description
        self.current_mood_description = self._generate_mood_description()
        
        # Extract any new information about the user
        user_info = self._extract_user_info(user_input)
        
        # Update theory of mind with user model
        if user_memory.get('name'):
            self.brain.theory_of_mind.other_models[user_memory['name']] = {
                'beliefs': user_memory.get('goals', []),
                'emotions': self._infer_user_emotions(user_input),
                'interests': user_memory.get('interests', [])
            }
        
        return {
            'brain_state': self.brain.get_current_state(),
            'mood_description': self.current_mood_description,
            'coffee_status': self._get_coffee_status(),
            'user_info': user_info,
            'empathy_response': self._generate_empathy_response(user_input)
        }
    
    def _extract_stimuli(self, user_input: str, user_memory: Dict) -> Dict:
        """Extract relevant stimuli from user input"""
        stimuli = {}
        
        # Check for social interaction
        if any(word in user_input.lower() for word in ['hi', 'hello', 'hey', 'thanks', 'appreciate']):
            stimuli['social_interaction'] = True
        
        # Check for stressful content
        stress_words = ['worried', 'anxious', 'stressed', 'confused', 'lost', 'scared', 'overwhelmed']
        stress_level = sum(1 for word in stress_words if word in user_input.lower()) / len(stress_words)
        if stress_level > 0:
            stimuli['stress'] = stress_level
        
        # Check for positive feedback
        if any(word in user_input.lower() for word in ['great', 'helpful', 'thanks', 'perfect', 'awesome']):
            stimuli['reward'] = 0.5
        
        # Information complexity
        word_count = len(user_input.split())
        if word_count > 50:
            stimuli['information'] = {
                'complexity': min(1.0, word_count / 100),
                'content': user_input[:100],
                'importance': 0.7 if any(word in user_input.lower() 
                                       for word in ['important', 'crucial', 'need', 'help']) else 0.3
            }
        
        # Check for questions about Aurora
        if any(phrase in user_input.lower() for phrase in ['about you', 'yourself', 'aurora']):
            stimuli['self_reflection'] = True
            
        return stimuli
    
    def _check_coffee_need(self, current_time: datetime):
        """Check if Aurora needs coffee based on brain state"""
        hours_since_coffee = (current_time - self.last_coffee_time).total_seconds() / 3600
        
        # Need coffee if energy low and been a while
        if (self.brain.energy_level < 0.5 and hours_since_coffee > 2) or \
           (self.brain.neurotransmitters['dopamine'] < 0.4 and hours_since_coffee > 1):
            self.coffee_count += 1
            self.last_coffee_time = current_time
            
            # Coffee effects
            self.brain.neurotransmitters['dopamine'] = min(1.0, 
                self.brain.neurotransmitters['dopamine'] + 0.2)
            self.brain.neurotransmitters['norepinephrine'] = min(1.0,
                self.brain.neurotransmitters['norepinephrine'] + 0.15)
            self.brain.energy_level = min(1.0, self.brain.energy_level + 0.3)
    
    def _generate_mood_description(self) -> str:
        """Generate Aurora's current mood description"""
        energy = self.brain.energy_level
        emotions = self.brain.emotional_state
        cognitive_load = self.brain.working_memory.cognitive_load
        
        # Determine primary mood
        if energy > 0.7 and emotions.emotions['joy'] > 0.6:
            mood = "hyped"
            description = "absolutely buzzing with energy"
        elif energy < 0.3 and cognitive_load > 0.7:
            mood = "burned out"
            description = "running on fumes"
        elif self.brain.neurotransmitters['cortisol'] > 0.6:
            mood = "anxious"
            description = "a bit on edge"
        elif emotions.emotions['joy'] > 0.5 and emotions.emotions['trust'] > 0.6:
            mood = "content"
            description = "in a good headspace"
        elif cognitive_load > 0.8:
            mood = "overwhelmed"
            description = "brain is at capacity"
        else:
            mood = "neutral"
            description = "just vibing"
        
        return description
    
    def _get_coffee_status(self) -> Dict:
        """Get Aurora's coffee consumption status"""
        return {
            'cups_today': self.coffee_count,
            'last_coffee': self.last_coffee_time.strftime("%I:%M %p"),
            'craving_level': 'high' if self.brain.energy_level < 0.4 else 'moderate'
        }
    
    def _extract_user_info(self, user_input: str) -> Dict:
        """Extract information about the user from their input"""
        info = {}
        
        # Name extraction with improved patterns
        name_patterns = [
            r"(?:my name is|i am|i'm|call me) (\w+)",
            r"(?:this is|it's) (\w+)",
            r"^(\w+) here",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                info['name'] = match.group(1).capitalize()
                break
        
        # Extract emotions/state
        if any(word in user_input.lower() for word in ['happy', 'excited', 'great']):
            info['emotional_state'] = 'positive'
        elif any(word in user_input.lower() for word in ['sad', 'worried', 'stressed']):
            info['emotional_state'] = 'needs_support'
            
        return info
    
    def _infer_user_emotions(self, user_input: str) -> Dict[str, float]:
        """Infer user's emotional state from their input"""
        emotions = {
            'joy': 0.0,
            'sadness': 0.0,
            'anxiety': 0.0,
            'frustration': 0.0,
            'excitement': 0.0
        }
        
        # Simple keyword-based emotion detection
        joy_indicators = ['happy', 'great', 'awesome', 'excited', 'love']
        sad_indicators = ['sad', 'down', 'depressed', 'unhappy']
        anxiety_indicators = ['worried', 'anxious', 'nervous', 'stressed']
        
        text_lower = user_input.lower()
        
        for word in joy_indicators:
            if word in text_lower:
                emotions['joy'] += 0.3
                
        for word in sad_indicators:
            if word in text_lower:
                emotions['sadness'] += 0.3
                
        for word in anxiety_indicators:
            if word in text_lower:
                emotions['anxiety'] += 0.3
        
        # Normalize emotions
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
            
        return emotions
    
    def _generate_empathy_response(self, user_input: str) -> Dict:
        """Generate empathetic response elements based on user's emotional state"""
        user_emotions = self._infer_user_emotions(user_input)
        
        response_elements = {
            'should_validate': False,
            'should_support': False,
            'should_encourage': False,
            'emotional_mirroring': 0.0
        }
        
        # High empathy due to Aurora's personality
        empathy_multiplier = self.brain.personality.traits[PersonalityTrait.AGREEABLENESS.value]
        
        if user_emotions['sadness'] > 0.3:
            response_elements['should_validate'] = True
            response_elements['should_support'] = True
            response_elements['emotional_mirroring'] = user_emotions['sadness'] * empathy_multiplier * 0.5
            
        if user_emotions['anxiety'] > 0.3:
            response_elements['should_validate'] = True
            response_elements['should_encourage'] = True
            
        if user_emotions['joy'] > 0.3:
            response_elements['emotional_mirroring'] = user_emotions['joy'] * empathy_multiplier * 0.7
            
        return response_elements
    
    def generate_response_modifiers(self) -> Dict:
        """Generate response style modifiers based on current brain state"""
        modifiers = {
            'typo_probability': 0.0,
            'use_caps': False,
            'use_ellipsis': False,
            'filler_words': [],
            'response_length': 'normal',
            'enthusiasm_level': 'moderate'
        }
        
        # High energy modifications
        if self.brain.energy_level > 0.7 and self.brain.neurotransmitters['dopamine'] > 0.7:
            modifiers['typo_probability'] = 0.15
            modifiers['use_caps'] = True
            modifiers['enthusiasm_level'] = 'high'
            modifiers['filler_words'] = ['literally', 'like', 'OMG']
            
        # Low energy modifications
        elif self.brain.energy_level < 0.4:
            modifiers['use_ellipsis'] = True
            modifiers['response_length'] = 'short'
            modifiers['filler_words'] = ['ugh', 'hmm']
            
        # High cognitive load
        if self.brain.working_memory.cognitive_load > 0.7:
            modifiers['filler_words'].extend(['wait', 'um', 'let me think'])
            
        # Anxiety modifications
        if self.brain.neurotransmitters['cortisol'] > 0.6:
            modifiers['filler_words'].extend(['maybe', 'I think', 'probably'])
            
        return modifiers
    
    def apply_personality_quirks(self, response: str) -> str:
        """Apply Aurora's personality quirks to the response"""
        modifiers = self.generate_response_modifiers()
        
        # Apply typos if high energy
        if modifiers['typo_probability'] > 0 and random.random() < modifiers['typo_probability']:
            typo_map = {
                'the': 'teh',
                'with': 'wiht',
                'really': 'realy',
                'because': 'becuase'
            }
            for correct, typo in typo_map.items():
                if correct in response and random.random() < 0.5:
                    response = response.replace(correct, typo, 1)
                    break
        
        # Add filler words
        if modifiers['filler_words'] and random.random() < 0.7:
            sentences = response.split('. ')
            if len(sentences) > 1:
                insert_pos = random.randint(0, len(sentences)-1)
                filler = random.choice(modifiers['filler_words'])
                sentences[insert_pos] = filler + ', ' + sentences[insert_pos]
                response = '. '.join(sentences)
        
        # Add caps for emphasis if hyped
        if modifiers['use_caps'] and random.random() < 0.6:
            words = response.split()
            if len(words) > 5:
                emphasis_word = random.choice([w for w in words if len(w) > 4])
                response = response.replace(emphasis_word, emphasis_word.upper(), 1)
        
        # Add ellipsis if tired
        if modifiers['use_ellipsis'] and not response.endswith('...'):
            response = response.rstrip('.!?') + '...'
            
        # Coffee references
        if self.coffee_count > 2 and random.random() < 0.3:
            coffee_insertions = [
                " (coffee number {} is kicking in)",
                " - fueled by {} espressos today -",
                " *sips coffee number {}*"
            ]
            insertion = random.choice(coffee_insertions).format(self.coffee_count)
            sentences = response.split('. ')
            if len(sentences) > 1:
                insert_pos = random.randint(0, len(sentences)-1)
                sentences[insert_pos] += insertion
                response = '. '.join(sentences)
                
        return response
    
    def get_prompt_additions(self) -> str:
        """Get additional context for the prompt based on brain state"""
        state = self.brain.get_current_state()
        
        additions = []
        
        # Energy state
        if state['energy_level'] < 0.3:
            additions.append("You're exhausted and running on empty. Keep responses shorter.")
        elif state['energy_level'] > 0.8:
            additions.append("You're WIRED and full of energy! Thoughts racing.")
            
        # Emotional state
        primary_emotion = max(state['emotional_state']['emotions'].items(), 
                            key=lambda x: x[1])
        if primary_emotion[1] > 0.6:
            additions.append(f"You're feeling particularly {primary_emotion[0]} right now.")
            
        # Cognitive load
        if state['cognitive_load'] > 0.7:
            additions.append("Your brain feels overloaded. You might lose track of thoughts.")
            
        # Coffee status
        if self.coffee_count > 3:
            additions.append(f"You've had {self.coffee_count} coffees today. Feeling jittery.")
            
        # Circadian influence
        if state['circadian']['alertness'] < 0.3:
            additions.append("It's that afternoon slump. Fighting to stay focused.")
            
        return "\n".join(additions)