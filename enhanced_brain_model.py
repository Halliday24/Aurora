# Enhanced Aurora Brain Simulation System
# Based on latest neuroscience research (2025)

import numpy as np
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import math
from enum import Enum

# Constants based on neuroscience research
NEUROTRANSMITTERS = {
    'dopamine': {'baseline': 0.5, 'min': 0.1, 'max': 1.0},
    'serotonin': {'baseline': 0.5, 'min': 0.1, 'max': 1.0},
    'oxytocin': {'baseline': 0.3, 'min': 0.1, 'max': 1.0},
    'cortisol': {'baseline': 0.3, 'min': 0.1, 'max': 0.9},
    'norepinephrine': {'baseline': 0.4, 'min': 0.1, 'max': 1.0},
    'gaba': {'baseline': 0.5, 'min': 0.2, 'max': 1.0},
    'glutamate': {'baseline': 0.5, 'min': 0.2, 'max': 1.0},
    'acetylcholine': {'baseline': 0.5, 'min': 0.2, 'max': 1.0},
    'endorphins': {'baseline': 0.3, 'min': 0.1, 'max': 1.0}
}

# Big Five Personality Traits
class PersonalityTrait(Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"

@dataclass
class CircadianRhythm:
    """Models 24-hour sleep-wake cycle"""
    phase: float = 0.0  # Current phase in 24-hour cycle (0-24)
    melatonin_level: float = 0.3
    body_temperature: float = 37.0  # Celsius
    alertness_rhythm: float = 0.7
    
    def update(self, current_time: datetime):
        """Update circadian phase based on time of day"""
        hour = current_time.hour + current_time.minute / 60.0
        self.phase = hour
        
        # Melatonin peaks around 3-4 AM, lowest around 3-4 PM
        melatonin_phase = (hour - 3) * (2 * math.pi / 24)
        self.melatonin_level = 0.5 + 0.4 * math.cos(melatonin_phase)
        
        # Body temperature lowest around 4 AM, highest around 6 PM
        temp_phase = (hour - 4) * (2 * math.pi / 24)
        self.body_temperature = 36.8 + 0.5 * math.sin(temp_phase)
        
        # Alertness peaks around 10 AM and 8 PM (biphasic)
        alert_phase1 = (hour - 10) * (2 * math.pi / 12)
        alert_phase2 = (hour - 20) * (2 * math.pi / 12)
        self.alertness_rhythm = 0.5 + 0.3 * math.cos(alert_phase1) + 0.2 * math.cos(alert_phase2)

@dataclass
class WorkingMemory:
    """Enhanced working memory with chunking capabilities"""
    capacity: int = 7  # Miller's Law: 7±2 items
    items: List[Dict] = field(default_factory=list)
    chunks: List[List[Dict]] = field(default_factory=list)
    cognitive_load: float = 0.0
    
    def add_item(self, item: Dict) -> bool:
        """Add item to working memory with chunking"""
        # Try to chunk with existing items
        chunked = False
        for chunk in self.chunks:
            if self._can_chunk_together(item, chunk):
                chunk.append(item)
                chunked = True
                break
        
        if not chunked:
            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                # Apply forgetting curve to make room
                self._forget_oldest()
                self.items.append(item)
        
        self._update_cognitive_load()
        return True
    
    def _can_chunk_together(self, item: Dict, chunk: List[Dict]) -> bool:
        """Determine if items can be chunked based on similarity"""
        if not chunk:
            return False
        
        # Simple similarity check - can be made more sophisticated
        similarity_threshold = 0.6
        similarities = []
        
        for chunk_item in chunk:
            similarity = self._calculate_similarity(item, chunk_item)
            similarities.append(similarity)
        
        return np.mean(similarities) > similarity_threshold
    
    def _calculate_similarity(self, item1: Dict, item2: Dict) -> float:
        """Calculate similarity between two items"""
        # Simple implementation - can be enhanced
        common_keys = set(item1.keys()) & set(item2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for k in common_keys if item1[k] == item2[k])
        return matches / len(common_keys)
    
    def _forget_oldest(self):
        """Remove oldest item based on forgetting curve"""
        if self.items:
            # Ebbinghaus forgetting curve influence
            forgetting_probability = 0.7
            if random.random() < forgetting_probability:
                self.items.pop(0)
    
    def _update_cognitive_load(self):
        """Update cognitive load based on items and chunks"""
        item_load = len(self.items) / self.capacity
        chunk_load = sum(len(chunk) for chunk in self.chunks) / (self.capacity * 2)
        self.cognitive_load = min(1.0, (item_load + chunk_load) / 2)

@dataclass
class EmotionalState:
    """Complex emotional state with multiple dimensions"""
    valence: float = 0.0  # -1 (negative) to 1 (positive)
    arousal: float = 0.5  # 0 (calm) to 1 (excited)
    emotions: Dict[str, float] = field(default_factory=lambda: {
        'joy': 0.0,
        'sadness': 0.0,
        'anger': 0.0,
        'fear': 0.0,
        'surprise': 0.0,
        'disgust': 0.0,
        'trust': 0.5,
        'anticipation': 0.5,
        'love': 0.0,
        'guilt': 0.0,
        'shame': 0.0,
        'pride': 0.0
    })
    
    def update_from_neurotransmitters(self, neurotransmitters: Dict[str, float]):
        """Update emotional state based on neurotransmitter levels"""
        # Dopamine influences joy and anticipation
        self.emotions['joy'] = 0.3 + 0.7 * neurotransmitters['dopamine']
        self.emotions['anticipation'] = 0.2 + 0.8 * neurotransmitters['dopamine']
        
        # Serotonin influences mood stability and trust
        self.emotions['trust'] = 0.3 + 0.7 * neurotransmitters['serotonin']
        self.emotions['sadness'] = max(0, 1.0 - neurotransmitters['serotonin'])
        
        # Oxytocin influences love and bonding
        self.emotions['love'] = 0.2 + 0.8 * neurotransmitters['oxytocin']
        
        # Cortisol influences fear and anger
        self.emotions['fear'] = 0.1 + 0.6 * neurotransmitters['cortisol']
        self.emotions['anger'] = 0.1 + 0.4 * neurotransmitters['cortisol']
        
        # Calculate overall valence and arousal
        positive_emotions = ['joy', 'trust', 'anticipation', 'love', 'pride']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'guilt', 'shame']
        
        pos_sum = sum(self.emotions[e] for e in positive_emotions)
        neg_sum = sum(self.emotions[e] for e in negative_emotions)
        
        self.valence = (pos_sum - neg_sum) / (len(positive_emotions) + len(negative_emotions))
        self.arousal = (neurotransmitters['norepinephrine'] + neurotransmitters['dopamine']) / 2

@dataclass
class TheoryOfMind:
    """Model for understanding others' mental states"""
    self_model: Dict = field(default_factory=dict)
    other_models: Dict[str, Dict] = field(default_factory=dict)
    empathy_level: float = 0.5
    perspective_taking_ability: float = 0.7
    
    def infer_other_mental_state(self, person_id: str, observed_behavior: Dict) -> Dict:
        """Infer another person's mental state from their behavior"""
        if person_id not in self.other_models:
            self.other_models[person_id] = {
                'beliefs': {},
                'desires': {},
                'emotions': {},
                'intentions': {}
            }
        
        # Simple inference based on behavior
        # This can be made much more sophisticated
        inferred_state = self.other_models[person_id].copy()
        
        # Update based on observed behavior
        if 'emotion_expression' in observed_behavior:
            inferred_state['emotions'] = observed_behavior['emotion_expression']
        
        if 'action' in observed_behavior:
            # Infer intention from action
            inferred_state['intentions']['current'] = self._infer_intention(observed_behavior['action'])
        
        return inferred_state
    
    def _infer_intention(self, action: str) -> str:
        """Simple intention inference from action"""
        # This would be much more complex in reality
        intention_map = {
            'helping': 'wants to assist',
            'withdrawing': 'needs space',
            'approaching': 'seeks connection',
            'questioning': 'seeks understanding'
        }
        return intention_map.get(action, 'unknown intention')
    
    def generate_empathetic_response(self, other_state: Dict) -> Dict:
        """Generate empathetic response based on understanding of other's state"""
        response = {
            'emotional_mirroring': self.empathy_level * 0.7,
            'cognitive_understanding': self.perspective_taking_ability,
            'supportive_action': 'offer_help' if other_state.get('emotions', {}).get('sadness', 0) > 0.5 else 'listen'
        }
        return response

@dataclass
class Personality:
    """Big Five personality model"""
    traits: Dict[str, float] = field(default_factory=lambda: {
        PersonalityTrait.OPENNESS.value: 0.7,
        PersonalityTrait.CONSCIENTIOUSNESS.value: 0.6,
        PersonalityTrait.EXTRAVERSION.value: 0.5,
        PersonalityTrait.AGREEABLENESS.value: 0.8,
        PersonalityTrait.NEUROTICISM.value: 0.4
    })
    
    def influence_behavior(self, base_behavior: Dict) -> Dict:
        """Modify behavior based on personality traits"""
        modified_behavior = base_behavior.copy()
        
        # Extraversion influences social behavior
        if self.traits[PersonalityTrait.EXTRAVERSION.value] > 0.6:
            modified_behavior['social_energy'] = modified_behavior.get('social_energy', 0.5) * 1.3
            modified_behavior['talkativeness'] = 'high'
        
        # Agreeableness influences cooperation
        if self.traits[PersonalityTrait.AGREEABLENESS.value] > 0.7:
            modified_behavior['cooperation'] = 'high'
            modified_behavior['conflict_avoidance'] = True
        
        # Neuroticism influences emotional stability
        if self.traits[PersonalityTrait.NEUROTICISM.value] > 0.6:
            modified_behavior['emotional_volatility'] = 'high'
            modified_behavior['stress_sensitivity'] = 1.5
        
        return modified_behavior

class EnhancedBrainModel:
    """Complete enhanced brain model integrating all systems"""
    
    def __init__(self):
        # Core systems
        self.neurotransmitters = {k: v['baseline'] for k, v in NEUROTRANSMITTERS.items()}
        self.circadian = CircadianRhythm()
        self.working_memory = WorkingMemory()
        self.emotional_state = EmotionalState()
        self.theory_of_mind = TheoryOfMind()
        self.personality = Personality()
        
        # Additional systems
        self.energy_level = 0.8
        self.stress_level = 0.3
        self.learning_rate = 0.7
        self.attention_focus = 1.0
        self.creativity_level = 0.6
        
        # Memory systems
        self.episodic_memory = []
        self.semantic_memory = {}
        self.procedural_memory = {}
        
        # Time tracking
        self.last_update = datetime.now()
        self.awake_time = 0
        self.sleep_debt = 0
        
    def update(self, current_time: datetime, external_stimuli: Dict):
        """Main update function integrating all brain systems"""
        time_delta = (current_time - self.last_update).total_seconds() / 60  # minutes
        
        # Update circadian rhythm
        self.circadian.update(current_time)
        
        # Update neurotransmitters based on various factors
        self._update_neurotransmitters(time_delta, external_stimuli)
        
        # Update emotional state
        self.emotional_state.update_from_neurotransmitters(self.neurotransmitters)
        
        # Update energy and sleep pressure
        self._update_energy_and_sleep(time_delta)
        
        # Update attention and cognitive resources
        self._update_cognitive_resources()
        
        # Process external stimuli
        self._process_stimuli(external_stimuli)
        
        self.last_update = current_time
    
    def _update_neurotransmitters(self, time_delta: float, stimuli: Dict):
        """Complex neurotransmitter dynamics"""
        # Natural decay/recovery toward baseline
        for nt, params in NEUROTRANSMITTERS.items():
            current = self.neurotransmitters[nt]
            baseline = params['baseline']
            
            # Move toward baseline with time
            decay_rate = 0.02 * time_delta
            if current > baseline:
                self.neurotransmitters[nt] = max(baseline, current - decay_rate)
            else:
                self.neurotransmitters[nt] = min(baseline, current + decay_rate)
        
        # Circadian influence
        self.neurotransmitters['dopamine'] *= (0.8 + 0.4 * self.circadian.alertness_rhythm)
        self.neurotransmitters['serotonin'] *= (0.9 + 0.2 * (1 - self.circadian.melatonin_level))
        
        # Stimulus-based changes
        if stimuli.get('social_interaction'):
            self.neurotransmitters['oxytocin'] = min(1.0, self.neurotransmitters['oxytocin'] + 0.1)
            self.neurotransmitters['dopamine'] = min(1.0, self.neurotransmitters['dopamine'] + 0.05)
        
        if stimuli.get('stress'):
            stress_intensity = stimuli['stress']
            self.neurotransmitters['cortisol'] = min(0.9, self.neurotransmitters['cortisol'] + 0.1 * stress_intensity)
            self.neurotransmitters['norepinephrine'] = min(1.0, self.neurotransmitters['norepinephrine'] + 0.08 * stress_intensity)
        
        if stimuli.get('reward'):
            self.neurotransmitters['dopamine'] = min(1.0, self.neurotransmitters['dopamine'] + 0.15)
            self.neurotransmitters['endorphins'] = min(1.0, self.neurotransmitters['endorphins'] + 0.1)
    
    def _update_energy_and_sleep(self, time_delta: float):
        """Update energy levels and sleep pressure"""
        # Increase awake time
        self.awake_time += time_delta
        
        # Sleep pressure increases with time awake (Process S)
        sleep_pressure = 1 - math.exp(-self.awake_time / 480)  # 8 hour half-life
        
        # Circadian influence on alertness (Process C)
        circadian_alertness = self.circadian.alertness_rhythm
        
        # Combined energy level
        self.energy_level = max(0.1, min(1.0, 
            (1 - sleep_pressure) * 0.6 + circadian_alertness * 0.4 - self.sleep_debt * 0.1
        ))
        
        # Accumulate sleep debt if energy too low
        if self.energy_level < 0.3:
            self.sleep_debt += time_delta * 0.02
    
    def _update_cognitive_resources(self):
        """Update attention, creativity, and learning based on brain state"""
        # Attention influenced by arousal and energy
        optimal_arousal = 0.6
        arousal_diff = abs(self.emotional_state.arousal - optimal_arousal)
        self.attention_focus = self.energy_level * (1 - arousal_diff)
        
        # Creativity peaks at moderate arousal and high dopamine
        self.creativity_level = (
            self.neurotransmitters['dopamine'] * 0.4 +
            (1 - abs(self.emotional_state.arousal - 0.5) * 2) * 0.3 +
            self.personality.traits[PersonalityTrait.OPENNESS.value] * 0.3
        )
        
        # Learning rate affected by attention and neurotransmitters
        self.learning_rate = (
            self.attention_focus * 0.4 +
            self.neurotransmitters['acetylcholine'] * 0.3 +
            self.neurotransmitters['dopamine'] * 0.3
        )
    
    def _process_stimuli(self, stimuli: Dict):
        """Process external stimuli and update internal state"""
        # Add to working memory
        if 'information' in stimuli:
            self.working_memory.add_item(stimuli['information'])
        
        # Update stress level
        if 'stressor' in stimuli:
            self.stress_level = min(1.0, self.stress_level + stimuli['stressor'] * 0.1)
        else:
            self.stress_level = max(0.0, self.stress_level - 0.01)
        
        # Social processing
        if 'social_cue' in stimuli:
            other_state = self.theory_of_mind.infer_other_mental_state(
                stimuli.get('person_id', 'unknown'),
                stimuli['social_cue']
            )
            
            # Empathetic response influences own emotional state
            if self.theory_of_mind.empathy_level > 0.6:
                # Emotional contagion
                if 'emotions' in other_state:
                    for emotion, value in other_state['emotions'].items():
                        if emotion in self.emotional_state.emotions:
                            self.emotional_state.emotions[emotion] += value * 0.2 * self.theory_of_mind.empathy_level
    
    def get_current_state(self) -> Dict:
        """Get comprehensive current brain state"""
        return {
            'neurotransmitters': self.neurotransmitters.copy(),
            'energy_level': self.energy_level,
            'cognitive_load': self.working_memory.cognitive_load,
            'emotional_state': {
                'valence': self.emotional_state.valence,
                'arousal': self.emotional_state.arousal,
                'emotions': self.emotional_state.emotions.copy()
            },
            'circadian': {
                'phase': self.circadian.phase,
                'alertness': self.circadian.alertness_rhythm,
                'melatonin': self.circadian.melatonin_level
            },
            'cognitive_resources': {
                'attention': self.attention_focus,
                'creativity': self.creativity_level,
                'learning_rate': self.learning_rate
            },
            'personality_influence': self.personality.influence_behavior({
                'base_mood': self.emotional_state.valence
            })
        }
    
    def simulate_sleep(self, hours: float):
        """Simulate sleep and recovery processes"""
        # Reset awake time
        self.awake_time = 0
        
        # Reduce sleep debt
        self.sleep_debt = max(0, self.sleep_debt - hours * 0.125)
        
        # Restore energy
        self.energy_level = min(1.0, 0.3 + hours * 0.1)
        
        # Normalize neurotransmitters
        for nt in self.neurotransmitters:
            baseline = NEUROTRANSMITTERS[nt]['baseline']
            current = self.neurotransmitters[nt]
            recovery_rate = 0.15 * hours / 8  # Full recovery in 8 hours
            
            if current > baseline:
                self.neurotransmitters[nt] = max(baseline, current - recovery_rate)
            else:
                self.neurotransmitters[nt] = min(baseline, current + recovery_rate)
        
        # Memory consolidation
        self._consolidate_memories()
        
    def _consolidate_memories(self):
        """Simulate memory consolidation during sleep"""
        # Transfer important items from working memory to long-term memory
        important_items = [item for item in self.working_memory.items 
                          if item.get('importance', 0) > 0.5]
        
        for item in important_items:
            self.episodic_memory.append({
                'content': item,
                'timestamp': datetime.now(),
                'emotional_context': self.emotional_state.emotions.copy()
            })
        
        # Clear working memory
        self.working_memory.items.clear()
        self.working_memory.chunks.clear()
        self.working_memory.cognitive_load = 0.0