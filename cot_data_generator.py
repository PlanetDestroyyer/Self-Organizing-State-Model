"""
Chain-of-Thought Data Generator for SOSM
=========================================

Generates synthetic reasoning demonstrations for Wikipedia text.
Injects <think>...</think> blocks to help the model learn structured reasoning.
"""
import re
import random
from typing import List, Dict, Tuple


class CoTDataGenerator:
    """Generate Chain-of-Thought demonstrations from Wikipedia text."""
    
    # Topic keywords for categorization
    TOPIC_KEYWORDS = {
        'biography': ['born', 'died', 'was', 'politician', 'actor', 'scientist', 'writer'],
        'history': ['war', 'battle', 'century', 'ancient', 'empire', 'revolution'],
        'geography': ['country', 'city', 'mountain', 'river', 'ocean', 'continent'],
        'science': ['theory', 'experiment', 'discovered', 'formula', 'physics', 'chemistry'],
        'astronomy': ['planet', 'star', 'galaxy', 'solar', 'orbit', 'moon'],
        'sports': ['team', 'championship', 'player', 'game', 'season', 'league'],
        'music': ['album', 'song', 'band', 'musician', 'released', 'record']
    }
    
    def __init__(self):
        self.cot_templates = self._create_templates()
    
    def _create_templates(self) -> List[str]:
        """Create CoT templates for different text patterns."""
        return [
            # Topic identification
            "<think> topic: {topic}. </think>",
            
            # Entity-based
            "<think> about: {entity}. category: {category}. </think>",
            
            # Temporal
            "<think> time: {year}. event: {event}. </think>",
            
            # Factual
            "<think> fact: {subject} {verb} {object}. </think>"
        ]
    
    def _detect_topic(self, text: str) -> str:
        """Detect topic from text using keyword matching."""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        return "general"
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities (capitalized words/phrases)."""
        # Simple heuristic: find capitalized words
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        return entities[:3]  # Limit to 3 entities
    
    def _extract_year(self, text: str) -> str:
        """Extract year mentions."""
        years = re.findall(r'\b(1\d{3}|20\d{2})\b', text)
        if years:
            return years[0]
        return ""
    
    def generate_cot_simple(self, text: str) -> str:
        """
        Generate simple CoT by injecting topic detection.
        
        Strategy: Insert <think>topic: X</think> at the start of text.
        """
        topic = self._detect_topic(text)
        cot = f"<think> topic: {topic}. </think> "
        return cot + text
    
    def generate_cot_entity(self, text: str) -> str:
        """
        Generate entity-based CoT.
        
        Strategy: Detect entities and insert reasoning about them.
        """
        entities = self._extract_entities(text)
        topic = self._detect_topic(text)
        
        if entities:
            entity_list = ", ".join(entities[:2])
            cot = f"<think> entities: {entity_list}. topic: {topic}. </think> "
            return cot + text
        else:
            return self.generate_cot_simple(text)
    
    def generate_cot_temporal(self, text: str) -> str:
        """
        Generate temporal CoT for texts with dates/years.
        
        Strategy: Extract years and create time-based reasoning.
        """
        year = self._extract_year(text)
        topic = self._detect_topic(text)
        
        if year:
            cot = f"<think> year: {year}. topic: {topic}. </think> "
            return cot + text
        else:
            return self.generate_cot_entity(text)
    
    def generate_cot_mixed(self, text: str) -> str:
        """
        Generate mixed CoT using multiple strategies.
        
        Randomly selects which CoT strategy to use.
        """
        strategies = [
            self.generate_cot_simple,
            self.generate_cot_entity,
            self.generate_cot_temporal
        ]
        strategy = random.choice(strategies)
        return strategy(text)
    
    def process_batch(self, texts: List[str], cot_ratio: float = 0.5) -> List[str]:
        """
        Process a batch of texts, adding CoT to a portion.
        
        Args:
            texts: List of text strings
            cot_ratio: Fraction of texts to augment with CoT (0.0 to 1.0)
        
        Returns:
            List of texts (some with CoT, some without)
        """
        result = []
        for text in texts:
            if random.random() < cot_ratio:
                # Add CoT
                result.append(self.generate_cot_mixed(text))
            else:
                # Keep original
                result.append(text)
        return result


if __name__ == "__main__":
    # Test the generator
    generator = CoTDataGenerator()
    
    test_texts = [
        "The solar system consists of the Sun and planets that orbit around it.",
        "Albert Einstein was born in 1879 in Germany. He developed the theory of relativity.",
        "The Great Wall of China was built over many centuries to protect against invasions.",
        "Machine learning is a field of artificial intelligence focused on data-driven algorithms."
    ]
    
    print("=" * 70)
    print("CHAIN-OF-THOUGHT DATA GENERATOR TEST")
    print("=" * 70)
    
    for i, text in enumerate(test_texts):
        print(f"\n{i+1}. Original:")
        print(f"   {text}")
        
        cot_text = generator.generate_cot_mixed(text)
        print(f"   With CoT:")
        print(f"   {cot_text}")
