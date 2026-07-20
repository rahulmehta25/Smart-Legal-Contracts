import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class DetectionResult:
    """Container for detection results"""
    is_arbitration: bool
    confidence: float
    text_span: str
    start_idx: int
    end_idx: int
    pattern_matches: List[str]
    semantic_score: float

class LegalBERTDetector:
    def __init__(self, model_name: str = 'nlpaueb/legal-bert-base-uncased'):
        """Initialize Legal-BERT model for arbitration detection"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Load pre-trained arbitration classifier head
        self.classifier = self._initialize_classifier()
        
        # Initialize pattern matcher
        self.pattern_matcher = ArbitrationPatternMatcher()
        
    def _initialize_classifier(self):
        """Initialize or load fine-tuned classification head"""
        import torch.nn as nn
        
        classifier = nn.Sequential(
            nn.Linear(768, 256),  # Legal-BERT hidden size is 768
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Binary classification
        ).to(self.device)
        
        # Load pre-trained weights if available
        try:
            classifier.load_state_dict(
                torch.load('models/arbitration_classifier.pth', 
                          map_location=self.device)
            )
        except FileNotFoundError:
            print("No pre-trained classifier found. Using random initialization.")
            
        return classifier
    
    def detect(self, text: str, threshold: float = 0.7) -> DetectionResult:
        """
        Detect arbitration clause in text
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for positive detection
        """
        # Get pattern matching scores
        pattern_results = self.pattern_matcher.match(text)
        
        # Get semantic embedding
        embedding = self._get_embedding(text)
        
        # Run through classifier
        with torch.no_grad():
            logits = self.classifier(embedding)
            probs = torch.softmax(logits, dim=-1)
            semantic_score = probs[0, 1].item()  # Probability of arbitration class
        
        # Combine scores
        combined_confidence = self._combine_scores(
            semantic_score, 
            pattern_results['confidence']
        )
        
        return DetectionResult(
            is_arbitration=combined_confidence >= threshold,
            confidence=combined_confidence,
            text_span=text[:500],  # First 500 chars for preview
            start_idx=0,
            end_idx=len(text),
            pattern_matches=pattern_results['matches'],
            semantic_score=semantic_score
        )
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Generate Legal-BERT embedding for text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
            
        return embedding
    
    def _combine_scores(self, semantic_score: float, pattern_score: float) -> float:
        """Combine semantic and pattern matching scores"""
        # Weighted average with higher weight on semantic score
        return (0.7 * semantic_score + 0.3 * pattern_score)
