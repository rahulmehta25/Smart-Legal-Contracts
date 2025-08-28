"""Legal-BERT detector for arbitration clause detection."""
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Container for detection results."""
    is_arbitration: bool
    confidence: float
    text_span: str
    start_idx: int
    end_idx: int
    pattern_matches: List[str]
    semantic_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_arbitration': self.is_arbitration,
            'confidence': self.confidence,
            'text_span': self.text_span,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'pattern_matches': self.pattern_matches,
            'semantic_score': self.semantic_score
        }

class LegalBERTDetector:
    """Legal-BERT based arbitration clause detector."""
    
    def __init__(self, model_name: str = 'nlpaueb/legal-bert-base-uncased'):
        """Initialize Legal-BERT model for arbitration detection."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a smaller model if Legal-BERT fails
            model_name = 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info(f"Fallback to model: {model_name}")
        
        # Load pre-trained arbitration classifier head
        self.classifier = self._initialize_classifier()
        
        # Initialize pattern matcher
        from .pattern_matcher import ArbitrationPatternMatcher
        self.pattern_matcher = ArbitrationPatternMatcher()
        
    def _initialize_classifier(self):
        """Initialize or load fine-tuned classification head."""
        import torch.nn as nn
        
        # Get the hidden size from the model config
        hidden_size = self.model.config.hidden_size
        
        classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Binary classification
        ).to(self.device)
        
        # Load pre-trained weights if available
        try:
            import os
            model_path = os.path.join(
                os.path.dirname(__file__), 
                '../../models/classifiers/arbitration_classifier.pth'
            )
            if os.path.exists(model_path):
                classifier.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                logger.info("Loaded pre-trained classifier weights")
        except Exception as e:
            logger.warning(f"No pre-trained classifier found: {e}. Using random initialization.")
            
        return classifier
    
    def detect(self, text: str, threshold: float = 0.7) -> DetectionResult:
        """
        Detect arbitration clause in text.
        
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
        """Generate Legal-BERT embedding for text."""
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
        """Combine semantic and pattern matching scores."""
        # Weighted average with higher weight on semantic score
        return (0.7 * semantic_score + 0.3 * pattern_score)
    
    def batch_detect(self, texts: List[str], threshold: float = 0.7) -> List[DetectionResult]:
        """
        Detect arbitration clauses in multiple texts.
        
        Args:
            texts: List of texts to analyze
            threshold: Confidence threshold
        """
        results = []
        for text in texts:
            try:
                result = self.detect(text, threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                results.append(DetectionResult(
                    is_arbitration=False,
                    confidence=0.0,
                    text_span="",
                    start_idx=0,
                    end_idx=0,
                    pattern_matches=[],
                    semantic_score=0.0
                ))
        return results