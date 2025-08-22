"""
Named Entity Recognition for legal entities in arbitration clauses
Identifies key legal entities, organizations, and procedural terms
"""
import logging
import re
from typing import List, Dict, Tuple, Set, Optional
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import mlflow
import mlflow.spacy
from pathlib import Path
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalEntityRecognizer:
    """
    Specialized NER for legal entities in arbitration clauses
    """
    
    def __init__(self, 
                 model_name: str = "en_core_web_sm",
                 model_save_path: str = "backend/models/legal_ner",
                 experiment_name: str = "legal_ner"):
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.experiment_name = experiment_name
        self.nlp = None
        self.custom_patterns = self._create_legal_patterns()
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
        
        # Ensure save directory exists
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    def _create_legal_patterns(self) -> List[Dict]:
        """
        Create pattern-based rules for legal entity recognition
        """
        patterns = [
            # Arbitration organizations
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "american"}, {"LOWER": "arbitration"}, {"LOWER": "association"}]},
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "aaa"}]},
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "jams"}]},
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "international"}, {"LOWER": "chamber"}, {"LOWER": "of"}, {"LOWER": "commerce"}]},
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "icc"}]},
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "london"}, {"LOWER": "court"}, {"LOWER": "of"}, {"LOWER": "international"}, {"LOWER": "arbitration"}]},
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "lcia"}]},
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "singapore"}, {"LOWER": "international"}, {"LOWER": "arbitration"}, {"LOWER": "centre"}]},
            {"label": "ARBITRATION_ORG", "pattern": [{"LOWER": "siac"}]},
            
            # Legal procedures
            {"label": "ARBITRATION_PROC", "pattern": [{"LOWER": "binding"}, {"LOWER": "arbitration"}]},
            {"label": "ARBITRATION_PROC", "pattern": [{"LOWER": "final"}, {"LOWER": "arbitration"}]},
            {"label": "ARBITRATION_PROC", "pattern": [{"LOWER": "mandatory"}, {"LOWER": "arbitration"}]},
            {"label": "ARBITRATION_PROC", "pattern": [{"LOWER": "compulsory"}, {"LOWER": "arbitration"}]},
            {"label": "ARBITRATION_PROC", "pattern": [{"LOWER": "arbitration"}, {"LOWER": "proceedings"}]},
            {"label": "ARBITRATION_PROC", "pattern": [{"LOWER": "arbitral"}, {"LOWER": "tribunal"}]},
            {"label": "ARBITRATION_PROC", "pattern": [{"LOWER": "arbitral"}, {"LOWER": "proceedings"}]},
            
            # Legal jurisdictions
            {"label": "JURISDICTION", "pattern": [{"LOWER": "federal"}, {"LOWER": "court"}]},
            {"label": "JURISDICTION", "pattern": [{"LOWER": "state"}, {"LOWER": "court"}]},
            {"label": "JURISDICTION", "pattern": [{"LOWER": "district"}, {"LOWER": "court"}]},
            {"label": "JURISDICTION", "pattern": [{"LOWER": "supreme"}, {"LOWER": "court"}]},
            {"label": "JURISDICTION", "pattern": [{"LOWER": "court"}, {"LOWER": "of"}, {"LOWER": "competent"}, {"LOWER": "jurisdiction"}]},
            
            # Contract terms
            {"label": "CONTRACT_TERM", "pattern": [{"LOWER": "dispute"}, {"LOWER": "resolution"}]},
            {"label": "CONTRACT_TERM", "pattern": [{"LOWER": "alternative"}, {"LOWER": "dispute"}, {"LOWER": "resolution"}]},
            {"label": "CONTRACT_TERM", "pattern": [{"LOWER": "adr"}]},
            {"label": "CONTRACT_TERM", "pattern": [{"LOWER": "class"}, {"LOWER": "action"}, {"LOWER": "waiver"}]},
            {"label": "CONTRACT_TERM", "pattern": [{"LOWER": "jury"}, {"LOWER": "trial"}, {"LOWER": "waiver"}]},
            {"label": "CONTRACT_TERM", "pattern": [{"LOWER": "governing"}, {"LOWER": "law"}]},
            {"label": "CONTRACT_TERM", "pattern": [{"LOWER": "applicable"}, {"LOWER": "law"}]},
            
            # Legal parties
            {"label": "LEGAL_PARTY", "pattern": [{"LOWER": "claimant"}]},
            {"label": "LEGAL_PARTY", "pattern": [{"LOWER": "respondent"}]},
            {"label": "LEGAL_PARTY", "pattern": [{"LOWER": "plaintiff"}]},
            {"label": "LEGAL_PARTY", "pattern": [{"LOWER": "defendant"}]},
            {"label": "LEGAL_PARTY", "pattern": [{"LOWER": "contracting"}, {"LOWER": "parties"}]},
            
            # Arbitration rules
            {"label": "ARBITRATION_RULES", "pattern": [{"LOWER": "uncitral"}, {"LOWER": "rules"}]},
            {"label": "ARBITRATION_RULES", "pattern": [{"LOWER": "icc"}, {"LOWER": "rules"}]},
            {"label": "ARBITRATION_RULES", "pattern": [{"LOWER": "aaa"}, {"LOWER": "rules"}]},
            {"label": "ARBITRATION_RULES", "pattern": [{"LOWER": "jams"}, {"LOWER": "rules"}]},
            {"label": "ARBITRATION_RULES", "pattern": [{"LOWER": "lcia"}, {"LOWER": "rules"}]},
        ]
        
        return patterns
    
    def load_base_model(self):
        """
        Load base spaCy model and add custom patterns
        """
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logger.warning(f"Model {self.model_name} not found. Installing...")
            spacy.cli.download(self.model_name)
            self.nlp = spacy.load(self.model_name)
        
        # Add entity ruler for pattern matching
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(self.custom_patterns)
        
        logger.info("Base model loaded with custom legal patterns")
    
    def create_training_data(self) -> List[Tuple[str, Dict]]:
        """
        Create training data for legal NER
        """
        training_data = [
            (
                "Any dispute arising under this agreement shall be resolved through binding arbitration administered by the American Arbitration Association.",
                {
                    "entities": [
                        (87, 106, "ARBITRATION_PROC"),
                        (122, 154, "ARBITRATION_ORG")
                    ]
                }
            ),
            (
                "The parties agree to submit any controversy to arbitration under the rules of JAMS.",
                {
                    "entities": [
                        (53, 64, "ARBITRATION_PROC"),
                        (85, 89, "ARBITRATION_ORG")
                    ]
                }
            ),
            (
                "All disputes shall be finally settled by arbitration administered by ICC.",
                {
                    "entities": [
                        (40, 51, "ARBITRATION_PROC"),
                        (68, 71, "ARBITRATION_ORG")
                    ]
                }
            ),
            (
                "Any claim or dispute shall be resolved exclusively through binding arbitration under UNCITRAL rules.",
                {
                    "entities": [
                        (66, 85, "ARBITRATION_PROC"),
                        (92, 106, "ARBITRATION_RULES")
                    ]
                }
            ),
            (
                "The arbitral tribunal shall have exclusive jurisdiction over any disputes.",
                {
                    "entities": [
                        (4, 21, "ARBITRATION_PROC")
                    ]
                }
            ),
            (
                "Disputes will be resolved in the federal court of the Southern District of New York.",
                {
                    "entities": [
                        (33, 46, "JURISDICTION"),
                        (54, 83, "JURISDICTION")
                    ]
                }
            ),
            (
                "This agreement includes a class action waiver and jury trial waiver.",
                {
                    "entities": [
                        (28, 47, "CONTRACT_TERM"),
                        (52, 67, "CONTRACT_TERM")
                    ]
                }
            ),
            (
                "The claimant and respondent shall each appoint one arbitrator.",
                {
                    "entities": [
                        (4, 13, "LEGAL_PARTY"),
                        (18, 28, "LEGAL_PARTY")
                    ]
                }
            ),
            (
                "This contract shall be governed by the laws of Delaware and disputes resolved under LCIA rules.",
                {
                    "entities": [
                        (87, 97, "ARBITRATION_RULES")
                    ]
                }
            ),
            (
                "Any arbitration proceedings shall be conducted in accordance with Singapore International Arbitration Centre procedures.",
                {
                    "entities": [
                        (4, 28, "ARBITRATION_PROC"),
                        (69, 108, "ARBITRATION_ORG")
                    ]
                }
            )
        ]
        
        return training_data
    
    def train_custom_ner(self, 
                        training_data: List[Tuple[str, Dict]] = None,
                        iterations: int = 100,
                        dropout: float = 0.5,
                        learning_rate: float = 0.001) -> spacy.Language:
        """
        Train custom NER model for legal entities
        """
        if training_data is None:
            training_data = self.create_training_data()
        
        with mlflow.start_run(run_name="legal_ner_training"):
            mlflow.log_params({
                "base_model": self.model_name,
                "iterations": iterations,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "training_samples": len(training_data)
            })
            
            # Load base model
            self.load_base_model()
            
            # Get the NER component
            ner = self.nlp.get_pipe("ner")
            
            # Add new entity labels
            for _, annotations in training_data:
                for ent in annotations.get("entities"):
                    ner.add_label(ent[2])
            
            # Disable other pipeline components during training
            pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
            unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]
            
            # Training loop
            with self.nlp.disable_pipes(*unaffected_pipes):
                # Initialize the model
                self.nlp.begin_training()
                
                for iteration in range(iterations):
                    random.shuffle(training_data)
                    losses = {}
                    
                    # Create batches and train
                    batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                    
                    for batch in batches:
                        examples = []
                        for text, annotations in batch:
                            doc = self.nlp.make_doc(text)
                            example = Example.from_dict(doc, annotations)
                            examples.append(example)
                        
                        self.nlp.update(examples, drop=dropout, losses=losses)
                    
                    if iteration % 10 == 0:
                        logger.info(f"Iteration {iteration}, Losses: {losses}")
                        mlflow.log_metric("training_loss", losses.get("ner", 0), step=iteration)
            
            # Save the trained model
            self.nlp.to_disk(self.model_save_path)
            
            # Log model to MLflow
            mlflow.spacy.log_model(
                self.nlp,
                "legal_ner_model",
                registered_model_name="legal_ner_classifier"
            )
            
            logger.info(f"Model trained and saved to {self.model_save_path}")
            return self.nlp
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract legal entities from text
        """
        if self.nlp is None:
            self.load_base_model()
        
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, "_.confidence", 1.0)
            })
        
        return entities
    
    def extract_arbitration_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract arbitration-specific entities and categorize them
        """
        entities = self.extract_entities(text)
        
        categorized_entities = {
            "arbitration_organizations": [],
            "arbitration_procedures": [],
            "arbitration_rules": [],
            "jurisdictions": [],
            "contract_terms": [],
            "legal_parties": []
        }
        
        label_mapping = {
            "ARBITRATION_ORG": "arbitration_organizations",
            "ARBITRATION_PROC": "arbitration_procedures", 
            "ARBITRATION_RULES": "arbitration_rules",
            "JURISDICTION": "jurisdictions",
            "CONTRACT_TERM": "contract_terms",
            "LEGAL_PARTY": "legal_parties"
        }
        
        for entity in entities:
            category = label_mapping.get(entity["label"])
            if category:
                categorized_entities[category].append(entity["text"])
        
        # Remove duplicates while preserving order
        for category in categorized_entities:
            categorized_entities[category] = list(dict.fromkeys(categorized_entities[category]))
        
        return categorized_entities
    
    def evaluate_ner_model(self, test_data: List[Tuple[str, Dict]]) -> Dict[str, float]:
        """
        Evaluate NER model performance
        """
        if self.nlp is None:
            self.load_base_model()
        
        y_true = []
        y_pred = []
        
        for text, annotations in test_data:
            # Get predictions
            doc = self.nlp(text)
            pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            
            # Get ground truth
            true_entities = annotations.get("entities", [])
            
            # Convert to BIO tags for evaluation
            tokens = text.split()
            true_tags = self._entities_to_bio_tags(tokens, true_entities, text)
            pred_tags = self._entities_to_bio_tags(tokens, pred_entities, text)
            
            y_true.extend(true_tags)
            y_pred.extend(pred_tags)
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        metrics = {
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"]
        }
        
        logger.info(f"NER evaluation metrics: {metrics}")
        return metrics
    
    def _entities_to_bio_tags(self, tokens: List[str], entities: List[Tuple], text: str) -> List[str]:
        """
        Convert entity spans to BIO tags
        """
        tags = ["O"] * len(tokens)
        
        # Simple approximation - in practice, need proper tokenization alignment
        for start, end, label in entities:
            entity_text = text[start:end].lower()
            for i, token in enumerate(tokens):
                if token.lower() in entity_text:
                    if tags[i] == "O":
                        tags[i] = f"B-{label}"
                    else:
                        tags[i] = f"I-{label}"
        
        return tags
    
    def create_entity_patterns_from_text(self, texts: List[str]) -> List[Dict]:
        """
        Automatically discover entity patterns from legal texts
        """
        if self.nlp is None:
            self.load_base_model()
        
        # Keywords that often indicate legal entities
        arbitration_keywords = [
            "arbitration", "arbitral", "tribunal", "arbitrator", "mediation",
            "adr", "dispute resolution", "binding", "final", "conclusive"
        ]
        
        org_patterns = []
        procedure_patterns = []
        
        for text in texts:
            doc = self.nlp(text.lower())
            
            # Find patterns around arbitration keywords
            for token in doc:
                if any(keyword in token.text for keyword in arbitration_keywords):
                    # Capture surrounding context
                    start = max(0, token.i - 2)
                    end = min(len(doc), token.i + 3)
                    context = doc[start:end]
                    
                    # Create pattern if it looks like an organization or procedure
                    pattern_text = " ".join([t.text for t in context])
                    
                    if any(org_word in pattern_text for org_word in ["association", "centre", "center", "chamber", "institute"]):
                        org_patterns.append(pattern_text)
                    elif any(proc_word in pattern_text for proc_word in ["binding", "final", "mandatory", "proceedings"]):
                        procedure_patterns.append(pattern_text)
        
        # Convert to spaCy patterns (simplified)
        patterns = []
        for pattern_text in set(org_patterns):
            tokens = pattern_text.split()
            pattern = [{"LOWER": token} for token in tokens]
            patterns.append({"label": "ARBITRATION_ORG", "pattern": pattern})
        
        for pattern_text in set(procedure_patterns):
            tokens = pattern_text.split()
            pattern = [{"LOWER": token} for token in tokens]
            patterns.append({"label": "ARBITRATION_PROC", "pattern": pattern})
        
        return patterns
    
    def load_trained_model(self, model_path: str = None):
        """
        Load a trained NER model
        """
        if model_path is None:
            model_path = self.model_save_path
        
        self.nlp = spacy.load(model_path)
        logger.info(f"Loaded trained model from {model_path}")


if __name__ == "__main__":
    # Example usage
    ner = LegalEntityRecognizer()
    
    # Train the model
    trained_model = ner.train_custom_ner()
    
    # Test entity extraction
    test_text = """
    Any dispute arising under this agreement shall be resolved through binding arbitration 
    administered by the American Arbitration Association under its Commercial Arbitration Rules.
    The arbitral tribunal shall consist of three arbitrators.
    """
    
    entities = ner.extract_arbitration_entities(test_text)
    print("Extracted entities:", entities)