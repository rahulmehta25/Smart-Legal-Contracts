"""
Fine-tuning module for legal text embeddings
Specializes embeddings for arbitration clause detection
"""
import logging
import os
from typing import List, Tuple, Dict, Optional
import torch
from transformers import (
    AutoTokenizer, AutoModel, TrainingArguments, Trainer,
    DataCollatorWithPadding, AutoConfig
)
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalTextDataset(Dataset):
    """Dataset for fine-tuning on legal text pairs"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LegalEmbeddingFineTuner:
    """Fine-tune embeddings specifically for legal arbitration detection"""
    
    def __init__(self, 
                 base_model: str = "nlpaueb/legal-bert-base-uncased",
                 model_save_path: str = "backend/models/legal_embeddings",
                 experiment_name: str = "legal_embedding_finetune"):
        self.base_model = base_model
        self.model_save_path = model_save_path
        self.experiment_name = experiment_name
        self.tokenizer = None
        self.model = None
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
    
    def prepare_contrastive_data(self, 
                               arbitration_texts: List[str], 
                               non_arbitration_texts: List[str]) -> List[InputExample]:
        """
        Prepare contrastive learning data for sentence transformers
        """
        examples = []
        
        # Positive pairs (arbitration clauses)
        for i, text1 in enumerate(arbitration_texts):
            for j, text2 in enumerate(arbitration_texts[i+1:], i+1):
                examples.append(InputExample(texts=[text1, text2], label=1.0))
        
        # Negative pairs (arbitration vs non-arbitration)
        for arb_text in arbitration_texts[:50]:  # Limit for performance
            for non_arb_text in non_arbitration_texts[:50]:
                examples.append(InputExample(texts=[arb_text, non_arb_text], label=0.0))
        
        # Negative pairs (non-arbitration clauses)
        for i, text1 in enumerate(non_arbitration_texts[:30]):
            for j, text2 in enumerate(non_arbitration_texts[i+1:31], i+1):
                examples.append(InputExample(texts=[text1, text2], label=0.0))
        
        logger.info(f"Created {len(examples)} contrastive examples")
        return examples
    
    def fine_tune_sentence_transformer(self, 
                                     training_data: List[InputExample],
                                     epochs: int = 3,
                                     batch_size: int = 16,
                                     learning_rate: float = 2e-5) -> SentenceTransformer:
        """
        Fine-tune sentence transformer for legal similarity
        """
        with mlflow.start_run(run_name="sentence_transformer_finetune"):
            mlflow.log_params({
                "base_model": self.base_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "training_samples": len(training_data)
            })
            
            # Initialize model
            model = SentenceTransformer(self.base_model)
            
            # Prepare training
            train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
            train_loss = losses.CosineSimilarityLoss(model)
            
            # Training
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=int(len(train_dataloader) * 0.1),
                output_path=f"{self.model_save_path}_sentence_transformer",
                save_best_model=True
            )
            
            # Log model
            mlflow.pytorch.log_model(
                model, 
                "sentence_transformer_model",
                registered_model_name="legal_sentence_transformer"
            )
            
            return model
    
    def fine_tune_classification_head(self,
                                    texts: List[str],
                                    labels: List[int],
                                    epochs: int = 3,
                                    batch_size: int = 16,
                                    learning_rate: float = 2e-5,
                                    validation_split: float = 0.2) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Fine-tune classification head for arbitration detection
        """
        with mlflow.start_run(run_name="classification_head_finetune"):
            mlflow.log_params({
                "base_model": self.base_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_split": validation_split,
                "training_samples": len(texts)
            })
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=validation_split, stratify=labels, random_state=42
            )
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            config = AutoConfig.from_pretrained(self.base_model)
            config.num_labels = 2  # Binary classification
            self.model = AutoModel.from_pretrained(self.base_model, config=config)
            
            # Create datasets
            train_dataset = LegalTextDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = LegalTextDataset(val_texts, val_labels, self.tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{self.model_save_path}_classification",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(self.tokenizer)
            )
            
            # Train
            trainer.train()
            
            # Save model
            trainer.save_model(f"{self.model_save_path}_classification")
            self.tokenizer.save_pretrained(f"{self.model_save_path}_classification")
            
            # Log metrics
            eval_results = trainer.evaluate()
            mlflow.log_metrics({
                "final_eval_loss": eval_results["eval_loss"],
                "final_train_loss": trainer.state.log_history[-1]["train_loss"]
            })
            
            # Log model
            mlflow.pytorch.log_model(
                self.model,
                "classification_model", 
                registered_model_name="legal_classification_head"
            )
            
            return self.model, self.tokenizer
    
    def create_domain_specific_vocabulary(self, legal_texts: List[str]) -> Dict[str, int]:
        """
        Create domain-specific vocabulary for legal texts
        """
        from collections import Counter
        import re
        
        # Legal-specific terms and patterns
        legal_patterns = [
            r'\b(?:arbitrat\w*|mediat\w*|dispute\w*|resolution|tribunal|binding|final|conclusive)\b',
            r'\b(?:contract\w*|agreement\w*|clause\w*|provision\w*|section\w*)\b',
            r'\b(?:party|parties|plaintiff\w*|defendant\w*|claimant\w*)\b',
            r'\b(?:court\w*|litigation|lawsuit\w*|proceeding\w*)\b',
            r'\b(?:governing\s+law|applicable\s+law|jurisdiction)\b'
        ]
        
        legal_vocab = Counter()
        
        for text in legal_texts:
            text_lower = text.lower()
            for pattern in legal_patterns:
                matches = re.findall(pattern, text_lower)
                legal_vocab.update(matches)
        
        # Return most common legal terms
        return dict(legal_vocab.most_common(1000))
    
    def evaluate_embeddings(self, 
                          model: SentenceTransformer,
                          test_arbitration_texts: List[str],
                          test_non_arbitration_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate quality of fine-tuned embeddings
        """
        # Generate embeddings
        arb_embeddings = model.encode(test_arbitration_texts)
        non_arb_embeddings = model.encode(test_non_arbitration_texts)
        
        # Calculate intra-class similarity (should be high for arbitration clauses)
        arb_similarities = []
        for i in range(len(arb_embeddings)):
            for j in range(i+1, len(arb_embeddings)):
                sim = np.dot(arb_embeddings[i], arb_embeddings[j]) / (
                    np.linalg.norm(arb_embeddings[i]) * np.linalg.norm(arb_embeddings[j])
                )
                arb_similarities.append(sim)
        
        # Calculate inter-class similarity (should be low)
        inter_similarities = []
        for arb_emb in arb_embeddings:
            for non_arb_emb in non_arb_embeddings:
                sim = np.dot(arb_emb, non_arb_emb) / (
                    np.linalg.norm(arb_emb) * np.linalg.norm(non_arb_emb)
                )
                inter_similarities.append(sim)
        
        metrics = {
            "intra_class_similarity": np.mean(arb_similarities),
            "inter_class_similarity": np.mean(inter_similarities),
            "separation_score": np.mean(arb_similarities) - np.mean(inter_similarities)
        }
        
        logger.info(f"Embedding evaluation metrics: {metrics}")
        return metrics


def create_synthetic_legal_data() -> Tuple[List[str], List[str]]:
    """
    Create synthetic legal training data for fine-tuning
    """
    arbitration_templates = [
        "Any dispute arising under this agreement shall be resolved through binding arbitration.",
        "The parties agree to submit any controversy to arbitration under the rules of {}.",
        "All disputes shall be finally settled by arbitration administered by {}.",
        "Any claim or dispute shall be resolved exclusively through binding arbitration.",
        "The parties hereby waive their right to trial by jury and agree to arbitration.",
        "Disputes will be resolved through confidential binding arbitration.",
        "Any controversy arising out of this contract shall be settled by arbitration.",
        "The parties agree that arbitration shall be the sole remedy for disputes.",
        "All claims must be resolved through individual arbitration proceedings.",
        "Any legal action shall be subject to binding arbitration under {} rules."
    ]
    
    non_arbitration_templates = [
        "Any dispute shall be resolved in the courts of {}.",
        "The parties retain the right to seek judicial remedies.",
        "This agreement shall be governed by the laws of {}.",
        "Any legal action must be brought in the appropriate court.",
        "The parties may seek injunctive relief in any court of competent jurisdiction.",
        "Disputes may be resolved through negotiation or mediation.",
        "This contract does not waive the right to jury trial.",
        "Legal proceedings shall be conducted in accordance with court rules.",
        "The parties reserve all rights to pursue legal remedies.",
        "Any lawsuit must be filed in the designated jurisdiction."
    ]
    
    arbitration_texts = []
    non_arbitration_texts = []
    
    # Generate variations
    arbitrators = ["AAA", "JAMS", "ICC", "LCIA", "the American Arbitration Association"]
    jurisdictions = ["New York", "California", "Delaware", "Texas", "Florida"]
    
    for template in arbitration_templates:
        if "{}" in template:
            for entity in arbitrators:
                arbitration_texts.append(template.format(entity))
        else:
            arbitration_texts.append(template)
    
    for template in non_arbitration_templates:
        if "{}" in template:
            for jurisdiction in jurisdictions:
                non_arbitration_texts.append(template.format(jurisdiction))
        else:
            non_arbitration_texts.append(template)
    
    return arbitration_texts, non_arbitration_texts


if __name__ == "__main__":
    # Example usage
    fine_tuner = LegalEmbeddingFineTuner()
    
    # Create synthetic data
    arb_texts, non_arb_texts = create_synthetic_legal_data()
    
    # Prepare contrastive learning data
    contrastive_data = fine_tuner.prepare_contrastive_data(arb_texts, non_arb_texts)
    
    # Fine-tune sentence transformer
    fine_tuned_model = fine_tuner.fine_tune_sentence_transformer(contrastive_data)
    
    # Evaluate
    metrics = fine_tuner.evaluate_embeddings(fine_tuned_model, arb_texts[:10], non_arb_texts[:10])
    print(f"Embedding quality metrics: {metrics}")