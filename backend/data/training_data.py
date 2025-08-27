"""
Training Data Generator for Arbitration Clause Detection

This module generates synthetic training data for machine learning models
that detect arbitration clauses in legal documents.
"""

import json
import random
import re
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd

class TrainingDataGenerator:
    def __init__(self, base_data_path: str = "/Users/rahulmehta/Desktop/Test/backend/data"):
        self.base_path = Path(base_data_path)
        self.positive_examples = []
        self.negative_examples = []
        self.arbitration_keywords = [
            "arbitration", "arbitrator", "arbitral", "mediation", "binding arbitration",
            "mandatory arbitration", "individual arbitration", "AAA", "JAMS", "ICC",
            "American Arbitration Association", "dispute resolution", "jury trial waiver",
            "class action waiver", "arbitrable", "arbitrate", "arbitrated"
        ]
        self.legal_keywords = [
            "dispute", "controversy", "claim", "breach", "enforcement", "interpretation",
            "validity", "violation", "settlement", "resolution", "proceeding", "litigation",
            "court", "jurisdiction", "venue", "governing law", "applicable law"
        ]
        self.non_arbitration_keywords = [
            "court", "judicial", "litigation", "trial", "jury", "judge", "verdict",
            "lawsuit", "legal action", "federal court", "state court", "district court"
        ]
        
    def load_existing_data(self):
        """Load existing positive and negative examples"""
        try:
            with open(self.base_path / "positive_examples.json", 'r') as f:
                pos_data = json.load(f)
                self.positive_examples = pos_data.get("arbitration_clauses", [])
        except FileNotFoundError:
            print("No positive examples file found")
            
        try:
            with open(self.base_path / "negative_examples.json", 'r') as f:
                neg_data = json.load(f)
                self.negative_examples = neg_data.get("non_arbitration_clauses", [])
        except FileNotFoundError:
            print("No negative examples file found")
    
    def generate_arbitration_variations(self, base_text: str, num_variations: int = 5) -> List[str]:
        """Generate variations of arbitration clauses"""
        variations = []
        
        # Define replacement patterns
        replacements = {
            "San Francisco, California": [
                "New York, New York", "Delaware", "Chicago, Illinois", "Los Angeles, California",
                "Seattle, Washington", "Austin, Texas", "Boston, Massachusetts"
            ],
            "JAMS": ["AAA", "ICC", "American Arbitration Association", "International Chamber of Commerce"],
            "binding arbitration": [
                "mandatory arbitration", "final and binding arbitration", "individual arbitration",
                "confidential arbitration", "expedited arbitration"
            ],
            "dispute": ["claim", "controversy", "disagreement", "conflict", "legal matter"],
            "shall be": ["will be", "must be", "are to be", "should be"],
            "you agree": ["you acknowledge", "you consent", "you understand and agree", "by using this service you agree"],
        }
        
        for i in range(num_variations):
            variant = base_text
            for original, alternatives in replacements.items():
                if original.lower() in variant.lower():
                    replacement = random.choice(alternatives)
                    variant = re.sub(re.escape(original), replacement, variant, flags=re.IGNORECASE)
            
            # Add some structural variations
            if random.random() < 0.3:
                variant = variant.upper() if "PLEASE READ" not in variant else variant
            if random.random() < 0.2:
                variant = f"IMPORTANT: {variant}"
            
            variations.append(variant)
        
        return variations
    
    def generate_synthetic_arbitration_clauses(self, num_clauses: int = 200) -> List[Dict]:
        """Generate synthetic arbitration clauses"""
        templates = [
            "Any {dispute_term} arising out of or relating to this {agreement_type} shall be resolved through {arbitration_type} arbitration in {location}.",
            "You agree that all {dispute_term}s will be settled by {arbitration_type} arbitration administered by {provider}.",
            "{emphasis}You waive your right to a jury trial{emphasis_end} All disputes will be resolved through mandatory arbitration.",
            "Any controversy or claim arising from this agreement shall be settled by arbitration in accordance with the rules of {provider}.",
            "Binding arbitration will be the exclusive remedy for any {dispute_term} related to this service.",
            "You and the Company agree that any {dispute_term} will be resolved through individual arbitration, and you waive your right to participate in class actions.",
            "All claims must be resolved through arbitration conducted by {provider} in {location}.",
            "Any legal action must be brought through arbitration only. Court proceedings are specifically excluded.",
            "Disputes will be resolved by a single arbitrator appointed under {provider} rules.",
            "You agree to binding arbitration for all {dispute_term}s and waive your right to jury trial."
        ]
        
        dispute_terms = ["dispute", "claim", "controversy", "disagreement", "conflict", "legal matter"]
        agreement_types = ["Agreement", "Terms of Service", "contract", "User Agreement", "Terms and Conditions"]
        arbitration_types = ["binding", "mandatory", "final and binding", "individual", "confidential"]
        providers = ["AAA", "JAMS", "ICC", "American Arbitration Association", "International Chamber of Commerce"]
        locations = ["Delaware", "New York", "California", "the state where the Company is headquartered"]
        emphasis_markers = [("", ""), ("**", "**"), ("PLEASE NOTE: ", ""), ("IMPORTANT: ", "")]
        
        synthetic_clauses = []
        
        for i in range(num_clauses):
            template = random.choice(templates)
            emphasis, emphasis_end = random.choice(emphasis_markers)
            
            clause_text = template.format(
                dispute_term=random.choice(dispute_terms),
                agreement_type=random.choice(agreement_types),
                arbitration_type=random.choice(arbitration_types),
                provider=random.choice(providers),
                location=random.choice(locations),
                emphasis=emphasis,
                emphasis_end=emphasis_end
            )
            
            # Add some complexity
            if random.random() < 0.4:
                clause_text += " The arbitration shall be governed by the laws of the State of Delaware."
            if random.random() < 0.3:
                clause_text += " You waive your right to participate in class action lawsuits."
            if random.random() < 0.2:
                clause_text += " Small claims court proceedings are excluded from this arbitration requirement."
            
            features = []
            if "binding" in clause_text.lower():
                features.append("binding")
            if "mandatory" in clause_text.lower():
                features.append("mandatory")
            if "waive" in clause_text.lower() and "jury" in clause_text.lower():
                features.append("jury_waiver")
            if "class action" in clause_text.lower():
                features.append("class_action_waiver")
            if any(provider.lower() in clause_text.lower() for provider in providers):
                features.append("provider_specified")
            if any(loc.lower() in clause_text.lower() for loc in locations):
                features.append("location_specified")
                
            synthetic_clauses.append({
                "id": f"synthetic_{i+1:03d}",
                "text": clause_text,
                "type": "binding" if "binding" in clause_text.lower() else "mandatory",
                "features": features,
                "synthetic": True
            })
        
        return synthetic_clauses
    
    def generate_negative_examples(self, num_examples: int = 200) -> List[Dict]:
        """Generate negative examples (non-arbitration clauses)"""
        templates = [
            "Any {dispute_term} shall be resolved exclusively in the courts of {location}.",
            "You agree that any legal action may be brought in federal or state court in {location}.",
            "These Terms are governed by the laws of {location} and any disputes will be heard in court.",
            "Legal proceedings may be initiated in any court of competent jurisdiction.",
            "You consent to the exclusive jurisdiction of the courts located in {location}.",
            "Any lawsuit or legal action must be filed in the appropriate court in {location}.",
            "Court proceedings are the exclusive remedy for any {dispute_term} arising from this agreement.",
            "You may bring legal action in small claims court or any other court of competent jurisdiction.",
            "Litigation may be pursued in state or federal court as appropriate.",
            "Judicial proceedings are available for the resolution of any {dispute_term}."
        ]
        
        dispute_terms = ["dispute", "claim", "controversy", "disagreement", "legal matter", "conflict"]
        locations = ["Delaware", "New York", "California", "the state of incorporation", "San Francisco County"]
        
        negative_examples = []
        
        for i in range(num_examples):
            template = random.choice(templates)
            clause_text = template.format(
                dispute_term=random.choice(dispute_terms),
                location=random.choice(locations)
            )
            
            # Add complexity to make it more realistic
            if random.random() < 0.3:
                clause_text += " You waive any objection to venue in such courts."
            if random.random() < 0.2:
                clause_text += " The prevailing party shall be entitled to reasonable attorney's fees."
            
            features = []
            if "court" in clause_text.lower():
                features.append("court_jurisdiction")
            if "exclusive" in clause_text.lower():
                features.append("exclusive_jurisdiction")
            if "venue" in clause_text.lower():
                features.append("venue_selection")
            features.append("no_arbitration")
            
            negative_examples.append({
                "id": f"negative_synthetic_{i+1:03d}",
                "text": clause_text,
                "type": "court_jurisdiction",
                "features": features,
                "synthetic": True
            })
        
        return negative_examples
    
    def add_ambiguous_examples(self, num_examples: int = 50) -> Tuple[List[Dict], List[Dict]]:
        """Generate ambiguous examples that are harder to classify"""
        
        # Ambiguous positive examples (mention arbitration but unclear if mandatory)
        ambiguous_positive = []
        templates_pos = [
            "Disputes may be resolved through arbitration or court proceedings at the party's option.",
            "The parties may agree to submit disputes to arbitration if both parties consent.",
            "Arbitration is available as an alternative dispute resolution mechanism.",
            "You may choose arbitration or litigation to resolve any {dispute_term}.",
            "Optional arbitration procedures are available through {provider}."
        ]
        
        for i in range(num_examples // 2):
            template = random.choice(templates_pos)
            text = template.format(
                dispute_term=random.choice(["dispute", "claim", "controversy"]),
                provider=random.choice(["AAA", "JAMS", "mediation services"])
            )
            
            ambiguous_positive.append({
                "id": f"ambiguous_pos_{i+1:03d}",
                "text": text,
                "type": "optional",
                "features": ["optional_arbitration", "choice_available"],
                "ambiguous": True
            })
        
        # Ambiguous negative examples (legal terms but no arbitration)
        ambiguous_negative = []
        templates_neg = [
            "Mediation may be used to resolve disputes before pursuing legal action.",
            "Alternative dispute resolution is encouraged but not required.",
            "The parties may engage in settlement discussions or mediation.",
            "Disputes should first be addressed through informal resolution procedures.",
            "You may seek equitable relief in court for intellectual property violations."
        ]
        
        for i in range(num_examples // 2):
            template = random.choice(templates_neg)
            text = template
            
            ambiguous_negative.append({
                "id": f"ambiguous_neg_{i+1:03d}",
                "text": text,
                "type": "mediation_or_other",
                "features": ["alternative_dispute_resolution", "no_mandatory_arbitration"],
                "ambiguous": True
            })
        
        return ambiguous_positive, ambiguous_negative
    
    def create_training_dataset(self, 
                              num_synthetic_positive: int = 300,
                              num_synthetic_negative: int = 300,
                              num_ambiguous: int = 100,
                              include_variations: bool = True) -> pd.DataFrame:
        """Create complete training dataset"""
        
        # Load existing data
        self.load_existing_data()
        
        # Generate synthetic data
        synthetic_positive = self.generate_synthetic_arbitration_clauses(num_synthetic_positive)
        synthetic_negative = self.generate_negative_examples(num_synthetic_negative)
        ambiguous_pos, ambiguous_neg = self.add_ambiguous_examples(num_ambiguous)
        
        # Combine all positive examples
        all_positive = self.positive_examples + synthetic_positive + ambiguous_pos
        all_negative = self.negative_examples + synthetic_negative + ambiguous_neg
        
        # Generate variations if requested
        if include_variations:
            variations_positive = []
            for example in self.positive_examples[:5]:  # Only vary real examples
                variations = self.generate_arbitration_variations(example['text'], 3)
                for j, var_text in enumerate(variations):
                    variations_positive.append({
                        "id": f"{example['id']}_var_{j+1}",
                        "text": var_text,
                        "type": example.get('type', 'binding'),
                        "features": example.get('features', []),
                        "variation": True
                    })
            all_positive.extend(variations_positive)
        
        # Create training dataset
        training_data = []
        
        # Add positive examples
        for example in all_positive:
            training_data.append({
                "text": example["text"],
                "label": 1,
                "has_arbitration": True,
                "arbitration_type": example.get("type", "unknown"),
                "features": ",".join(example.get("features", [])),
                "source": "synthetic" if example.get("synthetic") else ("variation" if example.get("variation") else "original"),
                "ambiguous": example.get("ambiguous", False),
                "text_length": len(example["text"]),
                "word_count": len(example["text"].split()),
                "arbitration_keyword_count": sum(1 for keyword in self.arbitration_keywords if keyword.lower() in example["text"].lower()),
                "legal_keyword_count": sum(1 for keyword in self.legal_keywords if keyword.lower() in example["text"].lower())
            })
        
        # Add negative examples
        for example in all_negative:
            training_data.append({
                "text": example["text"],
                "label": 0,
                "has_arbitration": False,
                "arbitration_type": "none",
                "features": ",".join(example.get("features", [])),
                "source": "synthetic" if example.get("synthetic") else "original",
                "ambiguous": example.get("ambiguous", False),
                "text_length": len(example["text"]),
                "word_count": len(example["text"].split()),
                "arbitration_keyword_count": sum(1 for keyword in self.arbitration_keywords if keyword.lower() in example["text"].lower()),
                "legal_keyword_count": sum(1 for keyword in self.legal_keywords if keyword.lower() in example["text"].lower())
            })
        
        # Shuffle the dataset
        random.shuffle(training_data)
        
        df = pd.DataFrame(training_data)
        print(f"Generated dataset with {len(df)} examples:")
        print(f"- Positive examples: {len(df[df['label'] == 1])}")
        print(f"- Negative examples: {len(df[df['label'] == 0])}")
        print(f"- Ambiguous examples: {len(df[df['ambiguous'] == True])}")
        
        return df
    
    def save_training_data(self, df: pd.DataFrame, filename: str = "training_data.csv"):
        """Save training dataset to file"""
        output_path = self.base_path / filename
        df.to_csv(output_path, index=False)
        print(f"Training data saved to {output_path}")
        
        # Also save as JSON for easier inspection
        json_path = self.base_path / filename.replace('.csv', '.json')
        df.to_json(json_path, orient='records', indent=2)
        print(f"Training data also saved as JSON to {json_path}")
        
        return output_path
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            "total_examples": len(df),
            "positive_examples": len(df[df['label'] == 1]),
            "negative_examples": len(df[df['label'] == 0]),
            "class_balance": len(df[df['label'] == 1]) / len(df),
            "ambiguous_examples": len(df[df['ambiguous'] == True]),
            "avg_text_length": df['text_length'].mean(),
            "avg_word_count": df['word_count'].mean(),
            "avg_arbitration_keywords": df['arbitration_keyword_count'].mean(),
            "avg_legal_keywords": df['legal_keyword_count'].mean(),
            "sources": df['source'].value_counts().to_dict(),
            "arbitration_types": df[df['label'] == 1]['arbitration_type'].value_counts().to_dict()
        }
        return stats


def main():
    """Generate training data"""
    generator = TrainingDataGenerator()
    
    # Create training dataset with 1000+ examples
    df = generator.create_training_dataset(
        num_synthetic_positive=400,
        num_synthetic_negative=400,
        num_ambiguous=120,
        include_variations=True
    )
    
    # Save the dataset
    output_path = generator.save_training_data(df)
    
    # Print statistics
    stats = generator.get_dataset_statistics(df)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return df, stats


if __name__ == "__main__":
    dataset, statistics = main()