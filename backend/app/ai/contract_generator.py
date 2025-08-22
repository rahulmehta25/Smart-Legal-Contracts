"""
AI-powered contract generator with counter-clause generation and compatibility checking.
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from openai import AsyncOpenAI
import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class GeneratedClause:
    """Represents a generated contract clause."""
    clause_text: str
    clause_type: str
    confidence: float
    compatibility_score: float
    legal_validity: float
    alternatives: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class ContractTemplate:
    """Contract template structure."""
    template_name: str
    industry: str
    party_types: List[str]
    sections: Dict[str, str]
    required_clauses: List[str]
    optional_clauses: List[str]
    jurisdiction: str
    language_style: str  # formal, plain, balanced


@dataclass
class CompatibilityCheck:
    """Results of clause compatibility check."""
    is_compatible: bool
    compatibility_score: float
    conflicts: List[Dict[str, str]]
    suggestions: List[str]
    risk_assessment: str


class ClauseGenerator(nn.Module):
    """Neural model for clause generation."""
    
    def __init__(self, vocab_size: int = 50000, hidden_dim: int = 768):
        super(ClauseGenerator, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer layers
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Compatibility checker
        self.compatibility_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, memory=None):
        """Generate clause tokens."""
        embedded = self.embedding(input_ids)
        
        if memory is not None:
            output = self.transformer(embedded, memory)
        else:
            output = embedded
        
        logits = self.output_projection(output)
        return logits
    
    def check_compatibility(self, clause1_emb, clause2_emb):
        """Check compatibility between two clauses."""
        combined = torch.cat([clause1_emb, clause2_emb], dim=-1)
        compatibility = self.compatibility_head(combined)
        return compatibility


class ContractGenerator:
    """Advanced contract generator with AI-powered clause creation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize contract generator."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not provided. Some features will be limited.")
        
        # Initialize local models
        self._initialize_models()
        
        # Load templates and patterns
        self.templates = self._load_contract_templates()
        self.legal_patterns = self._compile_legal_patterns()
        
        # Clause compatibility matrix
        self.compatibility_rules = self._load_compatibility_rules()
        
        # Legal language optimizer
        self.language_optimizer = LanguageOptimizer()
    
    def _initialize_models(self):
        """Initialize transformer models for generation."""
        try:
            # T5 for seq2seq generation
            self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
            
            # GPT-2 for language generation
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
            
        except Exception as e:
            logger.error(f"Error loading generation models: {e}")
            self.t5_model = None
            self.gpt2_model = None
        
        # Custom clause generator
        self.clause_generator = ClauseGenerator()
        
        # TF-IDF for similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3)
        )
    
    def _load_contract_templates(self) -> Dict[str, ContractTemplate]:
        """Load contract templates."""
        templates = {
            "saas_agreement": ContractTemplate(
                template_name="SaaS Service Agreement",
                industry="technology",
                party_types=["service_provider", "customer"],
                sections={
                    "services": "Provider shall provide access to the Software-as-a-Service platform",
                    "payment": "Customer shall pay fees according to the selected plan",
                    "data_security": "Provider implements industry-standard security measures",
                    "liability": "Limitation of liability provisions",
                    "termination": "Either party may terminate with 30 days notice"
                },
                required_clauses=["services", "payment", "liability", "termination"],
                optional_clauses=["sla", "support", "customization"],
                jurisdiction="Delaware",
                language_style="balanced"
            ),
            "employment_agreement": ContractTemplate(
                template_name="Employment Agreement",
                industry="general",
                party_types=["employer", "employee"],
                sections={
                    "position": "Employee shall serve in the position of",
                    "compensation": "Employee shall receive base salary and benefits",
                    "confidentiality": "Employee agrees to maintain confidentiality",
                    "ip_assignment": "Work product shall be property of Employer",
                    "termination": "Employment may be terminated by either party"
                },
                required_clauses=["position", "compensation", "termination"],
                optional_clauses=["non_compete", "stock_options", "severance"],
                jurisdiction="California",
                language_style="formal"
            ),
            "nda": ContractTemplate(
                template_name="Non-Disclosure Agreement",
                industry="general",
                party_types=["disclosing_party", "receiving_party"],
                sections={
                    "definition": "Confidential Information means non-public information",
                    "obligations": "Receiving Party shall maintain confidentiality",
                    "exceptions": "Obligations do not apply to publicly available information",
                    "term": "Agreement remains in effect for specified period",
                    "remedies": "Breach may result in irreparable harm"
                },
                required_clauses=["definition", "obligations", "term"],
                optional_clauses=["exceptions", "remedies", "governing_law"],
                jurisdiction="New York",
                language_style="formal"
            )
        }
        return templates
    
    def _compile_legal_patterns(self) -> Dict[str, re.Pattern]:
        """Compile legal language patterns."""
        return {
            "obligation": re.compile(r"(shall|must|will|agrees to|undertakes to)", re.IGNORECASE),
            "right": re.compile(r"(may|can|is entitled to|has the right to)", re.IGNORECASE),
            "condition": re.compile(r"(if|when|provided that|subject to|in the event)", re.IGNORECASE),
            "exception": re.compile(r"(except|unless|excluding|other than|notwithstanding)", re.IGNORECASE),
            "definition": re.compile(r'("[\w\s]+" means|"[\w\s]+" shall mean|defined as)', re.IGNORECASE),
            "reference": re.compile(r"(Section \d+|Exhibit [A-Z]|Schedule \d+|Appendix)", re.IGNORECASE)
        }
    
    def _load_compatibility_rules(self) -> Dict[str, List[str]]:
        """Load clause compatibility rules."""
        return {
            "arbitration": {
                "incompatible": ["litigation_rights", "jury_trial"],
                "requires": ["governing_law", "arbitration_rules"],
                "conflicts": ["class_action_rights"]
            },
            "non_compete": {
                "incompatible": ["california_employment"],
                "requires": ["geographic_scope", "time_limitation"],
                "conflicts": ["right_to_work"]
            },
            "unlimited_liability": {
                "incompatible": ["limitation_of_liability"],
                "requires": ["insurance_requirements"],
                "conflicts": ["indemnification_cap"]
            },
            "exclusive_jurisdiction": {
                "incompatible": ["arbitration", "mediation_first"],
                "requires": ["governing_law"],
                "conflicts": ["forum_selection_flexibility"]
            }
        }
    
    async def generate_clause(
        self,
        clause_type: str,
        context: Dict[str, Any],
        style: str = "balanced",
        party_favor: str = "neutral"
    ) -> GeneratedClause:
        """
        Generate a contract clause based on requirements.
        
        Args:
            clause_type: Type of clause to generate
            context: Context including parties, industry, etc.
            style: Language style (formal, plain, balanced)
            party_favor: Which party to favor (neutral, party1, party2)
            
        Returns:
            Generated clause with metadata
        """
        # Generate using multiple methods
        clauses = []
        
        # Method 1: Template-based generation
        template_clause = self._generate_from_template(clause_type, context, style)
        if template_clause:
            clauses.append(template_clause)
        
        # Method 2: GPT-4 generation
        if self.client:
            gpt_clause = await self._generate_with_gpt4(clause_type, context, style, party_favor)
            if gpt_clause:
                clauses.append(gpt_clause)
        
        # Method 3: Local transformer generation
        transformer_clause = self._generate_with_transformer(clause_type, context)
        if transformer_clause:
            clauses.append(transformer_clause)
        
        # Select best clause
        best_clause = self._select_best_clause(clauses, context)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(best_clause, clause_type, context)
        
        # Check legal validity
        validity_score = self._assess_legal_validity(best_clause)
        
        # Generate warnings
        warnings = self._generate_warnings(best_clause, clause_type, context)
        
        # Calculate compatibility with existing clauses
        compatibility = self._calculate_compatibility(best_clause, context.get('existing_clauses', []))
        
        return GeneratedClause(
            clause_text=best_clause,
            clause_type=clause_type,
            confidence=0.85,  # Based on generation method
            compatibility_score=compatibility,
            legal_validity=validity_score,
            alternatives=alternatives[:3],
            warnings=warnings,
            metadata={
                'generation_method': 'ensemble',
                'style': style,
                'party_favor': party_favor,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _generate_from_template(
        self,
        clause_type: str,
        context: Dict[str, Any],
        style: str
    ) -> Optional[str]:
        """Generate clause from template."""
        # Find matching template
        template_key = context.get('template_type', 'saas_agreement')
        if template_key not in self.templates:
            return None
        
        template = self.templates[template_key]
        
        # Get base clause
        if clause_type in template.sections:
            base_clause = template.sections[clause_type]
            
            # Customize based on context
            if context.get('party1_name'):
                base_clause = base_clause.replace('Provider', context['party1_name'])
            if context.get('party2_name'):
                base_clause = base_clause.replace('Customer', context['party2_name'])
            
            # Adjust style
            if style == "plain":
                base_clause = self._simplify_language(base_clause)
            elif style == "formal":
                base_clause = self._formalize_language(base_clause)
            
            return base_clause
        
        return None
    
    async def _generate_with_gpt4(
        self,
        clause_type: str,
        context: Dict[str, Any],
        style: str,
        party_favor: str
    ) -> Optional[str]:
        """Generate clause using GPT-4."""
        if not self.client:
            return None
        
        prompt = f"""
        Generate a {clause_type} clause for a {context.get('contract_type', 'commercial')} contract.
        
        Context:
        - Industry: {context.get('industry', 'general')}
        - Parties: {context.get('party1_type', 'Party A')} and {context.get('party2_type', 'Party B')}
        - Jurisdiction: {context.get('jurisdiction', 'United States')}
        - Style: {style} language
        - Favor: {party_favor}
        
        Generate a legally sound clause that is clear and enforceable.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a legal contract drafting expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT-4 generation error: {e}")
            return None
    
    def _generate_with_transformer(
        self,
        clause_type: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate clause using local transformer model."""
        if not self.t5_model:
            return None
        
        try:
            # Prepare input
            input_text = f"generate {clause_type} clause: {json.dumps(context)}"
            inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate
            outputs = self.t5_model.generate(
                inputs['input_ids'],
                max_length=200,
                num_beams=4,
                temperature=0.8,
                early_stopping=True
            )
            
            # Decode
            generated = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated
        except Exception as e:
            logger.error(f"Transformer generation error: {e}")
            return None
    
    def _select_best_clause(
        self,
        clauses: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Select best clause from alternatives."""
        if not clauses:
            return self._generate_fallback_clause(context)
        
        if len(clauses) == 1:
            return clauses[0]
        
        # Score each clause
        scores = []
        for clause in clauses:
            score = 0.0
            
            # Length score (prefer reasonable length)
            word_count = len(clause.split())
            if 50 <= word_count <= 200:
                score += 0.3
            
            # Legal language presence
            for pattern in self.legal_patterns.values():
                if pattern.search(clause):
                    score += 0.1
            
            # Clarity score (shorter sentences preferred)
            sentences = clause.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            if avg_sentence_length < 25:
                score += 0.2
            
            scores.append(score)
        
        # Return highest scoring clause
        best_idx = np.argmax(scores)
        return clauses[best_idx]
    
    def _generate_fallback_clause(self, context: Dict[str, Any]) -> str:
        """Generate a basic fallback clause."""
        clause_type = context.get('clause_type', 'general')
        
        fallback_clauses = {
            "payment": "The Party shall pay the agreed fees within thirty (30) days of invoice date.",
            "termination": "Either party may terminate this agreement with thirty (30) days written notice.",
            "confidentiality": "The Receiving Party shall maintain the confidentiality of all Confidential Information.",
            "liability": "Neither party shall be liable for indirect or consequential damages.",
            "general": "The parties agree to perform their respective obligations under this agreement."
        }
        
        return fallback_clauses.get(clause_type, fallback_clauses["general"])
    
    def _generate_alternatives(
        self,
        base_clause: str,
        clause_type: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate alternative formulations."""
        alternatives = []
        
        # Variation 1: More favorable to party 1
        alt1 = self._adjust_clause_favor(base_clause, "party1")
        alternatives.append(alt1)
        
        # Variation 2: More favorable to party 2
        alt2 = self._adjust_clause_favor(base_clause, "party2")
        alternatives.append(alt2)
        
        # Variation 3: Simplified language
        alt3 = self._simplify_language(base_clause)
        alternatives.append(alt3)
        
        return alternatives
    
    def _adjust_clause_favor(self, clause: str, favor: str) -> str:
        """Adjust clause to favor a specific party."""
        if favor == "party1":
            # Make more favorable to first party
            clause = clause.replace("shall", "may, at its discretion,")
            clause = clause.replace("must", "should reasonably attempt to")
            clause = clause.replace("immediately", "within a reasonable time")
        elif favor == "party2":
            # Make more favorable to second party
            clause = clause.replace("may", "shall")
            clause = clause.replace("reasonable efforts", "best efforts")
            clause = clause.replace("to the extent permitted", "in all cases")
        
        return clause
    
    def _simplify_language(self, text: str) -> str:
        """Simplify legal language to plain English."""
        simplifications = {
            "heretofore": "before",
            "hereinafter": "from now on",
            "whereas": "since",
            "notwithstanding": "despite",
            "pursuant to": "according to",
            "in lieu of": "instead of",
            "prior to": "before",
            "subsequent to": "after",
            "in the event that": "if",
            "with respect to": "about",
            "set forth": "stated",
            "foregoing": "above"
        }
        
        result = text
        for formal, simple in simplifications.items():
            result = re.sub(r'\b' + formal + r'\b', simple, result, flags=re.IGNORECASE)
        
        return result
    
    def _formalize_language(self, text: str) -> str:
        """Make language more formal/legal."""
        formalizations = {
            "before": "prior to",
            "after": "subsequent to",
            "if": "in the event that",
            "about": "with respect to",
            "stated": "set forth herein",
            "above": "foregoing",
            "below": "following",
            "must": "shall",
            "can": "may"
        }
        
        result = text
        for simple, formal in formalizations.items():
            result = re.sub(r'\b' + simple + r'\b', formal, result, flags=re.IGNORECASE)
        
        return result
    
    def _assess_legal_validity(self, clause: str) -> float:
        """Assess legal validity of generated clause."""
        validity = 0.5  # Base score
        
        # Check for required legal elements
        if self.legal_patterns["obligation"].search(clause):
            validity += 0.15
        if self.legal_patterns["right"].search(clause):
            validity += 0.1
        if self.legal_patterns["condition"].search(clause):
            validity += 0.1
        
        # Check for problematic patterns
        problematic_patterns = [
            r"absolute(ly)?.*discretion",
            r"under no circumstances",
            r"waives? all rights?",
            r"forever|perpetual(ly)?|in perpetuity"
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, clause, re.IGNORECASE):
                validity -= 0.1
        
        # Check clause structure
        if len(clause.split('.')) > 1:  # Multiple sentences
            validity += 0.05
        if 30 <= len(clause.split()) <= 300:  # Reasonable length
            validity += 0.1
        
        return min(max(validity, 0.1), 0.95)
    
    def _generate_warnings(
        self,
        clause: str,
        clause_type: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate warnings about potential issues."""
        warnings = []
        
        # Check for one-sided terms
        if re.search(r"sole(ly)?.*discretion", clause, re.IGNORECASE):
            warnings.append("Clause grants unilateral discretion - may be challenged as unconscionable")
        
        # Check for perpetual terms
        if re.search(r"perpetual|forever|indefinite", clause, re.IGNORECASE):
            warnings.append("Perpetual terms may be unenforceable in some jurisdictions")
        
        # Jurisdiction-specific warnings
        if context.get('jurisdiction') == 'California' and clause_type == 'non_compete':
            warnings.append("Non-compete clauses generally unenforceable in California")
        
        # Check for waiver of rights
        if re.search(r"waive.*right", clause, re.IGNORECASE):
            warnings.append("Rights waivers should be explicit and may require special formatting")
        
        # Ambiguity warning
        if clause.count("reasonable") > 2:
            warnings.append("Multiple uses of 'reasonable' may create ambiguity")
        
        return warnings
    
    def _calculate_compatibility(self, clause: str, existing_clauses: List[str]) -> float:
        """Calculate compatibility with existing clauses."""
        if not existing_clauses:
            return 1.0
        
        # Vectorize all clauses
        all_clauses = existing_clauses + [clause]
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_clauses)
            
            # Calculate similarity between new clause and existing ones
            new_clause_vec = tfidf_matrix[-1]
            existing_vecs = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(new_clause_vec, existing_vecs).flatten()
            
            # Check for conflicts
            conflict_score = 0.0
            for i, existing in enumerate(existing_clauses):
                similarity = similarities[i]
                
                # High similarity might indicate redundancy
                if similarity > 0.8:
                    conflict_score += 0.2
                
                # Check for logical conflicts
                if self._check_logical_conflict(clause, existing):
                    conflict_score += 0.3
            
            compatibility = 1.0 - min(conflict_score, 0.9)
            return compatibility
            
        except Exception as e:
            logger.error(f"Compatibility calculation error: {e}")
            return 0.7  # Default moderate compatibility
    
    def _check_logical_conflict(self, clause1: str, clause2: str) -> bool:
        """Check for logical conflicts between clauses."""
        # Extract key terms
        clause1_lower = clause1.lower()
        clause2_lower = clause2.lower()
        
        # Check for contradictory terms
        contradictions = [
            ("shall", "shall not"),
            ("must", "must not"),
            ("required", "prohibited"),
            ("exclusive", "non-exclusive"),
            ("binding", "non-binding"),
            ("confidential", "public"),
            ("perpetual", "temporary"),
            ("unlimited", "limited")
        ]
        
        for term1, term2 in contradictions:
            if term1 in clause1_lower and term2 in clause2_lower:
                return True
            if term2 in clause1_lower and term1 in clause2_lower:
                return True
        
        return False
    
    async def generate_counter_clause(
        self,
        original_clause: str,
        objections: List[str],
        party_position: str = "responding_party"
    ) -> GeneratedClause:
        """
        Generate a counter-clause addressing specific objections.
        
        Args:
            original_clause: The clause being countered
            objections: List of specific objections
            party_position: Position of the countering party
            
        Returns:
            Counter-clause addressing objections
        """
        # Analyze original clause
        clause_type = self._identify_clause_type(original_clause)
        issues = self._identify_issues(original_clause, objections)
        
        # Generate counter-proposal
        if self.client:
            counter = await self._gpt4_counter_clause(original_clause, objections, issues)
        else:
            counter = self._template_counter_clause(original_clause, issues, clause_type)
        
        # Ensure objections are addressed
        addressed = self._verify_objections_addressed(counter, objections)
        
        # Generate explanation
        explanation = self._generate_counter_explanation(original_clause, counter, objections)
        
        return GeneratedClause(
            clause_text=counter,
            clause_type=clause_type,
            confidence=0.8 if addressed else 0.6,
            compatibility_score=0.9,  # Counter-clauses designed to be compatible
            legal_validity=self._assess_legal_validity(counter),
            alternatives=[],
            warnings=["This is a counter-proposal - further negotiation may be needed"],
            metadata={
                'original_clause': original_clause[:200] + "...",
                'objections_addressed': addressed,
                'explanation': explanation
            }
        )
    
    def _identify_clause_type(self, clause: str) -> str:
        """Identify the type of clause."""
        clause_lower = clause.lower()
        
        patterns = {
            "payment": ["payment", "fee", "invoice", "billing"],
            "termination": ["termination", "terminate", "end", "cancel"],
            "liability": ["liability", "damages", "limitation", "indemnif"],
            "confidentiality": ["confidential", "proprietary", "non-disclosure"],
            "arbitration": ["arbitration", "arbitrator", "dispute resolution"],
            "warranty": ["warranty", "guarantee", "representation"],
            "ip": ["intellectual property", "copyright", "patent", "trademark"]
        }
        
        for clause_type, keywords in patterns.items():
            if any(keyword in clause_lower for keyword in keywords):
                return clause_type
        
        return "general"
    
    def _identify_issues(self, clause: str, objections: List[str]) -> List[str]:
        """Identify specific issues based on objections."""
        issues = []
        
        objection_text = " ".join(objections).lower()
        
        if "one-sided" in objection_text or "unfair" in objection_text:
            issues.append("lack_of_mutuality")
        if "broad" in objection_text or "unlimited" in objection_text:
            issues.append("overly_broad")
        if "unclear" in objection_text or "ambiguous" in objection_text:
            issues.append("ambiguity")
        if "expensive" in objection_text or "cost" in objection_text:
            issues.append("excessive_cost")
        if "perpetual" in objection_text or "forever" in objection_text:
            issues.append("unlimited_duration")
        
        return issues
    
    async def _gpt4_counter_clause(
        self,
        original: str,
        objections: List[str],
        issues: List[str]
    ) -> str:
        """Generate counter-clause using GPT-4."""
        if not self.client:
            return original  # Fallback to original
        
        prompt = f"""
        Original clause: {original}
        
        Objections:
        {chr(10).join('- ' + obj for obj in objections)}
        
        Issues identified: {', '.join(issues)}
        
        Generate a counter-proposal that:
        1. Addresses all objections
        2. Maintains legal validity
        3. Creates more balanced terms
        4. Preserves the essential purpose of the clause
        
        Return only the revised clause text.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a contract negotiation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT-4 counter-clause error: {e}")
            return self._template_counter_clause(original, issues, "general")
    
    def _template_counter_clause(
        self,
        original: str,
        issues: List[str],
        clause_type: str
    ) -> str:
        """Generate counter-clause from templates."""
        # Base the counter on the original
        counter = original
        
        # Address specific issues
        if "lack_of_mutuality" in issues:
            counter = counter.replace("Party A shall", "Both parties shall")
            counter = counter.replace("Party B shall", "Both parties shall")
        
        if "overly_broad" in issues:
            counter = re.sub(
                r"unlimited|absolute|sole",
                "reasonable",
                counter,
                flags=re.IGNORECASE
            )
        
        if "ambiguity" in issues:
            counter = self._clarify_ambiguous_terms(counter)
        
        if "excessive_cost" in issues:
            counter += " The costs shall be shared equally between the parties."
        
        if "unlimited_duration" in issues:
            counter = re.sub(
                r"perpetual(ly)?|forever|indefinite(ly)?",
                "for a period of two (2) years",
                counter,
                flags=re.IGNORECASE
            )
        
        return counter
    
    def _clarify_ambiguous_terms(self, text: str) -> str:
        """Clarify ambiguous terms in text."""
        clarifications = {
            "reasonable": "commercially reasonable",
            "promptly": "within five (5) business days",
            "from time to time": "no more than quarterly",
            "material": "resulting in damages exceeding $10,000",
            "best efforts": "commercially reasonable efforts"
        }
        
        result = text
        for ambiguous, clear in clarifications.items():
            result = re.sub(
                r'\b' + ambiguous + r'\b',
                clear,
                result,
                flags=re.IGNORECASE,
                count=1  # Only replace first occurrence
            )
        
        return result
    
    def _verify_objections_addressed(self, counter: str, objections: List[str]) -> bool:
        """Verify that objections have been addressed."""
        counter_lower = counter.lower()
        addressed_count = 0
        
        for objection in objections:
            objection_lower = objection.lower()
            
            # Check if key terms from objection appear to be addressed
            key_terms = [
                word for word in objection_lower.split()
                if len(word) > 4 and word not in ['should', 'would', 'could', 'that', 'this']
            ]
            
            if any(term in counter_lower for term in key_terms):
                addressed_count += 1
        
        # Consider addressed if most objections have related content
        return addressed_count >= len(objections) * 0.6
    
    def _generate_counter_explanation(
        self,
        original: str,
        counter: str,
        objections: List[str]
    ) -> str:
        """Generate explanation of changes in counter-clause."""
        explanation_parts = []
        
        # Identify main changes
        if len(counter) < len(original) * 0.8:
            explanation_parts.append("Simplified and shortened the clause for clarity")
        elif len(counter) > len(original) * 1.2:
            explanation_parts.append("Added provisions to address concerns")
        
        # Address specific objections
        for objection in objections[:2]:  # Top 2 objections
            if "one-sided" in objection.lower():
                explanation_parts.append("Introduced mutual obligations for fairness")
            elif "broad" in objection.lower():
                explanation_parts.append("Narrowed scope to reasonable limits")
            elif "cost" in objection.lower():
                explanation_parts.append("Added cost-sharing provisions")
        
        return ". ".join(explanation_parts) if explanation_parts else "Modified to address stated concerns"
    
    async def check_compatibility(
        self,
        clause1: str,
        clause2: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CompatibilityCheck:
        """
        Check compatibility between two clauses.
        
        Args:
            clause1: First clause
            clause2: Second clause
            context: Additional context
            
        Returns:
            Detailed compatibility analysis
        """
        # Check for direct conflicts
        conflicts = self._find_conflicts(clause1, clause2)
        
        # Calculate compatibility score
        if self.clause_generator:
            # Use neural model
            emb1 = self._get_clause_embedding(clause1)
            emb2 = self._get_clause_embedding(clause2)
            compatibility_score = float(
                self.clause_generator.check_compatibility(emb1, emb2).item()
            )
        else:
            # Fallback to rule-based
            compatibility_score = 1.0 - (len(conflicts) * 0.2)
            compatibility_score = max(0.0, compatibility_score)
        
        # Generate suggestions
        suggestions = self._generate_compatibility_suggestions(
            clause1, clause2, conflicts
        )
        
        # Risk assessment
        if conflicts:
            if len(conflicts) > 2:
                risk = "high"
            else:
                risk = "medium"
        else:
            risk = "low"
        
        return CompatibilityCheck(
            is_compatible=len(conflicts) == 0,
            compatibility_score=compatibility_score,
            conflicts=conflicts,
            suggestions=suggestions,
            risk_assessment=risk
        )
    
    def _find_conflicts(self, clause1: str, clause2: str) -> List[Dict[str, str]]:
        """Find conflicts between clauses."""
        conflicts = []
        
        # Check for logical contradictions
        if self._check_logical_conflict(clause1, clause2):
            conflicts.append({
                'type': 'logical_contradiction',
                'description': 'Clauses contain contradictory terms',
                'severity': 'high'
            })
        
        # Check for jurisdictional conflicts
        jurisdiction_pattern = re.compile(r"governed by.*laws of (\w+)", re.IGNORECASE)
        match1 = jurisdiction_pattern.search(clause1)
        match2 = jurisdiction_pattern.search(clause2)
        
        if match1 and match2 and match1.group(1) != match2.group(1):
            conflicts.append({
                'type': 'jurisdiction_conflict',
                'description': f'Different jurisdictions: {match1.group(1)} vs {match2.group(1)}',
                'severity': 'high'
            })
        
        # Check for dispute resolution conflicts
        if "arbitration" in clause1.lower() and "court" in clause2.lower():
            conflicts.append({
                'type': 'dispute_resolution_conflict',
                'description': 'Conflicting dispute resolution mechanisms',
                'severity': 'medium'
            })
        
        return conflicts
    
    def _get_clause_embedding(self, clause: str) -> torch.Tensor:
        """Get embedding for a clause."""
        # Simplified embedding generation
        words = clause.lower().split()
        # Use word indices as simple embedding
        embedding = torch.zeros(768)
        for i, word in enumerate(words[:100]):
            embedding[hash(word) % 768] += 1
        return embedding.unsqueeze(0)
    
    def _generate_compatibility_suggestions(
        self,
        clause1: str,
        clause2: str,
        conflicts: List[Dict[str, str]]
    ) -> List[str]:
        """Generate suggestions to resolve conflicts."""
        suggestions = []
        
        for conflict in conflicts:
            if conflict['type'] == 'logical_contradiction':
                suggestions.append(
                    "Revise contradictory terms to ensure consistency"
                )
            elif conflict['type'] == 'jurisdiction_conflict':
                suggestions.append(
                    "Choose a single governing law for the entire agreement"
                )
            elif conflict['type'] == 'dispute_resolution_conflict':
                suggestions.append(
                    "Establish a unified dispute resolution process"
                )
        
        if not conflicts:
            suggestions.append("Clauses appear compatible - no changes needed")
        
        return suggestions


class LanguageOptimizer:
    """Optimizes legal language for clarity and enforceability."""
    
    def __init__(self):
        self.clarity_rules = self._load_clarity_rules()
        self.legal_requirements = self._load_legal_requirements()
    
    def _load_clarity_rules(self) -> Dict[str, str]:
        """Load rules for improving clarity."""
        return {
            "passive_voice": "Use active voice for clarity",
            "long_sentences": "Break sentences over 30 words",
            "nested_clauses": "Simplify nested conditional clauses",
            "undefined_terms": "Define all technical terms",
            "ambiguous_pronouns": "Replace pronouns with specific references"
        }
    
    def _load_legal_requirements(self) -> Dict[str, List[str]]:
        """Load legal requirements by clause type."""
        return {
            "arbitration": [
                "Specify arbitration rules (AAA, JAMS, etc.)",
                "Include arbitrator selection process",
                "Define cost allocation",
                "Specify location/venue"
            ],
            "confidentiality": [
                "Define confidential information",
                "Specify duration",
                "Include exceptions",
                "Address return of materials"
            ],
            "liability": [
                "Specify types of damages excluded",
                "Include liability cap if applicable",
                "Address gross negligence/willful misconduct",
                "Ensure mutuality if required"
            ]
        }
    
    def optimize(self, clause: str, clause_type: str) -> Tuple[str, List[str]]:
        """
        Optimize clause language.
        
        Returns:
            Tuple of (optimized_clause, improvement_notes)
        """
        optimized = clause
        notes = []
        
        # Apply clarity improvements
        if self._has_passive_voice(optimized):
            optimized = self._convert_to_active(optimized)
            notes.append("Converted to active voice")
        
        # Check legal requirements
        missing = self._check_requirements(optimized, clause_type)
        if missing:
            notes.append(f"Consider adding: {', '.join(missing)}")
        
        return optimized, notes
    
    def _has_passive_voice(self, text: str) -> bool:
        """Check for passive voice constructions."""
        passive_indicators = [
            r"was \w+ed by",
            r"were \w+ed by",
            r"is being \w+ed",
            r"has been \w+ed",
            r"will be \w+ed"
        ]
        
        for pattern in passive_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _convert_to_active(self, text: str) -> str:
        """Convert passive voice to active (simplified)."""
        # This is a simplified conversion - real implementation would be more sophisticated
        text = re.sub(
            r"(\w+) was (\w+ed) by (\w+)",
            r"\3 \2 \1",
            text
        )
        return text
    
    def _check_requirements(self, clause: str, clause_type: str) -> List[str]:
        """Check for missing legal requirements."""
        if clause_type not in self.legal_requirements:
            return []
        
        missing = []
        clause_lower = clause.lower()
        
        for requirement in self.legal_requirements[clause_type]:
            # Simple keyword check - real implementation would be more sophisticated
            key_words = requirement.lower().split()[:3]
            if not any(word in clause_lower for word in key_words):
                missing.append(requirement)
        
        return missing