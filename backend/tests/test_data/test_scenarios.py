"""
Test scenario generator for arbitration clause detection system.

This module generates comprehensive test scenarios including:
- Document generation with various arbitration clause patterns
- Edge cases and boundary conditions
- Performance test documents
- Accuracy validation scenarios
"""

import json
import random
import string
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestScenario:
    """Represents a test scenario with expected results."""
    id: str
    name: str
    description: str
    document_text: str
    expected_result: Dict[str, Any]
    category: str
    difficulty_level: str  # easy, medium, hard, expert
    tags: List[str]


class ArbitrationTestGenerator:
    """Generates test documents and scenarios for arbitration detection."""
    
    def __init__(self):
        self.arbitration_phrases = [
            "binding arbitration",
            "mandatory arbitration", 
            "arbitration agreement",
            "resolved through arbitration",
            "arbitration proceedings",
            "arbitral tribunal",
            "arbitrator's decision",
            "final and binding arbitration"
        ]
        
        self.arbitration_bodies = [
            "American Arbitration Association",
            "AAA",
            "JAMS",
            "International Chamber of Commerce",
            "ICC",
            "London Court of International Arbitration",
            "LCIA",
            "International Centre for Settlement of Investment Disputes",
            "ICSID",
            "CPR International"
        ]
        
        self.locations = [
            "New York, New York",
            "Delaware",
            "California",
            "London, England", 
            "Geneva, Switzerland",
            "Singapore",
            "Hong Kong",
            "Paris, France"
        ]
        
        self.false_positive_terms = [
            "arbitrage",
            "arbitrary",
            "arbitrarily",
            "arbitrate justice",
            "arbitrating committee"
        ]

    def generate_clear_arbitration_documents(self) -> List[TestScenario]:
        """Generate documents with clear arbitration clauses."""
        scenarios = []
        
        # Simple binding arbitration
        scenarios.append(TestScenario(
            id="clear_001",
            name="Simple Binding Arbitration",
            description="Basic mandatory binding arbitration clause",
            document_text="""
            ARBITRATION AGREEMENT
            
            Any dispute arising under this agreement shall be resolved through binding 
            arbitration administered by the American Arbitration Association in accordance 
            with its Commercial Arbitration Rules. The arbitration award shall be final 
            and binding upon both parties.
            """,
            expected_result={
                "has_arbitration": True,
                "confidence": 0.95,
                "clause_type": "mandatory_binding",
                "keywords": ["binding arbitration", "American Arbitration Association", "final and binding"],
                "arbitration_body": "American Arbitration Association"
            },
            category="clear_arbitration",
            difficulty_level="easy",
            tags=["binding", "AAA", "simple"]
        ))
        
        # International arbitration
        scenarios.append(TestScenario(
            id="clear_002",
            name="ICC International Arbitration",
            description="International arbitration with ICC rules",
            document_text="""
            DISPUTE RESOLUTION CLAUSE
            
            All disputes arising out of or in connection with this contract shall be 
            finally settled under the Rules of Arbitration of the International Chamber 
            of Commerce by three arbitrators appointed in accordance with the said Rules. 
            The seat of arbitration shall be Geneva, Switzerland. The language of the 
            arbitration shall be English.
            """,
            expected_result={
                "has_arbitration": True,
                "confidence": 0.98,
                "clause_type": "international_arbitration",
                "keywords": ["International Chamber of Commerce", "three arbitrators", "Geneva"],
                "arbitration_body": "International Chamber of Commerce",
                "location": "Geneva, Switzerland",
                "number_of_arbitrators": 3,
                "language": "English"
            },
            category="clear_arbitration",
            difficulty_level="medium",
            tags=["international", "ICC", "three_arbitrators"]
        ))
        
        return scenarios

    def generate_hidden_arbitration_documents(self) -> List[TestScenario]:
        """Generate documents with arbitration clauses buried in text."""
        scenarios = []
        
        # Buried in employment contract
        scenarios.append(TestScenario(
            id="hidden_001",
            name="Employment Contract Hidden Clause",
            description="Arbitration clause buried in employment agreement",
            document_text="""
            EMPLOYMENT AGREEMENT
            
            This Employment Agreement is entered into between Company and Employee.
            
            Section 1: Position and Duties
            Employee agrees to serve as Senior Manager with the following responsibilities...
            
            Section 2: Compensation
            Base salary shall be $75,000 per year...
            
            Section 8: Benefits
            Employee shall be entitled to health insurance, dental coverage...
            
            Section 15: Miscellaneous Provisions
            15.1 This Agreement shall be governed by Delaware law.
            15.2 Any modification must be in writing.
            15.3 Any claim or dispute between Employee and Company, whether arising under 
            this Agreement or otherwise, including claims for discrimination, harassment, 
            or wrongful termination, shall be resolved exclusively through final and 
            binding arbitration before a single arbitrator selected pursuant to the 
            Employment Arbitration Rules of the American Arbitration Association.
            15.4 This Agreement represents the complete agreement between the parties.
            """,
            expected_result={
                "has_arbitration": True,
                "confidence": 0.88,
                "clause_type": "employment_arbitration",
                "keywords": ["exclusively through", "final and binding arbitration", "single arbitrator"],
                "arbitration_body": "American Arbitration Association",
                "employment_context": True,
                "exclusivity": True
            },
            category="hidden_arbitration",
            difficulty_level="medium",
            tags=["employment", "buried", "exclusive"]
        ))
        
        return scenarios

    def generate_negative_documents(self) -> List[TestScenario]:
        """Generate documents without arbitration clauses."""
        scenarios = []
        
        # Court jurisdiction only
        scenarios.append(TestScenario(
            id="negative_001",
            name="Court Jurisdiction Only", 
            description="Document specifying court jurisdiction without arbitration",
            document_text="""
            DISPUTE RESOLUTION
            
            Any disputes arising under this agreement shall be resolved exclusively 
            in the state and federal courts located in New York County, New York. 
            The parties hereby consent to the personal jurisdiction of such courts 
            and waive any objection to venue in such courts. Each party shall bear 
            its own attorney's fees and costs.
            """,
            expected_result={
                "has_arbitration": False,
                "confidence": 0.05,
                "clause_type": None,
                "keywords": [],
                "jurisdiction": "New York County, New York",
                "court_selection": True
            },
            category="no_arbitration",
            difficulty_level="easy",
            tags=["court_jurisdiction", "no_arbitration"]
        ))
        
        return scenarios

    def generate_ambiguous_documents(self) -> List[TestScenario]:
        """Generate documents with ambiguous arbitration language."""
        scenarios = []
        
        # Optional arbitration
        scenarios.append(TestScenario(
            id="ambiguous_001",
            name="Optional Arbitration Clause",
            description="Document with optional rather than mandatory arbitration",
            document_text="""
            DISPUTE RESOLUTION OPTIONS
            
            In the event of a dispute, the parties may elect to resolve the matter 
            through any of the following methods: (a) direct negotiation, (b) mediation 
            through a neutral third party, or (c) binding arbitration under the rules 
            of JAMS. If no election is made within thirty (30) days of notice of 
            dispute, the matter shall proceed to litigation in the appropriate court.
            """,
            expected_result={
                "has_arbitration": False,
                "confidence": 0.35,
                "clause_type": "optional",
                "keywords": ["may elect", "binding arbitration", "JAMS"],
                "arbitration_optional": True,
                "default_to_litigation": True
            },
            category="ambiguous",
            difficulty_level="hard",
            tags=["optional", "ambiguous", "litigation_default"]
        ))
        
        return scenarios

    def generate_false_positive_documents(self) -> List[TestScenario]:
        """Generate documents designed to test false positive prevention."""
        scenarios = []
        
        # Arbitrage financial terms
        scenarios.append(TestScenario(
            id="false_pos_001",
            name="Financial Arbitrage Terms",
            description="Document using arbitrage in financial context",
            document_text="""
            INVESTMENT STRATEGY
            
            The fund manager may engage in arbitrage strategies to exploit price 
            differentials in various markets. These arbitrage opportunities may include 
            currency arbitrage, merger arbitrage, and statistical arbitrage. The 
            arbitrage desk will operate under strict risk management protocols.
            
            Any disputes arising under this investment agreement shall be resolved 
            in the courts of New York.
            """,
            expected_result={
                "has_arbitration": False,
                "confidence": 0.05,
                "clause_type": None,
                "keywords": ["arbitrage"],
                "financial_context": True,
                "court_jurisdiction": True
            },
            category="false_positive_prevention",
            difficulty_level="medium",
            tags=["arbitrage", "financial", "false_positive"]
        ))
        
        return scenarios

    def generate_complex_documents(self) -> List[TestScenario]:
        """Generate complex multi-tier arbitration documents."""
        scenarios = []
        
        # Multi-tier with escalation
        scenarios.append(TestScenario(
            id="complex_001",
            name="Multi-Tier Escalation Process",
            description="Complex dispute resolution with multiple tiers",
            document_text="""
            COMPREHENSIVE DISPUTE RESOLUTION PROCEDURE
            
            The parties agree to the following escalating dispute resolution process:
            
            TIER 1: DIRECT NEGOTIATION (30 Days)
            The parties shall first attempt to resolve any dispute through good faith, 
            direct negotiations between senior executives of each party.
            
            TIER 2: MEDIATION (60 Days) 
            If Tier 1 is unsuccessful, the dispute shall be submitted to mediation 
            administered by CPR International Institute for Conflict Prevention and 
            Resolution under its Mediation Procedure.
            
            TIER 3: EXPEDITED ARBITRATION (For Disputes < $500,000)
            For disputes involving amounts less than $500,000, the matter shall proceed 
            to expedited arbitration under JAMS Expedited Arbitration Rules before a 
            single arbitrator.
            
            TIER 4: COMPREHENSIVE ARBITRATION (For Disputes ≥ $500,000)
            For disputes involving $500,000 or more, the matter shall be finally resolved 
            by binding arbitration administered by the International Chamber of Commerce 
            under its Rules of Arbitration. The arbitration shall be conducted by three 
            arbitrators, with each party appointing one arbitrator and the third selected 
            by the two party-appointed arbitrators. The seat of arbitration shall be 
            New York, New York, and the proceedings shall be conducted in English.
            
            All arbitration awards under Tiers 3 and 4 shall be final and binding, 
            with no right of appeal except on grounds specified in the Federal 
            Arbitration Act.
            """,
            expected_result={
                "has_arbitration": True,
                "confidence": 0.97,
                "clause_type": "multi_tier_complex",
                "keywords": ["multi-tier", "expedited arbitration", "binding arbitration", "ICC", "JAMS"],
                "multi_tier": True,
                "amount_based_tiers": True,
                "arbitration_bodies": ["JAMS", "International Chamber of Commerce"],
                "mediation_first": True,
                "negotiation_first": True
            },
            category="complex_arbitration",
            difficulty_level="expert",
            tags=["multi_tier", "complex", "amount_based", "escalation"]
        ))
        
        return scenarios

    def generate_performance_test_documents(self) -> List[TestScenario]:
        """Generate documents for performance testing."""
        scenarios = []
        
        # Very long document
        base_text = "This is a comprehensive legal agreement. " * 100
        arbitration_clause = """
        
        DISPUTE RESOLUTION: Any dispute arising under this agreement shall be 
        resolved through binding arbitration administered by the American 
        Arbitration Association.
        """
        long_document = base_text + arbitration_clause + base_text
        
        scenarios.append(TestScenario(
            id="perf_001",
            name="Large Document Performance Test",
            description="Very long document to test processing speed",
            document_text=long_document,
            expected_result={
                "has_arbitration": True,
                "confidence": 0.85,
                "max_processing_time": 5.0
            },
            category="performance",
            difficulty_level="medium",
            tags=["large_document", "performance"]
        ))
        
        return scenarios

    def generate_edge_case_documents(self) -> List[TestScenario]:
        """Generate edge case test documents.""" 
        scenarios = []
        
        # Document with special characters
        scenarios.append(TestScenario(
            id="edge_001",
            name="Special Characters Document",
            description="Document with Unicode and special characters",
            document_text="""
            ACUERDO DE ARBITRAJE / ARBITRATION AGREEMENT
            
            Cualquier disputa será resuelta mediante arbitraje vinculante / 
            Any dispute shall be resolved through binding arbitration administered 
            by the American Arbitration Association (AAA) in accordance with its 
            Commercial Rules.
            
            Special characters test: àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ
            Currency symbols: $ € £ ¥ ₹
            Mathematical: ∑ ∆ π ∞ ≤ ≥ ≠
            """,
            expected_result={
                "has_arbitration": True,
                "confidence": 0.88,
                "clause_type": "mandatory_binding",
                "unicode_handling": True
            },
            category="edge_cases",
            difficulty_level="medium",
            tags=["unicode", "special_characters", "multilingual"]
        ))
        
        return scenarios

    def generate_all_scenarios(self) -> List[TestScenario]:
        """Generate all test scenarios."""
        all_scenarios = []
        
        all_scenarios.extend(self.generate_clear_arbitration_documents())
        all_scenarios.extend(self.generate_hidden_arbitration_documents())
        all_scenarios.extend(self.generate_negative_documents())
        all_scenarios.extend(self.generate_ambiguous_documents())
        all_scenarios.extend(self.generate_false_positive_documents())
        all_scenarios.extend(self.generate_complex_documents())
        all_scenarios.extend(self.generate_performance_test_documents())
        all_scenarios.extend(self.generate_edge_case_documents())
        
        return all_scenarios

    def save_scenarios_to_files(self, output_dir: Path):
        """Save test scenarios to JSON files organized by category."""
        scenarios = self.generate_all_scenarios()
        
        # Group by category
        categories = {}
        for scenario in scenarios:
            if scenario.category not in categories:
                categories[scenario.category] = []
            
            categories[scenario.category].append({
                "id": scenario.id,
                "name": scenario.name,
                "description": scenario.description,
                "document_text": scenario.document_text.strip(),
                "expected_result": scenario.expected_result,
                "difficulty_level": scenario.difficulty_level,
                "tags": scenario.tags
            })
        
        # Save each category to separate file
        for category, category_scenarios in categories.items():
            file_path = output_dir / f"{category}_scenarios.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "category": category,
                    "scenario_count": len(category_scenarios),
                    "scenarios": category_scenarios
                }, f, indent=2, ensure_ascii=False)
        
        # Save summary file
        summary = {
            "total_scenarios": len(scenarios),
            "categories": {
                category: len(category_scenarios) 
                for category, category_scenarios in categories.items()
            },
            "difficulty_distribution": {
                level: len([s for s in scenarios if s.difficulty_level == level])
                for level in ["easy", "medium", "hard", "expert"]
            }
        }
        
        summary_path = output_dir / "test_scenarios_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return categories, summary


class AccuracyTestSuite:
    """Test suite for measuring arbitration detection accuracy."""
    
    def __init__(self, scenarios: List[TestScenario]):
        self.scenarios = scenarios
    
    def calculate_accuracy_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for scenario, result in zip(self.scenarios, results):
            expected = scenario.expected_result["has_arbitration"]
            predicted = result.get("has_arbitration", False)
            
            if expected and predicted:
                true_positives += 1
            elif not expected and predicted:
                false_positives += 1
            elif not expected and not predicted:
                true_negatives += 1
            elif expected and not predicted:
                false_negatives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(self.scenarios)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }
    
    def generate_accuracy_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive accuracy report."""
        overall_metrics = self.calculate_accuracy_metrics(results)
        
        # Category-specific metrics
        category_metrics = {}
        categories = set(scenario.category for scenario in self.scenarios)
        
        for category in categories:
            category_scenarios = [s for s in self.scenarios if s.category == category]
            category_results = [results[i] for i, s in enumerate(self.scenarios) if s.category == category]
            
            if category_results:
                suite = AccuracyTestSuite(category_scenarios)
                category_metrics[category] = suite.calculate_accuracy_metrics(category_results)
        
        # Difficulty-specific metrics
        difficulty_metrics = {}
        difficulties = set(scenario.difficulty_level for scenario in self.scenarios)
        
        for difficulty in difficulties:
            difficulty_scenarios = [s for s in self.scenarios if s.difficulty_level == difficulty]
            difficulty_results = [results[i] for i, s in enumerate(self.scenarios) if s.difficulty_level == difficulty]
            
            if difficulty_results:
                suite = AccuracyTestSuite(difficulty_scenarios)
                difficulty_metrics[difficulty] = suite.calculate_accuracy_metrics(difficulty_results)
        
        return {
            "overall_metrics": overall_metrics,
            "category_metrics": category_metrics,
            "difficulty_metrics": difficulty_metrics,
            "total_scenarios": len(self.scenarios)
        }


def generate_test_data():
    """Main function to generate all test data."""
    generator = ArbitrationTestGenerator()
    output_dir = Path(__file__).parent
    
    categories, summary = generator.save_scenarios_to_files(output_dir)
    
    print(f"Generated {summary['total_scenarios']} test scenarios")
    print(f"Categories: {', '.join(categories.keys())}")
    print(f"Difficulty distribution: {summary['difficulty_distribution']}")
    
    return categories, summary


if __name__ == "__main__":
    generate_test_data()