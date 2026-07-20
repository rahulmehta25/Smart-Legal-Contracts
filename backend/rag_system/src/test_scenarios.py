#!/usr/bin/env python3
"""
Additional test scenarios for arbitration detection accuracy testing.

This module provides comprehensive test cases including edge cases,
ambiguous language scenarios, and specific legal contexts.
"""

from typing import List, Dict

def get_class_action_waiver_scenarios() -> List[Dict]:
    """Test scenarios focused on class action waivers."""
    return [
        {
            'id': 'class_001',
            'text': 'YOU WAIVE ANY RIGHT TO PARTICIPATE IN A CLASS ACTION LAWSUIT OR CLASS-WIDE ARBITRATION.',
            'expected': True,
            'description': 'Clear class action waiver',
            'confidence_expected': 'high'
        },
        {
            'id': 'class_002',
            'text': 'This agreement prohibits class actions and requires individual arbitration.',
            'expected': True,
            'description': 'Class action prohibition with arbitration requirement',
            'confidence_expected': 'high'
        },
        {
            'id': 'class_003',
            'text': 'No class or representative actions are permitted under this agreement.',
            'expected': True,
            'description': 'Implicit arbitration through class action prohibition',
            'confidence_expected': 'medium'
        }
    ]

def get_opt_out_scenarios() -> List[Dict]:
    """Test scenarios focused on arbitration opt-out provisions."""
    return [
        {
            'id': 'opt_001',
            'text': 'You may opt out of this arbitration agreement by sending written notice within 30 days.',
            'expected': True,
            'description': 'Opt-out provision (still indicates arbitration exists)',
            'confidence_expected': 'medium'
        },
        {
            'id': 'opt_002',
            'text': 'To reject this arbitration provision, email legal@company.com within sixty days.',
            'expected': True,
            'description': 'Rejection option for arbitration',
            'confidence_expected': 'medium'
        },
        {
            'id': 'opt_003',
            'text': 'This opt-out provision allows you to decline mandatory arbitration.',
            'expected': True,
            'description': 'Explicit opt-out with mandatory arbitration mention',
            'confidence_expected': 'high'
        }
    ]

def get_ambiguous_language_scenarios() -> List[Dict]:
    """Test scenarios with ambiguous or unclear language."""
    return [
        {
            'id': 'ambiguous_001',
            'text': 'Disputes may be resolved through various means including arbitration.',
            'expected': False,
            'description': 'Optional/suggestive language',
            'confidence_expected': 'low'
        },
        {
            'id': 'ambiguous_002',
            'text': 'The parties prefer arbitration but retain all legal remedies.',
            'expected': False,
            'description': 'Preference without mandate',
            'confidence_expected': 'low'
        },
        {
            'id': 'ambiguous_003',
            'text': 'Subject to applicable law, arbitration may be required.',
            'expected': False,
            'description': 'Conditional arbitration requirement',
            'confidence_expected': 'low'
        },
        {
            'id': 'ambiguous_004',
            'text': 'The arbitration requirement depends on the nature of the dispute.',
            'expected': False,
            'description': 'Situational arbitration',
            'confidence_expected': 'low'
        }
    ]

def get_jurisdiction_confusion_scenarios() -> List[Dict]:
    """Test scenarios that might confuse jurisdiction with arbitration."""
    return [
        {
            'id': 'jurisdiction_001',
            'text': 'All disputes shall be resolved in the courts of New York, New York.',
            'expected': False,
            'description': 'Court jurisdiction clause (not arbitration)',
            'confidence_expected': 'very_low'
        },
        {
            'id': 'jurisdiction_002',
            'text': 'This agreement is subject to the exclusive jurisdiction of Delaware courts.',
            'expected': False,
            'description': 'Exclusive court jurisdiction',
            'confidence_expected': 'very_low'
        },
        {
            'id': 'jurisdiction_003',
            'text': 'Venue for any legal proceedings shall be in California state or federal court.',
            'expected': False,
            'description': 'Venue selection for court proceedings',
            'confidence_expected': 'very_low'
        }
    ]

def get_mediation_only_scenarios() -> List[Dict]:
    """Test scenarios with mediation but no arbitration."""
    return [
        {
            'id': 'mediation_001',
            'text': 'Disputes shall first be submitted to mediation before any court proceedings.',
            'expected': False,
            'description': 'Mediation requirement without arbitration',
            'confidence_expected': 'low'
        },
        {
            'id': 'mediation_002',
            'text': 'The parties agree to attempt mediation in good faith prior to litigation.',
            'expected': False,
            'description': 'Mediation attempt before litigation',
            'confidence_expected': 'very_low'
        }
    ]

def get_historical_or_academic_scenarios() -> List[Dict]:
    """Test scenarios discussing arbitration academically or historically."""
    return [
        {
            'id': 'academic_001',
            'text': 'The history of arbitration shows its effectiveness in commercial disputes.',
            'expected': False,
            'description': 'Academic discussion of arbitration',
            'confidence_expected': 'very_low'
        },
        {
            'id': 'academic_002',
            'text': 'Studies indicate that mandatory arbitration clauses are becoming more common.',
            'expected': False,
            'description': 'Research about arbitration clauses',
            'confidence_expected': 'very_low'
        },
        {
            'id': 'academic_003',
            'text': 'The Supreme Court has ruled on several arbitration cases in recent years.',
            'expected': False,
            'description': 'Legal analysis of arbitration cases',
            'confidence_expected': 'very_low'
        }
    ]

def get_complex_mixed_scenarios() -> List[Dict]:
    """Test scenarios with complex, mixed language."""
    return [
        {
            'id': 'complex_001',
            'text': '''
            DISPUTE RESOLUTION: Any claim, dispute, or controversy arising out of or relating to this Agreement
            shall be resolved through binding arbitration administered by the American Arbitration Association
            under its Commercial Arbitration Rules. However, either party may seek injunctive relief in court
            for the protection of intellectual property rights. The arbitration shall take place in San Francisco,
            California, and shall be governed by California law. YOU WAIVE YOUR RIGHT TO A JURY TRIAL AND TO
            PARTICIPATE IN ANY CLASS ACTION.
            ''',
            'expected': True,
            'description': 'Complex arbitration clause with exceptions',
            'confidence_expected': 'very_high'
        },
        {
            'id': 'complex_002',
            'text': '''
            This agreement contains dispute resolution procedures. First, the parties must attempt informal
            negotiation. If unsuccessful, disputes will be submitted to mediation. Only if mediation fails
            shall the matter proceed to binding arbitration under JAMS rules. Small claims court remains
            available for qualifying disputes under $5,000.
            ''',
            'expected': True,
            'description': 'Multi-step process ending in arbitration',
            'confidence_expected': 'high'
        }
    ]

def get_false_positive_traps() -> List[Dict]:
    """Test scenarios designed to trigger false positives."""
    return [
        {
            'id': 'trap_001',
            'text': 'We reserve the right to modify these terms without arbitration or prior notice.',
            'expected': False,
            'description': 'Arbitration mentioned but not as dispute resolution',
            'confidence_expected': 'very_low'
        },
        {
            'id': 'trap_002',
            'text': 'The binding nature of this agreement does not require arbitration.',
            'expected': False,
            'description': 'Binding language without arbitration requirement',
            'confidence_expected': 'very_low'
        },
        {
            'id': 'trap_003',
            'text': 'Class action lawsuits are discussed in the news but not relevant to this agreement.',
            'expected': False,
            'description': 'Class action mention without waiver',
            'confidence_expected': 'very_low'
        }
    ]

def get_international_scenarios() -> List[Dict]:
    """Test scenarios with international arbitration references."""
    return [
        {
            'id': 'intl_001',
            'text': 'Disputes shall be finally settled under the ICC Rules of Arbitration in Paris, France.',
            'expected': True,
            'description': 'ICC international arbitration',
            'confidence_expected': 'very_high'
        },
        {
            'id': 'intl_002',
            'text': 'Any controversy shall be resolved by UNCITRAL arbitration rules in Geneva.',
            'expected': True,
            'description': 'UNCITRAL arbitration',
            'confidence_expected': 'very_high'
        },
        {
            'id': 'intl_003',
            'text': 'This agreement is governed by English law but disputes go to LCIA arbitration.',
            'expected': True,
            'description': 'LCIA arbitration with English law',
            'confidence_expected': 'very_high'
        }
    ]

def get_all_test_scenarios() -> Dict[str, List[Dict]]:
    """Get all test scenarios organized by category."""
    return {
        'class_action_waivers': get_class_action_waiver_scenarios(),
        'opt_out_provisions': get_opt_out_scenarios(),
        'ambiguous_language': get_ambiguous_language_scenarios(),
        'jurisdiction_confusion': get_jurisdiction_confusion_scenarios(),
        'mediation_only': get_mediation_only_scenarios(),
        'academic_discussion': get_historical_or_academic_scenarios(),
        'complex_mixed': get_complex_mixed_scenarios(),
        'false_positive_traps': get_false_positive_traps(),
        'international': get_international_scenarios()
    }

def create_comprehensive_test_set() -> List[Dict]:
    """Create a comprehensive test set combining all scenarios."""
    all_scenarios = []
    
    for category, scenarios in get_all_test_scenarios().items():
        for scenario in scenarios:
            scenario['category'] = category
            all_scenarios.append(scenario)
    
    return all_scenarios

if __name__ == "__main__":
    # Print summary of test scenarios
    scenarios = get_all_test_scenarios()
    
    total_scenarios = 0
    expected_positive = 0
    
    print("Test Scenario Summary:")
    print("=" * 50)
    
    for category, category_scenarios in scenarios.items():
        positive_count = sum(1 for s in category_scenarios if s['expected'])
        total_scenarios += len(category_scenarios)
        expected_positive += positive_count
        
        print(f"{category:20s}: {len(category_scenarios):2d} scenarios ({positive_count} positive)")
    
    print("=" * 50)
    print(f"Total scenarios: {total_scenarios}")
    print(f"Expected positive: {expected_positive}")
    print(f"Expected negative: {total_scenarios - expected_positive}")