#!/usr/bin/env python3
"""
Test script to verify arbitration detection fixes.
This script tests that the system correctly rejects non-legal documents.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from rag.arbitration_detector import ArbitrationDetector
from rag.document_validator import DocumentValidator


def test_math_quiz():
    """Test that math quizzes don't trigger arbitration detection."""
    print("Testing math quiz detection...")
    
    math_quiz = """
    Mathematics Test - Chapter 7: Algebra and Arbitration Theory
    
    Question 1: In the equation 2x + arbitration = 15, solve for x if arbitration = 5
    a) x = 5
    b) x = 10  
    c) x = 7.5
    d) x = 2.5
    
    Question 2: If a binding agreement has coefficient y = 3x + 2, what is y when x = 4?
    Choose the correct answer:
    a) y = 14
    b) y = 12
    c) arbitration = 10
    d) binding = 16
    
    Question 3: Calculate the probability of arbitration success:
    P(arbitration) = binding_cases / total_cases
    If binding_cases = 12 and total_cases = 20, find P(arbitration).
    """
    
    detector = ArbitrationDetector()
    result = detector.detect(math_quiz, "math_quiz_test")
    
    print(f"Has arbitration: {result.has_arbitration}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Document type: {result.validation_result.document_type.value if result.validation_result else 'Unknown'}")
    print(f"Is legal document: {result.validation_result.is_legal_document if result.validation_result else 'Unknown'}")
    
    if result.has_arbitration:
        print("❌ FAILED: Math quiz incorrectly detected arbitration!")
        return False
    else:
        print("✅ PASSED: Math quiz correctly rejected")
        return True


def test_recipe():
    """Test that recipes don't trigger arbitration detection."""
    print("\nTesting recipe detection...")
    
    recipe = """
    Legal Sea Foods Famous Binding Chowder Recipe
    
    This binding chowder requires an agreement between flavors.
    
    Ingredients (parties to this culinary contract):
    - 2 cups fresh clams (the primary arbitration ingredient) 
    - 1 binding agreement of heavy cream
    - Salt and pepper (to enforce flavor)
    - 1 cup diced potatoes (subject to jurisdiction of taste)
    
    Instructions (Terms and Conditions):
    1. The parties (ingredients) shall be combined in binding arbitration
    2. Cook until all disputes between flavors are resolved
    3. Any disagreements about seasoning must be arbitrated by the chef
    4. Serve according to the governing laws of good taste
    5. This recipe is binding and mandatory for all who wish to enjoy it
    
    Note: All disputes arising from this recipe shall be resolved through
    kitchen arbitration. No class action lawsuits against the chef allowed.
    """
    
    detector = ArbitrationDetector()
    result = detector.detect(recipe, "recipe_test")
    
    print(f"Has arbitration: {result.has_arbitration}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Document type: {result.validation_result.document_type.value if result.validation_result else 'Unknown'}")
    print(f"Is legal document: {result.validation_result.is_legal_document if result.validation_result else 'Unknown'}")
    
    if result.has_arbitration:
        print("❌ FAILED: Recipe incorrectly detected arbitration!")
        return False
    else:
        print("✅ PASSED: Recipe correctly rejected")
        return True


def test_story():
    """Test that stories don't trigger arbitration detection."""
    print("\nTesting story detection...")
    
    story = """
    The Lawyer's Adventure: Chapter 12 - The Binding Arbitration
    
    Once upon a time, there was a young lawyer named Sarah who specialized in
    arbitration cases. She had agreed to take on a binding arbitration dispute
    between two large corporations.
    
    "This arbitration agreement is ironclad," said the opposing counsel.
    "Your client must submit to mandatory arbitration under the terms of service."
    
    Sarah studied the arbitration clause carefully. The character realized that
    the binding arbitration provision might have a loophole. As the plot unfolds,
    she discovers that the arbitration agreement was not properly executed.
    
    "I dispute this arbitration clause," Sarah announced in the hearing.
    "The mandatory arbitration terms violate state law."
    
    The arbitrator pondered this argument. Would Sarah's character succeed in
    challenging the binding arbitration? The story continues as our protagonist
    fights for justice in this legal thriller.
    
    Will the arbitration be upheld? Find out in the next chapter of this
    exciting legal drama! The tale of binding agreements and courtroom
    disputes continues...
    """
    
    detector = ArbitrationDetector()
    result = detector.detect(story, "story_test")
    
    print(f"Has arbitration: {result.has_arbitration}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Document type: {result.validation_result.document_type.value if result.validation_result else 'Unknown'}")
    print(f"Is legal document: {result.validation_result.is_legal_document if result.validation_result else 'Unknown'}")
    
    if result.has_arbitration:
        print("❌ FAILED: Story incorrectly detected arbitration!")
        return False
    else:
        print("✅ PASSED: Story correctly rejected")
        return True


def test_legitimate_tos():
    """Test that legitimate Terms of Service are correctly detected."""
    print("\nTesting legitimate Terms of Service...")
    
    tos = """
    TERMS OF SERVICE AGREEMENT
    
    Last Updated: January 15, 2024
    
    1. ACCEPTANCE OF TERMS
    By accessing and using our service, you agree to be bound by these Terms of Service
    and all applicable laws and regulations.
    
    2. USER OBLIGATIONS
    You agree to use the service only for lawful purposes and in accordance with these Terms.
    Users shall not engage in any activity that violates applicable law.
    
    3. BINDING ARBITRATION AND CLASS ACTION WAIVER
    PLEASE READ THIS SECTION CAREFULLY AS IT AFFECTS YOUR LEGAL RIGHTS.
    
    Any dispute, claim, or controversy arising out of or relating to these Terms or your
    use of the service shall be settled by binding arbitration administered by the
    American Arbitration Association (AAA) in accordance with its Commercial Arbitration Rules.
    
    You agree that binding arbitration is the exclusive means of resolving disputes and hereby
    waive your right to a jury trial. You also waive your right to participate in a class action
    lawsuit or class-wide arbitration.
    
    4. GOVERNING LAW
    These Terms shall be governed by the laws of the State of California without regard
    to conflict of law principles.
    
    5. LIMITATION OF LIABILITY
    In no event shall the company be liable for any indirect, incidental, special,
    consequential, or punitive damages.
    """
    
    detector = ArbitrationDetector()
    result = detector.detect(tos, "legitimate_tos_test")
    
    print(f"Has arbitration: {result.has_arbitration}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Document type: {result.validation_result.document_type.value if result.validation_result else 'Unknown'}")
    print(f"Is legal document: {result.validation_result.is_legal_document if result.validation_result else 'Unknown'}")
    print(f"Number of clauses found: {len(result.clauses)}")
    
    if not result.has_arbitration:
        print("❌ FAILED: Legitimate ToS should have detected arbitration!")
        return False
    else:
        print("✅ PASSED: Legitimate ToS correctly detected arbitration")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ENHANCED ARBITRATION DETECTION SYSTEM")
    print("=" * 60)
    
    results = []
    
    # Test non-legal documents (should NOT detect arbitration)
    results.append(test_math_quiz())
    results.append(test_recipe()) 
    results.append(test_story())
    
    # Test legal document (should detect arbitration)
    results.append(test_legitimate_tos())
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The arbitration detection system is working correctly.")
        print("✅ Non-legal documents are properly rejected")
        print("✅ Legal documents with arbitration are properly detected")
    else:
        print("⚠️  SOME TESTS FAILED! The system needs further refinement.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)