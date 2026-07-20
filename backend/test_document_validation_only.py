#!/usr/bin/env python3
"""
Test script to verify document validation fixes work.
This script only tests document validation without embeddings to avoid dependency issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from rag.document_validator import DocumentValidator, DocumentType


def test_math_quiz_validation():
    """Test that math quizzes are correctly identified as non-legal."""
    print("Testing math quiz validation...")
    
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
    
    validator = DocumentValidator()
    result = validator.validate_document(math_quiz)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    print(f"Reasons: {result.reasons}")
    
    if result.is_legal_document:
        print("❌ FAILED: Math quiz incorrectly classified as legal!")
        return False
    else:
        print("✅ PASSED: Math quiz correctly rejected as non-legal")
        return True


def test_recipe_validation():
    """Test that recipes are correctly identified as non-legal."""
    print("\nTesting recipe validation...")
    
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
    3. Preheat oven to 350 degrees
    4. Mix ingredients until combined
    5. Bake for 30 minutes
    6. Serve hot according to the governing laws of good taste
    """
    
    validator = DocumentValidator()
    result = validator.validate_document(recipe)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    
    if result.is_legal_document:
        print("❌ FAILED: Recipe incorrectly classified as legal!")
        return False
    else:
        print("✅ PASSED: Recipe correctly rejected as non-legal")
        return True


def test_story_validation():
    """Test that stories are correctly identified as non-legal."""
    print("\nTesting story validation...")
    
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
    
    The story continues as our protagonist fights for justice in this legal thriller.
    """
    
    validator = DocumentValidator()
    result = validator.validate_document(story)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    
    if result.is_legal_document:
        print("❌ FAILED: Story incorrectly classified as legal!")
        return False
    else:
        print("✅ PASSED: Story correctly rejected as non-legal")
        return True


def test_legitimate_tos_validation():
    """Test that legitimate Terms of Service are correctly identified as legal."""
    print("\nTesting legitimate Terms of Service validation...")
    
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
    
    validator = DocumentValidator()
    result = validator.validate_document(tos)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    
    if not result.is_legal_document:
        print("❌ FAILED: Legitimate ToS should be classified as legal!")
        return False
    else:
        print("✅ PASSED: Legitimate ToS correctly identified as legal")
        return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("TESTING ENHANCED DOCUMENT VALIDATION SYSTEM")
    print("=" * 60)
    
    results = []
    
    # Test non-legal documents (should NOT be classified as legal)
    results.append(test_math_quiz_validation())
    results.append(test_recipe_validation()) 
    results.append(test_story_validation())
    
    # Test legal document (should be classified as legal)
    results.append(test_legitimate_tos_validation())
    
    print("\n" + "=" * 60)
    print("VALIDATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Non-legal documents are properly rejected")
        print("✅ Legal documents are properly identified")
        print("\nThe enhanced validation system should prevent arbitration")
        print("detection from running on non-legal documents like math quizzes.")
    else:
        print("⚠️  SOME VALIDATION TESTS FAILED!")
        print("The document validator needs further refinement.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)