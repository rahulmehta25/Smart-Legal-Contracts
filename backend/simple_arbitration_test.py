#!/usr/bin/env python3
"""
Simple test script to verify document validation logic without ML dependencies.
This tests the core document validation that prevents false positives.
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
    
    Question 3: Calculate the probability of arbitration success.
    """
    
    validator = DocumentValidator()
    result = validator.validate_document(math_quiz)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    print(f"Reasons: {result.reasons}")
    
    if result.is_legal_document:
        print("❌ FAILED: Math quiz incorrectly identified as legal!")
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
    3. Preheat oven to 375 degrees
    4. Bake for 20 minutes or until binding is achieved
    5. Serve according to the governing laws of good taste
    """
    
    validator = DocumentValidator()
    result = validator.validate_document(recipe)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    print(f"Reasons: {result.reasons}")
    
    if result.is_legal_document:
        print("❌ FAILED: Recipe incorrectly identified as legal!")
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
    "Your client must submit to mandatory arbitration."
    
    Sarah studied the arbitration clause carefully. The character realized that
    the binding arbitration provision might have a loophole. As the plot unfolds,
    she discovers that the arbitration agreement was not properly executed.
    
    The story continues as our protagonist fights for justice in this legal thriller.
    This tale of binding agreements and courtroom disputes continues in the next chapter...
    """
    
    validator = DocumentValidator()
    result = validator.validate_document(story)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    print(f"Reasons: {result.reasons}")
    
    if result.is_legal_document:
        print("❌ FAILED: Story incorrectly identified as legal!")
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
    Any dispute, claim, or controversy arising out of or relating to these Terms shall be 
    settled by binding arbitration administered by the American Arbitration Association.
    
    You hereby agree that binding arbitration is the exclusive means of resolving disputes
    and waive your right to a jury trial and class action participation.
    
    4. GOVERNING LAW
    These Terms shall be governed by the laws of the State of California.
    
    5. LIMITATION OF LIABILITY
    The company shall not be liable for any indirect, incidental, or consequential damages.
    """
    
    validator = DocumentValidator()
    result = validator.validate_document(tos)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    print(f"Reasons: {result.reasons}")
    
    if not result.is_legal_document:
        print("❌ FAILED: Legitimate ToS should be identified as legal!")
        return False
    else:
        print("✅ PASSED: Legitimate ToS correctly identified as legal")
        return True


def test_academic_paper_validation():
    """Test that academic papers are correctly identified as non-legal."""
    print("\nTesting academic paper validation...")
    
    academic_paper = """
    Research Paper: Arbitration in Game Theory
    
    Abstract:
    This paper examines the role of arbitration in game theory and mechanism design.
    We analyze various arbitration mechanisms and their efficiency properties.
    
    1. Introduction
    Arbitration has become an important topic in the field of economics and game theory.
    This study investigates the theoretical foundations of binding arbitration mechanisms.
    
    2. Methodology
    We conducted experiments using the following approach:
    - Model 1: Nash arbitration solution
    - Model 2: Kalai-Smorodinsky solution
    - Data analysis using regression techniques
    
    3. Results  
    Our analysis shows that binding arbitration can achieve efficient outcomes
    in 87% of the test cases. The arbitration mechanism outperformed litigation
    in terms of cost and time efficiency.
    
    4. Conclusion
    The results demonstrate that arbitration theory provides valuable insights
    for practical dispute resolution mechanisms.
    
    References:
    [1] Nash, J. (1950). The Bargaining Problem. Econometrica.
    """
    
    validator = DocumentValidator()
    result = validator.validate_document(academic_paper)
    
    print(f"Is legal document: {result.is_legal_document}")
    print(f"Document type: {result.document_type.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Warning flags: {result.warning_flags}")
    print(f"Reasons: {result.reasons}")
    
    if result.is_legal_document:
        print("❌ FAILED: Academic paper incorrectly identified as legal!")
        return False
    else:
        print("✅ PASSED: Academic paper correctly rejected as non-legal")
        return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("TESTING DOCUMENT VALIDATION FOR FALSE POSITIVE PREVENTION")
    print("=" * 70)
    
    results = []
    
    # Test non-legal documents (should be rejected)
    results.append(test_math_quiz_validation())
    results.append(test_recipe_validation())
    results.append(test_story_validation())
    results.append(test_academic_paper_validation())
    
    # Test legal document (should be accepted)
    results.append(test_legitimate_tos_validation())
    
    print("\n" + "=" * 70)
    print("VALIDATION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Non-legal documents are properly identified and rejected")
        print("✅ Legal documents are properly identified and accepted")
        print("")
        print("The document validator will prevent arbitration detection from running")
        print("on math quizzes, recipes, stories, and other non-legal content.")
    else:
        print("⚠️  SOME VALIDATION TESTS FAILED!")
        print("The document validator needs further refinement.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)