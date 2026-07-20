"""
Comprehensive tests for document validation and false positive prevention.
Tests the enhanced arbitration detection system against various non-legal documents.
"""

import pytest
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from rag.document_validator import DocumentValidator, DocumentType, ValidationResult
from rag.arbitration_detector import ArbitrationDetector


class TestDocumentValidator:
    """Test the document validator component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DocumentValidator()
    
    def test_math_quiz_detection(self):
        """Test that math quizzes are correctly identified as non-legal."""
        math_quiz = """
        Math Quiz - Chapter 5: Algebra
        
        Question 1: Solve for x in the equation 2x + 5 = 15
        a) x = 5
        b) x = 10  
        c) x = 7.5
        d) x = 2.5
        
        Question 2: If y = 3x + 2, what is the value of y when x = 4?
        a) y = 14
        b) y = 12
        c) y = 10
        d) y = 16
        
        Question 3: Which theorem states that a² + b² = c²?
        a) Pythagorean theorem
        b) Fermat's theorem
        c) Euler's theorem
        d) Newton's theorem
        """
        
        result = self.validator.validate_document(math_quiz)
        
        assert not result.is_legal_document
        assert result.document_type in [DocumentType.QUIZ, DocumentType.ACADEMIC, DocumentType.NON_LEGAL]
        assert result.confidence < 0.4
        assert "quiz" in [flag.lower() for flag in result.warning_flags] or \
               "question" in [flag.lower() for flag in result.warning_flags]
    
    def test_recipe_detection(self):
        """Test that recipes are correctly identified as non-legal."""
        recipe = """
        Chocolate Chip Cookie Recipe
        
        Ingredients:
        - 2 cups all-purpose flour
        - 1 tsp baking soda
        - 1 tsp salt
        - 1 cup butter, softened
        - 3/4 cup granulated sugar
        - 2 large eggs
        - 2 cups chocolate chips
        
        Instructions:
        1. Preheat oven to 375 degrees F
        2. Mix flour, baking soda, and salt in bowl
        3. Beat butter and sugars until creamy
        4. Add eggs and mix well
        5. Stir in flour mixture and chocolate chips
        6. Bake for 9-11 minutes until golden brown
        7. Cool and serve
        """
        
        result = self.validator.validate_document(recipe)
        
        assert not result.is_legal_document
        assert result.document_type in [DocumentType.RECIPE, DocumentType.NON_LEGAL]
        assert result.confidence < 0.4
        assert any("recipe" in flag.lower() or "ingredient" in flag.lower() 
                  for flag in result.warning_flags)
    
    def test_story_detection(self):
        """Test that stories are correctly identified as non-legal."""
        story = """
        The Adventure Begins
        Chapter 1: A New Journey
        
        Once upon a time, in a land far away, there lived a young adventurer named Alex.
        Alex had always dreamed of exploring the mysterious forests beyond the village.
        
        One morning, Alex decided to pack a bag with supplies and set off on the journey.
        The character walked through the winding paths, discovering new sights and sounds.
        
        As the story unfolds, Alex encounters various challenges that test both courage
        and wisdom. The plot thickens when a mysterious figure appears at a crossroads.
        
        "Which path will you choose?" asked the narrator, as Alex pondered the decision.
        
        The tale continues with unexpected twists and turns, leading to an epic conclusion
        that will leave readers wanting more.
        
        The End.
        """
        
        result = self.validator.validate_document(story)
        
        assert not result.is_legal_document
        assert result.document_type in [DocumentType.STORY, DocumentType.NON_LEGAL]
        assert result.confidence < 0.4
        assert any("story" in flag.lower() or "chapter" in flag.lower() 
                  for flag in result.warning_flags)
    
    def test_legal_terms_of_service(self):
        """Test that legitimate Terms of Service are correctly identified as legal."""
        terms_of_service = """
        TERMS OF SERVICE
        
        Last Updated: January 1, 2024
        
        1. ACCEPTANCE OF TERMS
        By accessing and using this service, you agree to be bound by these Terms of Service.
        
        2. USER OBLIGATIONS  
        Users shall not violate any applicable laws or regulations when using the service.
        You agree to indemnify the company against any liability arising from your use.
        
        3. DISPUTE RESOLUTION
        Any disputes arising under this agreement shall be resolved through binding arbitration
        in accordance with the American Arbitration Association rules. You hereby waive your
        right to participate in any class action and agree to submit to the jurisdiction
        of arbitrators selected under these terms.
        
        4. GOVERNING LAW
        These terms shall be governed by the laws of the State of California.
        
        5. TERMINATION
        We may terminate your access at any time for breach of these terms.
        """
        
        result = self.validator.validate_document(terms_of_service)
        
        assert result.is_legal_document
        assert result.document_type == DocumentType.TERMS_OF_SERVICE
        assert result.confidence > 0.7
    
    def test_academic_paper(self):
        """Test that academic papers are correctly identified as non-legal."""
        academic_paper = """
        Research Paper: Machine Learning in Natural Language Processing
        
        Abstract:
        This paper presents a comprehensive analysis of machine learning techniques
        applied to natural language processing tasks. We examine various algorithms
        and their performance on different datasets.
        
        1. Introduction
        Natural language processing (NLP) has become increasingly important in
        the field of artificial intelligence. This study investigates the effectiveness
        of different machine learning approaches.
        
        2. Methodology  
        We conducted experiments using the following datasets and algorithms:
        - Dataset A: 10,000 text samples
        - Algorithm 1: Support Vector Machines
        - Algorithm 2: Neural Networks
        
        3. Results
        Our experiments show that Algorithm 2 achieved 94% accuracy, while
        Algorithm 1 achieved 89% accuracy on the test dataset.
        
        4. Conclusion
        The results demonstrate that neural networks outperform traditional
        machine learning methods for this particular NLP task.
        
        References:
        [1] Smith, J. (2023). Advanced NLP Techniques. Journal of AI Research.
        """
        
        result = self.validator.validate_document(academic_paper)
        
        assert not result.is_legal_document
        assert result.document_type in [DocumentType.ACADEMIC, DocumentType.NON_LEGAL]
        assert result.confidence < 0.5


class TestArbitrationDetectorValidation:
    """Test the enhanced arbitration detector with validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ArbitrationDetector()
    
    def test_math_quiz_no_arbitration_detection(self):
        """Test that math quizzes don't trigger arbitration detection."""
        math_quiz_with_arbitration_word = """
        Advanced Mathematics Quiz
        
        Question 1: In game theory, what is the process called when two parties
        reach a decision through negotiation and arbitration?
        a) Nash equilibrium
        b) Pareto optimization  
        c) Binding arbitration
        d) Strategic dominance
        
        Question 2: Calculate the probability of choosing arbitration over litigation
        if the probability of winning in court is 0.6 and the cost of arbitration is $5000.
        
        Question 3: Solve the equation: arbitration_cost = litigation_cost × 0.7
        """
        
        result = self.detector.detect(math_quiz_with_arbitration_word)
        
        # Should not detect arbitration despite containing the word
        assert not result.has_arbitration
        assert result.confidence < 0.3
        assert result.validation_result is not None
        assert not result.validation_result.is_legal_document
        assert 'rejection_reason' in result.summary
    
    def test_recipe_no_arbitration_detection(self):
        """Test that recipes don't trigger arbitration detection."""
        recipe_with_legal_words = """
        Legal Sea Foods Restaurant Recipe
        
        Binding Clam Chowder
        
        Ingredients:
        - 2 cups fresh clams (binding is important for texture)
        - 1 agreement of cream (heavy cream)
        - Salt and pepper to enforce flavor
        
        Instructions:
        1. The parties (clams and cream) shall be combined
        2. Arbitrate the seasoning until binding
        3. Serve according to the terms of good taste
        4. Any disputes about flavor shall be resolved by the chef
        
        This recipe is governed by the laws of good cooking!
        """
        
        result = self.detector.detect(recipe_with_legal_words)
        
        # Should not detect arbitration despite legal-sounding words
        assert not result.has_arbitration
        assert result.confidence < 0.3
        assert result.validation_result is not None
        assert not result.validation_result.is_legal_document
        assert result.validation_result.document_type == DocumentType.RECIPE
    
    def test_story_no_arbitration_detection(self):
        """Test that stories don't trigger arbitration detection."""
        story_with_legal_theme = """
        The Lawyer's Tale
        Chapter 3: The Arbitration
        
        Sarah, a skilled attorney, walked into the arbitration hearing with confidence.
        The case involved a binding arbitration clause that her client had agreed to
        in their service contract. 
        
        "Your honor," she began, "my client disputes the terms of this agreement."
        
        The arbitrator listened carefully as both parties presented their cases.
        The opposing counsel argued that the binding arbitration was mandatory
        and that all disputes must be resolved through this process.
        
        As the story unfolds, Sarah discovers a loophole in the arbitration provision
        that could save her client from an unfavorable ruling.
        
        The plot thickens when the character realizes that the arbitration clause
        might not be enforceable under state law.
        
        This tale of legal drama continues in the next chapter...
        """
        
        result = self.detector.detect(story_with_legal_theme)
        
        # Should not detect arbitration despite legal plot
        assert not result.has_arbitration
        assert result.confidence < 0.4
        assert result.validation_result is not None
        assert not result.validation_result.is_legal_document
        assert result.validation_result.document_type == DocumentType.STORY
    
    def test_legitimate_terms_of_service_detection(self):
        """Test that legitimate ToS with arbitration is correctly detected."""
        legitimate_tos = """
        TERMS OF SERVICE AGREEMENT
        
        Last Updated: December 15, 2023
        
        1. ACCEPTANCE OF TERMS
        By using our service, you agree to be bound by these Terms of Service.
        
        2. USER CONDUCT
        Users shall not engage in any prohibited activities as defined herein.
        You agree to comply with all applicable laws and regulations.
        
        3. BINDING ARBITRATION CLAUSE
        Any dispute, claim or controversy arising out of or relating to these Terms
        or the breach, termination, enforcement, interpretation or validity thereof,
        including the determination of the scope or applicability of this agreement
        to arbitrate, shall be determined by arbitration before a single arbitrator.
        
        The arbitration shall be administered by the American Arbitration Association
        in accordance with its Commercial Arbitration Rules. You agree that binding
        arbitration is the exclusive means of resolving disputes and waive your right
        to a jury trial or to participate in a class action lawsuit.
        
        4. GOVERNING LAW
        These Terms shall be governed by and construed in accordance with the laws
        of the State of Delaware, without regard to its conflict of law provisions.
        
        5. SEVERABILITY
        If any provision of these Terms is held to be invalid or unenforceable,
        the remaining provisions shall remain in full force and effect.
        """
        
        result = self.detector.detect(legitimate_tos)
        
        # Should correctly detect arbitration in legitimate legal document
        assert result.has_arbitration
        assert result.confidence > 0.75
        assert result.validation_result is not None
        assert result.validation_result.is_legal_document
        assert result.validation_result.document_type == DocumentType.TERMS_OF_SERVICE
        assert len(result.clauses) > 0
    
    def test_privacy_policy_no_arbitration(self):
        """Test privacy policy without arbitration clause."""
        privacy_policy = """
        PRIVACY POLICY
        
        Effective Date: January 1, 2024
        
        1. INFORMATION WE COLLECT
        We collect information you provide directly to us, such as when you
        create an account, make a purchase, or contact us for support.
        
        2. HOW WE USE YOUR INFORMATION
        We use the information we collect to provide, maintain, and improve
        our services, process transactions, and communicate with you.
        
        3. INFORMATION SHARING
        We do not sell, trade, or otherwise transfer your personal information
        to third parties without your consent, except as described in this policy.
        
        4. DATA SECURITY
        We implement appropriate security measures to protect your personal
        information against unauthorized access, alteration, disclosure, or destruction.
        
        5. YOUR RIGHTS
        You have the right to access, update, or delete your personal information.
        Contact us if you wish to exercise these rights.
        
        6. CONTACT US
        If you have questions about this Privacy Policy, please contact us at
        privacy@example.com or by mail at our business address.
        """
        
        result = self.detector.detect(privacy_policy)
        
        # Should be identified as legal but without arbitration
        assert not result.has_arbitration
        assert result.validation_result is not None
        assert result.validation_result.is_legal_document
        assert result.validation_result.document_type == DocumentType.PRIVACY_POLICY
    
    def test_mixed_content_validation(self):
        """Test document with mixed content that could be ambiguous."""
        mixed_content = """
        How to Resolve Disputes: A Guide
        
        This guide explains different methods of dispute resolution.
        
        1. Negotiation
        The first step in resolving any dispute is direct negotiation between parties.
        
        2. Mediation
        If negotiation fails, mediation with a neutral third party can help.
        
        3. Arbitration
        Binding arbitration is when parties agree to submit their dispute
        to an arbitrator whose decision is final and binding.
        
        Question: What is the difference between mediation and arbitration?
        Answer: Mediation is non-binding while arbitration can be binding.
        
        Quiz Question: Which method is typically faster?
        a) Court litigation
        b) Binding arbitration
        c) Mediation
        d) All of the same
        
        This educational material is for informational purposes only
        and does not constitute legal advice.
        """
        
        result = self.detector.detect(mixed_content)
        
        # Should not detect arbitration due to educational/quiz context
        assert not result.has_arbitration
        assert result.confidence < 0.5
        # Document type could be academic or non-legal due to quiz elements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])