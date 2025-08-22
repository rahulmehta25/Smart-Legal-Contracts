#!/usr/bin/env python3
"""
Example script demonstrating multi-language arbitration clause detection.

This example shows how to use the multilingual NLP system to detect
arbitration clauses across different languages and legal systems.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Import our multilingual components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.nlp.multilingual import MultilingualProcessor
from app.nlp.legal_translator import LegalTranslator, LegalDomain
from app.nlp.language_models import LanguageModelManager, ModelType


async def demonstrate_multilingual_detection():
    """Demonstrate comprehensive multilingual arbitration detection."""
    
    print("=" * 80)
    print("MULTILINGUAL ARBITRATION CLAUSE DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    print("\n1. Initializing multilingual components...")
    config_path = str(Path(__file__).parent.parent / "config" / "languages.yaml")
    
    multilingual_processor = MultilingualProcessor(config_path)
    legal_translator = LegalTranslator(config_path)
    language_model_manager = LanguageModelManager(config_path)
    
    print("✓ Components initialized successfully")
    
    # Sample texts in different languages
    test_documents = {
        'english': """
        Any dispute arising out of or relating to this Agreement shall be settled 
        by binding arbitration administered by the American Arbitration Association 
        under its Commercial Arbitration Rules. You waive your right to a jury trial 
        and class action proceedings.
        """,
        
        'spanish': """
        Cualquier disputa que surja de o se relacione con este Acuerdo será resuelta 
        mediante arbitraje vinculante administrado por la Asociación Americana de 
        Arbitraje bajo sus Reglas de Arbitraje Comercial. Usted renuncia a su derecho 
        a un juicio por jurado y procedimientos de acción colectiva.
        """,
        
        'french': """
        Tout différend découlant de ou se rapportant à cet Accord sera réglé par 
        arbitrage contraignant administré par l'Association Américaine d'Arbitrage 
        sous ses Règles d'Arbitrage Commercial. Vous renoncez à votre droit à un 
        procès devant jury et aux procédures d'action collective.
        """,
        
        'german': """
        Jede Streitigkeit, die aus oder in Bezug auf diese Vereinbarung entsteht, 
        wird durch bindendes Schiedsverfahren beigelegt, das von der American 
        Arbitration Association unter ihren Commercial Arbitration Rules verwaltet 
        wird. Sie verzichten auf Ihr Recht auf ein Geschworenenprozess und 
        Sammelklageverfahren.
        """,
        
        'chinese': """
        因本协议或与本协议相关而产生的任何争议应通过美国仲裁协会根据其商业仲裁规则
        管理的约束性仲裁来解决。您放弃陪审团审判和集体诉讼程序的权利。
        """,
        
        'japanese': """
        本契約から生じるまたは本契約に関連する紛争は、米国仲裁協会の商事仲裁規則の下で
        管理される拘束力のある仲裁により解決されるものとします。陪審裁判および集団訴訟
        手続きへの権利を放棄します。
        """
    }
    
    # Step 2: Language Detection
    print("\n2. Performing language detection...")
    detection_results = {}
    
    for lang_name, text in test_documents.items():
        result = multilingual_processor.language_detector.detect_language(text.strip())
        detection_results[lang_name] = result
        print(f"   {lang_name.title()}: {result.language} (confidence: {result.confidence:.3f}) "
              f"[{result.detected_by}]")
    
    # Step 3: Document Processing and Translation
    print("\n3. Processing documents with translation...")
    processing_results = {}
    
    for lang_name, text in test_documents.items():
        print(f"   Processing {lang_name} document...")
        result = await multilingual_processor.process_document(text.strip(), target_language="en")
        processing_results[lang_name] = result
        
        if result['translation_needed']:
            print(f"     ✓ Translated from {result['source_language']} to {result['target_language']}")
        else:
            print(f"     ✓ No translation needed (already in {result['target_language']})")
    
    # Step 4: Legal Translation with Term Preservation
    print("\n4. Legal translation with term preservation...")
    legal_translations = {}
    
    for lang_name, text in test_documents.items():
        if lang_name != 'english':  # Skip English as it's our target
            print(f"   Translating {lang_name} with legal term preservation...")
            
            # Detect source language
            detection = multilingual_processor.language_detector.detect_language(text.strip())
            
            # Perform legal translation
            translation_result = await legal_translator.translate(
                text.strip(),
                detection.language,
                target_language="en",
                domain=LegalDomain.ARBITRATION,
                jurisdiction_aware=True
            )
            
            legal_translations[lang_name] = translation_result
            print(f"     ✓ Translation confidence: {translation_result.confidence_score:.3f}")
            if translation_result.legal_terms_preserved:
                print(f"     ✓ Preserved {len(translation_result.legal_terms_preserved)} legal terms")
    
    # Step 5: Cross-lingual Similarity Analysis
    print("\n5. Cross-lingual similarity analysis...")
    
    # Use English as reference
    english_text = test_documents['english'].strip()
    
    similarity_results = {}
    for lang_name, text in test_documents.items():
        if lang_name != 'english':
            print(f"   Comparing {lang_name} with English...")
            
            # Get similarity result
            similarity = await language_model_manager.similarity_engine.compute_similarity(
                english_text,
                text.strip(),
                language1="en",
                language2=detection_results[lang_name].language,
                method="cosine",
                model_type=ModelType.SENTENCE_TRANSFORMER
            )
            
            similarity_results[lang_name] = similarity
            print(f"     ✓ Semantic similarity: {similarity.similarity_score:.3f}")
            print(f"     ✓ Cross-lingual: {similarity.cross_lingual}")
    
    # Step 6: Arbitration Pattern Detection
    print("\n6. Arbitration pattern detection across languages...")
    
    pattern_results = {}
    for lang_name, text in test_documents.items():
        print(f"   Detecting arbitration patterns in {lang_name}...")
        
        detection_lang = detection_results[lang_name].language
        pattern_result = await language_model_manager.pattern_detector.detect_patterns(
            text.strip(),
            language=detection_lang
        )
        
        pattern_results[lang_name] = pattern_result
        print(f"     ✓ Arbitration probability: {pattern_result.arbitration_probability:.3f}")
        print(f"     ✓ Patterns detected: {', '.join(pattern_result.patterns_detected)}")
        if pattern_result.key_phrases:
            print(f"     ✓ Key phrases: {pattern_result.key_phrases[:2]}")  # Show first 2
    
    # Step 7: Comprehensive Analysis
    print("\n7. Comprehensive document analysis...")
    
    comprehensive_results = {}
    for lang_name, text in test_documents.items():
        print(f"   Analyzing {lang_name} document comprehensively...")
        
        analysis = await language_model_manager.analyze_document(
            text.strip(),
            language=detection_results[lang_name].language,
            include_embeddings=True,
            include_similarity=False
        )
        
        comprehensive_results[lang_name] = analysis
        
        # Print key findings
        patterns = analysis['pattern_detection']
        print(f"     ✓ Arbitration probability: {patterns['arbitration_probability']:.3f}")
        print(f"     ✓ Legal domain: {patterns['legal_domain']}")
        print(f"     ✓ Confidence scores: {list(patterns['confidence_scores'].values())}")
    
    # Step 8: Performance Benchmarking
    print("\n8. Performance benchmarking...")
    
    test_texts = list(test_documents.values())
    test_languages = [detection_results[lang].language for lang in test_documents.keys()]
    
    benchmark_results = await language_model_manager.benchmark_models(
        test_texts, test_languages
    )
    
    print("   Model performance comparison:")
    for model_name, metrics in benchmark_results.items():
        print(f"     {model_name}:")
        print(f"       - Average time per text: {metrics['average_time_per_text']:.3f}s")
        print(f"       - Average confidence: {metrics['average_confidence']:.3f}")
    
    # Step 9: Summary Report
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\nDocuments processed: {len(test_documents)}")
    print(f"Languages detected: {len(set(detection_results[lang].language for lang in detection_results))}")
    print(f"Legal translations performed: {len(legal_translations)}")
    
    print("\nArbitration Detection Results:")
    for lang_name in test_documents.keys():
        prob = comprehensive_results[lang_name]['pattern_detection']['arbitration_probability']
        status = "✓ DETECTED" if prob >= 0.7 else "✗ NOT DETECTED"
        print(f"  {lang_name.title()}: {status} (probability: {prob:.3f})")
    
    print("\nCross-lingual Similarity (vs English):")
    for lang_name, similarity in similarity_results.items():
        print(f"  {lang_name.title()}: {similarity.similarity_score:.3f}")
    
    # Performance statistics
    stats = language_model_manager.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Embeddings generated: {stats['embeddings_generated']}")
    print(f"  Similarities computed: {stats['similarities_computed']}")
    print(f"  Patterns detected: {stats['patterns_detected']}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return {
        'detection_results': detection_results,
        'processing_results': processing_results,
        'legal_translations': legal_translations,
        'similarity_results': similarity_results,
        'pattern_results': pattern_results,
        'comprehensive_results': comprehensive_results,
        'benchmark_results': benchmark_results
    }


async def test_specific_scenarios():
    """Test specific edge cases and scenarios."""
    
    print("\n" + "=" * 80)
    print("TESTING SPECIFIC SCENARIOS")
    print("=" * 80)
    
    # Initialize components
    config_path = str(Path(__file__).parent.parent / "config" / "languages.yaml")
    processor = MultilingualProcessor(config_path)
    
    scenarios = {
        'hidden_arbitration': """
        This comprehensive agreement contains many provisions. Section 15.3 states that 
        all disputes shall be resolved through binding arbitration as the exclusive remedy. 
        Section 23.7 contains waiver provisions regarding jury trials and class actions.
        """,
        
        'conditional_arbitration': """
        Disputes related to billing may be resolved in small claims court. However, 
        all other disputes arising under this agreement must be submitted to binding 
        arbitration administered by the AAA.
        """,
        
        'no_arbitration': """
        This privacy policy explains how we collect and use your personal information. 
        We are committed to protecting your privacy and complying with applicable 
        data protection laws and regulations.
        """,
        
        'mixed_language': """
        This agreement shall be governed by US law. Any disputes will be resolved through 
        arbitraje vinculante administered by AAA. Les parties renoncent au procès devant jury.
        """
    }
    
    print("\nTesting scenario detection:")
    for scenario_name, text in scenarios.items():
        print(f"\n{scenario_name.upper().replace('_', ' ')}:")
        
        # Process the document
        result = await processor.process_document(text, target_language="en")
        
        # Extract key information
        keywords = result.get('legal_keywords', {})
        confidence = result.get('confidence_score', 0.0)
        
        print(f"  Language: {result['source_language']}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Legal keywords found: {sum(len(terms) for terms in keywords.values())}")
        
        # Show detected keywords by category
        for category, terms in keywords.items():
            if terms:
                print(f"    {category}: {', '.join(terms[:3])}{'...' if len(terms) > 3 else ''}")
    
    print("\n" + "=" * 40)
    print("SCENARIO TESTING COMPLETED")
    print("=" * 40)


async def main():
    """Main demonstration function."""
    start_time = time.time()
    
    try:
        # Run main demonstration
        results = await demonstrate_multilingual_detection()
        
        # Run specific scenario tests
        await test_specific_scenarios()
        
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        
        # Save results to file
        output_file = Path(__file__).parent / "multilingual_demo_results.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if key == 'detection_results':
                json_results[key] = {
                    lang: {
                        'language': res.language,
                        'confidence': res.confidence,
                        'detected_by': res.detected_by,
                        'is_supported': res.is_supported
                    }
                    for lang, res in value.items()
                }
            elif key == 'similarity_results':
                json_results[key] = {
                    lang: {
                        'similarity_score': res.similarity_score,
                        'cross_lingual': res.cross_lingual,
                        'confidence': res.confidence
                    }
                    for lang, res in value.items()
                }
            elif key == 'pattern_results':
                json_results[key] = {
                    lang: {
                        'patterns_detected': res.patterns_detected,
                        'arbitration_probability': res.arbitration_probability,
                        'key_phrases': res.key_phrases
                    }
                    for lang, res in value.items()
                }
            else:
                json_results[key] = value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ensure we have an event loop for async operations
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"Failed to run demonstration: {e}")
        import traceback
        traceback.print_exc()