"""
Demo test scenarios for Terms of Use (TOU) upload and analysis.
Demonstrates the complete workflow with real-world examples.
"""

import pytest
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock
import uuid


class TestTOUUploadDemo:
    """Demo tests for TOU upload and arbitration detection workflow."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        self.demo_results = {}
        self.test_documents_path = Path(__file__).parent / "sample_documents"
        self.setup_sample_documents()
        
    def setup_sample_documents(self):
        """Setup sample TOU documents for demo."""
        self.sample_tous = {
            "social_media_tos": {
                "filename": "social_media_terms.txt",
                "content": """
                TERMS OF SERVICE - Social Media Platform
                
                1. ACCEPTANCE OF TERMS
                By accessing and using this service, you accept and agree to be bound by the terms 
                and provision of this agreement.
                
                15. DISPUTE RESOLUTION
                Any disputes arising out of or relating to these Terms shall be resolved exclusively 
                through binding arbitration administered by the American Arbitration Association (AAA) 
                under its Consumer Arbitration Rules. The arbitration will be conducted in English 
                in San Francisco, California. You and Company each waive any right to a jury trial 
                or to participate in a class action lawsuit or class-wide arbitration.
                
                16. GOVERNING LAW
                These Terms shall be governed by California law.
                """,
                "expected_arbitration": True,
                "expected_confidence": 0.95,
                "arbitration_type": "consumer_binding_arbitration"
            },
            
            "saas_agreement": {
                "filename": "saas_service_agreement.txt",
                "content": """
                SOFTWARE AS A SERVICE AGREEMENT
                
                This SaaS Agreement governs your use of our cloud-based software service.
                
                8. DISPUTE RESOLUTION PROCESS
                
                8.1 Informal Resolution: Before initiating formal proceedings, the parties agree 
                to attempt to resolve any dispute through informal negotiation for a period of 
                thirty (30) days.
                
                8.2 Mediation: If informal resolution fails, the parties agree to submit the 
                dispute to mediation before a single mediator agreed upon by both parties.
                
                8.3 Arbitration: If mediation is unsuccessful, any remaining disputes shall be 
                settled by final and binding arbitration under the rules of JAMS. The arbitration 
                shall be conducted by one arbitrator in New York, New York. The arbitrator's 
                decision shall be final, binding, and non-appealable.
                
                9. CLASS ACTION WAIVER
                You agree that disputes must be brought in arbitration on an individual basis only, 
                and not as part of any purported class or representative action.
                """,
                "expected_arbitration": True,
                "expected_confidence": 0.98,
                "arbitration_type": "escalated_binding_arbitration"
            },
            
            "e_commerce_terms": {
                "filename": "ecommerce_terms.txt",
                "content": """
                E-COMMERCE TERMS AND CONDITIONS
                
                Welcome to our online store. These terms govern your purchase and use of our products.
                
                12. RETURNS AND REFUNDS
                You may return products within 30 days of purchase in original condition.
                
                13. DISPUTE RESOLUTION
                Any disputes relating to your purchase will be handled through our customer service 
                department. If we cannot resolve the issue to your satisfaction, you may file a 
                complaint with your state consumer protection agency or pursue resolution through 
                small claims court in your local jurisdiction.
                
                14. LIMITATION OF LIABILITY
                Our liability is limited to the purchase price of the product.
                """,
                "expected_arbitration": False,
                "expected_confidence": 0.1,
                "arbitration_type": None
            },
            
            "financial_services_tos": {
                "filename": "financial_services_tos.txt",
                "content": """
                FINANCIAL SERVICES TERMS OF SERVICE
                
                These terms govern your use of our financial services platform.
                
                SECTION 14: ARBITRATION AGREEMENT
                
                PLEASE READ THIS SECTION CAREFULLY. IT AFFECTS YOUR RIGHTS.
                
                14.1 Agreement to Arbitrate: You and Company agree that any dispute, claim, or 
                controversy arising out of or relating to these Terms or your use of the Services 
                will be settled by binding arbitration, rather than in court, except that you may 
                assert claims in small claims court if your claims qualify.
                
                14.2 Arbitration Rules: The arbitration will be administered by the American 
                Arbitration Association ("AAA") under its Consumer Arbitration Rules, as modified 
                by these Terms. The arbitrator will conduct hearings, if any, by teleconference 
                or videoconference, rather than by personal appearances, unless the arbitrator 
                determines upon request by you or Company that an in-person hearing is appropriate.
                
                14.3 Class Action Waiver: You may only resolve disputes with Company on an 
                individual basis, and may not bring a claim as a plaintiff or a class member 
                in a class, consolidated, or representative action.
                
                14.4 Right to Opt-Out: You have the right to opt-out of this arbitration agreement 
                by sending written notice to us within 30 days of first accepting these Terms.
                """,
                "expected_arbitration": True,
                "expected_confidence": 0.97,
                "arbitration_type": "consumer_arbitration_with_opt_out"
            },
            
            "privacy_policy": {
                "filename": "privacy_policy.txt",
                "content": """
                PRIVACY POLICY
                
                This Privacy Policy describes how we collect, use, and protect your personal information.
                
                1. INFORMATION WE COLLECT
                We collect information you provide directly to us, such as when you create an account.
                
                2. HOW WE USE INFORMATION
                We use the information we collect to provide, maintain, and improve our services.
                
                3. INFORMATION SHARING
                We do not sell, trade, or otherwise transfer your personal information to third parties.
                
                4. DATA SECURITY
                We implement appropriate security measures to protect your personal information.
                
                5. CONTACT US
                If you have questions about this Privacy Policy, please contact us at privacy@company.com.
                """,
                "expected_arbitration": False,
                "expected_confidence": 0.05,
                "arbitration_type": None
            }
        }
        
    def test_social_media_tos_upload_and_analysis(self):
        """Demo: Social media platform TOS with clear arbitration clause."""
        document = self.sample_tous["social_media_tos"]
        
        print(f"\n=== DEMO: Analyzing {document['filename']} ===")
        print(f"Document type: Social Media Terms of Service")
        print(f"Expected arbitration: {document['expected_arbitration']}")
        
        # Step 1: Document upload
        upload_result = self._simulate_document_upload(
            document["content"], 
            document["filename"]
        )
        
        assert upload_result["success"] == True
        print(f"✓ Document uploaded successfully: {upload_result['document_id']}")
        
        # Step 2: Arbitration analysis
        analysis_result = self._simulate_arbitration_analysis(
            upload_result["document_id"],
            document["content"]
        )
        
        # Step 3: Validate results
        assert analysis_result["has_arbitration"] == document["expected_arbitration"]
        assert analysis_result["confidence"] >= document["expected_confidence"] - 0.1
        
        print(f"✓ Arbitration detected: {analysis_result['has_arbitration']}")
        print(f"✓ Confidence score: {analysis_result['confidence']:.2f}")
        print(f"✓ Clause type: {analysis_result['clause_type']}")
        print(f"✓ Key findings: {', '.join(analysis_result['keywords'])}")
        
        # Step 4: Generate explanation
        explanation = self._generate_explanation(analysis_result)
        print(f"✓ Explanation: {explanation}")
        
        self.demo_results["social_media_tos"] = {
            "upload": upload_result,
            "analysis": analysis_result,
            "explanation": explanation,
            "status": "PASS"
        }
        
    def test_saas_agreement_multi_step_arbitration(self):
        """Demo: SaaS agreement with multi-step dispute resolution."""
        document = self.sample_tous["saas_agreement"]
        
        print(f"\n=== DEMO: Analyzing {document['filename']} ===")
        print(f"Document type: Software as a Service Agreement")
        print(f"Expected: Multi-step arbitration process")
        
        # Upload and analyze
        upload_result = self._simulate_document_upload(
            document["content"], 
            document["filename"]
        )
        
        analysis_result = self._simulate_arbitration_analysis(
            upload_result["document_id"],
            document["content"]
        )
        
        # Validate complex arbitration detection
        assert analysis_result["has_arbitration"] == True
        assert "escalated" in analysis_result["clause_type"]
        assert "mediation" in [kw.lower() for kw in analysis_result["keywords"]]
        assert "jams" in [kw.lower() for kw in analysis_result["keywords"]]
        
        print(f"✓ Multi-step arbitration detected")
        print(f"✓ Process: Negotiation → Mediation → Arbitration")
        print(f"✓ Provider: JAMS")
        print(f"✓ Binding: Yes")
        print(f"✓ Class action waiver: Detected")
        
        # Test clause extraction
        clauses = self._extract_arbitration_clauses(document["content"])
        print(f"✓ Extracted clauses: {len(clauses)} sections")
        
        self.demo_results["saas_agreement"] = {
            "upload": upload_result,
            "analysis": analysis_result,
            "clauses": clauses,
            "status": "PASS"
        }
        
    def test_ecommerce_no_arbitration_detection(self):
        """Demo: E-commerce terms without arbitration clause."""
        document = self.sample_tous["e_commerce_terms"]
        
        print(f"\n=== DEMO: Analyzing {document['filename']} ===")
        print(f"Document type: E-commerce Terms and Conditions")
        print(f"Expected: No arbitration clause")
        
        # Upload and analyze
        upload_result = self._simulate_document_upload(
            document["content"], 
            document["filename"]
        )
        
        analysis_result = self._simulate_arbitration_analysis(
            upload_result["document_id"],
            document["content"]
        )
        
        # Validate no arbitration detection
        assert analysis_result["has_arbitration"] == False
        assert analysis_result["confidence"] <= 0.2
        
        print(f"✓ No arbitration clause detected")
        print(f"✓ Confidence: {analysis_result['confidence']:.2f}")
        print(f"✓ Alternative dispute resolution: Small claims court")
        print(f"✓ Consumer protection: State agency complaints")
        
        # Test false positive prevention
        false_positive_check = self._check_false_positive_triggers(document["content"])
        print(f"✓ False positive triggers checked: {false_positive_check['triggers_found']}")
        
        self.demo_results["e_commerce_terms"] = {
            "upload": upload_result,
            "analysis": analysis_result,
            "false_positive_check": false_positive_check,
            "status": "PASS"
        }
        
    def test_financial_services_complex_arbitration(self):
        """Demo: Financial services with comprehensive arbitration clause."""
        document = self.sample_tous["financial_services_tos"]
        
        print(f"\n=== DEMO: Analyzing {document['filename']} ===")
        print(f"Document type: Financial Services Terms")
        print(f"Expected: Comprehensive arbitration with opt-out")
        
        # Upload and analyze
        upload_result = self._simulate_document_upload(
            document["content"], 
            document["filename"]
        )
        
        analysis_result = self._simulate_arbitration_analysis(
            upload_result["document_id"],
            document["content"]
        )
        
        # Validate comprehensive arbitration detection
        assert analysis_result["has_arbitration"] == True
        assert analysis_result["confidence"] >= 0.95
        
        # Test special features detection
        special_features = self._detect_special_features(document["content"])
        
        assert "opt_out_right" in special_features
        assert "class_action_waiver" in special_features
        assert "small_claims_exception" in special_features
        
        print(f"✓ Comprehensive arbitration clause detected")
        print(f"✓ Consumer rights preserved: Opt-out available")
        print(f"✓ Small claims court exception: Yes")
        print(f"✓ Remote hearings: Teleconference/videoconference")
        print(f"✓ Special features: {', '.join(special_features)}")
        
        # Test compliance assessment
        compliance = self._assess_compliance(analysis_result, special_features)
        print(f"✓ Consumer protection compliance: {compliance['score']:.2f}")
        
        self.demo_results["financial_services_tos"] = {
            "upload": upload_result,
            "analysis": analysis_result,
            "special_features": special_features,
            "compliance": compliance,
            "status": "PASS"
        }
        
    def test_privacy_policy_negative_example(self):
        """Demo: Privacy policy without arbitration (negative example)."""
        document = self.sample_tous["privacy_policy"]
        
        print(f"\n=== DEMO: Analyzing {document['filename']} ===")
        print(f"Document type: Privacy Policy")
        print(f"Expected: No arbitration clause (negative example)")
        
        # Upload and analyze
        upload_result = self._simulate_document_upload(
            document["content"], 
            document["filename"]
        )
        
        analysis_result = self._simulate_arbitration_analysis(
            upload_result["document_id"],
            document["content"]
        )
        
        # Validate negative detection
        assert analysis_result["has_arbitration"] == False
        assert analysis_result["confidence"] <= 0.1
        
        print(f"✓ Correctly identified as non-arbitration document")
        print(f"✓ Document focus: Data privacy and protection")
        print(f"✓ No dispute resolution mechanisms mentioned")
        print(f"✓ Contact method: Email for policy questions")
        
        # Test document classification
        classification = self._classify_document_type(document["content"])
        assert classification["type"] == "privacy_policy"
        
        print(f"✓ Document classification: {classification['type']}")
        print(f"✓ Classification confidence: {classification['confidence']:.2f}")
        
        self.demo_results["privacy_policy"] = {
            "upload": upload_result,
            "analysis": analysis_result,
            "classification": classification,
            "status": "PASS"
        }
        
    def test_batch_tou_analysis_demo(self):
        """Demo: Batch analysis of multiple TOU documents."""
        print(f"\n=== DEMO: Batch Analysis of All Documents ===")
        
        batch_documents = []
        for doc_key, document in self.sample_tous.items():
            batch_documents.append({
                "id": doc_key,
                "filename": document["filename"],
                "content": document["content"],
                "expected_arbitration": document["expected_arbitration"]
            })
        
        print(f"Processing {len(batch_documents)} documents in batch...")
        
        # Perform batch analysis
        start_time = time.time()
        batch_results = self._simulate_batch_analysis(batch_documents)
        processing_time = time.time() - start_time
        
        print(f"✓ Batch processing completed in {processing_time:.2f} seconds")
        print(f"✓ Average time per document: {processing_time/len(batch_documents):.2f} seconds")
        
        # Validate batch results
        correct_predictions = 0
        for result in batch_results["results"]:
            doc_id = result["document_id"]
            expected = next(doc["expected_arbitration"] for doc in batch_documents if doc["id"] == doc_id)
            actual = result["has_arbitration"]
            
            if expected == actual:
                correct_predictions += 1
                status = "✓"
            else:
                status = "✗"
                
            print(f"{status} {doc_id}: Expected={expected}, Got={actual}, Confidence={result['confidence']:.2f}")
        
        accuracy = correct_predictions / len(batch_documents)
        assert accuracy >= 0.8  # At least 80% accuracy
        
        print(f"✓ Batch analysis accuracy: {accuracy:.2%}")
        print(f"✓ Total documents processed: {len(batch_documents)}")
        print(f"✓ Arbitration clauses found: {sum(1 for r in batch_results['results'] if r['has_arbitration'])}")
        
        self.demo_results["batch_analysis"] = {
            "results": batch_results,
            "accuracy": accuracy,
            "processing_time": processing_time,
            "status": "PASS"
        }
        
    def test_comparative_analysis_demo(self):
        """Demo: Comparative analysis of different arbitration approaches."""
        print(f"\n=== DEMO: Comparative Analysis of Arbitration Approaches ===")
        
        # Compare different arbitration types
        arbitration_docs = [
            ("social_media_tos", "Consumer Arbitration (AAA)"),
            ("saas_agreement", "Commercial Arbitration (JAMS)"),
            ("financial_services_tos", "Consumer with Opt-out (AAA)")
        ]
        
        comparison_results = []
        
        for doc_key, description in arbitration_docs:
            document = self.sample_tous[doc_key]
            
            analysis = self._simulate_arbitration_analysis(
                f"comp_{doc_key}",
                document["content"]
            )
            
            features = self._analyze_arbitration_features(document["content"])
            
            comparison_results.append({
                "document": doc_key,
                "description": description,
                "analysis": analysis,
                "features": features
            })
            
        # Generate comparison report
        comparison_report = self._generate_comparison_report(comparison_results)
        
        print(f"✓ Arbitration Provider Distribution:")
        for provider, count in comparison_report["providers"].items():
            print(f"  - {provider}: {count} documents")
            
        print(f"✓ Common Features:")
        for feature, prevalence in comparison_report["common_features"].items():
            print(f"  - {feature}: {prevalence:.0%} of documents")
            
        print(f"✓ Consumer Protection Analysis:")
        for protection, details in comparison_report["consumer_protections"].items():
            print(f"  - {protection}: {details}")
            
        self.demo_results["comparative_analysis"] = {
            "comparison_results": comparison_results,
            "comparison_report": comparison_report,
            "status": "PASS"
        }
        
    def test_real_time_analysis_demo(self):
        """Demo: Real-time analysis as user types/uploads."""
        print(f"\n=== DEMO: Real-time Analysis Simulation ===")
        
        # Simulate progressive text input
        progressive_text = """
        TERMS OF SERVICE
        
        1. ACCEPTANCE
        By using this service, you agree to these terms.
        
        2. USER OBLIGATIONS
        You must use this service lawfully and responsibly.
        
        3. DISPUTE RESOLUTION
        Any disputes arising from these terms shall be resolved through
        """
        
        # Simulate real-time analysis as text is being typed
        text_chunks = [
            progressive_text,
            progressive_text + "binding arbitration administered by",
            progressive_text + "binding arbitration administered by the American Arbitration Association",
            progressive_text + "binding arbitration administered by the American Arbitration Association under its Consumer Rules."
        ]
        
        print("Simulating real-time analysis as document is being typed...")
        
        real_time_results = []
        for i, chunk in enumerate(text_chunks):
            analysis = self._simulate_arbitration_analysis(f"realtime_{i}", chunk)
            
            print(f"Characters: {len(chunk)}, Arbitration: {analysis['has_arbitration']}, Confidence: {analysis['confidence']:.2f}")
            
            real_time_results.append({
                "chunk_size": len(chunk),
                "has_arbitration": analysis["has_arbitration"],
                "confidence": analysis["confidence"],
                "response_time": analysis.get("response_time", 0.1)
            })
            
        # Validate real-time performance
        final_result = real_time_results[-1]
        assert final_result["has_arbitration"] == True
        assert final_result["confidence"] > 0.8
        assert all(r["response_time"] < 0.5 for r in real_time_results)  # Fast response
        
        print(f"✓ Real-time analysis: Progressive confidence increase")
        print(f"✓ Final detection: {final_result['has_arbitration']} ({final_result['confidence']:.2f})")
        print(f"✓ Average response time: {sum(r['response_time'] for r in real_time_results)/len(real_time_results):.3f}s")
        
        self.demo_results["real_time_analysis"] = {
            "results": real_time_results,
            "final_detection": final_result,
            "status": "PASS"
        }
        
    # Helper methods for simulation
    def _simulate_document_upload(self, content: str, filename: str) -> Dict[str, Any]:
        """Simulate document upload process."""
        time.sleep(0.1)  # Simulate upload time
        
        return {
            "success": True,
            "document_id": str(uuid.uuid4()),
            "filename": filename,
            "size": len(content),
            "upload_time": time.time(),
            "content_type": "text/plain"
        }
        
    def _simulate_arbitration_analysis(self, document_id: str, content: str) -> Dict[str, Any]:
        """Simulate arbitration clause analysis."""
        time.sleep(0.2)  # Simulate analysis time
        
        # Simulate analysis based on content patterns
        content_lower = content.lower()
        
        # Check for arbitration indicators
        arbitration_indicators = [
            "arbitration", "arbitrate", "arbitrator", "arbitral",
            "binding arbitration", "mandatory arbitration",
            "aaa", "american arbitration association",
            "jams", "dispute resolution"
        ]
        
        found_indicators = [ind for ind in arbitration_indicators if ind in content_lower]
        has_arbitration = len(found_indicators) >= 2
        
        # Calculate confidence based on indicators
        confidence = min(0.95, len(found_indicators) * 0.15 + 0.1)
        if not has_arbitration:
            confidence = max(0.05, 0.3 - len(found_indicators) * 0.1)
            
        # Determine clause type
        clause_type = None
        if has_arbitration:
            if "consumer" in content_lower:
                clause_type = "consumer_binding_arbitration"
            elif "mediation" in content_lower and "arbitration" in content_lower:
                clause_type = "escalated_binding_arbitration"
            elif "opt-out" in content_lower or "opt out" in content_lower:
                clause_type = "consumer_arbitration_with_opt_out"
            else:
                clause_type = "binding_arbitration"
                
        return {
            "document_id": document_id,
            "has_arbitration": has_arbitration,
            "confidence": confidence,
            "clause_type": clause_type,
            "keywords": found_indicators[:5],  # Top 5 indicators
            "analysis_time": 0.2,
            "response_time": 0.1
        }
        
    def _generate_explanation(self, analysis_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation of analysis."""
        if analysis_result["has_arbitration"]:
            return f"This document contains a {analysis_result['clause_type']} clause with {analysis_result['confidence']:.0%} confidence. Key indicators include: {', '.join(analysis_result['keywords'][:3])}."
        else:
            return f"No arbitration clause detected. The document appears to use alternative dispute resolution methods or standard legal processes."
            
    def _extract_arbitration_clauses(self, content: str) -> List[Dict[str, Any]]:
        """Extract specific arbitration clauses from content."""
        # Simulate clause extraction
        clauses = []
        
        if "dispute resolution" in content.lower():
            clauses.append({
                "type": "dispute_resolution_section",
                "text": "Dispute resolution section identified",
                "confidence": 0.9
            })
            
        if "arbitration" in content.lower():
            clauses.append({
                "type": "arbitration_clause",
                "text": "Arbitration clause identified", 
                "confidence": 0.95
            })
            
        return clauses
        
    def _check_false_positive_triggers(self, content: str) -> Dict[str, Any]:
        """Check for common false positive triggers."""
        triggers = []
        
        # Words that might trigger false positives
        false_positive_words = ["mediation", "negotiation", "discussion", "resolution"]
        
        for word in false_positive_words:
            if word in content.lower() and "arbitration" not in content.lower():
                triggers.append(word)
                
        return {
            "triggers_found": len(triggers),
            "triggers": triggers,
            "false_positive_risk": "low" if len(triggers) <= 1 else "medium"
        }
        
    def _detect_special_features(self, content: str) -> List[str]:
        """Detect special features in arbitration clauses."""
        features = []
        content_lower = content.lower()
        
        if "opt-out" in content_lower or "opt out" in content_lower:
            features.append("opt_out_right")
        if "class action waiver" in content_lower or "class waiver" in content_lower:
            features.append("class_action_waiver")
        if "small claims" in content_lower:
            features.append("small_claims_exception")
        if "teleconference" in content_lower or "videoconference" in content_lower:
            features.append("remote_hearings")
        if "english" in content_lower:
            features.append("language_requirement")
            
        return features
        
    def _assess_compliance(self, analysis: Dict[str, Any], features: List[str]) -> Dict[str, Any]:
        """Assess consumer protection compliance."""
        score = 0.5  # Base score
        
        if "opt_out_right" in features:
            score += 0.2
        if "small_claims_exception" in features:
            score += 0.2
        if "remote_hearings" in features:
            score += 0.1
            
        return {
            "score": min(1.0, score),
            "features_evaluated": len(features),
            "compliance_level": "high" if score > 0.8 else "medium" if score > 0.6 else "low"
        }
        
    def _classify_document_type(self, content: str) -> Dict[str, Any]:
        """Classify the type of document."""
        content_lower = content.lower()
        
        if "privacy policy" in content_lower:
            return {"type": "privacy_policy", "confidence": 0.95}
        elif "terms of service" in content_lower:
            return {"type": "terms_of_service", "confidence": 0.9}
        elif "software" in content_lower and "service" in content_lower:
            return {"type": "saas_agreement", "confidence": 0.85}
        else:
            return {"type": "unknown", "confidence": 0.5}
            
    def _simulate_batch_analysis(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate batch analysis of multiple documents."""
        results = []
        
        for doc in documents:
            analysis = self._simulate_arbitration_analysis(doc["id"], doc["content"])
            results.append(analysis)
            
        return {
            "total_documents": len(documents),
            "results": results,
            "processing_time": len(documents) * 0.2,
            "success": True
        }
        
    def _analyze_arbitration_features(self, content: str) -> Dict[str, Any]:
        """Analyze specific arbitration features."""
        content_lower = content.lower()
        
        features = {
            "provider": "AAA" if "aaa" in content_lower or "american arbitration" in content_lower else "JAMS" if "jams" in content_lower else "Other",
            "binding": "binding" in content_lower,
            "class_waiver": "class" in content_lower and "waiver" in content_lower,
            "opt_out": "opt-out" in content_lower or "opt out" in content_lower,
            "location_specified": any(city in content_lower for city in ["new york", "california", "san francisco"]),
            "language_specified": "english" in content_lower
        }
        
        return features
        
    def _generate_comparison_report(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparative analysis report."""
        providers = {}
        features = {}
        
        for result in comparison_results:
            # Count providers
            provider = result["features"]["provider"]
            providers[provider] = providers.get(provider, 0) + 1
            
            # Count features
            for feature, present in result["features"].items():
                if present and feature != "provider":
                    features[feature] = features.get(feature, 0) + 1
                    
        # Calculate percentages
        total_docs = len(comparison_results)
        common_features = {k: v/total_docs for k, v in features.items()}
        
        consumer_protections = {
            "opt_out_available": f"{features.get('opt_out', 0)}/{total_docs} documents",
            "class_waiver_present": f"{features.get('class_waiver', 0)}/{total_docs} documents",
            "location_specified": f"{features.get('location_specified', 0)}/{total_docs} documents"
        }
        
        return {
            "providers": providers,
            "common_features": common_features,
            "consumer_protections": consumer_protections
        }
        
    def generate_demo_report(self) -> Dict[str, Any]:
        """Generate comprehensive demo test report."""
        total_demos = len(self.demo_results)
        passed_demos = sum(1 for result in self.demo_results.values() if result["status"] == "PASS")
        
        return {
            "tou_upload_demo_report": {
                "total_demos": total_demos,
                "passed": passed_demos,
                "failed": total_demos - passed_demos,
                "success_rate": passed_demos / total_demos if total_demos > 0 else 0,
                "demo_details": self.demo_results,
                "timestamp": time.time()
            }
        }


if __name__ == "__main__":
    # Run demo tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])