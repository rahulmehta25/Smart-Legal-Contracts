-- Demo Database Setup for Arbitration Clause Detector
-- This file contains sample data for the interactive demo environment

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Demo Users
INSERT INTO users (id, email, password_hash, first_name, last_name, role, is_active, created_at, updated_at, demo_account) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'admin@demo.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LDFXCzxs1ZKbS.Q6C', 'Demo', 'Administrator', 'admin', true, NOW(), NOW(), true),
('550e8400-e29b-41d4-a716-446655440002', 'lawyer@demo.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LDFXCzxs1ZKbS.Q6C', 'Legal', 'Expert', 'legal_expert', true, NOW(), NOW(), true),
('550e8400-e29b-41d4-a716-446655440003', 'business@demo.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LDFXCzxs1ZKbS.Q6C', 'Business', 'User', 'business_user', true, NOW(), NOW(), true),
('550e8400-e29b-41d4-a716-446655440004', 'analyst@demo.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LDFXCzxs1ZKbS.Q6C', 'Data', 'Analyst', 'analyst', true, NOW(), NOW(), true),
('550e8400-e29b-41d4-a716-446655440005', 'developer@demo.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LDFXCzxs1ZKbS.Q6C', 'API', 'Developer', 'developer', true, NOW(), NOW(), true);

-- Demo API Keys
INSERT INTO api_keys (id, key_value, user_id, name, permissions, rate_limit, is_active, created_at, expires_at) VALUES
('660e8400-e29b-41d4-a716-446655440001', 'demo-api-key-12345', '550e8400-e29b-41d4-a716-446655440001', 'Demo Admin Key', ARRAY['read', 'write', 'analyze', 'export', 'admin'], 10000, true, NOW(), NOW() + INTERVAL '1 year'),
('660e8400-e29b-41d4-a716-446655440002', 'demo-api-key-legal-67890', '550e8400-e29b-41d4-a716-446655440002', 'Legal Expert Key', ARRAY['read', 'analyze', 'export'], 5000, true, NOW(), NOW() + INTERVAL '1 year'),
('660e8400-e29b-41d4-a716-446655440003', 'demo-api-key-business-54321', '550e8400-e29b-41d4-a716-446655440003', 'Business User Key', ARRAY['read', 'analyze'], 1000, true, NOW(), NOW() + INTERVAL '1 year'),
('660e8400-e29b-41d4-a716-446655440004', 'demo-public-key-99999', '550e8400-e29b-41d4-a716-446655440005', 'Public Demo Key', ARRAY['read', 'analyze'], 100, true, NOW(), NOW() + INTERVAL '1 month');

-- Sample Documents
INSERT INTO documents (id, filename, original_filename, file_path, file_size, mime_type, language, document_type, user_id, upload_date, processed, processing_status, is_demo) VALUES
('770e8400-e29b-41d4-a716-446655440001', 'tos_mandatory_arbitration.pdf', 'Terms of Service - Mandatory Arbitration.pdf', '/demo-data/documents/tos_mandatory_arbitration.pdf', 247680, 'application/pdf', 'en', 'terms_of_service', '550e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '7 days', true, 'completed', true),
('770e8400-e29b-41d4-a716-446655440002', 'enterprise_license.docx', 'Enterprise License Agreement.docx', '/demo-data/documents/enterprise_license.docx', 467456, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'en', 'software_license', '550e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '5 days', true, 'completed', true),
('770e8400-e29b-41d4-a716-446655440003', 'german_software_license.pdf', 'Software Lizenzvertrag.pdf', '/demo-data/documents/german_software_license.pdf', 312576, 'application/pdf', 'de', 'software_license', '550e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '3 days', true, 'completed', true),
('770e8400-e29b-41d4-a716-446655440004', 'privacy_policy_no_arbitration.pdf', 'Privacy Policy - No Arbitration.pdf', '/demo-data/documents/privacy_policy_no_arbitration.pdf', 189440, 'application/pdf', 'en', 'privacy_policy', '550e8400-e29b-41d4-a716-446655440003', NOW() - INTERVAL '2 days', true, 'completed', true),
('770e8400-e29b-41d4-a716-446655440005', 'saas_subscription_terms.pdf', 'SaaS Subscription Agreement.pdf', '/demo-data/documents/saas_subscription_terms.pdf', 398722, 'application/pdf', 'en', 'subscription_agreement', '550e8400-e29b-41d4-a716-446655440003', NOW() - INTERVAL '1 day', true, 'completed', true),
('770e8400-e29b-41d4-a716-446655440006', 'spanish_ecommerce_terms.pdf', 'Términos de Comercio Electrónico.pdf', '/demo-data/documents/spanish_ecommerce_terms.pdf', 267890, 'application/pdf', 'es', 'ecommerce_terms', '550e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '4 days', true, 'completed', true),
('770e8400-e29b-41d4-a716-446655440007', 'french_privacy_policy.pdf', 'Politique de Confidentialité.pdf', '/demo-data/documents/french_privacy_policy.pdf', 234567, 'application/pdf', 'fr', 'privacy_policy', '550e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '6 days', true, 'completed', true),
('770e8400-e29b-41d4-a716-446655440008', 'japanese_user_agreement.pdf', 'ユーザー規約.pdf', '/demo-data/documents/japanese_user_agreement.pdf', 445321, 'application/pdf', 'ja', 'user_agreement', '550e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '8 days', true, 'completed', true);

-- Analysis Results
INSERT INTO analysis_results (id, document_id, user_id, analysis_type, confidence_score, processing_time_ms, result_data, created_at, is_demo) VALUES
('880e8400-e29b-41d4-a716-446655440001', '770e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', 'arbitration_detection', 94.5, 2340, '{
  "clauses_found": 5,
  "high_confidence_clauses": 5,
  "clauses": [
    {
      "id": "clause_1",
      "type": "mandatory",
      "confidence": 94,
      "text": "You and the Company agree that any dispute, claim, or controversy arising out of or relating to these Terms will be settled by binding arbitration.",
      "start_position": 245,
      "end_position": 425,
      "severity": "high",
      "recommendations": ["This is a strong mandatory arbitration clause", "Consider reviewing AAA arbitration rules referenced"]
    },
    {
      "id": "clause_2", 
      "type": "binding",
      "confidence": 97,
      "text": "The arbitrator decision shall be final and binding upon all parties, with no right of appeal.",
      "start_position": 890,
      "end_position": 1020,
      "severity": "high",
      "recommendations": ["Waives right to appeal arbitration decisions", "Creates finality in dispute resolution"]
    },
    {
      "id": "clause_3",
      "type": "mandatory", 
      "confidence": 89,
      "text": "YOU AND THE COMPANY HEREBY WAIVE ANY CONSTITUTIONAL AND STATUTORY RIGHTS TO SUE IN COURT AND HAVE A TRIAL IN FRONT OF A JUDGE OR A JURY.",
      "start_position": 1250,
      "end_position": 1420,
      "severity": "high",
      "recommendations": ["Explicit jury trial waiver", "Constitutional rights waiver language"]
    },
    {
      "id": "clause_4",
      "type": "mandatory",
      "confidence": 92,
      "text": "The arbitration will be conducted by the American Arbitration Association (AAA) under its Consumer Arbitration Rules.",
      "start_position": 1500,
      "end_position": 1650,
      "severity": "medium",
      "recommendations": ["Specifies AAA arbitration rules", "Consumer protection provisions apply"]
    },
    {
      "id": "clause_5",
      "type": "class_action_waiver",
      "confidence": 95,
      "text": "YOU AND THE COMPANY AGREE THAT EACH MAY BRING CLAIMS AGAINST THE OTHER ONLY IN YOUR OR ITS INDIVIDUAL CAPACITY AND NOT AS A PLAINTIFF OR CLASS MEMBER.",
      "start_position": 1800,
      "end_position": 2000,
      "severity": "high", 
      "recommendations": ["Prohibits class action lawsuits", "Limits collective legal action"]
    }
  ],
  "summary": {
    "risk_level": "high",
    "enforceability": 85,
    "clarity": 78,
    "completeness": 92
  },
  "recommendations": [
    "This document contains strong mandatory arbitration provisions",
    "Users waive significant legal rights including jury trials",
    "Consider providing opt-out mechanisms where legally required"
  ]
}', NOW(), true),

('880e8400-e29b-41d4-a716-446655440002', '770e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', 'arbitration_detection', 87.3, 3120, '{
  "clauses_found": 3,
  "high_confidence_clauses": 2,
  "clauses": [
    {
      "id": "clause_1",
      "type": "mandatory",
      "confidence": 89,
      "text": "Any dispute, controversy, or claim arising out of or relating to this Agreement shall be settled by final and binding arbitration.",
      "start_position": 1200,
      "end_position": 1380,
      "severity": "high",
      "recommendations": ["Strong binding arbitration clause", "Covers all disputes related to agreement"]
    },
    {
      "id": "clause_2",
      "type": "binding",
      "confidence": 91,
      "text": "The arbitration shall be conducted by a single arbitrator in San Francisco, California.",
      "start_position": 1400,
      "end_position": 1520,
      "severity": "medium",
      "recommendations": ["Specifies single arbitrator", "Geographic location requirement"]
    },
    {
      "id": "clause_3",
      "type": "class_action_waiver",
      "confidence": 82,
      "text": "User agrees that any arbitration shall be conducted in Users individual capacity only and not as a class action.",
      "start_position": 1600,
      "end_position": 1750,
      "severity": "medium",
      "recommendations": ["Prevents class action arbitration", "Individual arbitration required"]
    }
  ],
  "summary": {
    "risk_level": "medium",
    "enforceability": 88,
    "clarity": 85,
    "completeness": 75
  }
}', NOW(), true),

('880e8400-e29b-41d4-a716-446655440003', '770e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440002', 'arbitration_detection', 91.2, 2890, '{
  "clauses_found": 2,
  "high_confidence_clauses": 2,
  "clauses": [
    {
      "id": "clause_1",
      "type": "mandatory",
      "confidence": 93,
      "text": "Jeder Streit, Kontroverse oder Anspruch, der sich aus oder in Bezug auf diese Vereinbarung ergibt, wird durch endgültige und bindende Schiedsverfahren beigelegt.",
      "start_position": 890,
      "end_position": 1120,
      "severity": "high",
      "recommendations": ["German mandatory arbitration clause", "Final and binding arbitration specified"]
    },
    {
      "id": "clause_2", 
      "type": "class_action_waiver",
      "confidence": 89,
      "text": "Der Benutzer stimmt zu, dass jedes Schiedsverfahren nur in der individuellen Eigenschaft des Benutzers durchgeführt wird.",
      "start_position": 1200,
      "end_position": 1350,
      "severity": "medium",
      "recommendations": ["German class action waiver", "Individual arbitration requirement"]
    }
  ],
  "summary": {
    "risk_level": "high",
    "enforceability": 90,
    "clarity": 87,
    "completeness": 82
  },
  "language_detection": {
    "detected_language": "de",
    "confidence": 0.98,
    "translation_provided": true
  }
}', NOW(), true),

('880e8400-e29b-41d4-a716-446655440004', '770e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440003', 'arbitration_detection', 15.2, 1450, '{
  "clauses_found": 0,
  "high_confidence_clauses": 0,
  "clauses": [],
  "summary": {
    "risk_level": "low",
    "enforceability": 0,
    "clarity": 95,
    "completeness": 0
  },
  "recommendations": [
    "No arbitration clauses detected in this privacy policy",
    "Document appears to be standard privacy policy without dispute resolution terms",
    "Consider adding dispute resolution mechanisms if needed"
  ]
}', NOW(), true);

-- Detected Clauses (detailed breakdown)
INSERT INTO detected_clauses (id, analysis_result_id, clause_type, confidence_score, start_position, end_position, clause_text, severity, recommendations, created_at) VALUES
('990e8400-e29b-41d4-a716-446655440001', '880e8400-e29b-41d4-a716-446655440001', 'mandatory', 94, 245, 425, 'You and the Company agree that any dispute, claim, or controversy arising out of or relating to these Terms will be settled by binding arbitration.', 'high', ARRAY['This is a strong mandatory arbitration clause', 'Consider reviewing AAA arbitration rules referenced'], NOW()),
('990e8400-e29b-41d4-a716-446655440002', '880e8400-e29b-41d4-a716-446655440001', 'binding', 97, 890, 1020, 'The arbitrator decision shall be final and binding upon all parties, with no right of appeal.', 'high', ARRAY['Waives right to appeal arbitration decisions', 'Creates finality in dispute resolution'], NOW()),
('990e8400-e29b-41d4-a716-446655440003', '880e8400-e29b-41d4-a716-446655440001', 'jury_waiver', 89, 1250, 1420, 'YOU AND THE COMPANY HEREBY WAIVE ANY CONSTITUTIONAL AND STATUTORY RIGHTS TO SUE IN COURT AND HAVE A TRIAL IN FRONT OF A JUDGE OR A JURY.', 'high', ARRAY['Explicit jury trial waiver', 'Constitutional rights waiver language'], NOW()),
('990e8400-e29b-41d4-a716-446655440004', '880e8400-e29b-41d4-a716-446655440001', 'procedural', 92, 1500, 1650, 'The arbitration will be conducted by the American Arbitration Association (AAA) under its Consumer Arbitration Rules.', 'medium', ARRAY['Specifies AAA arbitration rules', 'Consumer protection provisions apply'], NOW()),
('990e8400-e29b-41d4-a716-446655440005', '880e8400-e29b-41d4-a716-446655440001', 'class_action_waiver', 95, 1800, 2000, 'YOU AND THE COMPANY AGREE THAT EACH MAY BRING CLAIMS AGAINST THE OTHER ONLY IN YOUR OR ITS INDIVIDUAL CAPACITY AND NOT AS A PLAINTIFF OR CLASS MEMBER.', 'high', ARRAY['Prohibits class action lawsuits', 'Limits collective legal action'], NOW());

-- Demo Analytics Data
INSERT INTO analytics_metrics (id, metric_name, metric_value, metric_type, user_id, document_id, recorded_at, is_demo) VALUES
('aa0e8400-e29b-41d4-a716-446655440001', 'processing_time', 2340, 'performance', '550e8400-e29b-41d4-a716-446655440001', '770e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '7 days', true),
('aa0e8400-e29b-41d4-a716-446655440002', 'accuracy_score', 94.5, 'quality', '550e8400-e29b-41d4-a716-446655440001', '770e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '7 days', true),
('aa0e8400-e29b-41d4-a716-446655440003', 'clauses_detected', 5, 'analysis', '550e8400-e29b-41d4-a716-446655440001', '770e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '7 days', true),
('aa0e8400-e29b-41d4-a716-446655440004', 'processing_time', 3120, 'performance', '550e8400-e29b-41d4-a716-446655440002', '770e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '5 days', true),
('aa0e8400-e29b-41d4-a716-446655440005', 'accuracy_score', 87.3, 'quality', '550e8400-e29b-41d4-a716-446655440002', '770e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '5 days', true),
('aa0e8400-e29b-41d4-a716-446655440006', 'language_detection_confidence', 98, 'language', '550e8400-e29b-41d4-a716-446655440002', '770e8400-e29b-41d4-a716-446655440003', NOW() - INTERVAL '3 days', true);

-- Contract Templates for Demo
INSERT INTO contract_templates (id, name, description, category, language, industry, clauses_count, template_data, created_by, created_at, is_demo) VALUES
('bb0e8400-e29b-41d4-a716-446655440001', 'SaaS Terms of Service', 'Standard Terms of Service template for SaaS applications with arbitration clauses', 'terms_of_service', 'en', 'technology', 12, '{
  "sections": [
    {"id": "acceptance", "title": "Acceptance of Terms", "required": true},
    {"id": "service_description", "title": "Description of Service", "required": true},
    {"id": "user_accounts", "title": "User Accounts and Registration", "required": true},
    {"id": "payment", "title": "Payment and Billing", "required": true},
    {"id": "intellectual_property", "title": "Intellectual Property Rights", "required": true},
    {"id": "arbitration", "title": "Dispute Resolution and Arbitration", "required": true},
    {"id": "limitation", "title": "Limitation of Liability", "required": true},
    {"id": "termination", "title": "Termination", "required": true},
    {"id": "privacy", "title": "Privacy Policy", "required": true},
    {"id": "modifications", "title": "Modifications to Terms", "required": true},
    {"id": "governing_law", "title": "Governing Law", "required": true},
    {"id": "contact", "title": "Contact Information", "required": true}
  ],
  "arbitration_clauses": [
    {
      "type": "mandatory",
      "text": "Any dispute arising out of or relating to this Agreement shall be resolved through binding arbitration.",
      "severity": "high"
    },
    {
      "type": "class_action_waiver", 
      "text": "You agree to bring claims only in your individual capacity and not as part of any class action.",
      "severity": "high"
    }
  ]
}', '550e8400-e29b-41d4-a716-446655440001', NOW(), true),

('bb0e8400-e29b-41d4-a716-446655440002', 'Software License Agreement', 'Enterprise software license with arbitration provisions', 'software_license', 'en', 'technology', 9, '{
  "sections": [
    {"id": "grant", "title": "Grant of License", "required": true},
    {"id": "restrictions", "title": "License Restrictions", "required": true},
    {"id": "support", "title": "Support and Maintenance", "required": true},
    {"id": "warranty", "title": "Warranty and Disclaimers", "required": true},
    {"id": "liability", "title": "Limitation of Liability", "required": true},
    {"id": "arbitration", "title": "Dispute Resolution", "required": true},
    {"id": "termination", "title": "Termination", "required": true},
    {"id": "governing_law", "title": "Governing Law", "required": true},
    {"id": "entire_agreement", "title": "Entire Agreement", "required": true}
  ]
}', '550e8400-e29b-41d4-a716-446655440001', NOW(), true);

-- Collaboration Sessions (for real-time collaboration demo)
INSERT INTO collaboration_sessions (id, document_id, session_name, created_by, participants, status, created_at, updated_at, is_demo) VALUES
('cc0e8400-e29b-41d4-a716-446655440001', '770e8400-e29b-41d4-a716-446655440001', 'Legal Review Session - TOS', '550e8400-e29b-41d4-a716-446655440002', ARRAY['550e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440003'], 'active', NOW() - INTERVAL '2 hours', NOW(), true),
('cc0e8400-e29b-41d4-a716-446655440002', '770e8400-e29b-41d4-a716-446655440002', 'Contract Review - Enterprise License', '550e8400-e29b-41d4-a716-446655440001', ARRAY['550e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440003'], 'active', NOW() - INTERVAL '1 hour', NOW(), true);

-- Comments and Annotations for Collaboration Demo
INSERT INTO document_comments (id, document_id, user_id, collaboration_session_id, comment_text, position_start, position_end, comment_type, created_at, is_demo) VALUES
('dd0e8400-e29b-41d4-a716-446655440001', '770e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440002', 'cc0e8400-e29b-41d4-a716-446655440001', 'This arbitration clause seems overly broad. Should we consider adding carve-outs for intellectual property disputes?', 245, 425, 'question', NOW() - INTERVAL '30 minutes', true),
('dd0e8400-e29b-41d4-a716-446655440002', '770e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', 'cc0e8400-e29b-41d4-a716-446655440001', 'Good point. We should also ensure this complies with California consumer protection laws.', 245, 425, 'response', NOW() - INTERVAL '25 minutes', true),
('dd0e8400-e29b-41d4-a716-446655440003', '770e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440003', 'cc0e8400-e29b-41d4-a716-446655440001', 'From a business perspective, we need to balance legal protection with customer satisfaction.', 245, 425, 'suggestion', NOW() - INTERVAL '20 minutes', true);

-- Export History for Demo
INSERT INTO export_history (id, user_id, document_id, analysis_result_id, export_format, export_status, file_path, created_at, is_demo) VALUES
('ee0e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', '770e8400-e29b-41d4-a716-446655440001', '880e8400-e29b-41d4-a716-446655440001', 'PDF', 'completed', '/demo-data/exports/tos_analysis_report.pdf', NOW() - INTERVAL '1 day', true),
('ee0e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', '770e8400-e29b-41d4-a716-446655440002', '880e8400-e29b-41d4-a716-446655440002', 'JSON', 'completed', '/demo-data/exports/enterprise_license_data.json', NOW() - INTERVAL '2 days', true),
('ee0e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440003', '770e8400-e29b-41d4-a716-446655440003', '880e8400-e29b-41d4-a716-446655440003', 'CSV', 'completed', '/demo-data/exports/german_license_clauses.csv', NOW() - INTERVAL '3 days', true);

-- System Configuration for Demo
INSERT INTO system_config (key, value, description, category, is_demo) VALUES
('demo_mode', 'true', 'Enable demo mode features', 'system', true),
('demo_auto_login', 'true', 'Allow automatic login for demo users', 'authentication', true),
('demo_sample_data', 'true', 'Load sample data for demonstration', 'data', true),
('demo_rate_limit_bypass', 'true', 'Bypass rate limits for demo accounts', 'api', true),
('demo_analytics_enabled', 'true', 'Enable analytics collection for demo', 'analytics', true),
('demo_collaboration_enabled', 'true', 'Enable real-time collaboration features', 'collaboration', true),
('demo_voice_interface_enabled', 'true', 'Enable voice interface for accessibility demo', 'accessibility', true),
('demo_blockchain_enabled', 'true', 'Enable blockchain verification features', 'blockchain', true),
('demo_multilingual_enabled', 'true', 'Enable multi-language processing', 'nlp', true),
('demo_contract_builder_enabled', 'true', 'Enable contract builder functionality', 'contracts', true);

-- Demo Feedback and Usage Statistics
INSERT INTO demo_feedback (id, user_id, session_id, feature_rating, overall_rating, feedback_text, created_at) VALUES
('ff0e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440003', 'demo-session-1', 5, 5, 'Excellent tool! The AI detection accuracy is impressive and the interface is very user-friendly.', NOW() - INTERVAL '1 day'),
('ff0e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', 'demo-session-2', 4, 4, 'Great multi-language support. The German document analysis worked perfectly.', NOW() - INTERVAL '2 days'),
('ff0e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440001', 'demo-session-3', 5, 5, 'The collaboration features are exactly what our legal team needs. Real-time editing is smooth.', NOW() - INTERVAL '3 days');

-- Create indexes for better demo performance
CREATE INDEX IF NOT EXISTS idx_demo_documents_user_demo ON documents(user_id, is_demo);
CREATE INDEX IF NOT EXISTS idx_demo_analysis_results_demo ON analysis_results(is_demo, created_at);
CREATE INDEX IF NOT EXISTS idx_demo_clauses_analysis_result ON detected_clauses(analysis_result_id);
CREATE INDEX IF NOT EXISTS idx_demo_analytics_demo ON analytics_metrics(is_demo, recorded_at);
CREATE INDEX IF NOT EXISTS idx_demo_collaboration_status ON collaboration_sessions(status, is_demo);

-- Update statistics for the demo environment
ANALYZE;

-- Demo environment setup complete
SELECT 'Demo database setup completed successfully!' as status;