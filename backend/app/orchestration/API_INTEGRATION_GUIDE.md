# API Integration Guide

## Overview

This guide provides comprehensive documentation for integrating with all services through the orchestration layer. The system provides unified APIs for document analysis, voice interaction, compliance automation, visualization, and white-label functionality.

## Authentication

All API requests must include authentication headers:

```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
X-Request-ID: <unique_request_id>
X-Tenant-ID: <tenant_id> (for multi-tenant requests)
```

## Core Document Analysis APIs

### Upload Document

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

{
  "file": <document_file>,
  "jurisdiction": "US",
  "document_type": "terms_of_service",
  "language": "en"
}
```

**Response**:
```json
{
  "document_id": "doc_123456",
  "status": "uploaded",
  "analysis_triggered": true,
  "saga_id": "saga_789012"
}
```

### Analyze Document

```http
POST /api/v1/analysis/analyze
Content-Type: application/json

{
  "document_id": "doc_123456",
  "analysis_type": "full",
  "options": {
    "include_risk_assessment": true,
    "include_suggestions": true,
    "target_jurisdiction": "US"
  }
}
```

**Response**:
```json
{
  "analysis_id": "analysis_456789",
  "status": "processing",
  "estimated_completion": "2024-01-15T10:30:00Z",
  "progress": {
    "stage": "clause_extraction",
    "percentage": 25
  }
}
```

### Get Analysis Results

```http
GET /api/v1/analysis/{analysis_id}/results
```

**Response**:
```json
{
  "analysis_id": "analysis_456789",
  "document_id": "doc_123456",
  "status": "completed",
  "results": {
    "has_arbitration_clause": true,
    "confidence_score": 0.92,
    "clauses_found": [
      {
        "type": "mandatory_arbitration",
        "text": "Any dispute arising out of...",
        "position": {"start": 1250, "end": 1456},
        "confidence": 0.95
      }
    ],
    "risk_assessment": {
      "overall_risk_level": "medium",
      "risk_factors": [
        {
          "factor": "broad_scope",
          "severity": "high",
          "description": "Arbitration clause covers all disputes"
        }
      ]
    },
    "suggestions": [
      {
        "type": "clause_modification",
        "priority": "high",
        "suggestion": "Consider adding opt-out provision"
      }
    ]
  },
  "metadata": {
    "processing_time_seconds": 45.2,
    "model_version": "1.2.0",
    "jurisdiction_specific_rules_applied": ["US_CONSUMER_PROTECTION"]
  }
}
```

## Voice Interface APIs

### Process Voice Command

```http
POST /api/v1/voice/command
Content-Type: multipart/form-data

{
  "audio": <audio_file>,
  "format": "wav",
  "language": "en-US",
  "context": {
    "document_id": "doc_123456",
    "user_intent": "analysis"
  }
}
```

**Response**:
```json
{
  "command_id": "cmd_789012",
  "status": "processing",
  "transcription": {
    "text": "Analyze this document for arbitration clauses",
    "confidence": 0.89
  },
  "intent": {
    "action": "analyze_document",
    "entities": {
      "analysis_type": "arbitration_clauses",
      "document_reference": "current"
    }
  },
  "triggered_actions": [
    {
      "action": "document_analysis",
      "saga_id": "saga_345678"
    }
  ]
}
```

### Text-to-Speech Synthesis

```http
POST /api/v1/voice/synthesize
Content-Type: application/json

{
  "text": "The document contains a mandatory arbitration clause with high confidence.",
  "voice": "professional_female",
  "language": "en-US",
  "format": "mp3"
}
```

**Response**:
```json
{
  "synthesis_id": "tts_123456",
  "audio_url": "/api/v1/voice/audio/tts_123456",
  "duration_seconds": 8.5,
  "format": "mp3"
}
```

### Voice-Enabled Document Navigation

```http
POST /api/v1/voice/navigate
Content-Type: application/json

{
  "document_id": "doc_123456",
  "command": "go to arbitration section",
  "current_position": 1000
}
```

**Response**:
```json
{
  "navigation_result": {
    "target_position": 1250,
    "section_title": "Dispute Resolution and Arbitration",
    "context": "Found arbitration clause at position 1250-1456",
    "audio_feedback_url": "/api/v1/voice/audio/nav_123456"
  }
}
```

## Document Comparison APIs

### Compare Documents

```http
POST /api/v1/documents/compare
Content-Type: application/json

{
  "document_ids": ["doc_123456", "doc_789012"],
  "comparison_type": "clause_analysis",
  "focus_areas": ["arbitration", "liability", "termination"],
  "output_format": "detailed"
}
```

**Response**:
```json
{
  "comparison_id": "comp_456789",
  "status": "processing",
  "progress": {
    "stage": "text_alignment",
    "percentage": 20
  },
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

### Get Comparison Results

```http
GET /api/v1/documents/compare/{comparison_id}/results
```

**Response**:
```json
{
  "comparison_id": "comp_456789",
  "status": "completed",
  "documents": {
    "doc_123456": {"title": "Service Agreement v1.0"},
    "doc_789012": {"title": "Service Agreement v2.0"}
  },
  "comparison_results": {
    "overall_similarity": 0.87,
    "clause_differences": [
      {
        "clause_type": "arbitration",
        "status": "modified",
        "document_1": {
          "text": "Disputes shall be resolved through binding arbitration",
          "position": {"start": 1250, "end": 1300}
        },
        "document_2": {
          "text": "Disputes may be resolved through arbitration or court proceedings",
          "position": {"start": 1280, "end": 1350}
        },
        "significance": "high",
        "impact": "Arbitration changed from mandatory to optional"
      }
    ],
    "new_clauses": [
      {
        "clause_type": "class_action_waiver",
        "document": "doc_789012",
        "text": "User waives right to participate in class action lawsuits",
        "position": {"start": 1400, "end": 1500},
        "risk_level": "high"
      }
    ],
    "removed_clauses": [
      {
        "clause_type": "cooling_off_period",
        "document": "doc_123456",
        "text": "30-day cooling off period before arbitration",
        "impact": "Reduced consumer protection"
      }
    ]
  },
  "recommendations": [
    {
      "priority": "high",
      "category": "compliance_risk",
      "message": "New class action waiver may not be enforceable in some jurisdictions"
    }
  ]
}
```

## White-label/Multi-tenant APIs

### Create Tenant

```http
POST /api/v1/tenants
Content-Type: application/json

{
  "tenant_name": "Legal Corp",
  "domain": "legalcorp.com",
  "subscription_plan": "enterprise",
  "customization": {
    "brand_colors": {
      "primary": "#1f4e79",
      "secondary": "#f8f9fa"
    },
    "logo_url": "https://legalcorp.com/logo.png",
    "custom_css": ".header { background: #1f4e79; }",
    "white_label_domain": "contracts.legalcorp.com"
  },
  "features": {
    "voice_interface": true,
    "document_comparison": true,
    "advanced_analytics": true,
    "api_access": true
  },
  "admin_user": {
    "email": "admin@legalcorp.com",
    "name": "John Smith",
    "role": "tenant_admin"
  }
}
```

**Response**:
```json
{
  "tenant_id": "tenant_123456",
  "status": "provisioning",
  "saga_id": "saga_tenant_789012",
  "provisioning_steps": [
    "infrastructure_setup",
    "database_initialization", 
    "customization_deployment",
    "admin_user_creation",
    "welcome_email_sending"
  ],
  "estimated_completion": "2024-01-15T11:00:00Z",
  "access_details": {
    "admin_portal_url": "https://admin.contracts.legalcorp.com",
    "api_endpoint": "https://api.contracts.legalcorp.com",
    "documentation_url": "https://docs.contracts.legalcorp.com"
  }
}
```

### Update Tenant Configuration

```http
PUT /api/v1/tenants/{tenant_id}/config
Content-Type: application/json

{
  "customization": {
    "theme": "dark_mode",
    "dashboard_layout": "compact",
    "default_language": "es"
  },
  "features": {
    "voice_interface": true,
    "document_comparison": true,
    "compliance_automation": true
  },
  "integration_settings": {
    "webhook_url": "https://legalcorp.com/webhooks/contracts",
    "api_rate_limits": {
      "requests_per_minute": 1000,
      "burst_requests": 2000
    }
  }
}
```

### Get Tenant Analytics

```http
GET /api/v1/tenants/{tenant_id}/analytics
```

**Response**:
```json
{
  "tenant_id": "tenant_123456",
  "period": "last_30_days",
  "metrics": {
    "documents_processed": 1250,
    "api_requests": 45000,
    "voice_commands": 320,
    "document_comparisons": 89,
    "compliance_checks": 156
  },
  "usage_trends": {
    "daily_active_users": [45, 52, 48, 61, 58],
    "document_processing_growth": 15.2,
    "feature_adoption": {
      "voice_interface": 0.34,
      "document_comparison": 0.67,
      "compliance_automation": 0.89
    }
  },
  "performance_metrics": {
    "average_analysis_time": 23.5,
    "success_rate": 0.987,
    "user_satisfaction_score": 4.6
  }
}
```

## Compliance Automation APIs

### Trigger Compliance Check

```http
POST /api/v1/compliance/check
Content-Type: application/json

{
  "document_id": "doc_123456",
  "jurisdiction": "EU",
  "compliance_frameworks": ["GDPR", "CONSUMER_RIGHTS_DIRECTIVE"],
  "check_type": "comprehensive",
  "notify_on_completion": true
}
```

**Response**:
```json
{
  "compliance_check_id": "compliance_789012",
  "status": "processing",
  "saga_id": "saga_compliance_345678",
  "frameworks_checked": ["GDPR", "CONSUMER_RIGHTS_DIRECTIVE"],
  "estimated_completion": "2024-01-15T10:45:00Z"
}
```

### Get Compliance Report

```http
GET /api/v1/compliance/check/{compliance_check_id}/report
```

**Response**:
```json
{
  "compliance_check_id": "compliance_789012",
  "document_id": "doc_123456",
  "status": "completed",
  "overall_compliance_score": 75,
  "frameworks": {
    "GDPR": {
      "compliance_score": 80,
      "status": "partially_compliant",
      "violations": [
        {
          "article": "Article 7",
          "description": "Consent requirements",
          "severity": "medium",
          "clause_location": {"start": 500, "end": 650},
          "violation_details": "Consent mechanism not clearly specified",
          "recommendations": [
            "Add explicit consent checkbox",
            "Provide clear opt-out mechanism"
          ]
        }
      ],
      "compliant_areas": [
        "Data subject rights clearly stated",
        "Data processing purposes specified",
        "Data controller contact information provided"
      ]
    },
    "CONSUMER_RIGHTS_DIRECTIVE": {
      "compliance_score": 70,
      "status": "partially_compliant",
      "violations": [
        {
          "article": "Article 6",
          "description": "Right of withdrawal",
          "severity": "high",
          "clause_location": {"start": 800, "end": 900},
          "violation_details": "14-day withdrawal period not mentioned",
          "recommendations": [
            "Add explicit 14-day withdrawal right",
            "Specify withdrawal procedure"
          ]
        }
      ]
    }
  },
  "recommendations": {
    "high_priority": [
      "Add consumer withdrawal rights clause",
      "Clarify consent mechanisms"
    ],
    "medium_priority": [
      "Review data retention periods",
      "Update privacy contact information"
    ]
  },
  "next_review_date": "2024-07-15T00:00:00Z"
}
```

### Automated Compliance Monitoring

```http
POST /api/v1/compliance/monitoring/setup
Content-Type: application/json

{
  "tenant_id": "tenant_123456",
  "monitoring_config": {
    "enabled": true,
    "check_frequency": "weekly",
    "jurisdictions": ["US", "EU", "UK"],
    "frameworks": ["GDPR", "CCPA", "CONSUMER_RIGHTS_DIRECTIVE"],
    "alert_thresholds": {
      "compliance_score_below": 80,
      "new_violations": true,
      "regulation_updates": true
    },
    "notification_channels": {
      "email": ["compliance@legalcorp.com"],
      "webhook": "https://legalcorp.com/compliance-alerts",
      "dashboard": true
    }
  }
}
```

## Visualization APIs

### Generate Analysis Dashboard

```http
POST /api/v1/visualizations/dashboard
Content-Type: application/json

{
  "data_sources": [
    {
      "type": "document_analysis",
      "document_ids": ["doc_123456", "doc_789012"],
      "time_range": "last_30_days"
    }
  ],
  "dashboard_type": "executive_summary",
  "widgets": [
    "risk_assessment_chart",
    "clause_distribution_pie", 
    "compliance_score_gauge",
    "trend_analysis_line"
  ],
  "filters": {
    "jurisdiction": "US",
    "document_type": "terms_of_service"
  }
}
```

**Response**:
```json
{
  "dashboard_id": "dash_456789",
  "status": "generating",
  "progress": {
    "stage": "data_aggregation",
    "percentage": 30
  },
  "estimated_completion": "2024-01-15T10:40:00Z"
}
```

### Get Generated Dashboard

```http
GET /api/v1/visualizations/dashboard/{dashboard_id}
```

**Response**:
```json
{
  "dashboard_id": "dash_456789",
  "status": "completed",
  "dashboard_url": "/dashboards/dash_456789",
  "embed_code": "<iframe src='/dashboards/embed/dash_456789'>",
  "widgets": [
    {
      "widget_id": "risk_chart_001",
      "type": "risk_assessment_chart",
      "data_url": "/api/v1/visualizations/data/risk_chart_001",
      "config": {
        "chart_type": "bar",
        "x_axis": "risk_level",
        "y_axis": "document_count"
      }
    }
  ],
  "metadata": {
    "generated_at": "2024-01-15T10:40:00Z",
    "data_points": 156,
    "refresh_rate": "hourly"
  }
}
```

### Export Report

```http
POST /api/v1/visualizations/export
Content-Type: application/json

{
  "dashboard_id": "dash_456789",
  "format": "pdf",
  "options": {
    "include_raw_data": false,
    "high_resolution": true,
    "template": "executive_report"
  },
  "delivery": {
    "method": "email",
    "recipients": ["executive@legalcorp.com"],
    "subject": "Monthly Compliance Report"
  }
}
```

## Webhook Integration

### Register Webhook

```http
POST /api/v1/webhooks
Content-Type: application/json

{
  "url": "https://yourapp.com/webhooks/arbitration-platform",
  "events": [
    "document.analysis.completed",
    "compliance.check.completed",
    "voice.command.processed",
    "comparison.completed"
  ],
  "secret": "webhook_secret_key",
  "active": true,
  "retry_policy": {
    "max_retries": 3,
    "retry_delay_seconds": 60
  }
}
```

### Webhook Payload Examples

#### Document Analysis Completed
```json
{
  "event": "document.analysis.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "analysis_id": "analysis_456789",
    "document_id": "doc_123456",
    "has_arbitration_clause": true,
    "confidence_score": 0.92,
    "risk_level": "medium",
    "tenant_id": "tenant_123456"
  },
  "signature": "sha256=abc123..."
}
```

#### Voice Command Processed
```json
{
  "event": "voice.command.processed",
  "timestamp": "2024-01-15T10:32:00Z",
  "data": {
    "command_id": "cmd_789012",
    "transcription": "Analyze this document for arbitration clauses",
    "intent": "analyze_document",
    "actions_triggered": ["document_analysis"],
    "user_id": "user_456789",
    "tenant_id": "tenant_123456"
  },
  "signature": "sha256=def456..."
}
```

## SDK Examples

### Python SDK

```python
from arbitration_platform_sdk import ArbitrationClient

# Initialize client
client = ArbitrationClient(
    api_key="your_api_key",
    base_url="https://api.arbitration-platform.com",
    tenant_id="tenant_123456"
)

# Upload and analyze document
with open("contract.pdf", "rb") as file:
    document = client.documents.upload(
        file=file,
        jurisdiction="US",
        document_type="terms_of_service"
    )

# Wait for analysis completion
analysis = client.analysis.wait_for_completion(
    document.analysis_id,
    timeout=300
)

# Process voice command
voice_result = client.voice.process_command(
    audio_file="command.wav",
    context={"document_id": document.document_id}
)

# Generate visualization
dashboard = client.visualizations.create_dashboard(
    data_sources=[{
        "type": "document_analysis",
        "document_ids": [document.document_id]
    }],
    dashboard_type="executive_summary"
)
```

### JavaScript SDK

```javascript
import { ArbitrationClient } from '@arbitration-platform/sdk';

// Initialize client
const client = new ArbitrationClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.arbitration-platform.com',
  tenantId: 'tenant_123456'
});

// Upload document
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('jurisdiction', 'US');

const document = await client.documents.upload(formData);

// Start voice recording and processing
const voiceProcessor = client.voice.createProcessor({
  onTranscription: (text) => console.log('Transcribed:', text),
  onIntent: (intent) => console.log('Intent detected:', intent),
  onAction: (action) => console.log('Action triggered:', action)
});

voiceProcessor.startRecording();

// Compare documents
const comparison = await client.documents.compare({
  documentIds: ['doc_123456', 'doc_789012'],
  comparisonType: 'clause_analysis'
});
```

## Rate Limits and Quotas

### Default Rate Limits

| Endpoint Category | Requests per Minute | Burst Limit |
|------------------|-------------------|-------------|
| Document Upload | 10 | 20 |
| Analysis Requests | 30 | 50 |
| Voice Commands | 20 | 40 |
| Comparison Requests | 5 | 10 |
| Visualization Generation | 15 | 30 |
| Webhook Management | 100 | 200 |

### Quota Limits by Plan

| Plan | Documents/Month | API Requests/Month | Voice Commands/Month |
|------|----------------|-------------------|-------------------|
| Starter | 100 | 10,000 | 500 |
| Professional | 1,000 | 100,000 | 2,000 |
| Enterprise | Unlimited | Unlimited | Unlimited |

## Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is invalid",
    "details": "Document file is required",
    "request_id": "req_123456",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `PROCESSING_ERROR` | 422 | Unable to process request |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Retry Logic

For transient errors (5xx status codes), implement exponential backoff:

```python
import time
import random

def make_request_with_retry(request_func, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            return request_func()
        except Exception as e:
            if attempt == max_retries:
                raise
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

## Testing and Debugging

### Test Environment

Test API endpoint: `https://test-api.arbitration-platform.com`

### Debug Headers

Include these headers for debugging:

```http
X-Debug-Mode: true
X-Trace-Request: true
X-Include-Performance-Metrics: true
```

### Health Check Endpoint

```http
GET /api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "document_service": "healthy",
    "analysis_service": "healthy", 
    "voice_interface": "healthy",
    "compliance_automation": "healthy"
  },
  "performance": {
    "average_response_time_ms": 145,
    "requests_per_second": 23.5
  }
}
```