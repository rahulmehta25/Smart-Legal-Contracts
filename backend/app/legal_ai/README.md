# Legal AI Services

Comprehensive AI-powered legal document analysis services for the Smart Legal Contracts platform.

## Overview

This module extends the existing arbitration clause detection system into a full legal document analysis platform with the following capabilities:

- **Clause Classification**: Detect and classify all types of legal clauses
- **Risk Scoring**: Assign risk scores and identify unfavorable terms
- **Contract Summarization**: Generate executive summaries with structured data
- **Compliance Checking**: Check against GDPR, SOX, HIPAA, and custom rules
- **Term Extraction**: Extract parties, dates, amounts, and obligations
- **Document Comparison**: Semantic diff with risk analysis
- **Template Matching**: Compare against standard templates
- **Natural Language Query**: Ask questions about documents

## Architecture

```
legal_ai/
├── interfaces/           # Abstract base classes
│   ├── base.py          # Base service interface
│   ├── clause_classification.py
│   ├── risk_scoring.py
│   ├── summarization.py
│   ├── compliance_checking.py
│   ├── term_extraction.py
│   ├── document_comparison.py
│   ├── template_matching.py
│   └── natural_language_query.py
├── providers/            # AI provider implementations
│   ├── config.py        # Provider configuration
│   ├── base_provider.py # Base provider interface
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── vertex_provider.py
└── services/            # Service implementations
    ├── clause_classification_service.py
    ├── risk_scoring_service.py
    ├── summarization_service.py
    ├── compliance_checking_service.py
    ├── term_extraction_service.py
    ├── document_comparison_service.py
    ├── template_matching_service.py
    └── natural_language_query_service.py
```

## Quick Start

```python
from app.legal_ai import (
    ClauseClassificationService,
    RiskScoringService,
    ContractSummarizationService,
    NaturalLanguageQueryService,
)

# Initialize a service
async with ClauseClassificationService() as classifier:
    result = await classifier.classify_document(contract_text)
    print(f"Found {result.total_clauses} clauses")
    for clause in result.clauses:
        print(f"- {clause.clause_type.value}: {clause.confidence:.2f}")
```

## Services

### 1. Clause Classification Service

Detects and classifies all types of legal clauses:

- Arbitration, Indemnification, Liability Limitation
- Termination, Non-Compete, Confidentiality
- Force Majeure, IP Assignment, Governing Law
- Dispute Resolution, Warranty, Payment Terms
- Data Protection, Audit Rights, and more

```python
from app.legal_ai import ClauseClassificationService, ClauseType

async with ClauseClassificationService() as service:
    # Classify entire document
    result = await service.classify_document(
        document_text=contract,
        target_clause_types=[ClauseType.ARBITRATION, ClauseType.INDEMNIFICATION],
        min_confidence=0.7
    )

    # Classify single clause
    clause = await service.classify_clause(
        clause_text=specific_clause,
        context=surrounding_text
    )
```

### 2. Risk Scoring Service

Analyzes contracts for risks and unfavorable terms:

```python
from app.legal_ai import RiskScoringService

async with RiskScoringService() as service:
    result = await service.score_document(
        document_text=contract,
        party_perspective="client",
        industry="technology",
        contract_type="SaaS"
    )

    print(f"Overall Risk: {result.overall_score:.2f} ({result.overall_level.value})")

    for factor in result.get_critical_risks():
        print(f"CRITICAL: {factor.name}")
        print(f"  Mitigation: {factor.mitigation_suggestions}")

    # Generate report
    report = await service.generate_risk_report(result)
```

### 3. Contract Summarization Service

Generates executive summaries with structured extraction:

```python
from app.legal_ai import ContractSummarizationService

async with ContractSummarizationService() as service:
    summary = await service.summarize_document(
        document_text=contract,
        summary_length="detailed",
        focus_areas=["financial", "termination"]
    )

    print(f"Title: {summary.title}")
    print(f"Parties: {', '.join(summary.parties)}")
    print(f"Value: {summary.total_value}")

    # Get obligations by party
    for obligation in summary.get_obligations_by_party("Client"):
        print(f"- {obligation.description} (Due: {obligation.deadline})")

    # Get upcoming deadlines
    for deadline in summary.get_upcoming_deadlines(days=30):
        print(f"- {deadline.description}: {deadline.days_until} days")
```

### 4. Compliance Checking Service

Checks documents against compliance frameworks:

```python
from app.legal_ai import ComplianceCheckingService, ComplianceFramework

async with ComplianceCheckingService() as service:
    result = await service.check_compliance(
        document_text=contract,
        frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
        industry="healthcare"
    )

    print(f"Compliance Score: {result.compliance_score:.1%}")
    print(f"Status: {result.overall_status.value}")

    for violation in result.get_critical_violations():
        print(f"VIOLATION: {violation.rule.name}")
        print(f"  Fix: {violation.remediation_steps}")

    # Generate remediation plan
    plan = await service.generate_remediation_plan(result)
```

### 5. Term Extraction Service

Extracts structured data from contracts:

```python
from app.legal_ai import TermExtractionService

async with TermExtractionService() as service:
    result = await service.extract_all(document_text=contract)

    # Access extracted data
    for party in result.parties:
        print(f"Party: {party.name} ({party.role})")

    for amount in result.amounts:
        print(f"Amount: {amount.currency} {amount.value:,.2f} - {amount.description}")

    for deadline in result.deadlines:
        print(f"Deadline: {deadline.due_date_text} - {deadline.action_required}")

    # Generate term sheet
    term_sheet = await service.generate_term_sheet(result)
```

### 6. Document Comparison Service

Semantic diff between contract versions:

```python
from app.legal_ai import DocumentComparisonService

async with DocumentComparisonService() as service:
    result = await service.compare_documents(
        original_text=version1,
        modified_text=version2,
        party_perspective="client"
    )

    print(f"Similarity: {result.similarity_metrics.text_similarity:.1%}")
    print(f"Total Changes: {result.total_changes}")
    print(f"Risk Delta: {result.overall_risk_delta:+.2f}")

    for change in result.get_critical_changes():
        print(f"CRITICAL: {change.semantic_description}")

    # Generate redline
    redline = await service.generate_redline(result, format="markdown")
```

### 7. Template Matching Service

Compare against standard templates:

```python
from app.legal_ai import TemplateMatchingService, Template

async with TemplateMatchingService() as service:
    template = Template(
        template_id="nda-standard",
        name="Standard NDA",
        version="2.0",
        contract_type="NDA",
        industry="general",
        description="Standard mutual NDA template",
        template_text=template_text,
        required_clauses=["confidentiality", "term", "return_of_materials"],
        optional_clauses=["non_solicitation"],
        variable_fields=["party_names", "effective_date"]
    )

    result = await service.match_template(
        document_text=contract,
        template=template,
        strict_mode=True
    )

    print(f"Match Score: {result.overall_match_score:.1%}")
    print(f"Missing Required: {result.missing_required_clauses}")
    print(f"Requires Approval: {result.requires_approval}")
```

### 8. Natural Language Query Service

Ask questions about documents:

```python
from app.legal_ai import NaturalLanguageQueryService

async with NaturalLanguageQueryService() as service:
    # Start a conversation
    context = await service.start_conversation(
        document_text=contract,
        user_role="legal_counsel"
    )

    # Ask a question
    result = await service.query(
        question="What are the termination conditions?",
        document_text=contract,
        context=context
    )

    print(f"Answer: {result.answer.answer}")
    print(f"Confidence: {result.answer.confidence.value}")

    for citation in result.answer.citations:
        print(f"Source: {citation.text[:100]}...")

    # Get suggested questions
    suggestions = await service.suggest_queries(contract, num_suggestions=5)
```

## Configuration

### Provider Configuration

Configure which AI provider to use per service:

```python
from app.legal_ai.providers import configure_service, AIProviderType

# Use Anthropic Claude for risk scoring
configure_service(
    "risk_scoring",
    provider=AIProviderType.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    temperature=0.0,
    max_tokens=8192
)

# Use OpenAI for summarization
configure_service(
    "summarization",
    provider=AIProviderType.OPENAI,
    model="gpt-4o",
    temperature=0.0
)
```

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Vertex AI
VERTEX_PROJECT_ID=your-project
VERTEX_LOCATION=us-central1
# OR
GOOGLE_API_KEY=...  # For direct Gemini API
```

### Default Provider Configuration

| Service | Default Provider | Default Model |
|---------|-----------------|---------------|
| Clause Classification | OpenAI | gpt-4o |
| Risk Scoring | Anthropic | claude-sonnet-4 |
| Summarization | OpenAI | gpt-4o |
| Compliance Checking | Anthropic | claude-sonnet-4 |
| Term Extraction | OpenAI | gpt-4o |
| Document Comparison | OpenAI | gpt-4o |
| Template Matching | OpenAI | gpt-4o |
| Natural Language Query | Anthropic | claude-sonnet-4 |

## Data Models

### Common Structures

All services use dataclasses with `to_dict()` methods for serialization:

```python
@dataclass
class ClassifiedClause:
    text: str
    clause_type: ClauseType
    confidence: float
    start_position: int
    end_position: int
    key_terms: List[str]
    risk_indicators: List[str]

    def to_dict(self) -> dict:
        ...
```

### Enums

- `ClauseType`: All supported clause types
- `RiskLevel`: critical, high, medium, low, minimal
- `RiskCategory`: legal, financial, operational, compliance, reputational, strategic
- `ComplianceFramework`: GDPR, SOX, HIPAA, CCPA, PCI_DSS, etc.
- `ComplianceStatus`: compliant, partially_compliant, non_compliant, requires_review
- `ChangeType`: addition, deletion, modification, moved, semantic_change
- `ChangeImpact`: critical, significant, moderate, minor, cosmetic

## Error Handling

All services raise `RuntimeError` for operation failures and `ValueError` for invalid inputs:

```python
try:
    result = await service.score_document(text)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Operation failed: {e}")
```

## Health Checks

All services implement health checking:

```python
health = await service.health_check()
print(f"Healthy: {health.healthy}")
print(f"Provider: {health.provider.value}")
print(f"Latency: {health.latency_ms}ms")
```

## Async Context Manager

All services support async context managers for proper resource management:

```python
async with RiskScoringService() as service:
    # Service is initialized
    result = await service.score_document(text)
# Service is automatically shut down
```

Or manage lifecycle manually:

```python
service = RiskScoringService()
await service.initialize()
try:
    result = await service.score_document(text)
finally:
    await service.shutdown()
```

## Integration with Existing System

These services complement the existing arbitration clause detection. To integrate:

```python
from app.legal_ai import (
    ClauseClassificationService,
    RiskScoringService,
)

# Use alongside existing RAG system
async def analyze_document(document_text: str):
    # Existing arbitration detection
    arbitration_result = existing_rag_analysis(document_text)

    # New comprehensive analysis
    async with ClauseClassificationService() as classifier:
        clauses = await classifier.classify_document(document_text)

    async with RiskScoringService() as scorer:
        risks = await scorer.score_document(document_text)

    return {
        "arbitration": arbitration_result,
        "all_clauses": clauses.to_dict(),
        "risk_analysis": risks.to_dict(),
    }
```

## Performance Considerations

- Services cache provider clients for connection reuse
- Use `classify_document()` for batch processing instead of multiple `classify_clause()` calls
- Set appropriate `max_tokens` limits based on document size
- Consider using `gpt-4o-mini` or `claude-3-haiku` for high-volume, simple queries

## License

Part of the Smart Legal Contracts platform.
