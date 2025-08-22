"""
Main GraphQL Application Entry Point
"""

from fastapi import FastAPI
from .graphql.server import integrate_with_existing_fastapi, create_graphql_app
from .main import app as existing_app  # Import existing FastAPI app


# Option 1: Integrate with existing FastAPI app
def add_graphql_to_existing_app():
    """Add GraphQL to the existing FastAPI application"""
    integrate_with_existing_fastapi(existing_app, prefix="/graphql")
    return existing_app


# Option 2: Create standalone GraphQL app
def create_standalone_graphql_app():
    """Create a standalone GraphQL application"""
    return create_graphql_app()


# Example usage documentation
USAGE_EXAMPLES = """
# GraphQL API Usage Examples

## 1. Starting the Server

### Development Server
```bash
python -m app.graphql.server dev
```

### Production Server
```bash
python -m app.graphql.server prod
```

### Federation Gateway
```bash
python -m app.graphql.server federation
```

## 2. GraphQL Queries

### Basic Document Query
```graphql
query GetDocuments {
  documents(first: 10) {
    edges {
      node {
        id
        filename
        fileType
        processingStatus
        hasArbitrationClauses
      }
    }
    totalCount
  }
}
```

### Document Analysis
```graphql
query GetDocumentAnalysis($id: ID!) {
  document(id: $id) {
    id
    filename
    
    analyses(first: 1) {
      edges {
        node {
          id
          hasArbitrationClause
          confidenceScore
          riskLevel
          
          clauses(first: 10) {
            edges {
              node {
                clauseText
                clauseType
                relevanceScore
                severityScore
                isBinding
              }
            }
          }
        }
      }
    }
  }
}
```

### Search Documents
```graphql
query SearchDocuments($query: String!) {
  searchDocuments(query: $query, first: 20) {
    edges {
      node {
        id
        filename
        hasArbitrationClauses
        averageConfidenceScore
      }
    }
    totalCount
  }
}
```

## 3. Mutations

### Upload Document
```graphql
mutation UploadDocument($input: DocumentUploadInput!) {
  uploadDocument(input: $input) {
    success
    message
    document {
      id
      filename
      processingStatus
    }
    errors
  }
}
```

Variables:
```json
{
  "input": {
    "filename": "terms_of_service.txt",
    "content": "Any disputes will be resolved through binding arbitration...",
    "fileType": "text/plain"
  }
}
```

### Request Analysis
```graphql
mutation RequestAnalysis($input: AnalysisRequestInput!) {
  requestAnalysis(input: $input) {
    success
    message
    analysis {
      id
      hasArbitrationClause
      confidenceScore
      riskLevel
    }
    processingTimeMs
    errors
  }
}
```

### Quick Analysis
```graphql
mutation QuickAnalysis($input: QuickAnalysisInput!) {
  quickAnalysis(input: $input) {
    hasArbitrationClause
    confidenceScore
    summary
    clausesFound {
      text
      type
      confidence
      startPosition
      endPosition
    }
  }
}
```

## 4. Subscriptions

### Document Processing Updates
```graphql
subscription DocumentProcessing($documentId: ID) {
  documentProcessing(documentId: $documentId) {
    documentId
    status
    progress
    message
  }
}
```

### Analysis Progress
```graphql
subscription AnalysisProgress($documentId: ID) {
  analysisProgress(documentId: $documentId) {
    analysisId
    documentId
    status
    progress
    results {
      hasArbitrationClause
      confidenceScore
      riskLevel
    }
  }
}
```

## 5. Authentication

Include JWT token in Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

### Login
```graphql
mutation LoginUser($username: String!, $password: String!) {
  loginUser(username: $username, password: $password) {
    success
    user {
      id
      username
      email
      role
    }
    token
    errors
  }
}
```

## 6. Error Handling

All mutations return standardized error responses:
```json
{
  "success": false,
  "message": "Validation failed",
  "errors": [
    "Filename is required",
    "Content is required"
  ]
}
```

## 7. Rate Limiting

API requests are rate-limited by operation type:
- Queries: 100 requests/minute
- Mutations: 20 requests/minute  
- Analysis operations: 10 requests/5 minutes
- Uploads: 5 requests/minute

## 8. Query Complexity

Query complexity is automatically analyzed and limited:
- Maximum complexity: 1000 points
- Maximum depth: 10 levels
- Introspection queries: 1000 points

## 9. Apollo Federation

For microservices architecture, use federation support:
```graphql
extend type Document @key(fields: "id") {
  id: ID! @external
  analysisResults: [AnalysisResult!]!
}
```

## 10. Client Generation

Generate TypeScript client:
```python
from app.graphql.codegen import ClientGenerator, GeneratorConfig, ClientLanguage

config = GeneratorConfig(
    language=ClientLanguage.TYPESCRIPT,
    output_dir="./generated",
    api_endpoint="http://localhost:8000/graphql"
)

generator = ClientGenerator(schema_introspection, config)
files = generator.generate_client()
```

## 11. GraphQL Playground

Access interactive playground at:
- Development: http://localhost:8000/playground
- Schema docs: http://localhost:8000/graphql/schema

## 12. WebSocket Subscriptions

Connect to WebSocket endpoint for real-time updates:
- WebSocket URL: ws://localhost:8000/graphql/ws
- Protocol: graphql-transport-ws or graphql-ws
"""

if __name__ == "__main__":
    print(USAGE_EXAMPLES)