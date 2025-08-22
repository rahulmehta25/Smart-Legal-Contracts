"""
GraphQL Playground Implementation
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
from typing import Dict, Any, Optional


def create_playground_app(
    endpoint: str = "/graphql",
    subscriptions_endpoint: str = "/graphql/ws",
    title: str = "Arbitration Detection GraphQL API",
    version: str = "1.0.0"
) -> FastAPI:
    """Create FastAPI app with GraphQL Playground"""
    
    app = FastAPI(
        title=f"{title} - Playground",
        description="Interactive GraphQL Playground for the Arbitration Detection API",
        version=version
    )
    
    @app.get("/", response_class=HTMLResponse)
    async def playground_html():
        """Serve GraphQL Playground HTML"""
        return get_playground_html(endpoint, subscriptions_endpoint, title)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "graphql-playground"}
    
    return app


def get_playground_html(
    endpoint: str,
    subscriptions_endpoint: str,
    title: str
) -> str:
    """Generate GraphQL Playground HTML"""
    
    config = {
        "endpoint": endpoint,
        "subscriptionEndpoint": subscriptions_endpoint,
        "settings": {
            "editor.theme": "dark",
            "editor.fontSize": 14,
            "editor.fontFamily": "'Source Code Pro', 'Consolas', 'Inconsolata', 'Droid Sans Mono', 'Monaco', monospace",
            "editor.reuseHeaders": True,
            "request.credentials": "include",
            "schema.polling.enable": True,
            "schema.polling.endpointFilter": "*localhost*",
            "schema.polling.interval": 20000,
            "tracing.hideTracingResponse": True,
            "queryPlan.hideQueryPlanResponse": True,
            "prettier.printWidth": 80,
            "prettier.tabWidth": 2,
            "prettier.useTabs": False
        },
        "tabs": [
            {
                "name": "Welcome",
                "endpoint": endpoint,
                "query": get_welcome_query(),
                "variables": get_welcome_variables()
            },
            {
                "name": "Documents",
                "endpoint": endpoint,
                "query": get_documents_query(),
                "variables": "{}"
            },
            {
                "name": "Analysis",
                "endpoint": endpoint,
                "query": get_analysis_query(),
                "variables": get_analysis_variables()
            },
            {
                "name": "Mutations",
                "endpoint": endpoint,
                "query": get_mutations_query(),
                "variables": get_mutations_variables()
            },
            {
                "name": "Subscriptions",
                "endpoint": subscriptions_endpoint,
                "query": get_subscriptions_query(),
                "variables": "{}"
            }
        ]
    }
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title}</title>
        <link
            rel="stylesheet"
            href="https://cdn.jsdelivr.net/npm/graphql-playground-react@1.7.28/build/static/css/index.css"
        />
        <link
            rel="shortcut icon"
            href="https://cdn.jsdelivr.net/npm/graphql-playground-react@1.7.28/build/favicon.png"
        />
        <script
            src="https://cdn.jsdelivr.net/npm/graphql-playground-react@1.7.28/build/static/js/middleware.js"
        ></script>
        <style>
            body {{
                margin: 0;
                overflow: hidden;
            }}
            
            .playground-wrapper {{
                position: relative;
                height: 100vh;
            }}
            
            .custom-header {{
                background: #1a1a2e;
                color: white;
                padding: 10px 20px;
                font-family: 'Source Code Pro', monospace;
                border-bottom: 1px solid #16213e;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .api-info {{
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}
            
            .api-title {{
                font-size: 18px;
                font-weight: 600;
                margin: 0;
            }}
            
            .api-description {{
                font-size: 12px;
                opacity: 0.8;
                margin: 0;
            }}
            
            .api-version {{
                background: #0f3460;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
            }}
            
            .playground-container {{
                height: calc(100vh - 70px);
            }}
        </style>
    </head>
    <body>
        <div class="playground-wrapper">
            <div class="custom-header">
                <div class="api-info">
                    <h1 class="api-title">{title}</h1>
                    <p class="api-description">
                        Powerful GraphQL API for arbitration clause detection and analysis
                    </p>
                </div>
                <div class="api-version">v1.0.0</div>
            </div>
            <div id="root" class="playground-container">
                <style>
                    body {{
                        background-color: rgb(23, 42, 58);
                        font-family: Open Sans, sans-serif;
                        height: 90vh;
                    }}
                    #root {{
                        height: 100%;
                        width: 100%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-size: 32px;
                    }}
                </style>
                Loading GraphQL Playground...
            </div>
        </div>
        <script>
            window.addEventListener('load', function (event) {{
                GraphQLPlayground.init(document.getElementById('root'), {json.dumps(config, indent=2)});
            }});
        </script>
    </body>
    </html>
    """


def get_welcome_query() -> str:
    """Get welcome query for playground"""
    return '''# Welcome to Arbitration Detection GraphQL API!
# 
# This API provides powerful tools for detecting and analyzing arbitration clauses
# in legal documents using advanced AI and machine learning techniques.
#
# Key Features:
# - Document upload and processing
# - AI-powered arbitration clause detection
# - Real-time analysis results
# - Pattern management for custom detection rules
# - User management and collaboration tools
# - Comprehensive statistics and analytics
#
# Getting Started:
# 1. Explore the schema using the "Docs" panel
# 2. Try the example queries in different tabs
# 3. Upload a document and request analysis
# 4. Subscribe to real-time updates
#
# Example: Get system statistics
query GetSystemOverview {
  systemStats {
    documents {
      totalDocuments
      processedDocuments
      documentsWithArbitration
      processingRate
    }
    detections {
      totalDetections
      highConfidenceDetections
      averageConfidenceScore
    }
    patterns {
      totalPatterns
      activePatterns
      averageEffectiveness
    }
    version
    uptime
  }
}

# Example: Search for documents
query SearchDocuments($query: String!) {
  searchDocuments(query: $query, first: 10) {
    edges {
      node {
        id
        filename
        fileType
        processingStatus
        hasArbitrationClauses
        uploadDate
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
    totalCount
  }
}'''


def get_welcome_variables() -> str:
    """Get welcome query variables"""
    return '''{
  "query": "terms of service"
}'''


def get_documents_query() -> str:
    """Get documents query for playground"""
    return '''# Document Management Queries
# 
# These queries help you manage and explore documents in the system

# Get paginated list of documents
query GetDocuments($first: Int, $after: String, $filter: DocumentFilter) {
  documents(first: $first, after: $after, filter: $filter) {
    edges {
      node {
        id
        filename
        fileType
        fileSize
        processingStatus
        uploadDate
        lastProcessed
        isProcessed
        totalChunks
        detectionCount
        hasArbitrationClauses
        averageConfidenceScore
      }
      cursor
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
    totalCount
  }
}

# Get document with detailed analysis
query GetDocumentDetails($id: ID!) {
  document(id: $id) {
    id
    filename
    fileType
    fileSize
    processingStatus
    uploadDate
    lastProcessed
    metadata
    
    # Get document chunks
    chunks(first: 5) {
      edges {
        node {
          id
          chunkIndex
          content
          pageNumber
          sectionTitle
          hasDetections
        }
      }
    }
    
    # Get detections found
    detections(first: 10) {
      edges {
        node {
          id
          detectionType
          confidenceScore
          matchedText
          contextSnippet
          isHighConfidence
          isValidated
        }
      }
    }
    
    # Get analyses performed
    analyses(first: 5) {
      edges {
        node {
          id
          hasArbitrationClause
          confidenceScore
          analysisSummary
          analyzedAt
          analysisVersion
          riskLevel
          clauseCount
        }
      }
    }
  }
}

# Search documents by content
query SearchDocuments($query: String!, $filter: DocumentFilter) {
  searchDocuments(query: $query, first: 20, filter: $filter) {
    edges {
      node {
        id
        filename
        fileType
        hasArbitrationClauses
        averageConfidenceScore
        uploadDate
      }
    }
    totalCount
  }
}'''


def get_analysis_query() -> str:
    """Get analysis query for playground"""
    return '''# Analysis Queries
# 
# These queries help you explore analysis results and patterns

# Get recent analyses
query GetRecentAnalyses($first: Int) {
  analyses(first: $first, orderBy: "analyzedAt", orderDirection: "DESC") {
    edges {
      node {
        id
        documentId
        hasArbitrationClause
        confidenceScore
        analysisSummary
        analyzedAt
        analysisVersion
        processingTimeMs
        riskLevel
        clauseCount
        averageClauseScore
        
        # Get associated document
        document {
          id
          filename
          fileType
        }
        
        # Get found clauses
        clauses(first: 5) {
          edges {
            node {
              id
              clauseText
              clauseType
              relevanceScore
              severityScore
              riskLevel
              isBinding
            }
          }
        }
      }
    }
  }
}

# Get detections with filtering
query GetDetections($filter: DetectionFilter) {
  detections(first: 20, filter: $filter) {
    edges {
      node {
        id
        documentId
        chunkId
        detectionType
        confidenceScore
        matchedText
        contextBefore
        contextAfter
        startPosition
        endPosition
        pageNumber
        detectionMethod
        isValidated
        isHighConfidence
        severity
        
        # Get associated chunk
        chunk {
          id
          content
          pageNumber
          sectionTitle
        }
        
        # Get associated pattern (if any)
        pattern {
          id
          patternName
          patternType
          category
          effectivenessScore
        }
      }
    }
  }
}

# Get patterns for detection
query GetPatterns($isActive: Boolean) {
  patterns(first: 50, isActive: $isActive) {
    edges {
      node {
        id
        patternName
        patternText
        patternType
        category
        language
        effectivenessScore
        usageCount
        lastUsed
        isActive
        isEffective
        averageConfidenceScore
        createdBy
      }
    }
  }
}

# Get comprehensive statistics
query GetAnalysisStatistics {
  documentStats {
    totalDocuments
    processedDocuments
    documentsWithArbitration
    averageProcessingTime
    processingRate
  }
  
  detectionStats {
    totalDetections
    highConfidenceDetections
    averageConfidenceScore
    detectionsByType {
      type
      count
      averageConfidence
    }
    detectionsByMethod {
      method
      count
      averageConfidence
    }
  }
}'''


def get_analysis_variables() -> str:
    """Get analysis query variables"""
    return '''{
  "first": 10,
  "filter": {
    "confidenceScore": {
      "min": 0.5
    },
    "isHighConfidence": true
  },
  "isActive": true
}'''


def get_mutations_query() -> str:
    """Get mutations query for playground"""
    return '''# Mutation Examples
# 
# These mutations demonstrate how to create, update, and delete resources

# Upload a new document
mutation UploadDocument($input: DocumentUploadInput!) {
  uploadDocument(input: $input) {
    success
    message
    document {
      id
      filename
      fileType
      fileSize
      processingStatus
      uploadDate
      totalChunks
    }
    errors
  }
}

# Request analysis for a document
mutation RequestAnalysis($input: AnalysisRequestInput!) {
  requestAnalysis(input: $input) {
    success
    message
    analysis {
      id
      documentId
      hasArbitrationClause
      confidenceScore
      analysisSummary
      analyzedAt
      riskLevel
      clauseCount
    }
    processingTimeMs
    errors
  }
}

# Quick analysis of text (no storage)
mutation QuickAnalysis($input: QuickAnalysisInput!) {
  quickAnalysis(input: $input) {
    hasArbitrationClause
    confidenceScore
    summary
    processingTimeMs
    clausesFound {
      text
      type
      confidence
      startPosition
      endPosition
    }
  }
}

# Create a new pattern (admin only)
mutation CreatePattern($input: PatternCreateInput!) {
  createPattern(input: $input) {
    success
    message
    pattern {
      id
      patternName
      patternType
      category
      language
      effectivenessScore
      isActive
    }
    errors
  }
}

# Validate a detection
mutation ValidateDetection($detectionId: ID!, $isValid: Boolean!) {
  validateDetection(detectionId: $detectionId, isValid: $isValid)
}

# User registration
mutation RegisterUser($input: UserCreateInput!) {
  registerUser(input: $input) {
    success
    message
    user {
      id
      username
      email
      fullName
      organization
      isActive
      isVerified
    }
    token
    errors
  }
}

# User login
mutation LoginUser($username: String!, $password: String!) {
  loginUser(username: $username, password: $password) {
    success
    message
    user {
      id
      username
      email
      fullName
      role
      lastLogin
    }
    token
    errors
  }
}'''


def get_mutations_variables() -> str:
    """Get mutations query variables"""
    return '''{
  "input": {
    "filename": "sample_terms_of_service.txt",
    "content": "TERMS OF SERVICE\\n\\nBy using our service, you agree to resolve any disputes through binding arbitration...",
    "fileType": "text/plain",
    "metadata": "{\\"source\\": \\"website\\", \\"category\\": \\"terms_of_service\\"}"
  },
  "analysisInput": {
    "documentId": "1",
    "forceReanalysis": false,
    "analysisOptions": {
      "includeContext": true,
      "confidenceThreshold": 0.5,
      "maxClauses": 50
    }
  },
  "quickAnalysisInput": {
    "text": "Any claim or dispute arising out of or relating to this Agreement shall be resolved through binding arbitration.",
    "includeContext": true
  },
  "patternInput": {
    "patternName": "Binding Arbitration Clause",
    "patternText": "binding arbitration",
    "patternType": "KEYWORD",
    "category": "arbitration",
    "language": "en"
  },
  "userInput": {
    "email": "user@example.com",
    "username": "testuser",
    "password": "SecurePass123!",
    "fullName": "Test User",
    "organization": "Example Corp"
  },
  "detectionId": "1",
  "isValid": true,
  "username": "testuser",
  "password": "SecurePass123!"
}'''


def get_subscriptions_query() -> str:
    """Get subscriptions query for playground"""
    return '''# Real-time Subscriptions
# 
# These subscriptions provide real-time updates for various operations

# Subscribe to document processing updates
subscription DocumentProcessing($documentId: ID) {
  documentProcessing(documentId: $documentId) {
    documentId
    status
    progress
    message
    errorMessage
  }
}

# Subscribe to analysis progress
subscription AnalysisProgress($documentId: ID) {
  analysisProgress(documentId: $documentId) {
    analysisId
    documentId
    status
    progress
    results {
      id
      hasArbitrationClause
      confidenceScore
      riskLevel
    }
  }
}

# Subscribe to document comments (collaboration)
subscription DocumentComments($documentId: ID!) {
  documentComments(documentId: $documentId) {
    documentId
    comment {
      id
      content
      isResolved
      createdAt
      user {
        username
        fullName
      }
    }
    action
  }
}

# Subscribe to collaboration updates
subscription DocumentCollaboration($documentId: ID!) {
  documentCollaboration(documentId: $documentId) {
    documentId
    userId
    action
    data
  }
}

# Subscribe to system statistics (admin only)
subscription SystemStats {
  systemStats {
    documents {
      totalDocuments
      processedDocuments
      documentsWithArbitration
    }
    detections {
      totalDetections
      highConfidenceDetections
      averageConfidenceScore
    }
    version
    uptime
  }
}

# Note: To test subscriptions:
# 1. Start the subscription in this tab
# 2. In another tab, upload a document or request analysis
# 3. Watch for real-time updates in this tab'''


def create_schema_documentation() -> str:
    """Create interactive schema documentation"""
    return '''
    # GraphQL Schema Documentation
    
    ## Overview
    The Arbitration Detection GraphQL API provides a comprehensive interface for:
    - Document management and processing
    - AI-powered arbitration clause detection
    - Pattern-based analysis rules
    - User management and collaboration
    - Real-time updates via subscriptions
    
    ## Authentication
    Most operations require authentication via Bearer tokens:
    ```
    Authorization: Bearer <your-jwt-token>
    ```
    
    ## Rate Limiting
    API requests are rate-limited by operation type:
    - Queries: 100 requests/minute
    - Mutations: 20 requests/minute
    - Analysis operations: 10 requests/5 minutes
    - Uploads: 5 requests/minute
    
    ## Complexity Analysis
    Query complexity is automatically analyzed and limited to prevent resource exhaustion.
    
    ## Error Handling
    All mutations return standardized error responses with success flags and error lists.
    '''