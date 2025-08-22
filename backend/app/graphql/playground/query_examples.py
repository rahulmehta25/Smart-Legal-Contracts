"""
GraphQL Query Examples for Documentation and Testing
"""

from typing import Dict, List, Any


def get_example_queries() -> Dict[str, Dict[str, Any]]:
    """Get collection of example queries organized by category"""
    
    return {
        "basic_queries": {
            "title": "Basic Queries",
            "description": "Simple queries to get started with the API",
            "examples": [
                {
                    "name": "Get Documents",
                    "description": "Fetch a list of documents with basic information",
                    "query": '''query GetDocuments {
  documents(first: 10) {
    edges {
      node {
        id
        filename
        fileType
        fileSize
        processingStatus
        uploadDate
        hasArbitrationClauses
      }
    }
    totalCount
  }
}''',
                    "variables": {}
                },
                {
                    "name": "Get Single Document",
                    "description": "Fetch detailed information about a specific document",
                    "query": '''query GetDocument($id: ID!) {
  document(id: $id) {
    id
    filename
    fileType
    fileSize
    processingStatus
    uploadDate
    lastProcessed
    totalChunks
    metadata
    isProcessed
    detectionCount
    hasArbitrationClauses
    averageConfidenceScore
  }
}''',
                    "variables": {"id": "1"}
                },
                {
                    "name": "Get Patterns",
                    "description": "Fetch active detection patterns",
                    "query": '''query GetPatterns {
  patterns(first: 20, isActive: true) {
    edges {
      node {
        id
        patternName
        patternType
        category
        effectivenessScore
        usageCount
        isEffective
      }
    }
  }
}''',
                    "variables": {}
                }
            ]
        },
        
        "analysis_queries": {
            "title": "Analysis Queries",
            "description": "Queries for exploring analysis results and detections",
            "examples": [
                {
                    "name": "Document Analysis Results",
                    "description": "Get comprehensive analysis results for a document",
                    "query": '''query GetDocumentAnalysis($documentId: ID!) {
  document(id: $documentId) {
    id
    filename
    
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
          averageClauseScore
          
          clauses(first: 10) {
            edges {
              node {
                id
                clauseText
                clauseType
                relevanceScore
                severityScore
                riskLevel
                isBinding
                startPosition
                endPosition
              }
            }
          }
        }
      }
    }
    
    detections(first: 20) {
      edges {
        node {
          id
          detectionType
          confidenceScore
          matchedText
          contextSnippet
          isHighConfidence
          isValidated
          severity
          pageNumber
          detectionMethod
        }
      }
    }
  }
}''',
                    "variables": {"documentId": "1"}
                },
                {
                    "name": "High Confidence Detections",
                    "description": "Find all high-confidence arbitration clause detections",
                    "query": '''query GetHighConfidenceDetections {
  detections(
    first: 50
    filter: {
      isHighConfidence: true
      confidenceScore: { min: 0.8 }
    }
  ) {
    edges {
      node {
        id
        documentId
        detectionType
        confidenceScore
        matchedText
        contextBefore
        contextAfter
        isValidated
        
        document {
          id
          filename
          fileType
        }
        
        pattern {
          id
          patternName
          category
        }
      }
    }
    totalCount
  }
}''',
                    "variables": {}
                },
                {
                    "name": "Analysis Statistics",
                    "description": "Get comprehensive analysis statistics",
                    "query": '''query GetAnalysisStats {
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
}''',
                    "variables": {}
                }
            ]
        },
        
        "search_queries": {
            "title": "Search Queries",
            "description": "Advanced search and filtering queries",
            "examples": [
                {
                    "name": "Search Documents by Content",
                    "description": "Search for documents containing specific terms",
                    "query": '''query SearchDocuments($query: String!, $hasArbitration: Boolean) {
  searchDocuments(
    query: $query
    first: 20
    filter: {
      hasArbitrationClauses: $hasArbitration
      processingStatus: COMPLETED
    }
  ) {
    edges {
      node {
        id
        filename
        fileType
        hasArbitrationClauses
        averageConfidenceScore
        uploadDate
        
        detections(first: 3) {
          edges {
            node {
              id
              detectionType
              confidenceScore
              matchedText
            }
          }
        }
      }
    }
    totalCount
  }
}''',
                    "variables": {
                        "query": "arbitration",
                        "hasArbitration": True
                    }
                },
                {
                    "name": "Filter Documents by Date Range",
                    "description": "Find documents uploaded within a specific time period",
                    "query": '''query GetRecentDocuments($startDate: DateTime!, $endDate: DateTime!) {
  documents(
    first: 30
    filter: {
      uploadedAfter: $startDate
      uploadedBefore: $endDate
      processingStatus: COMPLETED
    }
    orderBy: "uploadDate"
    orderDirection: "DESC"
  ) {
    edges {
      node {
        id
        filename
        uploadDate
        processingStatus
        hasArbitrationClauses
        detectionCount
      }
    }
    totalCount
  }
}''',
                    "variables": {
                        "startDate": "2024-01-01T00:00:00Z",
                        "endDate": "2024-12-31T23:59:59Z"
                    }
                },
                {
                    "name": "Search Arbitration Clauses",
                    "description": "Search for specific arbitration clause content",
                    "query": '''query SearchClauses($query: String!) {
  searchClauses(query: $query, first: 20) {
    id
    clauseText
    clauseType
    relevanceScore
    severityScore
    riskLevel
    isBinding
    
    analysis {
      id
      documentId
      confidenceScore
      
      document {
        id
        filename
      }
    }
  }
}''',
                    "variables": {
                        "query": "binding arbitration"
                    }
                }
            ]
        },
        
        "mutation_examples": {
            "title": "Mutation Examples",
            "description": "Examples of creating, updating, and deleting resources",
            "examples": [
                {
                    "name": "Upload Document",
                    "description": "Upload a new document for analysis",
                    "query": '''mutation UploadDocument($input: DocumentUploadInput!) {
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
}''',
                    "variables": {
                        "input": {
                            "filename": "terms_of_service.txt",
                            "content": "TERMS OF SERVICE\\n\\nBy using this service, you agree that any disputes will be resolved through binding arbitration administered by the American Arbitration Association...",
                            "fileType": "text/plain",
                            "metadata": "{\\"source\\": \\"website\\", \\"category\\": \\"legal\\"}"
                        }
                    }
                },
                {
                    "name": "Request Analysis",
                    "description": "Request arbitration analysis for a document",
                    "query": '''mutation RequestAnalysis($input: AnalysisRequestInput!) {
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
      analysisVersion
      riskLevel
      clauseCount
      averageClauseScore
    }
    processingTimeMs
    errors
  }
}''',
                    "variables": {
                        "input": {
                            "documentId": "1",
                            "forceReanalysis": false,
                            "analysisOptions": {
                                "includeContext": True,
                                "confidenceThreshold": 0.5,
                                "maxClauses": 50
                            }
                        }
                    }
                },
                {
                    "name": "Quick Analysis",
                    "description": "Perform quick analysis on text without storing",
                    "query": '''mutation QuickAnalysis($input: QuickAnalysisInput!) {
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
}''',
                    "variables": {
                        "input": {
                            "text": "Any dispute arising under this agreement shall be resolved exclusively through final and binding arbitration before a single arbitrator.",
                            "includeContext": True
                        }
                    }
                },
                {
                    "name": "Create Pattern",
                    "description": "Create a new detection pattern (admin only)",
                    "query": '''mutation CreatePattern($input: PatternCreateInput!) {
  createPattern(input: $input) {
    success
    message
    pattern {
      id
      patternName
      patternText
      patternType
      category
      language
      effectivenessScore
      isActive
      createdBy
    }
    errors
  }
}''',
                    "variables": {
                        "input": {
                            "patternName": "Mandatory Arbitration Clause",
                            "patternText": "mandatory arbitration|compulsory arbitration|required arbitration",
                            "patternType": "REGEX",
                            "category": "arbitration_mandatory",
                            "language": "en"
                        }
                    }
                },
                {
                    "name": "Validate Detection",
                    "description": "Validate or invalidate a detection result",
                    "query": '''mutation ValidateDetection($detectionId: ID!, $isValid: Boolean!) {
  validateDetection(detectionId: $detectionId, isValid: $isValid)
}''',
                    "variables": {
                        "detectionId": "123",
                        "isValid": True
                    }
                }
            ]
        },
        
        "subscription_examples": {
            "title": "Subscription Examples",
            "description": "Real-time subscription examples for live updates",
            "examples": [
                {
                    "name": "Document Processing Updates",
                    "description": "Subscribe to real-time document processing status",
                    "query": '''subscription DocumentProcessing($documentId: ID) {
  documentProcessing(documentId: $documentId) {
    documentId
    status
    progress
    message
    errorMessage
  }
}''',
                    "variables": {
                        "documentId": "1"
                    }
                },
                {
                    "name": "Analysis Progress",
                    "description": "Subscribe to analysis progress updates",
                    "query": '''subscription AnalysisProgress($documentId: ID) {
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
      clauseCount
    }
  }
}''',
                    "variables": {
                        "documentId": "1"
                    }
                },
                {
                    "name": "Document Collaboration",
                    "description": "Subscribe to collaboration updates on a document",
                    "query": '''subscription DocumentCollaboration($documentId: ID!) {
  documentCollaboration(documentId: $documentId) {
    documentId
    userId
    action
    data
  }
}''',
                    "variables": {
                        "documentId": "1"
                    }
                },
                {
                    "name": "System Statistics",
                    "description": "Subscribe to real-time system statistics (admin only)",
                    "query": '''subscription SystemStats {
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
    uptime
    version
  }
}''',
                    "variables": {}
                }
            ]
        },
        
        "user_management": {
            "title": "User Management",
            "description": "User authentication and management queries",
            "examples": [
                {
                    "name": "Register User",
                    "description": "Register a new user account",
                    "query": '''mutation RegisterUser($input: UserCreateInput!) {
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
      createdAt
    }
    token
    errors
  }
}''',
                    "variables": {
                        "input": {
                            "email": "john.doe@example.com",
                            "username": "johndoe",
                            "password": "SecurePassword123!",
                            "fullName": "John Doe",
                            "organization": "Example Corporation"
                        }
                    }
                },
                {
                    "name": "Login User",
                    "description": "Authenticate user and get access token",
                    "query": '''mutation LoginUser($username: String!, $password: String!) {
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
      documentCount
      analysisCount
    }
    token
    errors
  }
}''',
                    "variables": {
                        "username": "johndoe",
                        "password": "SecurePassword123!"
                    }
                },
                {
                    "name": "Get Current User",
                    "description": "Get information about the currently authenticated user",
                    "query": '''query GetCurrentUser {
  currentUser {
    id
    username
    email
    fullName
    organization
    role
    isActive
    isVerified
    lastLogin
    createdAt
    documentCount
    analysisCount
  }
}''',
                    "variables": {}
                },
                {
                    "name": "Get User Documents",
                    "description": "Get documents belonging to the current user",
                    "query": '''query GetUserDocuments($userId: ID!) {
  user(id: $userId) {
    id
    username
    fullName
    
    documents(first: 20) {
      edges {
        node {
          id
          filename
          fileType
          uploadDate
          processingStatus
          hasArbitrationClauses
          detectionCount
        }
      }
      totalCount
    }
    
    analyses(first: 10) {
      edges {
        node {
          id
          hasArbitrationClause
          confidenceScore
          analyzedAt
          riskLevel
          
          document {
            id
            filename
          }
        }
      }
    }
  }
}''',
                    "variables": {
                        "userId": "1"
                    }
                }
            ]
        },
        
        "advanced_queries": {
            "title": "Advanced Queries",
            "description": "Complex queries demonstrating advanced GraphQL features",
            "examples": [
                {
                    "name": "Comprehensive Document Analysis",
                    "description": "Get complete analysis data for documents with nested relationships",
                    "query": '''query ComprehensiveDocumentAnalysis($filter: DocumentFilter!) {
  documents(first: 10, filter: $filter) {
    edges {
      node {
        id
        filename
        fileType
        fileSize
        uploadDate
        processingStatus
        hasArbitrationClauses
        
        # Get document chunks with detections
        chunks(first: 5) {
          edges {
            node {
              id
              chunkIndex
              content
              pageNumber
              sectionTitle
              hasDetections
              
              detections(first: 3) {
                edges {
                  node {
                    id
                    detectionType
                    confidenceScore
                    matchedText
                    isHighConfidence
                    
                    pattern {
                      id
                      patternName
                      category
                    }
                  }
                }
              }
            }
          }
        }
        
        # Get latest analysis with clauses
        analyses(first: 1) {
          edges {
            node {
              id
              hasArbitrationClause
              confidenceScore
              analysisSummary
              riskLevel
              
              clauses(first: 10) {
                edges {
                  node {
                    id
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
    pageInfo {
      hasNextPage
      endCursor
    }
    totalCount
  }
}''',
                    "variables": {
                        "filter": {
                            "hasArbitrationClauses": True,
                            "processingStatus": "COMPLETED"
                        }
                    }
                },
                {
                    "name": "Pattern Effectiveness Analysis",
                    "description": "Analyze pattern effectiveness with detection statistics",
                    "query": '''query PatternEffectivenessAnalysis {
  patterns(first: 50, isActive: true) {
    edges {
      node {
        id
        patternName
        patternType
        category
        effectivenessScore
        usageCount
        lastUsed
        isEffective
        averageConfidenceScore
        
        detections(first: 100) {
          edges {
            node {
              id
              confidenceScore
              isValidated
              detectionMethod
              
              document {
                id
                filename
                fileType
              }
            }
          }
          totalCount
        }
      }
    }
  }
}''',
                    "variables": {}
                },
                {
                    "name": "Risk Assessment Dashboard",
                    "description": "Get data for a comprehensive risk assessment dashboard",
                    "query": '''query RiskAssessmentDashboard {
  # Overall statistics
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
      detectionsByType {
        type
        count
        averageConfidence
      }
    }
  }
  
  # High-risk documents
  documents(
    first: 10
    filter: { hasArbitrationClauses: true }
    orderBy: "averageConfidenceScore"
    orderDirection: "DESC"
  ) {
    edges {
      node {
        id
        filename
        averageConfidenceScore
        detectionCount
        
        analyses(first: 1) {
          edges {
            node {
              id
              riskLevel
              clauseCount
              averageClauseScore
            }
          }
        }
      }
    }
  }
  
  # Recent high-confidence detections
  detections(
    first: 20
    filter: {
      isHighConfidence: true
      confidenceScore: { min: 0.9 }
    }
  ) {
    edges {
      node {
        id
        detectionType
        confidenceScore
        matchedText
        
        document {
          id
          filename
        }
      }
    }
  }
}''',
                    "variables": {}
                }
            ]
        }
    }


def get_query_by_name(category: str, name: str) -> Dict[str, Any]:
    """Get specific query example by category and name"""
    examples = get_example_queries()
    
    if category not in examples:
        return None
    
    category_examples = examples[category].get("examples", [])
    
    for example in category_examples:
        if example.get("name") == name:
            return example
    
    return None


def get_all_query_names() -> List[str]:
    """Get list of all available query names"""
    examples = get_example_queries()
    names = []
    
    for category, category_data in examples.items():
        for example in category_data.get("examples", []):
            names.append(f"{category}.{example.get('name', '')}")
    
    return names


def format_query_for_playground(query: str, variables: Dict[str, Any]) -> str:
    """Format query and variables for GraphQL Playground"""
    return f"""# Query
{query}

# Variables
{json.dumps(variables, indent=2)}"""


def get_learning_path_queries() -> List[Dict[str, Any]]:
    """Get queries arranged in a learning path for new users"""
    return [
        {
            "step": 1,
            "title": "Hello GraphQL",
            "description": "Start with a simple query to understand the basics",
            "category": "basic_queries",
            "query_name": "Get Documents"
        },
        {
            "step": 2,
            "title": "Query with Variables",
            "description": "Learn to use variables in your queries",
            "category": "basic_queries",
            "query_name": "Get Single Document"
        },
        {
            "step": 3,
            "title": "Filtering and Search",
            "description": "Explore filtering and search capabilities",
            "category": "search_queries",
            "query_name": "Search Documents by Content"
        },
        {
            "step": 4,
            "title": "Creating Data",
            "description": "Learn to create resources with mutations",
            "category": "mutation_examples",
            "query_name": "Upload Document"
        },
        {
            "step": 5,
            "title": "Real-time Updates",
            "description": "Subscribe to real-time data changes",
            "category": "subscription_examples",
            "query_name": "Document Processing Updates"
        },
        {
            "step": 6,
            "title": "Complex Relationships",
            "description": "Work with nested data and relationships",
            "category": "analysis_queries",
            "query_name": "Document Analysis Results"
        },
        {
            "step": 7,
            "title": "Advanced Features",
            "description": "Explore advanced GraphQL features",
            "category": "advanced_queries",
            "query_name": "Comprehensive Document Analysis"
        }
    ]