"""
GraphQL Client Code Generator
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ClientLanguage(Enum):
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"


@dataclass
class GeneratorConfig:
    """Configuration for code generation"""
    language: ClientLanguage
    output_dir: str
    api_endpoint: str
    include_fragments: bool = True
    include_mutations: bool = True
    include_subscriptions: bool = True
    use_hooks: bool = True  # For React
    use_composition_api: bool = True  # For Vue
    generate_types: bool = True
    generate_queries: bool = True
    generate_tests: bool = False


class ClientGenerator:
    """Generate GraphQL client code for various languages/frameworks"""
    
    def __init__(self, schema_introspection: Dict[str, Any], config: GeneratorConfig):
        self.schema = schema_introspection
        self.config = config
        self.types = schema_introspection.get("__schema", {}).get("types", [])
        
    def generate_client(self) -> Dict[str, str]:
        """Generate client code files"""
        files = {}
        
        if self.config.language == ClientLanguage.TYPESCRIPT:
            files.update(self._generate_typescript_client())
        elif self.config.language == ClientLanguage.JAVASCRIPT:
            files.update(self._generate_javascript_client())
        elif self.config.language == ClientLanguage.PYTHON:
            files.update(self._generate_python_client())
        elif self.config.language == ClientLanguage.REACT:
            files.update(self._generate_react_client())
        elif self.config.language == ClientLanguage.VUE:
            files.update(self._generate_vue_client())
        elif self.config.language == ClientLanguage.ANGULAR:
            files.update(self._generate_angular_client())
        
        return files
    
    def _generate_typescript_client(self) -> Dict[str, str]:
        """Generate TypeScript client code"""
        files = {}
        
        # Generate types
        if self.config.generate_types:
            files["types.ts"] = self._generate_typescript_types()
        
        # Generate client class
        files["client.ts"] = self._generate_typescript_client_class()
        
        # Generate query hooks (if enabled)
        if self.config.use_hooks:
            files["hooks.ts"] = self._generate_typescript_hooks()
        
        # Generate fragments
        if self.config.include_fragments:
            files["fragments.ts"] = self._generate_typescript_fragments()
        
        # Generate queries
        if self.config.generate_queries:
            files["queries.ts"] = self._generate_typescript_queries()
        
        # Generate mutations
        if self.config.include_mutations:
            files["mutations.ts"] = self._generate_typescript_mutations()
        
        # Generate subscriptions
        if self.config.include_subscriptions:
            files["subscriptions.ts"] = self._generate_typescript_subscriptions()
        
        return files
    
    def _generate_typescript_types(self) -> str:
        """Generate TypeScript type definitions"""
        lines = []
        lines.append("// Generated GraphQL Types")
        lines.append("// DO NOT EDIT - This file is auto-generated")
        lines.append("")
        
        # Generate scalar types
        lines.append("// Scalar types")
        lines.append("export type DateTime = string;")
        lines.append("export type JSON = any;")
        lines.append("export type Upload = File;")
        lines.append("")
        
        # Generate enums
        enum_types = [t for t in self.types if t.get("kind") == "ENUM"]
        if enum_types:
            lines.append("// Enums")
            for enum_type in enum_types:
                lines.extend(self._generate_typescript_enum(enum_type))
            lines.append("")
        
        # Generate input types
        input_types = [t for t in self.types if t.get("kind") == "INPUT_OBJECT"]
        if input_types:
            lines.append("// Input types")
            for input_type in input_types:
                lines.extend(self._generate_typescript_input_type(input_type))
            lines.append("")
        
        # Generate object types
        object_types = [t for t in self.types if t.get("kind") == "OBJECT" and not t.get("name", "").startswith("__")]
        if object_types:
            lines.append("// Object types")
            for object_type in object_types:
                lines.extend(self._generate_typescript_object_type(object_type))
            lines.append("")
        
        return "\\n".join(lines)
    
    def _generate_typescript_enum(self, enum_type: Dict[str, Any]) -> List[str]:
        """Generate TypeScript enum"""
        lines = []
        name = enum_type.get("name", "")
        description = enum_type.get("description", "")
        
        if description:
            lines.append(f"/** {description} */")
        
        lines.append(f"export enum {name} {{")
        
        enum_values = enum_type.get("enumValues", [])
        for i, value in enumerate(enum_values):
            value_name = value.get("name", "")
            value_desc = value.get("description", "")
            
            if value_desc:
                lines.append(f"  /** {value_desc} */")
            
            comma = "," if i < len(enum_values) - 1 else ""
            lines.append(f"  {value_name} = '{value_name}'{comma}")
        
        lines.append("}")
        lines.append("")
        
        return lines
    
    def _generate_typescript_input_type(self, input_type: Dict[str, Any]) -> List[str]:
        """Generate TypeScript input interface"""
        lines = []
        name = input_type.get("name", "")
        description = input_type.get("description", "")
        
        if description:
            lines.append(f"/** {description} */")
        
        lines.append(f"export interface {name} {{")
        
        input_fields = input_type.get("inputFields", [])
        for field in input_fields:
            field_name = field.get("name", "")
            field_type = self._format_typescript_type(field.get("type", {}))
            field_desc = field.get("description", "")
            
            if field_desc:
                lines.append(f"  /** {field_desc} */")
            
            lines.append(f"  {field_name}: {field_type};")
        
        lines.append("}")
        lines.append("")
        
        return lines
    
    def _generate_typescript_object_type(self, object_type: Dict[str, Any]) -> List[str]:
        """Generate TypeScript object interface"""
        lines = []
        name = object_type.get("name", "")
        description = object_type.get("description", "")
        
        if description:
            lines.append(f"/** {description} */")
        
        lines.append(f"export interface {name} {{")
        
        fields = object_type.get("fields", [])
        for field in fields:
            field_name = field.get("name", "")
            field_type = self._format_typescript_type(field.get("type", {}))
            field_desc = field.get("description", "")
            
            if field_desc:
                lines.append(f"  /** {field_desc} */")
            
            lines.append(f"  {field_name}: {field_type};")
        
        lines.append("}")
        lines.append("")
        
        return lines
    
    def _format_typescript_type(self, type_ref: Dict[str, Any]) -> str:
        """Format GraphQL type reference as TypeScript type"""
        if not type_ref:
            return "unknown"
        
        kind = type_ref.get("kind", "")
        
        if kind == "NON_NULL":
            inner_type = self._format_typescript_type(type_ref.get("ofType", {}))
            return inner_type  # Remove undefined from union
        elif kind == "LIST":
            inner_type = self._format_typescript_type(type_ref.get("ofType", {}))
            return f"Array<{inner_type}>"
        elif kind in ["SCALAR", "OBJECT", "INTERFACE", "UNION", "ENUM", "INPUT_OBJECT"]:
            type_name = type_ref.get("name", "")
            
            # Map GraphQL scalars to TypeScript types
            scalar_map = {
                "String": "string",
                "Int": "number",
                "Float": "number",
                "Boolean": "boolean",
                "ID": "string",
                "DateTime": "DateTime",
                "JSON": "JSON",
                "Upload": "Upload"
            }
            
            return scalar_map.get(type_name, type_name)
        else:
            return "unknown"
    
    def _generate_typescript_client_class(self) -> str:
        """Generate TypeScript GraphQL client class"""
        return f'''// Generated GraphQL Client
// DO NOT EDIT - This file is auto-generated

import {{ gql, GraphQLClient }} from 'graphql-request';
import * as Types from './types';

export class ArbitrationDetectionClient {{
  private client: GraphQLClient;

  constructor(endpoint: string = '{self.config.api_endpoint}', headers?: Record<string, string>) {{
    this.client = new GraphQLClient(endpoint, {{ headers }});
  }}

  setAuthToken(token: string) {{
    this.client.setHeader('Authorization', `Bearer ${{token}}`);
  }}

  async query<T = any>(query: string, variables?: any): Promise<T> {{
    return this.client.request<T>(query, variables);
  }}

  async mutation<T = any>(mutation: string, variables?: any): Promise<T> {{
    return this.client.request<T>(mutation, variables);
  }}

  // Document operations
  async getDocuments(variables?: Types.DocumentFilter): Promise<Types.DocumentConnection> {{
    const query = gql`
      query GetDocuments($filter: DocumentFilter, $first: Int, $after: String) {{
        documents(filter: $filter, first: $first, after: $after) {{
          edges {{
            node {{
              id
              filename
              fileType
              fileSize
              processingStatus
              uploadDate
              hasArbitrationClauses
            }}
          }}
          pageInfo {{
            hasNextPage
            endCursor
          }}
          totalCount
        }}
      }}
    `;
    
    return this.query(query, variables);
  }}

  async uploadDocument(input: Types.DocumentUploadInput): Promise<Types.DocumentUploadResult> {{
    const mutation = gql`
      mutation UploadDocument($input: DocumentUploadInput!) {{
        uploadDocument(input: $input) {{
          success
          message
          document {{
            id
            filename
            fileType
            processingStatus
          }}
          errors
        }}
      }}
    `;
    
    return this.mutation(mutation, {{ input }});
  }}

  async requestAnalysis(input: Types.AnalysisRequestInput): Promise<Types.AnalysisResult> {{
    const mutation = gql`
      mutation RequestAnalysis($input: AnalysisRequestInput!) {{
        requestAnalysis(input: $input) {{
          success
          message
          analysis {{
            id
            hasArbitrationClause
            confidenceScore
            riskLevel
          }}
          errors
        }}
      }}
    `;
    
    return this.mutation(mutation, {{ input }});
  }}
}}

export default ArbitrationDetectionClient;'''
    
    def _generate_typescript_hooks(self) -> str:
        """Generate React hooks for GraphQL operations"""
        return '''// Generated React Hooks for GraphQL
// DO NOT EDIT - This file is auto-generated

import { useQuery, useMutation, useSubscription, UseQueryResult, UseMutationResult } from '@apollo/client';
import { gql } from '@apollo/client';
import * as Types from './types';

// Document hooks
export function useDocuments(
  variables?: { filter?: Types.DocumentFilter; first?: number; after?: string }
): UseQueryResult<{ documents: Types.DocumentConnection }> {
  const query = gql`
    query GetDocuments($filter: DocumentFilter, $first: Int, $after: String) {
      documents(filter: $filter, first: $first, after: $after) {
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
        pageInfo {
          hasNextPage
          endCursor
        }
        totalCount
      }
    }
  `;
  
  return useQuery(query, { variables });
}

export function useUploadDocument(): UseMutationResult<
  { uploadDocument: Types.DocumentUploadResult },
  { input: Types.DocumentUploadInput }
> {
  const mutation = gql`
    mutation UploadDocument($input: DocumentUploadInput!) {
      uploadDocument(input: $input) {
        success
        message
        document {
          id
          filename
          fileType
          processingStatus
        }
        errors
      }
    }
  `;
  
  return useMutation(mutation);
}

export function useRequestAnalysis(): UseMutationResult<
  { requestAnalysis: Types.AnalysisResult },
  { input: Types.AnalysisRequestInput }
> {
  const mutation = gql`
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
        errors
      }
    }
  `;
  
  return useMutation(mutation);
}

// Subscription hooks
export function useDocumentProcessing(documentId?: string) {
  const subscription = gql`
    subscription DocumentProcessing($documentId: ID) {
      documentProcessing(documentId: $documentId) {
        documentId
        status
        progress
        message
      }
    }
  `;
  
  return useSubscription(subscription, {
    variables: { documentId },
    skip: !documentId
  });
}'''
    
    def _generate_typescript_fragments(self) -> str:
        """Generate TypeScript GraphQL fragments"""
        return '''// Generated GraphQL Fragments
// DO NOT EDIT - This file is auto-generated

import { gql } from '@apollo/client';

export const DOCUMENT_FRAGMENT = gql`
  fragment DocumentFragment on Document {
    id
    filename
    fileType
    fileSize
    processingStatus
    uploadDate
    lastProcessed
    totalChunks
    hasArbitrationClauses
    averageConfidenceScore
  }
`;

export const ANALYSIS_FRAGMENT = gql`
  fragment AnalysisFragment on ArbitrationAnalysis {
    id
    documentId
    hasArbitrationClause
    confidenceScore
    analysisSummary
    analyzedAt
    analysisVersion
    riskLevel
    clauseCount
  }
`;

export const DETECTION_FRAGMENT = gql`
  fragment DetectionFragment on Detection {
    id
    documentId
    detectionType
    confidenceScore
    matchedText
    isHighConfidence
    isValidated
    severity
  }
`;

export const CLAUSE_FRAGMENT = gql`
  fragment ClauseFragment on ArbitrationClause {
    id
    clauseText
    clauseType
    relevanceScore
    severityScore
    riskLevel
    isBinding
  }
`;

export const USER_FRAGMENT = gql`
  fragment UserFragment on User {
    id
    username
    email
    fullName
    organization
    role
    isActive
    isVerified
  }
`;'''
    
    def _generate_typescript_queries(self) -> str:
        """Generate TypeScript GraphQL queries"""
        return '''// Generated GraphQL Queries
// DO NOT EDIT - This file is auto-generated

import { gql } from '@apollo/client';
import { DOCUMENT_FRAGMENT, ANALYSIS_FRAGMENT, DETECTION_FRAGMENT } from './fragments';

export const GET_DOCUMENTS = gql`
  ${DOCUMENT_FRAGMENT}
  query GetDocuments($filter: DocumentFilter, $first: Int, $after: String) {
    documents(filter: $filter, first: $first, after: $after) {
      edges {
        node {
          ...DocumentFragment
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
      totalCount
    }
  }
`;

export const GET_DOCUMENT = gql`
  ${DOCUMENT_FRAGMENT}
  ${ANALYSIS_FRAGMENT}
  ${DETECTION_FRAGMENT}
  query GetDocument($id: ID!) {
    document(id: $id) {
      ...DocumentFragment
      
      analyses(first: 5) {
        edges {
          node {
            ...AnalysisFragment
          }
        }
      }
      
      detections(first: 20) {
        edges {
          node {
            ...DetectionFragment
          }
        }
      }
    }
  }
`;

export const SEARCH_DOCUMENTS = gql`
  ${DOCUMENT_FRAGMENT}
  query SearchDocuments($query: String!, $filter: DocumentFilter) {
    searchDocuments(query: $query, first: 20, filter: $filter) {
      edges {
        node {
          ...DocumentFragment
        }
      }
      totalCount
    }
  }
`;

export const GET_SYSTEM_STATS = gql`
  query GetSystemStats {
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
      version
      uptime
    }
  }
`;'''
    
    def _generate_typescript_mutations(self) -> str:
        """Generate TypeScript GraphQL mutations"""
        return '''// Generated GraphQL Mutations
// DO NOT EDIT - This file is auto-generated

import { gql } from '@apollo/client';
import { DOCUMENT_FRAGMENT, ANALYSIS_FRAGMENT } from './fragments';

export const UPLOAD_DOCUMENT = gql`
  ${DOCUMENT_FRAGMENT}
  mutation UploadDocument($input: DocumentUploadInput!) {
    uploadDocument(input: $input) {
      success
      message
      document {
        ...DocumentFragment
      }
      errors
    }
  }
`;

export const REQUEST_ANALYSIS = gql`
  ${ANALYSIS_FRAGMENT}
  mutation RequestAnalysis($input: AnalysisRequestInput!) {
    requestAnalysis(input: $input) {
      success
      message
      analysis {
        ...AnalysisFragment
      }
      processingTimeMs
      errors
    }
  }
`;

export const QUICK_ANALYSIS = gql`
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
`;

export const VALIDATE_DETECTION = gql`
  mutation ValidateDetection($detectionId: ID!, $isValid: Boolean!) {
    validateDetection(detectionId: $detectionId, isValid: $isValid)
  }
`;

export const REGISTER_USER = gql`
  mutation RegisterUser($input: UserCreateInput!) {
    registerUser(input: $input) {
      success
      message
      user {
        id
        username
        email
        fullName
      }
      token
      errors
    }
  }
`;

export const LOGIN_USER = gql`
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
      }
      token
      errors
    }
  }
`;'''
    
    def _generate_typescript_subscriptions(self) -> str:
        """Generate TypeScript GraphQL subscriptions"""
        return '''// Generated GraphQL Subscriptions
// DO NOT EDIT - This file is auto-generated

import { gql } from '@apollo/client';

export const DOCUMENT_PROCESSING_SUBSCRIPTION = gql`
  subscription DocumentProcessing($documentId: ID) {
    documentProcessing(documentId: $documentId) {
      documentId
      status
      progress
      message
      errorMessage
    }
  }
`;

export const ANALYSIS_PROGRESS_SUBSCRIPTION = gql`
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
`;

export const DOCUMENT_COLLABORATION_SUBSCRIPTION = gql`
  subscription DocumentCollaboration($documentId: ID!) {
    documentCollaboration(documentId: $documentId) {
      documentId
      userId
      action
      data
    }
  }
`;

export const SYSTEM_STATS_SUBSCRIPTION = gql`
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
      uptime
      version
    }
  }
`;'''
    
    def _generate_python_client(self) -> Dict[str, str]:
        """Generate Python client code"""
        files = {}
        
        files["client.py"] = '''"""
Generated GraphQL Client for Python
DO NOT EDIT - This file is auto-generated
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json


@dataclass
class GraphQLError:
    """GraphQL error representation"""
    message: str
    locations: Optional[List[Dict[str, int]]] = None
    path: Optional[List[str]] = None
    extensions: Optional[Dict[str, Any]] = None


class GraphQLResponse:
    """GraphQL response wrapper"""
    
    def __init__(self, data: Optional[Dict[str, Any]], errors: Optional[List[Dict[str, Any]]] = None):
        self.data = data
        self.errors = [GraphQLError(**error) for error in (errors or [])]
        self.has_errors = len(self.errors) > 0


class ArbitrationDetectionClient:
    """Python GraphQL client for Arbitration Detection API"""
    
    def __init__(self, endpoint: str = "''' + self.config.api_endpoint + '''", headers: Optional[Dict[str, str]] = None):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def set_auth_token(self, token: str):
        """Set authentication token"""
        self.headers["Authorization"] = f"Bearer {token}"
    
    async def execute(self, query: str, variables: Optional[Dict[str, Any]] = None) -> GraphQLResponse:
        """Execute GraphQL query"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        async with self.session.post(
            self.endpoint,
            json=payload,
            headers=self.headers
        ) as response:
            result = await response.json()
            return GraphQLResponse(
                data=result.get("data"),
                errors=result.get("errors")
            )
    
    # Document operations
    async def get_documents(self, filter_params: Optional[Dict[str, Any]] = None) -> GraphQLResponse:
        """Get paginated list of documents"""
        query = """
        query GetDocuments($filter: DocumentFilter, $first: Int, $after: String) {
            documents(filter: $filter, first: $first, after: $after) {
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
                pageInfo {
                    hasNextPage
                    endCursor
                }
                totalCount
            }
        }
        """
        
        variables = {
            "filter": filter_params,
            "first": 20
        }
        
        return await self.execute(query, variables)
    
    async def upload_document(self, filename: str, content: str, file_type: str = "text/plain") -> GraphQLResponse:
        """Upload a new document"""
        mutation = """
        mutation UploadDocument($input: DocumentUploadInput!) {
            uploadDocument(input: $input) {
                success
                message
                document {
                    id
                    filename
                    fileType
                    processingStatus
                }
                errors
            }
        }
        """
        
        variables = {
            "input": {
                "filename": filename,
                "content": content,
                "fileType": file_type
            }
        }
        
        return await self.execute(mutation, variables)
    
    async def request_analysis(self, document_id: str, force_reanalysis: bool = False) -> GraphQLResponse:
        """Request analysis for a document"""
        mutation = """
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
                errors
            }
        }
        """
        
        variables = {
            "input": {
                "documentId": document_id,
                "forceReanalysis": force_reanalysis
            }
        }
        
        return await self.execute(mutation, variables)
    
    async def quick_analysis(self, text: str) -> GraphQLResponse:
        """Perform quick analysis on text"""
        mutation = """
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
        """
        
        variables = {
            "input": {
                "text": text,
                "includeContext": True
            }
        }
        
        return await self.execute(mutation, variables)


# Example usage
async def main():
    async with ArbitrationDetectionClient() as client:
        # Set authentication token
        client.set_auth_token("your-jwt-token")
        
        # Get documents
        response = await client.get_documents()
        if not response.has_errors:
            documents = response.data["documents"]["edges"]
            print(f"Found {len(documents)} documents")
        
        # Upload document
        response = await client.upload_document(
            filename="terms.txt",
            content="Any disputes will be resolved through binding arbitration."
        )
        
        if response.data["uploadDocument"]["success"]:
            document_id = response.data["uploadDocument"]["document"]["id"]
            
            # Request analysis
            analysis_response = await client.request_analysis(document_id)
            print("Analysis requested:", analysis_response.data)


if __name__ == "__main__":
    asyncio.run(main())
'''
        
        return files
    
    def _generate_react_client(self) -> Dict[str, str]:
        """Generate React-specific client code"""
        files = self._generate_typescript_client()
        
        # Add React-specific components
        files["components/DocumentUpload.tsx"] = '''// Generated React Component
// DO NOT EDIT - This file is auto-generated

import React, { useState } from 'react';
import { useUploadDocument } from '../hooks';
import * as Types from '../types';

interface DocumentUploadProps {
  onUploadSuccess?: (document: Types.Document) => void;
  onUploadError?: (errors: string[]) => void;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadSuccess,
  onUploadError
}) => {
  const [filename, setFilename] = useState('');
  const [content, setContent] = useState('');
  const [fileType, setFileType] = useState('text/plain');
  
  const [uploadDocument, { loading, error }] = useUploadDocument({
    onCompleted: (data) => {
      if (data.uploadDocument.success && data.uploadDocument.document) {
        onUploadSuccess?.(data.uploadDocument.document);
        setFilename('');
        setContent('');
      } else {
        onUploadError?.(data.uploadDocument.errors);
      }
    },
    onError: (error) => {
      onUploadError?.([error.message]);
    }
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    await uploadDocument({
      variables: {
        input: {
          filename,
          content,
          fileType
        }
      }
    });
  };

  return (
    <form onSubmit={handleSubmit} className="document-upload">
      <div className="form-group">
        <label htmlFor="filename">Filename:</label>
        <input
          id="filename"
          type="text"
          value={filename}
          onChange={(e) => setFilename(e.target.value)}
          required
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="fileType">File Type:</label>
        <select
          id="fileType"
          value={fileType}
          onChange={(e) => setFileType(e.target.value)}
        >
          <option value="text/plain">Text</option>
          <option value="application/pdf">PDF</option>
          <option value="application/vnd.openxmlformats-officedocument.wordprocessingml.document">
            Word Document
          </option>
        </select>
      </div>
      
      <div className="form-group">
        <label htmlFor="content">Content:</label>
        <textarea
          id="content"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={10}
          required
        />
      </div>
      
      <button type="submit" disabled={loading}>
        {loading ? 'Uploading...' : 'Upload Document'}
      </button>
      
      {error && (
        <div className="error">
          Error: {error.message}
        </div>
      )}
    </form>
  );
};'''
        
        return files
    
    def _generate_vue_client(self) -> Dict[str, str]:
        """Generate Vue-specific client code"""
        files = {}
        
        files["composables/useGraphQL.ts"] = '''// Generated Vue Composables
// DO NOT EDIT - This file is auto-generated

import { ref, computed } from 'vue';
import { useQuery, useMutation } from '@vue/apollo-composable';
import { gql } from '@apollo/client/core';
import * as Types from '../types';

export function useDocuments(variables?: { filter?: Types.DocumentFilter }) {
  const { result, loading, error, refetch } = useQuery(
    gql`
      query GetDocuments($filter: DocumentFilter, $first: Int) {
        documents(filter: $filter, first: $first) {
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
      }
    `,
    variables
  );

  const documents = computed(() => result.value?.documents?.edges?.map(edge => edge.node) || []);
  const totalCount = computed(() => result.value?.documents?.totalCount || 0);

  return {
    documents,
    totalCount,
    loading,
    error,
    refetch
  };
}

export function useUploadDocument() {
  const { mutate: uploadDocument, loading, error } = useMutation(
    gql`
      mutation UploadDocument($input: DocumentUploadInput!) {
        uploadDocument(input: $input) {
          success
          message
          document {
            id
            filename
            fileType
            processingStatus
          }
          errors
        }
      }
    `
  );

  const upload = async (input: Types.DocumentUploadInput) => {
    const result = await uploadDocument({ input });
    return result?.data?.uploadDocument;
  };

  return {
    upload,
    loading,
    error
  };
}'''
        
        return files
    
    def _generate_angular_client(self) -> Dict[str, str]:
        """Generate Angular-specific client code"""
        files = {}
        
        files["services/graphql.service.ts"] = '''// Generated Angular Service
// DO NOT EDIT - This file is auto-generated

import { Injectable } from '@angular/core';
import { Apollo } from 'apollo-angular';
import { gql } from '@apollo/client/core';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import * as Types from '../types';

@Injectable({
  providedIn: 'root'
})
export class GraphQLService {
  constructor(private apollo: Apollo) {}

  getDocuments(variables?: { filter?: Types.DocumentFilter }): Observable<Types.Document[]> {
    return this.apollo.query<{ documents: Types.DocumentConnection }>({
      query: gql`
        query GetDocuments($filter: DocumentFilter, $first: Int) {
          documents(filter: $filter, first: $first) {
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
          }
        }
      `,
      variables
    }).pipe(
      map(result => result.data.documents.edges.map(edge => edge.node))
    );
  }

  uploadDocument(input: Types.DocumentUploadInput): Observable<Types.DocumentUploadResult> {
    return this.apollo.mutate<{ uploadDocument: Types.DocumentUploadResult }>({
      mutation: gql`
        mutation UploadDocument($input: DocumentUploadInput!) {
          uploadDocument(input: $input) {
            success
            message
            document {
              id
              filename
              fileType
              processingStatus
            }
            errors
          }
        }
      `,
      variables: { input }
    }).pipe(
      map(result => result.data!.uploadDocument)
    );
  }
}'''
        
        return files
    
    def _generate_javascript_client(self) -> Dict[str, str]:
        """Generate JavaScript client code"""
        files = {}
        
        # Convert TypeScript files to JavaScript
        ts_files = self._generate_typescript_client()
        
        for filename, content in ts_files.items():
            if filename.endswith('.ts'):
                js_filename = filename.replace('.ts', '.js')
                # Simple TS to JS conversion (remove type annotations)
                js_content = self._convert_typescript_to_javascript(content)
                files[js_filename] = js_content
        
        return files
    
    def _convert_typescript_to_javascript(self, ts_content: str) -> str:
        """Convert TypeScript to JavaScript (basic conversion)"""
        lines = ts_content.split('\\n')
        js_lines = []
        
        for line in lines:
            # Remove type annotations (basic)
            line = line.replace(': string', '')
            line = line.replace(': number', '')
            line = line.replace(': boolean', '')
            line = line.replace(': any', '')
            line = line.replace('export interface', '// interface')
            line = line.replace('export enum', 'const')
            
            js_lines.append(line)
        
        return '\\n'.join(js_lines)