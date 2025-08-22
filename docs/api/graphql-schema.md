# API Marketplace GraphQL Schema

## Overview
This GraphQL schema provides a flexible and powerful way to interact with the API Marketplace, allowing complex querying and mutation capabilities.

## Types

### API Type
```graphql
type API {
  id: ID!
  name: String!
  description: String
  version: String!
  category: APICategory!
  endpoints: [Endpoint!]!
  pricing: PricingTier!
  analytics: APIAnalytics
  documentation: String
}
```

### Endpoint Type
```graphql
type Endpoint {
  path: String!
  method: HTTPMethod!
  description: String
  parameters: [Parameter!]
  responseType: String
}
```

### PricingTier Type
```graphql
type PricingTier {
  tier: TierLevel!
  monthlyRate: Float!
  rateLimit: Int!
  features: [String!]!
}
```

## Enums

### APICategory
```graphql
enum APICategory {
  FINANCE
  COMMUNICATION
  PRODUCTIVITY
  SOCIAL_MEDIA
  UTILITIES
  MACHINE_LEARNING
  OTHER
}
```

### HTTPMethod
```graphql
enum HTTPMethod {
  GET
  POST
  PUT
  DELETE
  PATCH
}
```

### TierLevel
```graphql
enum TierLevel {
  FREE
  BASIC
  PRO
  ENTERPRISE
}
```

## Query Root
```graphql
type Query {
  # Get a list of all APIs
  apis(
    category: APICategory
    searchTerm: String
    minRateLimit: Int
  ): [API!]!

  # Get details of a specific API
  api(id: ID!): API

  # Search APIs with advanced filtering
  searchAPIs(
    name: String
    category: APICategory
    minRating: Float
    maxPrice: Float
  ): [API!]!
}
```

## Mutation Root
```graphql
type Mutation {
  # Register a new API
  registerAPI(input: APIInput!): API!

  # Update an existing API
  updateAPI(id: ID!, input: APIUpdateInput!): API!

  # Delete an API
  deleteAPI(id: ID!): Boolean!

  # Purchase API access
  purchaseAPIAccess(
    apiId: ID!
    tierLevel: TierLevel!
  ): PurchaseReceipt!
}
```

## Input Types

### APIInput
```graphql
input APIInput {
  name: String!
  description: String
  category: APICategory!
  endpoints: [EndpointInput!]!
  pricing: PricingTierInput!
}
```

## Subscriptions
```graphql
type Subscription {
  # Real-time API marketplace events
  marketplaceEvents: MarketplaceEvent!
}
```

## Authorization
Authentication is handled via OAuth 2.0. Include the `Authorization` header with a Bearer token for all requests.

## Rate Limiting
- Free Tier: 100 requests/day
- Basic Tier: 1,000 requests/day
- Pro Tier: 10,000 requests/day
- Enterprise Tier: Unlimited requests