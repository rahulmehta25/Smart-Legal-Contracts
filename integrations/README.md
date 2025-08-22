# Partner Integration Platform

A comprehensive enterprise-grade partner integration ecosystem built with Spring Boot, Apache Camel, and Apache Kafka.

## Overview

The Partner Integration Platform provides seamless connectivity and data synchronization between multiple partner systems including Salesforce, Microsoft 365, Google Workspace, DocuSign, Slack, Box, SAP, and NetSuite.

## Features

### Core Integration Components
- **Partner Connectors**: Pre-built connectors for major SaaS platforms
- **Data Transformers**: Apache Camel-based ETL pipelines for data transformation
- **Workflow Orchestration**: Complex integration workflow management
- **Event Streaming**: Kafka-based real-time event processing
- **API Gateway**: Spring Cloud Gateway with rate limiting and circuit breakers
- **Monitoring & Health Checks**: Comprehensive system monitoring and alerting

### Supported Partners
- **Salesforce CRM**: Complete CRUD operations, bulk API support, real-time sync
- **Microsoft 365**: Graph API integration, Office suite connectivity
- **Google Workspace**: Admin SDK integration, user/group management  
- **DocuSign**: E-signature workflow integration
- **Slack**: Bot integration and notification management
- **Box**: Document management and file synchronization
- **SAP**: Enterprise system integration
- **NetSuite**: ERP and financial data synchronization

### Key Capabilities
- **Real-time Data Synchronization**: Bi-directional data sync with conflict resolution
- **Event-Driven Architecture**: Kafka-based event streaming and processing
- **Circuit Breaker Patterns**: Resilience4j integration for fault tolerance
- **Rate Limiting**: Per-partner rate limiting and throttling
- **Data Transformation**: Schema mapping and format conversion
- **Workflow Orchestration**: Complex multi-step integration processes
- **Monitoring & Observability**: Prometheus metrics, Grafana dashboards
- **Security**: OAuth2/JWT authentication, API key management

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  Orchestration  │────│   Connectors    │
│  (Rate Limit,   │    │   (Workflows)   │    │   (Partners)    │
│ Circuit Breaker)│    └─────────────────┘    └─────────────────┘
└─────────────────┘             │                       │
         │                      │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Event Streaming │    │  Transformers   │    │   Monitoring    │
│    (Kafka)      │    │  (Apache Camel) │    │ (Health Checks) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites
- Java 21 or higher
- Docker and Docker Compose
- Maven 3.8+

### Running with Docker Compose

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd integrations
   ```

2. **Start the infrastructure**
   ```bash
   docker-compose up -d postgres kafka redis
   ```

3. **Build and start the application**
   ```bash
   docker-compose up --build integration-platform
   ```

4. **Access the services**
   - Integration Platform: http://localhost:8080/integration-platform
   - Kafka UI: http://localhost:8090
   - Grafana: http://localhost:3000 (admin/integration_admin)
   - Prometheus: http://localhost:9090
   - Jaeger: http://localhost:16686

### Local Development

1. **Start infrastructure services**
   ```bash
   docker-compose up -d postgres kafka redis
   ```

2. **Run the application**
   ```bash
   ./mvnw spring-boot:run -Dspring-boot.run.profiles=local
   ```

## Configuration

### Environment Variables

Configure partner credentials using environment variables:

```bash
# Salesforce
export SALESFORCE_CLIENT_ID="your-client-id"
export SALESFORCE_CLIENT_SECRET="your-client-secret"
export SALESFORCE_USERNAME="your-username"
export SALESFORCE_PASSWORD="your-password"
export SALESFORCE_SECURITY_TOKEN="your-security-token"

# Microsoft 365
export MICROSOFT_CLIENT_ID="your-app-id"
export MICROSOFT_CLIENT_SECRET="your-client-secret"
export MICROSOFT_TENANT_ID="your-tenant-id"

# Google Workspace
export GOOGLE_CLIENT_ID="your-client-id"
export GOOGLE_CLIENT_SECRET="your-client-secret"
export GOOGLE_SERVICE_ACCOUNT_KEY_PATH="/path/to/service-account.json"
export GOOGLE_DOMAIN="your-domain.com"

# Additional partners...
```

### Application Configuration

Key configuration properties in `application.yml`:

```yaml
integration:
  platform:
    partners:
      salesforce:
        rate-limit-per-hour: 5000
        bulk-api-enabled: true
      microsoft:
        rate-limit-per-minute: 600
    gateway:
      global-rate-limit-per-second: 1000
      circuit-breaker-timeout: PT10S
    monitoring:
      health-check-interval: PT1M
      error-alert-threshold: 10
```

## API Reference

### Health Check
```bash
GET /integration-platform/actuator/health
```

### Partner Operations
```bash
# Salesforce operations
GET /api/v1/salesforce/accounts
POST /api/v1/salesforce/accounts
PUT /api/v1/salesforce/accounts/{id}

# Microsoft 365 operations  
GET /api/v1/microsoft/users
POST /api/v1/microsoft/users
```

### Workflow Management
```bash
# Start sync workflow
POST /api/v1/workflows/partner-sync
{
  "sourcePartner": "salesforce",
  "targetPartner": "microsoft365",
  "syncType": "incremental",
  "resourceType": "contacts"
}

# Get workflow status
GET /api/v1/workflows/executions/{executionId}/status
```

## Development

### Project Structure
```
src/
├── main/java/
│   ├── core/                 # Core platform components
│   ├── connectors/           # Partner connectors
│   ├── transformers/         # Data transformers
│   ├── orchestration/        # Workflow orchestration
│   ├── gateway/              # API gateway configuration
│   ├── streaming/            # Kafka event processing
│   └── monitoring/           # Health and metrics
├── test/java/                # Unit and integration tests
└── main/resources/
    ├── application.yml       # Application configuration
    └── db/migration/         # Database migrations
```

### Building
```bash
# Build with tests
./mvnw clean package

# Build without tests
./mvnw clean package -DskipTests

# Build Docker image
docker build -t integration-platform .
```

### Testing
```bash
# Run unit tests
./mvnw test

# Run integration tests
./mvnw verify

# Run specific test class
./mvnw test -Dtest=SalesforceConnectorTest
```

## Monitoring & Observability

### Metrics
The platform exposes Prometheus metrics at `/actuator/prometheus`:
- `integration_requests_total` - Total integration requests
- `integration_errors_total` - Total integration errors
- `integration_response_time` - Response time distribution
- `integration_health_score` - Overall health score (0-100)

### Health Checks
- Application health: `/actuator/health`
- Individual connector health checks
- Database connectivity
- Kafka connectivity
- Partner system availability

### Dashboards
Pre-configured Grafana dashboards provide visibility into:
- System performance and throughput
- Error rates and success rates
- Partner-specific metrics
- Resource utilization
- SLA compliance

## Security

### Authentication
- OAuth2/JWT token-based authentication
- Per-partner API key management
- Role-based access control

### API Security
- Rate limiting per partner and client
- Circuit breaker protection
- Input validation and sanitization
- Audit logging

## Deployment

### Production Deployment
1. Use production Docker Compose configuration
2. Configure external databases and message queues
3. Set up proper monitoring and alerting
4. Configure backup strategies
5. Implement CI/CD pipelines

### Kubernetes
Kubernetes manifests are available in the `k8s/` directory:
```bash
kubectl apply -f k8s/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Code Standards
- Follow Java naming conventions
- Write comprehensive JavaDoc comments
- Include unit and integration tests
- Maintain 80%+ test coverage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the Integration Platform Team
- Check the documentation wiki

## Changelog

### Version 1.0.0
- Initial release with core integration framework
- Support for 8 major partner systems
- Comprehensive monitoring and observability
- Production-ready Docker deployment