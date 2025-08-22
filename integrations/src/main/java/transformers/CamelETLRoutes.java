package com.enterprise.integrations.transformers;

import com.enterprise.integrations.core.IntegrationProperties;
import org.apache.camel.Exchange;
import org.apache.camel.LoggingLevel;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.kafka.KafkaConstants;
import org.apache.camel.model.dataformat.JsonLibrary;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Apache Camel ETL routes for data transformation and integration pipelines.
 * 
 * Defines data processing routes for:
 * - Partner data ingestion and transformation
 * - ETL pipeline orchestration
 * - Error handling and dead letter queues
 * - Data validation and cleansing
 * - Format conversion and routing
 * - Event-driven processing
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Component
public class CamelETLRoutes extends RouteBuilder {
    
    private final IntegrationProperties properties;
    
    /**
     * Constructor with dependency injection.
     * 
     * @param properties integration properties
     */
    @Autowired
    public CamelETLRoutes(IntegrationProperties properties) {
        this.properties = properties;
    }
    
    @Override
    public void configure() throws Exception {
        
        // Configure global error handler
        errorHandler(deadLetterChannel("kafka:" + properties.getKafka().getErrorsTopic())
                .maximumRedeliveries(3)
                .redeliveryDelay(5000)
                .exponentialBackOff()
                .logRetryAttempted(true)
                .logStackTrace(true)
                .onRedelivery(exchange -> {
                    int redeliveryCounter = exchange.getIn().getHeader(Exchange.REDELIVERY_COUNTER, Integer.class);
                    log.warn("Redelivery attempt {} for exchange: {}", redeliveryCounter, exchange.getExchangeId());
                }));
        
        // Main integration event processing route
        configureMainProcessingRoute();
        
        // Partner-specific transformation routes
        configureSalesforceTransformationRoute();
        configureMicrosoftTransformationRoute();
        configureGenericTransformationRoute();
        
        // Data validation routes
        configureDataValidationRoute();
        
        // Format conversion routes
        configureFormatConversionRoutes();
        
        // Error handling routes
        configureErrorHandlingRoutes();
        
        // Monitoring and metrics routes
        configureMonitoringRoutes();
        
        // Batch processing routes
        configureBatchProcessingRoutes();
    }
    
    /**
     * Configures the main integration event processing route.
     */
    private void configureMainProcessingRoute() {
        from("kafka:" + properties.getKafka().getIntegrationEventsTopic() + 
             "?groupId=integration-etl-processor")
            .routeId("main-integration-processor")
            .log(LoggingLevel.DEBUG, "Processing integration event: ${body}")
            .unmarshal().json(JsonLibrary.Jackson)
            .process(exchange -> {
                // Add correlation ID if not present
                if (exchange.getIn().getHeader("correlationId") == null) {
                    exchange.getIn().setHeader("correlationId", 
                        "corr-" + System.currentTimeMillis() + "-" + exchange.getExchangeId());
                }
                // Add processing timestamp
                exchange.getIn().setHeader("processingTimestamp", System.currentTimeMillis());
            })
            .choice()
                .when(header("partnerId").isEqualTo("salesforce"))
                    .to("direct:salesforce-transformation")
                .when(header("partnerId").isEqualTo("microsoft365"))
                    .to("direct:microsoft-transformation")
                .when(header("partnerId").isEqualTo("google"))
                    .to("direct:google-transformation")
                .when(header("partnerId").isEqualTo("docusign"))
                    .to("direct:docusign-transformation")
                .when(header("partnerId").isEqualTo("slack"))
                    .to("direct:slack-transformation")
                .otherwise()
                    .to("direct:generic-transformation")
            .end()
            .to("direct:data-validation")
            .to("kafka:" + properties.getKafka().getIntegrationEventsTopic() + "-processed");
    }
    
    /**
     * Configures Salesforce-specific transformation route.
     */
    private void configureSalesforceTransformationRoute() {
        from("direct:salesforce-transformation")
            .routeId("salesforce-transformation")
            .log(LoggingLevel.INFO, "Transforming Salesforce data: ${header.operationType}")
            .setHeader("transformerId", constant("salesforce-transformer"))
            .bean("salesforceTransformer", "transform")
            .process(exchange -> {
                // Add transformation metadata
                exchange.getIn().setHeader("transformedAt", System.currentTimeMillis());
                exchange.getIn().setHeader("sourceSystem", "salesforce");
            })
            .choice()
                .when(header("operationType").isEqualTo("BULK"))
                    .to("direct:bulk-processing")
                .otherwise()
                    .log(LoggingLevel.DEBUG, "Salesforce transformation completed successfully")
            .end();
    }
    
    /**
     * Configures Microsoft 365-specific transformation route.
     */
    private void configureMicrosoftTransformationRoute() {
        from("direct:microsoft-transformation")
            .routeId("microsoft-transformation")
            .log(LoggingLevel.INFO, "Transforming Microsoft 365 data: ${header.operationType}")
            .setHeader("transformerId", constant("microsoft-transformer"))
            .bean("microsoftTransformer", "transform")
            .process(exchange -> {
                // Add transformation metadata
                exchange.getIn().setHeader("transformedAt", System.currentTimeMillis());
                exchange.getIn().setHeader("sourceSystem", "microsoft365");
            })
            .log(LoggingLevel.DEBUG, "Microsoft 365 transformation completed successfully");
    }
    
    /**
     * Configures generic transformation route for other partners.
     */
    private void configureGenericTransformationRoute() {
        from("direct:generic-transformation")
            .routeId("generic-transformation")
            .log(LoggingLevel.INFO, "Transforming data for partner: ${header.partnerId}")
            .setHeader("transformerId", constant("generic-transformer"))
            .bean("genericTransformer", "transform")
            .process(exchange -> {
                // Add transformation metadata
                exchange.getIn().setHeader("transformedAt", System.currentTimeMillis());
                exchange.getIn().setHeader("sourceSystem", header("partnerId"));
            })
            .log(LoggingLevel.DEBUG, "Generic transformation completed successfully");
    }
    
    /**
     * Configures data validation route.
     */
    private void configureDataValidationRoute() {
        from("direct:data-validation")
            .routeId("data-validation")
            .log(LoggingLevel.DEBUG, "Validating transformed data")
            .bean("dataValidator", "validate")
            .choice()
                .when(header("validationResult").isEqualTo("FAILED"))
                    .log(LoggingLevel.WARN, "Data validation failed: ${header.validationErrors}")
                    .setHeader(KafkaConstants.KEY, header("correlationId"))
                    .to("kafka:" + properties.getKafka().getErrorsTopic())
                    .stop()
                .when(header("validationResult").isEqualTo("WARNING"))
                    .log(LoggingLevel.WARN, "Data validation warnings: ${header.validationWarnings}")
                    .setHeader("hasWarnings", constant(true))
                .otherwise()
                    .log(LoggingLevel.DEBUG, "Data validation passed")
            .end();
    }
    
    /**
     * Configures format conversion routes.
     */
    private void configureFormatConversionRoutes() {
        // JSON to XML conversion
        from("direct:json-to-xml")
            .routeId("json-to-xml-converter")
            .log(LoggingLevel.DEBUG, "Converting JSON to XML")
            .unmarshal().json(JsonLibrary.Jackson)
            .marshal().jacksonXml()
            .setHeader("contentType", constant("application/xml"));
        
        // XML to JSON conversion
        from("direct:xml-to-json")
            .routeId("xml-to-json-converter")
            .log(LoggingLevel.DEBUG, "Converting XML to JSON")
            .unmarshal().jacksonXml()
            .marshal().json(JsonLibrary.Jackson)
            .setHeader("contentType", constant("application/json"));
        
        // CSV processing
        from("direct:csv-processing")
            .routeId("csv-processor")
            .log(LoggingLevel.DEBUG, "Processing CSV data")
            .unmarshal().csv()
            .split(body())
            .process(exchange -> {
                // Convert CSV row to Map
                @SuppressWarnings("unchecked")
                java.util.List<String> row = exchange.getIn().getBody(java.util.List.class);
                java.util.Map<String, Object> record = new java.util.HashMap<>();
                
                // Assume first row contains headers (would be more sophisticated in production)
                for (int i = 0; i < row.size(); i++) {
                    record.put("field_" + i, row.get(i));
                }
                
                exchange.getIn().setBody(record);
            })
            .marshal().json(JsonLibrary.Jackson)
            .to("direct:data-validation");
    }
    
    /**
     * Configures error handling routes.
     */
    private void configureErrorHandlingRoutes() {
        // Dead letter queue processor
        from("kafka:" + properties.getKafka().getErrorsTopic() + "?groupId=error-processor")
            .routeId("error-processor")
            .log(LoggingLevel.ERROR, "Processing error message: ${body}")
            .unmarshal().json(JsonLibrary.Jackson)
            .process(exchange -> {
                // Extract error information
                @SuppressWarnings("unchecked")
                java.util.Map<String, Object> errorData = exchange.getIn().getBody(java.util.Map.class);
                
                // Determine if error is retryable
                boolean retryable = Boolean.TRUE.equals(errorData.get("retryable"));
                String errorCode = (String) errorData.get("errorCode");
                
                exchange.setProperty("retryable", retryable);
                exchange.setProperty("errorCode", errorCode);
            })
            .choice()
                .when(exchangeProperty("retryable").isEqualTo(true))
                    .log(LoggingLevel.INFO, "Scheduling retry for retryable error: ${exchangeProperty.errorCode}")
                    .delay(30000) // 30 second delay
                    .to("kafka:" + properties.getKafka().getIntegrationEventsTopic())
                .otherwise()
                    .log(LoggingLevel.ERROR, "Non-retryable error, sending to audit: ${exchangeProperty.errorCode}")
                    .to("kafka:" + properties.getKafka().getAuditTopic())
            .end();
        
        // Exception handler for transformation errors
        onException(TransformationException.class)
            .handled(true)
            .log(LoggingLevel.ERROR, "Transformation error: ${exception.message}")
            .process(exchange -> {
                TransformationException ex = exchange.getProperty(Exchange.EXCEPTION_CAUGHT, TransformationException.class);
                
                java.util.Map<String, Object> errorInfo = new java.util.HashMap<>();
                errorInfo.put("errorCode", ex.getErrorCode());
                errorInfo.put("errorMessage", ex.getMessage());
                errorInfo.put("transformerId", ex.getTransformerId());
                errorInfo.put("stage", ex.getStage().toString());
                errorInfo.put("retryable", ex.isRetryable());
                errorInfo.put("correlationId", exchange.getIn().getHeader("correlationId"));
                errorInfo.put("timestamp", System.currentTimeMillis());
                
                exchange.getIn().setBody(errorInfo);
            })
            .marshal().json(JsonLibrary.Jackson)
            .to("kafka:" + properties.getKafka().getErrorsTopic());
    }
    
    /**
     * Configures monitoring and metrics routes.
     */
    private void configureMonitoringRoutes() {
        // Metrics collection route
        from("timer://metrics-collector?period=30000") // Every 30 seconds
            .routeId("metrics-collector")
            .log(LoggingLevel.DEBUG, "Collecting integration metrics")
            .bean("metricsCollector", "collectMetrics")
            .marshal().json(JsonLibrary.Jackson)
            .to("kafka:" + properties.getKafka().getMetricsTopic());
        
        // Health check route
        from("timer://health-checker?period=60000") // Every minute
            .routeId("health-checker")
            .log(LoggingLevel.DEBUG, "Performing health checks")
            .bean("healthChecker", "performHealthChecks")
            .choice()
                .when(header("healthStatus").isEqualTo("UNHEALTHY"))
                    .log(LoggingLevel.WARN, "Unhealthy components detected: ${body}")
                    .marshal().json(JsonLibrary.Jackson)
                    .to("kafka:" + properties.getKafka().getErrorsTopic())
                .otherwise()
                    .log(LoggingLevel.DEBUG, "All components healthy")
            .end();
        
        // Audit trail route
        from("kafka:" + properties.getKafka().getAuditTopic() + "?groupId=audit-processor")
            .routeId("audit-processor")
            .log(LoggingLevel.INFO, "Processing audit event: ${body}")
            .unmarshal().json(JsonLibrary.Jackson)
            .bean("auditService", "recordAuditEvent")
            .log(LoggingLevel.DEBUG, "Audit event recorded successfully");
    }
    
    /**
     * Configures batch processing routes.
     */
    private void configureBatchProcessingRoutes() {
        // Batch aggregation route
        from("direct:bulk-processing")
            .routeId("bulk-processor")
            .log(LoggingLevel.INFO, "Processing bulk operation")
            .aggregate(header("bulkOperationId"))
            .completionSize(100) // Process in batches of 100
            .completionTimeout(30000) // Or after 30 seconds
            .aggregationStrategy((exchange1, exchange2) -> {
                if (exchange1 == null) {
                    return exchange2;
                }
                
                @SuppressWarnings("unchecked")
                java.util.List<Object> list1 = exchange1.getIn().getBody(java.util.List.class);
                if (list1 == null) {
                    list1 = new java.util.ArrayList<>();
                    list1.add(exchange1.getIn().getBody());
                }
                
                list1.add(exchange2.getIn().getBody());
                exchange1.getIn().setBody(list1);
                return exchange1;
            })
            .bean("bulkProcessor", "processBatch")
            .marshal().json(JsonLibrary.Jackson)
            .to("kafka:" + properties.getKafka().getIntegrationEventsTopic() + "-bulk-processed");
        
        // File-based ETL route
        from("file://data/incoming?delay=10000&delete=true")
            .routeId("file-etl-processor")
            .log(LoggingLevel.INFO, "Processing file: ${header.CamelFileName}")
            .choice()
                .when(header("CamelFileName").endsWith(".csv"))
                    .to("direct:csv-processing")
                .when(header("CamelFileName").endsWith(".xml"))
                    .to("direct:xml-to-json")
                .when(header("CamelFileName").endsWith(".json"))
                    .to("direct:data-validation")
                .otherwise()
                    .log(LoggingLevel.WARN, "Unsupported file format: ${header.CamelFileName}")
                    .to("file://data/unsupported")
            .end();
    }
}