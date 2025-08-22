package com.enterprise.integrations.streaming;

import com.enterprise.integrations.core.IntegrationProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.header.Header;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.handler.annotation.Header;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Integration event service for Kafka-based event streaming.
 * 
 * Provides event-driven integration capabilities including:
 * - Event publishing and consumption
 * - Event sourcing patterns
 * - CQRS implementation support
 * - Dead letter queue handling
 * - Event replay and recovery
 * - Real-time data synchronization
 * - Webhook event processing
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Service
public class IntegrationEventService {
    
    private static final Logger logger = LoggerFactory.getLogger(IntegrationEventService.class);
    
    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final ObjectMapper objectMapper;
    private final IntegrationProperties properties;
    private final EventProcessorRegistry eventProcessorRegistry;
    
    /**
     * Constructor with dependency injection.
     * 
     * @param kafkaTemplate Kafka template for publishing events
     * @param objectMapper JSON object mapper
     * @param properties integration properties
     * @param eventProcessorRegistry event processor registry
     */
    @Autowired
    public IntegrationEventService(KafkaTemplate<String, Object> kafkaTemplate,
                                  ObjectMapper objectMapper,
                                  IntegrationProperties properties,
                                  EventProcessorRegistry eventProcessorRegistry) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
        this.properties = properties;
        this.eventProcessorRegistry = eventProcessorRegistry;
    }
    
    /**
     * Publishes an integration event to Kafka.
     * 
     * @param event the integration event to publish
     * @return CompletableFuture with send result
     */
    public CompletableFuture<Void> publishEvent(IntegrationEvent event) {
        try {
            String topic = determineTopicForEvent(event);
            String key = generateEventKey(event);
            
            ProducerRecord<String, Object> record = new ProducerRecord<>(topic, key, event);
            
            // Add event headers
            record.headers().add("eventType", event.getEventType().getBytes());
            record.headers().add("eventVersion", event.getVersion().getBytes());
            record.headers().add("partnerId", event.getPartnerId().getBytes());
            record.headers().add("correlationId", event.getCorrelationId().getBytes());
            record.headers().add("timestamp", String.valueOf(event.getTimestamp()).getBytes());
            
            logger.debug("Publishing event: {} to topic: {}", event.getEventId(), topic);
            
            return kafkaTemplate.send(record)
                .thenAccept(result -> {
                    logger.info("Successfully published event: {} to partition: {} offset: {}",
                               event.getEventId(),
                               result.getRecordMetadata().partition(),
                               result.getRecordMetadata().offset());
                })
                .exceptionally(throwable -> {
                    logger.error("Failed to publish event: {}", event.getEventId(), throwable);
                    throw new EventPublishingException("Failed to publish event", event, throwable);
                });
                
        } catch (Exception e) {
            logger.error("Error publishing event: {}", event.getEventId(), e);
            throw new EventPublishingException("Error publishing event", event, e);
        }
    }
    
    /**
     * Publishes a data change event.
     * 
     * @param partnerId partner system identifier
     * @param resourceType type of resource changed
     * @param resourceId identifier of changed resource
     * @param changeType type of change (CREATE, UPDATE, DELETE)
     * @param oldData previous data state
     * @param newData new data state
     * @return CompletableFuture with send result
     */
    public CompletableFuture<Void> publishDataChangeEvent(String partnerId, String resourceType,
                                                         String resourceId, DataChangeType changeType,
                                                         Object oldData, Object newData) {
        DataChangeEvent dataChangeEvent = new DataChangeEvent(
            partnerId,
            resourceType,
            resourceId,
            changeType,
            oldData,
            newData
        );
        
        IntegrationEvent event = IntegrationEvent.builder()
            .eventType("DATA_CHANGE")
            .partnerId(partnerId)
            .source(partnerId)
            .data(dataChangeEvent)
            .correlationId(generateCorrelationId())
            .build();
        
        return publishEvent(event);
    }
    
    /**
     * Publishes a workflow event.
     * 
     * @param workflowId workflow identifier
     * @param executionId execution identifier
     * @param eventType workflow event type
     * @param eventData event data
     * @return CompletableFuture with send result
     */
    public CompletableFuture<Void> publishWorkflowEvent(String workflowId, String executionId,
                                                       String eventType, Object eventData) {
        WorkflowEvent workflowEvent = new WorkflowEvent(
            workflowId,
            executionId,
            eventType,
            eventData
        );
        
        IntegrationEvent event = IntegrationEvent.builder()
            .eventType("WORKFLOW_" + eventType)
            .partnerId("system")
            .source("workflow-engine")
            .data(workflowEvent)
            .correlationId(executionId)
            .build();
        
        return publishEvent(event);
    }
    
    /**
     * Publishes an error event.
     * 
     * @param partnerId partner identifier
     * @param errorCode error code
     * @param errorMessage error message
     * @param context error context
     * @return CompletableFuture with send result
     */
    public CompletableFuture<Void> publishErrorEvent(String partnerId, String errorCode,
                                                    String errorMessage, Map<String, Object> context) {
        ErrorEvent errorEvent = new ErrorEvent(
            partnerId,
            errorCode,
            errorMessage,
            context
        );
        
        IntegrationEvent event = IntegrationEvent.builder()
            .eventType("ERROR")
            .partnerId(partnerId)
            .source(partnerId)
            .data(errorEvent)
            .priority(EventPriority.HIGH)
            .correlationId(generateCorrelationId())
            .build();
        
        return publishEvent(event);
    }
    
    /**
     * Main integration events consumer.
     * 
     * @param record Kafka consumer record
     * @param acknowledgment message acknowledgment
     * @param partition partition number
     * @param offset message offset
     */
    @KafkaListener(topics = "${integration.platform.kafka.integration-events-topic}",
                   groupId = "${spring.kafka.consumer.group-id}")
    public void handleIntegrationEvent(@Payload ConsumerRecord<String, Object> record,
                                     Acknowledgment acknowledgment,
                                     @Header(KafkaHeaders.RECEIVED_PARTITION) int partition,
                                     @Header(KafkaHeaders.OFFSET) long offset) {
        try {
            logger.debug("Processing integration event from partition: {} offset: {}", partition, offset);
            
            IntegrationEvent event = parseIntegrationEvent(record);
            String eventType = event.getEventType();
            
            // Route event to appropriate processor
            EventProcessor processor = eventProcessorRegistry.getProcessor(eventType);
            if (processor != null) {
                processor.processEvent(event);
            } else {
                logger.warn("No processor found for event type: {}", eventType);
            }
            
            // Acknowledge message processing
            acknowledgment.acknowledge();
            
        } catch (Exception e) {
            logger.error("Error processing integration event from partition: {} offset: {}", 
                        partition, offset, e);
            
            // Send to dead letter queue
            sendToDeadLetterQueue(record, e);
            acknowledgment.acknowledge(); // Acknowledge to avoid infinite retry
        }
    }
    
    /**
     * Error events consumer.
     * 
     * @param record Kafka consumer record
     * @param acknowledgment message acknowledgment
     */
    @KafkaListener(topics = "${integration.platform.kafka.errors-topic}",
                   groupId = "${spring.kafka.consumer.group-id}-errors")
    public void handleErrorEvent(@Payload ConsumerRecord<String, Object> record,
                               Acknowledgment acknowledgment) {
        try {
            logger.info("Processing error event: {}", record.value());
            
            IntegrationEvent event = parseIntegrationEvent(record);
            
            // Process error event (logging, alerting, etc.)
            processErrorEvent(event);
            
            acknowledgment.acknowledge();
            
        } catch (Exception e) {
            logger.error("Error processing error event", e);
            acknowledgment.acknowledge();
        }
    }
    
    /**
     * Metrics events consumer.
     * 
     * @param record Kafka consumer record
     * @param acknowledgment message acknowledgment
     */
    @KafkaListener(topics = "${integration.platform.kafka.metrics-topic}",
                   groupId = "${spring.kafka.consumer.group-id}-metrics")
    public void handleMetricsEvent(@Payload ConsumerRecord<String, Object> record,
                                 Acknowledgment acknowledgment) {
        try {
            logger.debug("Processing metrics event");
            
            IntegrationEvent event = parseIntegrationEvent(record);
            
            // Process metrics event
            processMetricsEvent(event);
            
            acknowledgment.acknowledge();
            
        } catch (Exception e) {
            logger.error("Error processing metrics event", e);
            acknowledgment.acknowledge();
        }
    }
    
    /**
     * Audit events consumer.
     * 
     * @param record Kafka consumer record
     * @param acknowledgment message acknowledgment
     */
    @KafkaListener(topics = "${integration.platform.kafka.audit-topic}",
                   groupId = "${spring.kafka.consumer.group-id}-audit")
    public void handleAuditEvent(@Payload ConsumerRecord<String, Object> record,
                               Acknowledgment acknowledgment) {
        try {
            logger.debug("Processing audit event");
            
            IntegrationEvent event = parseIntegrationEvent(record);
            
            // Process audit event
            processAuditEvent(event);
            
            acknowledgment.acknowledge();
            
        } catch (Exception e) {
            logger.error("Error processing audit event", e);
            acknowledgment.acknowledge();
        }
    }
    
    /**
     * Determines the appropriate Kafka topic for an event.
     * 
     * @param event integration event
     * @return topic name
     */
    private String determineTopicForEvent(IntegrationEvent event) {
        String eventType = event.getEventType();
        
        return switch (eventType) {
            case "ERROR" -> properties.getKafka().getErrorsTopic();
            case "METRICS" -> properties.getKafka().getMetricsTopic();
            case "AUDIT" -> properties.getKafka().getAuditTopic();
            default -> {
                if (eventType.startsWith("WORKFLOW_")) {
                    yield properties.getKafka().getIntegrationEventsTopic() + "-workflows";
                } else if (eventType.equals("DATA_CHANGE")) {
                    yield properties.getKafka().getIntegrationEventsTopic() + "-data-changes";
                }
                yield properties.getKafka().getIntegrationEventsTopic();
            }
        };
    }
    
    /**
     * Generates an event key for partitioning.
     * 
     * @param event integration event
     * @return event key
     */
    private String generateEventKey(IntegrationEvent event) {
        // Use partner ID and resource type for partitioning
        StringBuilder keyBuilder = new StringBuilder();
        keyBuilder.append(event.getPartnerId());
        
        if (event.getData() instanceof DataChangeEvent dataChangeEvent) {
            keyBuilder.append(":").append(dataChangeEvent.resourceType());
        } else if (event.getData() instanceof WorkflowEvent workflowEvent) {
            keyBuilder.append(":").append(workflowEvent.workflowId());
        }
        
        return keyBuilder.toString();
    }
    
    /**
     * Generates a correlation ID for event tracking.
     * 
     * @return correlation ID
     */
    private String generateCorrelationId() {
        return "corr-" + System.currentTimeMillis() + "-" + 
               Integer.toHexString((int) (Math.random() * Integer.MAX_VALUE));
    }
    
    /**
     * Parses integration event from Kafka consumer record.
     * 
     * @param record Kafka consumer record
     * @return integration event
     */
    private IntegrationEvent parseIntegrationEvent(ConsumerRecord<String, Object> record) {
        try {
            Object value = record.value();
            
            if (value instanceof IntegrationEvent) {
                return (IntegrationEvent) value;
            } else if (value instanceof String) {
                return objectMapper.readValue((String) value, IntegrationEvent.class);
            } else if (value instanceof Map) {
                return objectMapper.convertValue(value, IntegrationEvent.class);
            } else {
                throw new IllegalArgumentException("Unsupported event format: " + value.getClass());
            }
            
        } catch (Exception e) {
            logger.error("Error parsing integration event from record", e);
            throw new EventParsingException("Failed to parse integration event", e);
        }
    }
    
    /**
     * Sends failed message to dead letter queue.
     * 
     * @param record original consumer record
     * @param error processing error
     */
    private void sendToDeadLetterQueue(ConsumerRecord<String, Object> record, Throwable error) {
        try {
            String dlqTopic = properties.getKafka().getErrorsTopic() + "-dlq";
            
            Map<String, Object> dlqMessage = new HashMap<>();
            dlqMessage.put("originalTopic", record.topic());
            dlqMessage.put("originalPartition", record.partition());
            dlqMessage.put("originalOffset", record.offset());
            dlqMessage.put("originalValue", record.value());
            dlqMessage.put("error", error.getMessage());
            dlqMessage.put("errorClass", error.getClass().getName());
            dlqMessage.put("timestamp", LocalDateTime.now());
            
            kafkaTemplate.send(dlqTopic, record.key(), dlqMessage);
            logger.info("Sent failed message to DLQ: {}", dlqTopic);
            
        } catch (Exception e) {
            logger.error("Failed to send message to DLQ", e);
        }
    }
    
    /**
     * Processes error events for monitoring and alerting.
     * 
     * @param event error event
     */
    private void processErrorEvent(IntegrationEvent event) {
        if (event.getData() instanceof ErrorEvent errorEvent) {
            logger.warn("Error event received: {} - {} (Partner: {})",
                       errorEvent.errorCode(),
                       errorEvent.errorMessage(),
                       errorEvent.partnerId());
            
            // In production, this would trigger alerts, update dashboards, etc.
        }
    }
    
    /**
     * Processes metrics events for monitoring and reporting.
     * 
     * @param event metrics event
     */
    private void processMetricsEvent(IntegrationEvent event) {
        logger.debug("Metrics event processed: {}", event.getEventId());
        // In production, this would update metrics databases, dashboards, etc.
    }
    
    /**
     * Processes audit events for compliance and tracking.
     * 
     * @param event audit event
     */
    private void processAuditEvent(IntegrationEvent event) {
        logger.info("Audit event recorded: {} (Partner: {})", 
                   event.getEventType(), event.getPartnerId());
        // In production, this would store in audit database
    }
    
    /**
     * Data change event record.
     */
    public record DataChangeEvent(
        String partnerId,
        String resourceType,
        String resourceId,
        DataChangeType changeType,
        Object oldData,
        Object newData
    ) {}
    
    /**
     * Workflow event record.
     */
    public record WorkflowEvent(
        String workflowId,
        String executionId,
        String eventType,
        Object eventData
    ) {}
    
    /**
     * Error event record.
     */
    public record ErrorEvent(
        String partnerId,
        String errorCode,
        String errorMessage,
        Map<String, Object> context
    ) {}
    
    /**
     * Data change types.
     */
    public enum DataChangeType {
        CREATE, UPDATE, DELETE, MERGE
    }
    
    /**
     * Event priorities.
     */
    public enum EventPriority {
        LOW, NORMAL, HIGH, CRITICAL
    }
}