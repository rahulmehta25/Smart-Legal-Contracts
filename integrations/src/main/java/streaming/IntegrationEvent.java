package com.enterprise.integrations.streaming;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Integration event for Kafka-based event streaming.
 * 
 * Represents events flowing through the integration platform including:
 * - Data change events from partner systems
 * - Workflow execution events
 * - System health and monitoring events
 * - Error and audit events
 * - Business rule processing events
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class IntegrationEvent {
    
    private String eventId;
    private String eventType;
    private String partnerId;
    private String source;
    private String version;
    private Object data;
    private LocalDateTime timestamp;
    private String correlationId;
    private String causationId;
    private Map<String, Object> metadata;
    private IntegrationEventService.EventPriority priority;
    private String schemaVersion;
    private Long ttlSeconds;
    
    /**
     * Default constructor.
     */
    public IntegrationEvent() {
        this.eventId = generateEventId();
        this.timestamp = LocalDateTime.now();
        this.version = "1.0";
        this.schemaVersion = "1.0.0";
        this.priority = IntegrationEventService.EventPriority.NORMAL;
        this.metadata = new HashMap<>();
    }
    
    /**
     * Constructor with basic parameters.
     * 
     * @param eventType the type of event
     * @param partnerId the partner system identifier
     * @param source the event source
     * @param data the event data
     */
    public IntegrationEvent(String eventType, String partnerId, String source, Object data) {
        this();
        this.eventType = eventType;
        this.partnerId = partnerId;
        this.source = source;
        this.data = data;
    }
    
    /**
     * Creates a builder for constructing integration events.
     * 
     * @return new event builder
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Generates a unique event ID.
     * 
     * @return unique event identifier
     */
    private String generateEventId() {
        return "evt-" + System.currentTimeMillis() + "-" + 
               Integer.toHexString((int) (Math.random() * Integer.MAX_VALUE));
    }
    
    /**
     * Adds metadata to the event.
     * 
     * @param key metadata key
     * @param value metadata value
     * @return this event for method chaining
     */
    public IntegrationEvent withMetadata(String key, Object value) {
        this.metadata.put(key, value);
        return this;
    }
    
    /**
     * Sets the correlation ID for event tracking.
     * 
     * @param correlationId correlation identifier
     * @return this event for method chaining
     */
    public IntegrationEvent withCorrelationId(String correlationId) {
        this.correlationId = correlationId;
        return this;
    }
    
    /**
     * Sets the causation ID for event causality tracking.
     * 
     * @param causationId causation identifier
     * @return this event for method chaining
     */
    public IntegrationEvent withCausationId(String causationId) {
        this.causationId = causationId;
        return this;
    }
    
    /**
     * Sets the event priority.
     * 
     * @param priority event priority
     * @return this event for method chaining
     */
    public IntegrationEvent withPriority(IntegrationEventService.EventPriority priority) {
        this.priority = priority;
        return this;
    }
    
    /**
     * Sets the TTL (time to live) for the event.
     * 
     * @param ttlSeconds TTL in seconds
     * @return this event for method chaining
     */
    public IntegrationEvent withTTL(Long ttlSeconds) {
        this.ttlSeconds = ttlSeconds;
        return this;
    }
    
    /**
     * Creates a data change event.
     * 
     * @param partnerId partner identifier
     * @param resourceType type of resource changed
     * @param resourceId identifier of changed resource
     * @param changeType type of change
     * @param oldData previous data state
     * @param newData new data state
     * @return data change integration event
     */
    public static IntegrationEvent dataChange(String partnerId, String resourceType,
                                            String resourceId, String changeType,
                                            Object oldData, Object newData) {
        IntegrationEventService.DataChangeEvent dataChangeEvent = 
            new IntegrationEventService.DataChangeEvent(
                partnerId,
                resourceType,
                resourceId,
                IntegrationEventService.DataChangeType.valueOf(changeType.toUpperCase()),
                oldData,
                newData
            );
        
        return new IntegrationEvent("DATA_CHANGE", partnerId, partnerId, dataChangeEvent)
            .withMetadata("resourceType", resourceType)
            .withMetadata("resourceId", resourceId)
            .withMetadata("changeType", changeType);
    }
    
    /**
     * Creates a workflow event.
     * 
     * @param workflowId workflow identifier
     * @param executionId execution identifier
     * @param eventType workflow event type
     * @param eventData event data
     * @return workflow integration event
     */
    public static IntegrationEvent workflowEvent(String workflowId, String executionId,
                                               String eventType, Object eventData) {
        IntegrationEventService.WorkflowEvent workflowEvent = 
            new IntegrationEventService.WorkflowEvent(workflowId, executionId, eventType, eventData);
        
        return new IntegrationEvent("WORKFLOW_" + eventType.toUpperCase(), "system", 
                                   "workflow-engine", workflowEvent)
            .withCorrelationId(executionId)
            .withMetadata("workflowId", workflowId)
            .withMetadata("executionId", executionId);
    }
    
    /**
     * Creates an error event.
     * 
     * @param partnerId partner identifier
     * @param errorCode error code
     * @param errorMessage error message
     * @param context error context
     * @return error integration event
     */
    public static IntegrationEvent errorEvent(String partnerId, String errorCode,
                                            String errorMessage, Map<String, Object> context) {
        IntegrationEventService.ErrorEvent errorEvent = 
            new IntegrationEventService.ErrorEvent(partnerId, errorCode, errorMessage, context);
        
        return new IntegrationEvent("ERROR", partnerId, partnerId, errorEvent)
            .withPriority(IntegrationEventService.EventPriority.HIGH)
            .withMetadata("errorCode", errorCode)
            .withMetadata("severity", "ERROR");
    }
    
    /**
     * Creates a system health event.
     * 
     * @param component system component
     * @param status health status
     * @param details health details
     * @return health integration event
     */
    public static IntegrationEvent healthEvent(String component, String status, 
                                             Map<String, Object> details) {
        Map<String, Object> healthData = new HashMap<>();
        healthData.put("component", component);
        healthData.put("status", status);
        healthData.put("details", details);
        healthData.put("timestamp", LocalDateTime.now());
        
        return new IntegrationEvent("HEALTH_CHECK", "system", component, healthData)
            .withMetadata("component", component)
            .withMetadata("healthStatus", status);
    }
    
    /**
     * Creates a metrics event.
     * 
     * @param metricName metric name
     * @param metricValue metric value
     * @param tags metric tags
     * @return metrics integration event
     */
    public static IntegrationEvent metricsEvent(String metricName, Object metricValue,
                                              Map<String, String> tags) {
        Map<String, Object> metricsData = new HashMap<>();
        metricsData.put("metricName", metricName);
        metricsData.put("metricValue", metricValue);
        metricsData.put("tags", tags);
        metricsData.put("timestamp", LocalDateTime.now());
        
        return new IntegrationEvent("METRICS", "system", "metrics-collector", metricsData)
            .withMetadata("metricName", metricName)
            .withMetadata("metricType", metricValue.getClass().getSimpleName());
    }
    
    // Getters and setters
    
    public String getEventId() {
        return eventId;
    }
    
    public void setEventId(String eventId) {
        this.eventId = eventId;
    }
    
    public String getEventType() {
        return eventType;
    }
    
    public void setEventType(String eventType) {
        this.eventType = eventType;
    }
    
    public String getPartnerId() {
        return partnerId;
    }
    
    public void setPartnerId(String partnerId) {
        this.partnerId = partnerId;
    }
    
    public String getSource() {
        return source;
    }
    
    public void setSource(String source) {
        this.source = source;
    }
    
    public String getVersion() {
        return version;
    }
    
    public void setVersion(String version) {
        this.version = version;
    }
    
    public Object getData() {
        return data;
    }
    
    public void setData(Object data) {
        this.data = data;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public String getCorrelationId() {
        return correlationId;
    }
    
    public void setCorrelationId(String correlationId) {
        this.correlationId = correlationId;
    }
    
    public String getCausationId() {
        return causationId;
    }
    
    public void setCausationId(String causationId) {
        this.causationId = causationId;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    public IntegrationEventService.EventPriority getPriority() {
        return priority;
    }
    
    public void setPriority(IntegrationEventService.EventPriority priority) {
        this.priority = priority;
    }
    
    public String getSchemaVersion() {
        return schemaVersion;
    }
    
    public void setSchemaVersion(String schemaVersion) {
        this.schemaVersion = schemaVersion;
    }
    
    public Long getTtlSeconds() {
        return ttlSeconds;
    }
    
    public void setTtlSeconds(Long ttlSeconds) {
        this.ttlSeconds = ttlSeconds;
    }
    
    /**
     * Builder pattern for constructing integration events.
     */
    public static class Builder {
        private final IntegrationEvent event;
        
        private Builder() {
            this.event = new IntegrationEvent();
        }
        
        public Builder eventType(String eventType) {
            event.setEventType(eventType);
            return this;
        }
        
        public Builder partnerId(String partnerId) {
            event.setPartnerId(partnerId);
            return this;
        }
        
        public Builder source(String source) {
            event.setSource(source);
            return this;
        }
        
        public Builder version(String version) {
            event.setVersion(version);
            return this;
        }
        
        public Builder data(Object data) {
            event.setData(data);
            return this;
        }
        
        public Builder correlationId(String correlationId) {
            event.setCorrelationId(correlationId);
            return this;
        }
        
        public Builder causationId(String causationId) {
            event.setCausationId(causationId);
            return this;
        }
        
        public Builder priority(IntegrationEventService.EventPriority priority) {
            event.setPriority(priority);
            return this;
        }
        
        public Builder metadata(String key, Object value) {
            event.withMetadata(key, value);
            return this;
        }
        
        public Builder ttl(Long ttlSeconds) {
            event.setTtlSeconds(ttlSeconds);
            return this;
        }
        
        public IntegrationEvent build() {
            if (event.getEventType() == null) {
                throw new IllegalStateException("Event type is required");
            }
            if (event.getPartnerId() == null) {
                throw new IllegalStateException("Partner ID is required");
            }
            if (event.getSource() == null) {
                event.setSource(event.getPartnerId());
            }
            return event;
        }
    }
    
    @Override
    public String toString() {
        return "IntegrationEvent{" +
                "eventId='" + eventId + '\'' +
                ", eventType='" + eventType + '\'' +
                ", partnerId='" + partnerId + '\'' +
                ", source='" + source + '\'' +
                ", timestamp=" + timestamp +
                ", correlationId='" + correlationId + '\'' +
                ", priority=" + priority +
                '}';
    }
}