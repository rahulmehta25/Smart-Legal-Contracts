package com.enterprise.integrations.connectors;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Event emitted by partner connectors.
 * 
 * Represents events from partner systems such as data changes,
 * status updates, webhooks, and system notifications.
 * 
 * @param <T> the type of event data
 * @author Integration Platform Team
 * @version 1.0.0
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ConnectorEvent<T> {
    
    private String eventId;
    private String eventType;
    private String partnerId;
    private String source;
    private T data;
    private LocalDateTime timestamp;
    private String correlationId;
    private Map<String, Object> metadata;
    private EventPriority priority;
    private String version;
    
    /**
     * Default constructor.
     */
    public ConnectorEvent() {
        this.metadata = new HashMap<>();
        this.timestamp = LocalDateTime.now();
        this.eventId = generateEventId();
        this.priority = EventPriority.NORMAL;
        this.version = "1.0";
    }
    
    /**
     * Constructor with basic parameters.
     * 
     * @param eventType the type of event
     * @param partnerId the partner system identifier
     * @param data the event data
     */
    public ConnectorEvent(String eventType, String partnerId, T data) {
        this();
        this.eventType = eventType;
        this.partnerId = partnerId;
        this.data = data;
        this.source = partnerId;
    }
    
    /**
     * Constructor with full parameters.
     * 
     * @param eventType the type of event
     * @param partnerId the partner system identifier
     * @param source the event source
     * @param data the event data
     * @param priority the event priority
     */
    public ConnectorEvent(String eventType, String partnerId, String source, T data, EventPriority priority) {
        this(eventType, partnerId, data);
        this.source = source;
        this.priority = priority;
    }
    
    /**
     * Generates a unique event ID.
     * 
     * @return unique event identifier
     */
    private String generateEventId() {
        return "evt-" + System.currentTimeMillis() + "-" + 
               Integer.toHexString(this.hashCode());
    }
    
    /**
     * Adds metadata to the event.
     * 
     * @param key metadata key
     * @param value metadata value
     * @return this event for method chaining
     */
    public ConnectorEvent<T> withMetadata(String key, Object value) {
        this.metadata.put(key, value);
        return this;
    }
    
    /**
     * Sets correlation ID for tracing.
     * 
     * @param correlationId correlation identifier
     * @return this event for method chaining
     */
    public ConnectorEvent<T> withCorrelationId(String correlationId) {
        this.correlationId = correlationId;
        return this;
    }
    
    /**
     * Creates a data change event.
     * 
     * @param partnerId partner identifier
     * @param resourceType type of resource changed
     * @param resourceId identifier of changed resource
     * @param changeType type of change (CREATE, UPDATE, DELETE)
     * @param data the changed data
     * @param <T> type of data
     * @return data change event
     */
    public static <T> ConnectorEvent<T> dataChange(String partnerId, String resourceType, 
                                                  String resourceId, String changeType, T data) {
        ConnectorEvent<T> event = new ConnectorEvent<>("DATA_CHANGE", partnerId, data);
        event.withMetadata("resourceType", resourceType)
             .withMetadata("resourceId", resourceId)
             .withMetadata("changeType", changeType);
        return event;
    }
    
    /**
     * Creates a status change event.
     * 
     * @param partnerId partner identifier
     * @param oldStatus previous status
     * @param newStatus new status
     * @param <T> type of data
     * @return status change event
     */
    public static <T> ConnectorEvent<T> statusChange(String partnerId, String oldStatus, String newStatus) {
        ConnectorEvent<T> event = new ConnectorEvent<>("STATUS_CHANGE", partnerId, null);
        event.withMetadata("oldStatus", oldStatus)
             .withMetadata("newStatus", newStatus);
        return event;
    }
    
    /**
     * Creates an error event.
     * 
     * @param partnerId partner identifier
     * @param errorCode error code
     * @param errorMessage error message
     * @param <T> type of data
     * @return error event
     */
    public static <T> ConnectorEvent<T> error(String partnerId, String errorCode, String errorMessage) {
        ConnectorEvent<T> event = new ConnectorEvent<>("ERROR", partnerId, null);
        event.priority = EventPriority.HIGH;
        event.withMetadata("errorCode", errorCode)
             .withMetadata("errorMessage", errorMessage);
        return event;
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
    
    public T getData() {
        return data;
    }
    
    public void setData(T data) {
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
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    public EventPriority getPriority() {
        return priority;
    }
    
    public void setPriority(EventPriority priority) {
        this.priority = priority;
    }
    
    public String getVersion() {
        return version;
    }
    
    public void setVersion(String version) {
        this.version = version;
    }
    
    /**
     * Event priority levels.
     */
    public enum EventPriority {
        LOW, NORMAL, HIGH, CRITICAL
    }
    
    @Override
    public String toString() {
        return "ConnectorEvent{" +
                "eventId='" + eventId + '\'' +
                ", eventType='" + eventType + '\'' +
                ", partnerId='" + partnerId + '\'' +
                ", source='" + source + '\'' +
                ", timestamp=" + timestamp +
                ", priority=" + priority +
                '}';
    }
}