package com.enterprise.integrations.streaming;

/**
 * Interface for processing integration events.
 * 
 * Event processors handle specific types of events and execute
 * appropriate business logic, transformations, or routing.
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
public interface EventProcessor {
    
    /**
     * Gets the event type this processor handles.
     * 
     * @return event type
     */
    String getEventType();
    
    /**
     * Processes an integration event.
     * 
     * @param event the integration event to process
     * @throws EventProcessingException if processing fails
     */
    void processEvent(IntegrationEvent event) throws EventProcessingException;
    
    /**
     * Checks if this processor can handle the given event type.
     * 
     * @param eventType event type to check
     * @return true if processor can handle the event type
     */
    default boolean canProcess(String eventType) {
        return getEventType().equals(eventType);
    }
}

/**
 * Exception thrown during event processing.
 */
class EventProcessingException extends RuntimeException {
    public EventProcessingException(String message) {
        super(message);
    }
    
    public EventProcessingException(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * Exception thrown during event publishing.
 */
class EventPublishingException extends RuntimeException {
    private final IntegrationEvent event;
    
    public EventPublishingException(String message, IntegrationEvent event) {
        super(message);
        this.event = event;
    }
    
    public EventPublishingException(String message, IntegrationEvent event, Throwable cause) {
        super(message, cause);
        this.event = event;
    }
    
    public IntegrationEvent getEvent() {
        return event;
    }
}

/**
 * Exception thrown during event parsing.
 */
class EventParsingException extends RuntimeException {
    public EventParsingException(String message, Throwable cause) {
        super(message, cause);
    }
}