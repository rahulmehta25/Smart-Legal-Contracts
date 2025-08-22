package com.enterprise.integrations.streaming;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Registry for event processors.
 * 
 * Manages the registration and discovery of event processors
 * for different event types in the integration platform.
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Component
public class EventProcessorRegistry {
    
    private static final Logger logger = LoggerFactory.getLogger(EventProcessorRegistry.class);
    
    private final Map<String, EventProcessor> processors;
    
    /**
     * Constructor with dependency injection.
     * 
     * @param eventProcessors list of available event processors
     */
    @Autowired
    public EventProcessorRegistry(List<EventProcessor> eventProcessors) {
        this.processors = new HashMap<>();
        registerProcessors(eventProcessors);
    }
    
    /**
     * Gets an event processor for the specified event type.
     * 
     * @param eventType event type
     * @return event processor or null if not found
     */
    public EventProcessor getProcessor(String eventType) {
        return processors.get(eventType);
    }
    
    /**
     * Registers an event processor.
     * 
     * @param processor event processor to register
     */
    public void registerProcessor(EventProcessor processor) {
        String eventType = processor.getEventType();
        processors.put(eventType, processor);
        logger.info("Registered event processor for type: {}", eventType);
    }
    
    /**
     * Unregisters an event processor.
     * 
     * @param eventType event type to unregister
     */
    public void unregisterProcessor(String eventType) {
        EventProcessor removed = processors.remove(eventType);
        if (removed != null) {
            logger.info("Unregistered event processor for type: {}", eventType);
        }
    }
    
    /**
     * Gets all registered event types.
     * 
     * @return set of registered event types
     */
    public java.util.Set<String> getRegisteredEventTypes() {
        return processors.keySet();
    }
    
    /**
     * Checks if a processor is registered for the given event type.
     * 
     * @param eventType event type to check
     * @return true if processor is registered
     */
    public boolean hasProcessor(String eventType) {
        return processors.containsKey(eventType);
    }
    
    /**
     * Registers all provided event processors.
     * 
     * @param eventProcessors list of event processors to register
     */
    private void registerProcessors(List<EventProcessor> eventProcessors) {
        for (EventProcessor processor : eventProcessors) {
            registerProcessor(processor);
        }
        
        logger.info("Registered {} event processors", eventProcessors.size());
    }
}