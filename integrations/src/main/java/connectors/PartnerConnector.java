package com.enterprise.integrations.connectors;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Base interface for all partner connectors.
 * 
 * Defines the contract that all partner integrations must implement:
 * - Connection management and health checks
 * - Data synchronization operations
 * - Event handling and webhooks
 * - Error handling and retry mechanisms
 * - Rate limiting compliance
 * 
 * @param <T> The type of data this connector handles
 * @author Integration Platform Team
 * @version 1.0.0
 */
public interface PartnerConnector<T> {

    /**
     * Gets the unique identifier for this connector.
     * 
     * @return connector identifier (e.g., "salesforce", "microsoft365")
     */
    String getConnectorId();

    /**
     * Gets the display name for this connector.
     * 
     * @return human-readable connector name
     */
    String getDisplayName();

    /**
     * Gets the current version of this connector.
     * 
     * @return connector version
     */
    String getVersion();

    /**
     * Establishes connection to the partner system.
     * 
     * @return CompletableFuture that completes when connection is established
     * @throws ConnectorException if connection cannot be established
     */
    CompletableFuture<Void> connect();

    /**
     * Closes connection to the partner system.
     * 
     * @return CompletableFuture that completes when connection is closed
     */
    CompletableFuture<Void> disconnect();

    /**
     * Checks if the connector is currently connected and healthy.
     * 
     * @return true if connected and healthy, false otherwise
     */
    boolean isHealthy();

    /**
     * Performs a health check against the partner system.
     * 
     * @return CompletableFuture<HealthStatus> with detailed health information
     */
    CompletableFuture<HealthStatus> healthCheck();

    /**
     * Retrieves data from the partner system.
     * 
     * @param query the query parameters for data retrieval
     * @return CompletableFuture containing the retrieved data
     * @throws ConnectorException if data retrieval fails
     */
    CompletableFuture<ConnectorResponse<T>> retrieveData(ConnectorRequest query);

    /**
     * Sends data to the partner system.
     * 
     * @param data the data to send
     * @return CompletableFuture containing the operation result
     * @throws ConnectorException if data sending fails
     */
    CompletableFuture<ConnectorResponse<T>> sendData(T data);

    /**
     * Updates existing data in the partner system.
     * 
     * @param id the identifier of the data to update
     * @param data the updated data
     * @return CompletableFuture containing the operation result
     * @throws ConnectorException if data update fails
     */
    CompletableFuture<ConnectorResponse<T>> updateData(String id, T data);

    /**
     * Deletes data from the partner system.
     * 
     * @param id the identifier of the data to delete
     * @return CompletableFuture containing the operation result
     * @throws ConnectorException if data deletion fails
     */
    CompletableFuture<ConnectorResponse<Void>> deleteData(String id);

    /**
     * Performs bulk operations on the partner system.
     * 
     * @param operation the bulk operation to perform
     * @return CompletableFuture containing the bulk operation result
     * @throws ConnectorException if bulk operation fails
     */
    CompletableFuture<BulkOperationResponse<T>> performBulkOperation(BulkOperation<T> operation);

    /**
     * Subscribes to events from the partner system.
     * 
     * @param eventTypes the types of events to subscribe to
     * @param callback the callback to invoke when events are received
     * @return subscription identifier for managing the subscription
     * @throws ConnectorException if subscription fails
     */
    String subscribeToEvents(String[] eventTypes, EventCallback<T> callback);

    /**
     * Unsubscribes from events.
     * 
     * @param subscriptionId the subscription identifier
     * @return CompletableFuture that completes when unsubscribed
     */
    CompletableFuture<Void> unsubscribeFromEvents(String subscriptionId);

    /**
     * Gets current rate limit status.
     * 
     * @return current rate limit information
     */
    RateLimitStatus getRateLimitStatus();

    /**
     * Gets connector-specific configuration.
     * 
     * @return configuration map
     */
    Map<String, Object> getConfiguration();

    /**
     * Gets connector statistics.
     * 
     * @return current connector metrics and statistics
     */
    ConnectorMetrics getMetrics();

    /**
     * Health status information for a connector.
     */
    record HealthStatus(
        boolean healthy,
        String status,
        LocalDateTime lastCheck,
        Map<String, Object> details
    ) {}

    /**
     * Rate limit status information.
     */
    record RateLimitStatus(
        int remaining,
        int limit,
        LocalDateTime resetTime,
        boolean throttled
    ) {}

    /**
     * Connector metrics and statistics.
     */
    record ConnectorMetrics(
        long totalRequests,
        long successfulRequests,
        long failedRequests,
        double averageResponseTime,
        LocalDateTime lastActivity
    ) {}

    /**
     * Functional interface for event callbacks.
     * 
     * @param <T> the type of event data
     */
    @FunctionalInterface
    interface EventCallback<T> {
        void onEvent(ConnectorEvent<T> event);
    }
}