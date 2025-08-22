package com.enterprise.integrations.monitoring;

import com.enterprise.integrations.connectors.PartnerConnector;
import com.enterprise.integrations.core.IntegrationProperties;
import com.enterprise.integrations.streaming.IntegrationEvent;
import com.enterprise.integrations.streaming.IntegrationEventService;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuator.health.Health;
import org.springframework.boot.actuator.health.HealthIndicator;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Integration health monitoring service.
 * 
 * Provides comprehensive health monitoring for the integration platform including:
 * - Partner connector health checks
 * - System performance monitoring
 * - SLA tracking and alerting
 * - Error rate monitoring
 * - Real-time metrics collection
 * - Automated health reporting
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Service
public class IntegrationHealthService implements HealthIndicator {
    
    private static final Logger logger = LoggerFactory.getLogger(IntegrationHealthService.class);
    
    private final Map<String, PartnerConnector<?>> connectors;
    private final IntegrationProperties properties;
    private final IntegrationEventService eventService;
    private final MeterRegistry meterRegistry;
    
    // Health metrics
    private final AtomicInteger healthyConnectors = new AtomicInteger(0);
    private final AtomicInteger unhealthyConnectors = new AtomicInteger(0);
    private final AtomicLong totalHealthChecks = new AtomicLong(0);
    private final AtomicLong failedHealthChecks = new AtomicLong(0);
    private final Map<String, LocalDateTime> lastHealthCheck = new HashMap<>();
    private final Map<String, Boolean> connectorHealthStatus = new HashMap<>();
    
    // Performance metrics
    private final Counter requestCounter;
    private final Counter errorCounter;
    private final Timer responseTimer;
    private final Gauge healthScore;
    
    /**
     * Constructor with dependency injection.
     * 
     * @param connectors available partner connectors
     * @param properties integration properties
     * @param eventService integration event service
     * @param meterRegistry metrics registry
     */
    @Autowired
    public IntegrationHealthService(Map<String, PartnerConnector<?>> connectors,
                                   IntegrationProperties properties,
                                   IntegrationEventService eventService,
                                   MeterRegistry meterRegistry) {
        this.connectors = connectors;
        this.properties = properties;
        this.eventService = eventService;
        this.meterRegistry = meterRegistry;
        
        // Initialize metrics
        this.requestCounter = Counter.builder("integration.requests.total")
            .description("Total number of integration requests")
            .register(meterRegistry);
        
        this.errorCounter = Counter.builder("integration.errors.total")
            .description("Total number of integration errors")
            .register(meterRegistry);
        
        this.responseTimer = Timer.builder("integration.response.time")
            .description("Integration response time")
            .register(meterRegistry);
        
        this.healthScore = Gauge.builder("integration.health.score")
            .description("Overall integration health score (0-100)")
            .register(meterRegistry, this, IntegrationHealthService::calculateHealthScore);
        
        // Initialize connector health status
        connectors.keySet().forEach(partnerId -> {
            connectorHealthStatus.put(partnerId, false);
            lastHealthCheck.put(partnerId, LocalDateTime.now().minusHours(1));
        });
    }
    
    @Override
    public Health health() {
        Health.Builder healthBuilder = new Health.Builder();
        
        try {
            Map<String, Object> details = new HashMap<>();
            
            // Overall system health
            boolean systemHealthy = isSystemHealthy();
            
            // Connector health details
            Map<String, Object> connectorHealth = new HashMap<>();
            for (Map.Entry<String, PartnerConnector<?>> entry : connectors.entrySet()) {
                String partnerId = entry.getKey();
                Boolean isHealthy = connectorHealthStatus.get(partnerId);
                LocalDateTime lastCheck = lastHealthCheck.get(partnerId);
                
                Map<String, Object> connectorDetails = new HashMap<>();
                connectorDetails.put("healthy", isHealthy != null ? isHealthy : false);
                connectorDetails.put("lastCheck", lastCheck);
                connectorDetails.put("status", isHealthy != null && isHealthy ? "UP" : "DOWN");
                
                connectorHealth.put(partnerId, connectorDetails);
            }
            
            details.put("connectors", connectorHealth);
            details.put("healthyConnectors", healthyConnectors.get());
            details.put("unhealthyConnectors", unhealthyConnectors.get());
            details.put("totalHealthChecks", totalHealthChecks.get());
            details.put("failedHealthChecks", failedHealthChecks.get());
            details.put("healthScore", calculateHealthScore());
            details.put("lastHealthCheck", LocalDateTime.now());
            
            if (systemHealthy) {
                return healthBuilder.up().withDetails(details).build();
            } else {
                return healthBuilder.down().withDetails(details).build();
            }
            
        } catch (Exception e) {
            logger.error("Error checking system health", e);
            return healthBuilder.down().withException(e).build();
        }
    }
    
    /**
     * Scheduled health check for all partner connectors.
     */
    @Scheduled(fixedRateString = "${integration.platform.monitoring.health-check-interval:60000}")
    public void performScheduledHealthChecks() {
        logger.debug("Performing scheduled health checks for {} connectors", connectors.size());
        
        for (Map.Entry<String, PartnerConnector<?>> entry : connectors.entrySet()) {
            String partnerId = entry.getKey();
            PartnerConnector<?> connector = entry.getValue();
            
            CompletableFuture.runAsync(() -> performConnectorHealthCheck(partnerId, connector))
                .exceptionally(throwable -> {
                    logger.error("Error during health check for connector: {}", partnerId, throwable);
                    return null;
                });
        }
    }
    
    /**
     * Performs health check for a specific partner connector.
     * 
     * @param partnerId partner identifier
     * @param connector partner connector
     */
    public void performConnectorHealthCheck(String partnerId, PartnerConnector<?> connector) {
        logger.debug("Performing health check for connector: {}", partnerId);
        
        totalHealthChecks.incrementAndGet();
        LocalDateTime checkTime = LocalDateTime.now();
        
        try {
            Timer.Sample sample = Timer.start(meterRegistry);
            
            CompletableFuture<PartnerConnector.HealthStatus> healthFuture = connector.healthCheck();
            PartnerConnector.HealthStatus healthStatus = healthFuture.get(30, java.util.concurrent.TimeUnit.SECONDS);
            
            sample.stop(Timer.builder("integration.health.check.duration")
                .tag("partner", partnerId)
                .register(meterRegistry));
            
            boolean isHealthy = healthStatus.healthy();
            connectorHealthStatus.put(partnerId, isHealthy);
            lastHealthCheck.put(partnerId, checkTime);
            
            if (isHealthy) {
                logger.debug("Connector {} is healthy", partnerId);
                updateHealthyConnectorCount();
            } else {
                logger.warn("Connector {} is unhealthy: {}", partnerId, healthStatus.status());
                failedHealthChecks.incrementAndGet();
                updateHealthyConnectorCount();
                
                // Publish health event
                publishHealthEvent(partnerId, "UNHEALTHY", healthStatus.details());
                
                // Trigger alert if error threshold exceeded
                if (shouldTriggerAlert(partnerId)) {
                    triggerHealthAlert(partnerId, healthStatus);
                }
            }
            
        } catch (Exception e) {
            logger.error("Health check failed for connector: {}", partnerId, e);
            failedHealthChecks.incrementAndGet();
            connectorHealthStatus.put(partnerId, false);
            lastHealthCheck.put(partnerId, checkTime);
            updateHealthyConnectorCount();
            
            // Publish error event
            Map<String, Object> errorContext = new HashMap<>();
            errorContext.put("partnerId", partnerId);
            errorContext.put("operation", "health-check");
            errorContext.put("error", e.getMessage());
            
            publishHealthEvent(partnerId, "HEALTH_CHECK_FAILED", errorContext);
        }
    }
    
    /**
     * Records a successful integration operation.
     * 
     * @param partnerId partner identifier
     * @param operation operation type
     * @param responseTime response time in milliseconds
     */
    public void recordSuccessfulOperation(String partnerId, String operation, long responseTime) {
        requestCounter.increment(
            "partner", partnerId,
            "operation", operation,
            "status", "success"
        );
        
        responseTimer.record(responseTime, java.util.concurrent.TimeUnit.MILLISECONDS,
            "partner", partnerId,
            "operation", operation
        );
    }
    
    /**
     * Records a failed integration operation.
     * 
     * @param partnerId partner identifier
     * @param operation operation type
     * @param errorCode error code
     * @param responseTime response time in milliseconds
     */
    public void recordFailedOperation(String partnerId, String operation, String errorCode, long responseTime) {
        requestCounter.increment(
            "partner", partnerId,
            "operation", operation,
            "status", "failed"
        );
        
        errorCounter.increment(
            "partner", partnerId,
            "operation", operation,
            "error.code", errorCode
        );
        
        responseTimer.record(responseTime, java.util.concurrent.TimeUnit.MILLISECONDS,
            "partner", partnerId,
            "operation", operation
        );
        
        // Check if error threshold exceeded
        if (isErrorThresholdExceeded(partnerId)) {
            triggerErrorAlert(partnerId, operation, errorCode);
        }
    }
    
    /**
     * Gets current health metrics for all connectors.
     * 
     * @return health metrics map
     */
    public Map<String, Object> getHealthMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        metrics.put("totalConnectors", connectors.size());
        metrics.put("healthyConnectors", healthyConnectors.get());
        metrics.put("unhealthyConnectors", unhealthyConnectors.get());
        metrics.put("healthScore", calculateHealthScore());
        metrics.put("totalHealthChecks", totalHealthChecks.get());
        metrics.put("failedHealthChecks", failedHealthChecks.get());
        metrics.put("successRate", calculateSuccessRate());
        metrics.put("lastUpdate", LocalDateTime.now());
        
        // Per-connector metrics
        Map<String, Object> connectorMetrics = new HashMap<>();
        for (String partnerId : connectors.keySet()) {
            Map<String, Object> partnerMetrics = new HashMap<>();
            partnerMetrics.put("healthy", connectorHealthStatus.getOrDefault(partnerId, false));
            partnerMetrics.put("lastCheck", lastHealthCheck.get(partnerId));
            partnerMetrics.put("rateLimitStatus", getRateLimitStatus(partnerId));
            connectorMetrics.put(partnerId, partnerMetrics);
        }
        metrics.put("connectorDetails", connectorMetrics);
        
        return metrics;
    }
    
    /**
     * Gets SLA compliance metrics.
     * 
     * @return SLA metrics
     */
    public Map<String, Object> getSLAMetrics() {
        Map<String, Object> slaMetrics = new HashMap<>();
        
        // Calculate uptime percentage
        double uptimePercentage = calculateUptimePercentage();
        slaMetrics.put("uptimePercentage", uptimePercentage);
        slaMetrics.put("slaCompliant", uptimePercentage >= 99.0); // 99% SLA threshold
        
        // Average response time
        Timer.Sample responseTimeSample = responseTimer.takeSnapshot();
        slaMetrics.put("averageResponseTime", responseTimeSample.mean());
        slaMetrics.put("p95ResponseTime", responseTimeSample.percentile(0.95));
        slaMetrics.put("p99ResponseTime", responseTimeSample.percentile(0.99));
        
        // Error rates
        double errorRate = calculateErrorRate();
        slaMetrics.put("errorRate", errorRate);
        slaMetrics.put("errorRateCompliant", errorRate <= 1.0); // 1% error rate threshold
        
        return slaMetrics;
    }
    
    /**
     * Checks if the overall system is healthy.
     * 
     * @return true if system is healthy
     */
    private boolean isSystemHealthy() {
        int totalConnectors = connectors.size();
        if (totalConnectors == 0) {
            return true; // No connectors to check
        }
        
        int healthy = healthyConnectors.get();
        double healthPercentage = (double) healthy / totalConnectors * 100;
        
        // System is healthy if at least 80% of connectors are healthy
        return healthPercentage >= 80.0;
    }
    
    /**
     * Calculates overall health score (0-100).
     * 
     * @return health score
     */
    private double calculateHealthScore() {
        int totalConnectors = connectors.size();
        if (totalConnectors == 0) {
            return 100.0;
        }
        
        int healthy = healthyConnectors.get();
        double baseScore = (double) healthy / totalConnectors * 100;
        
        // Adjust score based on error rate
        double errorRate = calculateErrorRate();
        double errorPenalty = Math.min(errorRate * 10, 20); // Max 20 point penalty
        
        return Math.max(0, baseScore - errorPenalty);
    }
    
    /**
     * Calculates success rate percentage.
     * 
     * @return success rate percentage
     */
    private double calculateSuccessRate() {
        long total = totalHealthChecks.get();
        long failed = failedHealthChecks.get();
        
        if (total == 0) {
            return 100.0;
        }
        
        return ((double) (total - failed) / total) * 100;
    }
    
    /**
     * Calculates uptime percentage.
     * 
     * @return uptime percentage
     */
    private double calculateUptimePercentage() {
        // Simplified calculation - in production would track actual uptime
        return calculateSuccessRate();
    }
    
    /**
     * Calculates error rate percentage.
     * 
     * @return error rate percentage
     */
    private double calculateErrorRate() {
        double requestCount = requestCounter.count();
        double errorCount = errorCounter.count();
        
        if (requestCount == 0) {
            return 0.0;
        }
        
        return (errorCount / requestCount) * 100;
    }
    
    /**
     * Updates healthy connector count.
     */
    private void updateHealthyConnectorCount() {
        int healthy = 0;
        for (Boolean status : connectorHealthStatus.values()) {
            if (Boolean.TRUE.equals(status)) {
                healthy++;
            }
        }
        
        healthyConnectors.set(healthy);
        unhealthyConnectors.set(connectors.size() - healthy);
    }
    
    /**
     * Gets rate limit status for a partner.
     * 
     * @param partnerId partner identifier
     * @return rate limit status
     */
    private Map<String, Object> getRateLimitStatus(String partnerId) {
        PartnerConnector<?> connector = connectors.get(partnerId);
        if (connector != null) {
            PartnerConnector.RateLimitStatus rateLimitStatus = connector.getRateLimitStatus();
            Map<String, Object> status = new HashMap<>();
            status.put("remaining", rateLimitStatus.remaining());
            status.put("limit", rateLimitStatus.limit());
            status.put("resetTime", rateLimitStatus.resetTime());
            status.put("throttled", rateLimitStatus.throttled());
            return status;
        }
        return new HashMap<>();
    }
    
    /**
     * Checks if alert should be triggered for a partner.
     * 
     * @param partnerId partner identifier
     * @return true if alert should be triggered
     */
    private boolean shouldTriggerAlert(String partnerId) {
        // Simple threshold-based alerting
        long recentFailures = failedHealthChecks.get();
        return recentFailures >= properties.getMonitoring().getErrorAlertThreshold();
    }
    
    /**
     * Checks if error threshold is exceeded for a partner.
     * 
     * @param partnerId partner identifier
     * @return true if error threshold is exceeded
     */
    private boolean isErrorThresholdExceeded(String partnerId) {
        // Simplified check - in production would track per-partner error rates
        return calculateErrorRate() > 5.0; // 5% error rate threshold
    }
    
    /**
     * Publishes a health event.
     * 
     * @param partnerId partner identifier
     * @param status health status
     * @param details health details
     */
    private void publishHealthEvent(String partnerId, String status, Map<String, Object> details) {
        try {
            IntegrationEvent healthEvent = IntegrationEvent.healthEvent(partnerId, status, details);
            eventService.publishEvent(healthEvent);
        } catch (Exception e) {
            logger.error("Failed to publish health event for partner: {}", partnerId, e);
        }
    }
    
    /**
     * Triggers health alert for a partner.
     * 
     * @param partnerId partner identifier
     * @param healthStatus health status
     */
    private void triggerHealthAlert(String partnerId, PartnerConnector.HealthStatus healthStatus) {
        logger.warn("Triggering health alert for partner: {} - Status: {}", partnerId, healthStatus.status());
        
        Map<String, Object> alertContext = new HashMap<>();
        alertContext.put("partnerId", partnerId);
        alertContext.put("healthStatus", healthStatus.status());
        alertContext.put("details", healthStatus.details());
        alertContext.put("alertType", "HEALTH_ALERT");
        
        try {
            eventService.publishErrorEvent(partnerId, "HEALTH_ALERT", 
                "Partner connector health alert triggered", alertContext);
        } catch (Exception e) {
            logger.error("Failed to publish health alert for partner: {}", partnerId, e);
        }
    }
    
    /**
     * Triggers error alert for a partner.
     * 
     * @param partnerId partner identifier
     * @param operation operation type
     * @param errorCode error code
     */
    private void triggerErrorAlert(String partnerId, String operation, String errorCode) {
        logger.warn("Triggering error alert for partner: {} - Operation: {} - Error: {}", 
                   partnerId, operation, errorCode);
        
        Map<String, Object> alertContext = new HashMap<>();
        alertContext.put("partnerId", partnerId);
        alertContext.put("operation", operation);
        alertContext.put("errorCode", errorCode);
        alertContext.put("errorRate", calculateErrorRate());
        alertContext.put("alertType", "ERROR_RATE_ALERT");
        
        try {
            eventService.publishErrorEvent(partnerId, "ERROR_RATE_ALERT", 
                "Error rate threshold exceeded", alertContext);
        } catch (Exception e) {
            logger.error("Failed to publish error alert for partner: {}", partnerId, e);
        }
    }
}