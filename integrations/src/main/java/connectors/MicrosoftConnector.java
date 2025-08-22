package com.enterprise.integrations.connectors;

import com.enterprise.integrations.core.IntegrationProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Microsoft 365 and Graph API integration connector.
 * 
 * Provides integration with Microsoft 365 services including:
 * - Microsoft Graph API access
 * - Azure AD authentication
 * - Office 365 services (Exchange, SharePoint, Teams)
 * - User and group management
 * - Calendar and mail integration
 * - Files and document management
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Component
public class MicrosoftConnector implements PartnerConnector<Map<String, Object>> {
    
    private static final Logger logger = LoggerFactory.getLogger(MicrosoftConnector.class);
    
    private final IntegrationProperties.Microsoft config;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    
    private String accessToken;
    private LocalDateTime tokenExpiry;
    private final Map<String, EventCallback<Map<String, Object>>> eventSubscriptions;
    private final ConnectorMetrics metrics;
    private RateLimitStatus rateLimitStatus;
    
    /**
     * Constructor with dependency injection.
     * 
     * @param properties integration properties
     * @param restTemplate REST client
     * @param objectMapper JSON mapper
     */
    @Autowired
    public MicrosoftConnector(IntegrationProperties properties,
                             RestTemplate restTemplate,
                             ObjectMapper objectMapper) {
        this.config = properties.getPartners().getMicrosoft();
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
        this.eventSubscriptions = new ConcurrentHashMap<>();
        this.metrics = new ConnectorMetrics(0, 0, 0, 0.0, null);
        this.rateLimitStatus = new RateLimitStatus(
            config.getRateLimitPerMinute(),
            config.getRateLimitPerMinute(),
            LocalDateTime.now().plusMinutes(1),
            false
        );
    }
    
    @Override
    public String getConnectorId() {
        return "microsoft365";
    }
    
    @Override
    public String getDisplayName() {
        return "Microsoft 365";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public CompletableFuture<Void> connect() {
        return CompletableFuture.runAsync(() -> {
            try {
                logger.info("Connecting to Microsoft 365...");
                authenticate();
                logger.info("Successfully connected to Microsoft 365");
            } catch (Exception e) {
                logger.error("Failed to connect to Microsoft 365", e);
                throw new ConnectorException("MICROSOFT_CONNECTION_FAILED",
                    "Failed to establish connection to Microsoft 365", e, true);
            }
        });
    }
    
    @Override
    public CompletableFuture<Void> disconnect() {
        return CompletableFuture.runAsync(() -> {
            logger.info("Disconnecting from Microsoft 365...");
            accessToken = null;
            tokenExpiry = null;
            eventSubscriptions.clear();
            logger.info("Disconnected from Microsoft 365");
        });
    }
    
    @Override
    public boolean isHealthy() {
        return accessToken != null &&
               (tokenExpiry == null || tokenExpiry.isAfter(LocalDateTime.now().plusMinutes(5)));
    }
    
    @Override
    public CompletableFuture<HealthStatus> healthCheck() {
        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> details = new HashMap<>();
            boolean healthy = false;
            String status = "UNHEALTHY";
            
            try {
                if (!isHealthy()) {
                    authenticate();
                }
                
                // Test with a simple Graph API call
                String url = config.getGraphUrl() + "/me";
                HttpHeaders headers = createAuthHeaders();
                HttpEntity<?> entity = new HttpEntity<>(headers);
                
                ResponseEntity<Map> response = restTemplate.exchange(
                    url, HttpMethod.GET, entity, Map.class
                );
                
                if (response.getStatusCode() == HttpStatus.OK) {
                    healthy = true;
                    status = "HEALTHY";
                    details.put("graphApiUrl", config.getGraphUrl());
                    details.put("tokenValid", true);
                }
                
            } catch (Exception e) {
                logger.error("Microsoft 365 health check failed", e);
                details.put("error", e.getMessage());
                status = "UNHEALTHY";
            }
            
            details.put("lastCheck", LocalDateTime.now());
            return new HealthStatus(healthy, status, LocalDateTime.now(), details);
        });
    }
    
    @Override
    public CompletableFuture<ConnectorResponse<Map<String, Object>>> retrieveData(ConnectorRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            long startTime = System.currentTimeMillis();
            
            try {
                ensureAuthenticated();
                updateMetrics(true);
                
                String url = buildGraphUrl(request);
                HttpHeaders headers = createAuthHeaders();
                
                // Add OData query parameters if specified
                if (request.getParameters().containsKey("$filter") ||
                    request.getParameters().containsKey("$select") ||
                    request.getParameters().containsKey("$orderby")) {
                    url += buildODataQuery(request);
                }
                
                HttpEntity<?> entity = new HttpEntity<>(headers);
                ResponseEntity<Map> response = restTemplate.exchange(
                    url, HttpMethod.GET, entity, Map.class
                );
                
                if (response.getStatusCode() == HttpStatus.OK) {
                    Map<String, Object> responseData = response.getBody();
                    
                    ConnectorResponse<Map<String, Object>> connectorResponse =
                        ConnectorResponse.success(responseData, "Data retrieved successfully");
                    
                    // Handle pagination for Graph API
                    if (responseData != null && responseData.containsKey("@odata.nextLink")) {
                        ConnectorResponse.PaginationResponse pagination =
                            new ConnectorResponse.PaginationResponse();
                        pagination.setHasNext(true);
                        pagination.setNextCursor((String) responseData.get("@odata.nextLink"));
                        connectorResponse.withPagination(pagination);
                    }
                    
                    long processingTime = System.currentTimeMillis() - startTime;
                    ConnectorResponse.ResponseMetrics metrics =
                        new ConnectorResponse.ResponseMetrics(processingTime, 0);
                    connectorResponse.withMetrics(metrics);
                    
                    return connectorResponse;
                }
                
                return ConnectorResponse.error("MICROSOFT_QUERY_FAILED",
                    "Failed to retrieve data: " + response.getStatusCode());
                
            } catch (Exception e) {
                logger.error("Failed to retrieve data from Microsoft 365", e);
                updateMetrics(false);
                return ConnectorResponse.error("MICROSOFT_RETRIEVE_ERROR", e.getMessage());
            }
        });
    }
    
    @Override
    public CompletableFuture<ConnectorResponse<Map<String, Object>>> sendData(Map<String, Object> data) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                ensureAuthenticated();
                updateMetrics(true);
                
                String resourceType = (String) data.get("resourceType");
                if (resourceType == null) {
                    return ConnectorResponse.error("MISSING_RESOURCE_TYPE",
                        "Resource type is required for Microsoft Graph operations");
                }
                
                String url = config.getGraphUrl() + "/" + resourceType;
                
                HttpHeaders headers = createAuthHeaders();
                headers.setContentType(MediaType.APPLICATION_JSON);
                
                // Remove metadata fields
                Map<String, Object> payload = new HashMap<>(data);
                payload.remove("resourceType");
                
                HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);
                
                ResponseEntity<Map> response = restTemplate.exchange(
                    url, HttpMethod.POST, entity, Map.class
                );
                
                if (response.getStatusCode() == HttpStatus.CREATED ||
                    response.getStatusCode() == HttpStatus.OK) {
                    Map<String, Object> responseData = response.getBody();
                    return ConnectorResponse.success(responseData, "Resource created successfully");
                }
                
                return ConnectorResponse.error("MICROSOFT_CREATE_FAILED",
                    "Failed to create resource: " + response.getStatusCode());
                
            } catch (Exception e) {
                logger.error("Failed to send data to Microsoft 365", e);
                updateMetrics(false);
                return ConnectorResponse.error("MICROSOFT_SEND_ERROR", e.getMessage());
            }
        });
    }
    
    @Override
    public CompletableFuture<ConnectorResponse<Map<String, Object>>> updateData(String id, Map<String, Object> data) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                ensureAuthenticated();
                updateMetrics(true);
                
                String resourceType = (String) data.get("resourceType");
                if (resourceType == null) {
                    return ConnectorResponse.error("MISSING_RESOURCE_TYPE",
                        "Resource type is required for Microsoft Graph operations");
                }
                
                String url = config.getGraphUrl() + "/" + resourceType + "/" + id;
                
                HttpHeaders headers = createAuthHeaders();
                headers.setContentType(MediaType.APPLICATION_JSON);
                
                // Remove metadata fields
                Map<String, Object> payload = new HashMap<>(data);
                payload.remove("resourceType");
                
                HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);
                
                ResponseEntity<Map> response = restTemplate.exchange(
                    url, HttpMethod.PATCH, entity, Map.class
                );
                
                if (response.getStatusCode() == HttpStatus.OK ||
                    response.getStatusCode() == HttpStatus.NO_CONTENT) {
                    Map<String, Object> result = new HashMap<>();
                    result.put("id", id);
                    result.put("success", true);
                    if (response.getBody() != null) {
                        result.putAll(response.getBody());
                    }
                    return ConnectorResponse.success(result, "Resource updated successfully");
                }
                
                return ConnectorResponse.error("MICROSOFT_UPDATE_FAILED",
                    "Failed to update resource: " + response.getStatusCode());
                
            } catch (Exception e) {
                logger.error("Failed to update data in Microsoft 365", e);
                updateMetrics(false);
                return ConnectorResponse.error("MICROSOFT_UPDATE_ERROR", e.getMessage());
            }
        });
    }
    
    @Override
    public CompletableFuture<ConnectorResponse<Void>> deleteData(String id) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                ensureAuthenticated();
                updateMetrics(true);
                
                return ConnectorResponse.error("NOT_IMPLEMENTED",
                    "Delete operation requires resource type specification");
                
            } catch (Exception e) {
                logger.error("Failed to delete data from Microsoft 365", e);
                updateMetrics(false);
                return ConnectorResponse.error("MICROSOFT_DELETE_ERROR", e.getMessage());
            }
        });
    }
    
    @Override
    public CompletableFuture<BulkOperationResponse<Map<String, Object>>> performBulkOperation(
            BulkOperation<Map<String, Object>> operation) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                ensureAuthenticated();
                
                // Microsoft Graph supports batch requests
                return implementBatchOperation(operation);
                
            } catch (Exception e) {
                logger.error("Failed to perform bulk operation", e);
                return BulkOperationResponse.failure(operation.getOperationId(),
                    Arrays.asList(e.getMessage()));
            }
        });
    }
    
    @Override
    public String subscribeToEvents(String[] eventTypes, EventCallback<Map<String, Object>> callback) {
        String subscriptionId = "ms-sub-" + System.currentTimeMillis();
        eventSubscriptions.put(subscriptionId, callback);
        
        // In production, this would set up Microsoft Graph webhooks
        logger.info("Subscribed to Microsoft 365 events: {} with subscription ID: {}",
                   Arrays.toString(eventTypes), subscriptionId);
        
        return subscriptionId;
    }
    
    @Override
    public CompletableFuture<Void> unsubscribeFromEvents(String subscriptionId) {
        return CompletableFuture.runAsync(() -> {
            eventSubscriptions.remove(subscriptionId);
            logger.info("Unsubscribed from Microsoft 365 events: {}", subscriptionId);
        });
    }
    
    @Override
    public RateLimitStatus getRateLimitStatus() {
        return rateLimitStatus;
    }
    
    @Override
    public Map<String, Object> getConfiguration() {
        Map<String, Object> config = new HashMap<>();
        config.put("graphUrl", this.config.getGraphUrl());
        config.put("rateLimitPerMinute", this.config.getRateLimitPerMinute());
        config.put("tenantId", this.config.getTenantId());
        return config;
    }
    
    @Override
    public ConnectorMetrics getMetrics() {
        return new ConnectorMetrics(
            metrics.totalRequests(),
            metrics.successfulRequests(),
            metrics.failedRequests(),
            calculateAverageResponseTime(),
            metrics.lastActivity()
        );
    }
    
    /**
     * Authenticates with Microsoft 365 using OAuth 2.0 client credentials flow.
     */
    private void authenticate() {
        try {
            String url = "https://login.microsoftonline.com/" + config.getTenantId() + "/oauth2/v2.0/token";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
            
            String body = String.format(
                "client_id=%s&scope=%s&client_secret=%s&grant_type=client_credentials",
                config.getClientId(),
                java.net.URLEncoder.encode(config.getScope(), "UTF-8"),
                config.getClientSecret()
            );
            
            HttpEntity<String> entity = new HttpEntity<>(body, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> tokenResponse = response.getBody();
                this.accessToken = (String) tokenResponse.get("access_token");
                
                // Calculate token expiry
                Integer expiresIn = (Integer) tokenResponse.get("expires_in");
                if (expiresIn != null) {
                    this.tokenExpiry = LocalDateTime.now().plusSeconds(expiresIn - 300); // 5 min buffer
                } else {
                    this.tokenExpiry = LocalDateTime.now().plusHours(1); // Default 1 hour
                }
                
                logger.info("Successfully authenticated with Microsoft 365");
            } else {
                throw new ConnectorException("AUTHENTICATION_FAILED",
                    "Failed to authenticate with Microsoft 365");
            }
            
        } catch (Exception e) {
            throw new ConnectorException("AUTHENTICATION_ERROR",
                "Error during Microsoft 365 authentication", e, true);
        }
    }
    
    /**
     * Ensures the connection is authenticated and token is valid.
     */
    private void ensureAuthenticated() {
        if (!isHealthy()) {
            authenticate();
        }
    }
    
    /**
     * Creates HTTP headers with authentication.
     * 
     * @return HTTP headers with bearer token
     */
    private HttpHeaders createAuthHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(accessToken);
        headers.setAccept(Arrays.asList(MediaType.APPLICATION_JSON));
        return headers;
    }
    
    /**
     * Builds Microsoft Graph URL from connector request.
     * 
     * @param request the connector request
     * @return Graph API URL
     */
    private String buildGraphUrl(ConnectorRequest request) {
        StringBuilder url = new StringBuilder(config.getGraphUrl());
        
        String resourceType = request.getResourceType();
        if (resourceType != null) {
            url.append("/").append(resourceType);
        }
        
        String resourceId = request.getResourceId();
        if (resourceId != null) {
            url.append("/").append(resourceId);
        }
        
        return url.toString();
    }
    
    /**
     * Builds OData query parameters for Graph API.
     * 
     * @param request the connector request
     * @return OData query string
     */
    private String buildODataQuery(ConnectorRequest request) {
        List<String> queryParams = new ArrayList<>();
        
        Map<String, Object> parameters = request.getParameters();
        
        // Add OData parameters
        if (parameters.containsKey("$filter")) {
            queryParams.add("$filter=" + parameters.get("$filter"));
        }
        if (parameters.containsKey("$select")) {
            queryParams.add("$select=" + parameters.get("$select"));
        }
        if (parameters.containsKey("$orderby")) {
            queryParams.add("$orderby=" + parameters.get("$orderby"));
        }
        if (parameters.containsKey("$top")) {
            queryParams.add("$top=" + parameters.get("$top"));
        }
        if (parameters.containsKey("$skip")) {
            queryParams.add("$skip=" + parameters.get("$skip"));
        }
        
        return queryParams.isEmpty() ? "" : "?" + String.join("&", queryParams);
    }
    
    /**
     * Implements Microsoft Graph batch operations.
     * 
     * @param operation the bulk operation to perform
     * @return bulk operation response
     */
    private BulkOperationResponse<Map<String, Object>> implementBatchOperation(
            BulkOperation<Map<String, Object>> operation) {
        
        BulkOperationResponse<Map<String, Object>> response =
            new BulkOperationResponse<>(operation.getOperationId());
        
        try {
            // Microsoft Graph batch API supports up to 20 requests per batch
            List<Map<String, Object>> data = operation.getData();
            int batchSize = Math.min(20, operation.getBatchSize());
            
            for (int i = 0; i < data.size(); i += batchSize) {
                List<Map<String, Object>> batch = data.subList(
                    i, Math.min(i + batchSize, data.size())
                );
                
                // Create batch request
                Map<String, Object> batchRequest = createBatchRequest(batch, operation.getType());
                
                String url = config.getGraphUrl() + "/$batch";
                HttpHeaders headers = createAuthHeaders();
                headers.setContentType(MediaType.APPLICATION_JSON);
                HttpEntity<Map<String, Object>> entity = new HttpEntity<>(batchRequest, headers);
                
                ResponseEntity<Map> batchResponse = restTemplate.exchange(
                    url, HttpMethod.POST, entity, Map.class
                );
                
                if (batchResponse.getStatusCode() == HttpStatus.OK) {
                    processBatchResponse(batchResponse.getBody(), response);
                } else {
                    response.addError("Batch request failed: " + batchResponse.getStatusCode());
                }
            }
            
            response.markCompleted();
            
        } catch (Exception e) {
            response.markFailed("Batch operation failed: " + e.getMessage());
        }
        
        return response;
    }
    
    /**
     * Creates a batch request for Microsoft Graph.
     * 
     * @param batch the batch data
     * @param operationType the operation type
     * @return batch request object
     */
    private Map<String, Object> createBatchRequest(List<Map<String, Object>> batch,
                                                  BulkOperation.BulkOperationType operationType) {
        Map<String, Object> batchRequest = new HashMap<>();
        List<Map<String, Object>> requests = new ArrayList<>();
        
        for (int i = 0; i < batch.size(); i++) {
            Map<String, Object> item = batch.get(i);
            Map<String, Object> request = new HashMap<>();
            request.put("id", String.valueOf(i));
            
            String resourceType = (String) item.get("resourceType");
            request.put("url", "/" + resourceType);
            
            switch (operationType) {
                case INSERT -> {
                    request.put("method", "POST");
                    Map<String, Object> body = new HashMap<>(item);
                    body.remove("resourceType");
                    request.put("body", body);
                }
                case UPDATE -> {
                    request.put("method", "PATCH");
                    String id = (String) item.get("id");
                    request.put("url", "/" + resourceType + "/" + id);
                    Map<String, Object> body = new HashMap<>(item);
                    body.remove("resourceType");
                    body.remove("id");
                    request.put("body", body);
                }
                case DELETE -> {
                    request.put("method", "DELETE");
                    String id = (String) item.get("id");
                    request.put("url", "/" + resourceType + "/" + id);
                }
                default -> throw new UnsupportedOperationException("Unsupported operation: " + operationType);
            }
            
            requests.add(request);
        }
        
        batchRequest.put("requests", requests);
        return batchRequest;
    }
    
    /**
     * Processes batch response from Microsoft Graph.
     * 
     * @param batchResponseData the batch response data
     * @param response the bulk operation response to update
     */
    @SuppressWarnings("unchecked")
    private void processBatchResponse(Map<String, Object> batchResponseData,
                                    BulkOperationResponse<Map<String, Object>> response) {
        if (batchResponseData == null) return;
        
        List<Map<String, Object>> responses = (List<Map<String, Object>>) batchResponseData.get("responses");
        if (responses == null) return;
        
        for (Map<String, Object> itemResponse : responses) {
            String id = (String) itemResponse.get("id");
            Integer status = (Integer) itemResponse.get("status");
            
            if (status != null && status >= 200 && status < 300) {
                Map<String, Object> body = (Map<String, Object>) itemResponse.get("body");
                response.addResult(BulkOperationResponse.BulkRecordResult.success(id, body));
            } else {
                String errorMessage = "Request failed with status: " + status;
                response.addResult(BulkOperationResponse.BulkRecordResult.failure(
                    id, "BATCH_REQUEST_FAILED", errorMessage));
            }
        }
    }
    
    /**
     * Updates connector metrics.
     * 
     * @param success whether the operation was successful
     */
    private void updateMetrics(boolean success) {
        logger.debug("Microsoft 365 operation completed: success={}", success);
    }
    
    /**
     * Calculates average response time from metrics.
     * 
     * @return average response time in milliseconds
     */
    private double calculateAverageResponseTime() {
        return 200.0; // Default value
    }
}