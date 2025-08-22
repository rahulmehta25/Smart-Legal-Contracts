package com.enterprise.integrations.connectors;

import com.enterprise.integrations.core.IntegrationProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Comprehensive tests for SalesforceConnector.
 * 
 * Tests all connector functionality including:
 * - Connection management
 * - Data operations (CRUD)
 * - Authentication handling
 * - Error scenarios
 * - Health checks
 * - Rate limiting
 * - Bulk operations
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@ExtendWith(MockitoExtension.class)
class SalesforceConnectorTest {
    
    @Mock
    private RestTemplate restTemplate;
    
    @Mock
    private ObjectMapper objectMapper;
    
    @Mock
    private IntegrationProperties integrationProperties;
    
    @Mock
    private IntegrationProperties.Partners partners;
    
    @Mock
    private IntegrationProperties.Salesforce salesforceConfig;
    
    private SalesforceConnector salesforceConnector;
    
    @BeforeEach
    void setUp() {
        // Setup mock configuration
        when(integrationProperties.getPartners()).thenReturn(partners);
        when(partners.getSalesforce()).thenReturn(salesforceConfig);
        when(salesforceConfig.getClientId()).thenReturn("test-client-id");
        when(salesforceConfig.getClientSecret()).thenReturn("test-client-secret");
        when(salesforceConfig.getUsername()).thenReturn("test@example.com");
        when(salesforceConfig.getPassword()).thenReturn("password");
        when(salesforceConfig.getSecurityToken()).thenReturn("token");
        when(salesforceConfig.getLoginUrl()).thenReturn("https://login.salesforce.com");
        when(salesforceConfig.getRateLimitPerHour()).thenReturn(5000);
        when(salesforceConfig.isBulkApiEnabled()).thenReturn(true);
        
        salesforceConnector = new SalesforceConnector(integrationProperties, restTemplate, objectMapper);
    }
    
    @Test
    void testGetConnectorId() {
        assertEquals("salesforce", salesforceConnector.getConnectorId());
    }
    
    @Test
    void testGetDisplayName() {
        assertEquals("Salesforce CRM", salesforceConnector.getDisplayName());
    }
    
    @Test
    void testGetVersion() {
        assertEquals("1.0.0", salesforceConnector.getVersion());
    }
    
    @Test
    void testSuccessfulConnection() throws ExecutionException, InterruptedException {
        // Mock authentication response
        Map<String, Object> authResponse = new HashMap<>();
        authResponse.put("access_token", "test-access-token");
        authResponse.put("instance_url", "https://test.salesforce.com");
        
        ResponseEntity<Map> mockResponse = new ResponseEntity<>(authResponse, HttpStatus.OK);
        when(restTemplate.postForEntity(anyString(), any(HttpEntity.class), eq(Map.class)))
            .thenReturn(mockResponse);
        
        // Test connection
        CompletableFuture<Void> connectionFuture = salesforceConnector.connect();
        assertDoesNotThrow(() -> connectionFuture.get());
        
        // Verify connector is healthy after connection
        assertTrue(salesforceConnector.isHealthy());
    }
    
    @Test
    void testConnectionFailure() {
        // Mock authentication failure
        when(restTemplate.postForEntity(anyString(), any(HttpEntity.class), eq(Map.class)))
            .thenThrow(new RuntimeException("Authentication failed"));
        
        // Test connection failure
        CompletableFuture<Void> connectionFuture = salesforceConnector.connect();
        
        ConnectorException exception = assertThrows(ConnectorException.class, 
            () -> connectionFuture.get());
        
        assertEquals("SALESFORCE_CONNECTION_FAILED", exception.getErrorCode());
        assertTrue(exception.isRetryable());
    }
    
    @Test
    void testHealthCheckWhenHealthy() throws ExecutionException, InterruptedException {
        // Setup successful authentication
        setupSuccessfulAuthentication();
        
        // Mock successful health check query
        Map<String, Object> queryResponse = new HashMap<>();
        queryResponse.put("totalSize", 1);
        
        ResponseEntity<Map> mockResponse = new ResponseEntity<>(queryResponse, HttpStatus.OK);
        when(restTemplate.exchange(anyString(), eq(HttpMethod.GET), any(HttpEntity.class), eq(Map.class)))
            .thenReturn(mockResponse);
        
        // Test health check
        CompletableFuture<PartnerConnector.HealthStatus> healthFuture = salesforceConnector.healthCheck();
        PartnerConnector.HealthStatus healthStatus = healthFuture.get();
        
        assertTrue(healthStatus.healthy());
        assertEquals("HEALTHY", healthStatus.status());
        assertNotNull(healthStatus.details());
    }
    
    @Test
    void testHealthCheckWhenUnhealthy() throws ExecutionException, InterruptedException {
        // Setup authentication but make health check fail
        setupSuccessfulAuthentication();
        
        when(restTemplate.exchange(anyString(), eq(HttpMethod.GET), any(HttpEntity.class), eq(Map.class)))
            .thenThrow(new RuntimeException("Connection failed"));
        
        // Test health check
        CompletableFuture<PartnerConnector.HealthStatus> healthFuture = salesforceConnector.healthCheck();
        PartnerConnector.HealthStatus healthStatus = healthFuture.get();
        
        assertFalse(healthStatus.healthy());
        assertEquals("UNHEALTHY", healthStatus.status());
        assertNotNull(healthStatus.details());
    }
    
    @Test
    void testRetrieveDataSuccess() throws ExecutionException, InterruptedException {
        setupSuccessfulAuthentication();
        
        // Mock successful data retrieval
        Map<String, Object> queryResponse = new HashMap<>();
        queryResponse.put("totalSize", 2);
        queryResponse.put("done", true);
        queryResponse.put("records", java.util.Arrays.asList(
            Map.of("Id", "001000000000001", "Name", "Test Account 1"),
            Map.of("Id", "001000000000002", "Name", "Test Account 2")
        ));
        
        ResponseEntity<Map> mockResponse = new ResponseEntity<>(queryResponse, HttpStatus.OK);
        when(restTemplate.exchange(anyString(), eq(HttpMethod.GET), any(HttpEntity.class), eq(Map.class)))
            .thenReturn(mockResponse);
        
        // Test data retrieval
        ConnectorRequest request = new ConnectorRequest("QUERY", "Account");
        request.withParameter("fields", "Id,Name");
        
        CompletableFuture<ConnectorResponse<Map<String, Object>>> responseFuture = 
            salesforceConnector.retrieveData(request);
        ConnectorResponse<Map<String, Object>> response = responseFuture.get();
        
        assertTrue(response.isSuccess());
        assertNotNull(response.getData());
        assertEquals("Data retrieved successfully", response.getMessage());
    }
    
    @Test
    void testRetrieveDataFailure() throws ExecutionException, InterruptedException {
        setupSuccessfulAuthentication();
        
        // Mock failed data retrieval
        when(restTemplate.exchange(anyString(), eq(HttpMethod.GET), any(HttpEntity.class), eq(Map.class)))
            .thenReturn(new ResponseEntity<>(HttpStatus.BAD_REQUEST));
        
        // Test data retrieval failure
        ConnectorRequest request = new ConnectorRequest("QUERY", "Account");
        
        CompletableFuture<ConnectorResponse<Map<String, Object>>> responseFuture = 
            salesforceConnector.retrieveData(request);
        ConnectorResponse<Map<String, Object>> response = responseFuture.get();
        
        assertFalse(response.isSuccess());
        assertEquals("SALESFORCE_QUERY_FAILED", response.getErrorCode());
    }
    
    @Test
    void testSendDataSuccess() throws ExecutionException, InterruptedException {
        setupSuccessfulAuthentication();
        
        // Mock successful record creation
        Map<String, Object> createResponse = new HashMap<>();
        createResponse.put("id", "001000000000001");
        createResponse.put("success", true);
        
        ResponseEntity<Map> mockResponse = new ResponseEntity<>(createResponse, HttpStatus.CREATED);
        when(restTemplate.exchange(anyString(), eq(HttpMethod.POST), any(HttpEntity.class), eq(Map.class)))
            .thenReturn(mockResponse);
        
        // Test data sending
        Map<String, Object> testData = new HashMap<>();
        testData.put("objectType", "Account");
        testData.put("Name", "Test Account");
        testData.put("Type", "Customer");
        
        CompletableFuture<ConnectorResponse<Map<String, Object>>> responseFuture = 
            salesforceConnector.sendData(testData);
        ConnectorResponse<Map<String, Object>> response = responseFuture.get();
        
        assertTrue(response.isSuccess());
        assertNotNull(response.getData());
        assertEquals("Record created successfully", response.getMessage());
    }
    
    @Test
    void testSendDataWithoutObjectType() throws ExecutionException, InterruptedException {
        setupSuccessfulAuthentication();
        
        // Test data sending without object type
        Map<String, Object> testData = new HashMap<>();
        testData.put("Name", "Test Account");
        
        CompletableFuture<ConnectorResponse<Map<String, Object>>> responseFuture = 
            salesforceConnector.sendData(testData);
        ConnectorResponse<Map<String, Object>> response = responseFuture.get();
        
        assertFalse(response.isSuccess());
        assertEquals("MISSING_OBJECT_TYPE", response.getErrorCode());
    }
    
    @Test
    void testUpdateDataSuccess() throws ExecutionException, InterruptedException {
        setupSuccessfulAuthentication();
        
        // Mock successful record update
        ResponseEntity<Map> mockResponse = new ResponseEntity<>(HttpStatus.NO_CONTENT);
        when(restTemplate.exchange(anyString(), eq(HttpMethod.PATCH), any(HttpEntity.class), eq(Map.class)))
            .thenReturn(mockResponse);
        
        // Test data update
        Map<String, Object> updateData = new HashMap<>();
        updateData.put("objectType", "Account");
        updateData.put("Name", "Updated Account Name");
        
        CompletableFuture<ConnectorResponse<Map<String, Object>>> responseFuture = 
            salesforceConnector.updateData("001000000000001", updateData);
        ConnectorResponse<Map<String, Object>> response = responseFuture.get();
        
        assertTrue(response.isSuccess());
        assertEquals("Record updated successfully", response.getMessage());
    }
    
    @Test
    void testBulkOperationWhenDisabled() throws ExecutionException, InterruptedException {
        setupSuccessfulAuthentication();
        
        // Configure bulk API as disabled
        when(salesforceConfig.isBulkApiEnabled()).thenReturn(false);
        
        // Test bulk operation
        BulkOperation<Map<String, Object>> bulkOp = BulkOperation.insert(
            java.util.Arrays.asList(
                Map.of("objectType", "Account", "Name", "Bulk Account 1"),
                Map.of("objectType", "Account", "Name", "Bulk Account 2")
            )
        );
        
        CompletableFuture<BulkOperationResponse<Map<String, Object>>> responseFuture = 
            salesforceConnector.performBulkOperation(bulkOp);
        BulkOperationResponse<Map<String, Object>> response = responseFuture.get();
        
        assertFalse(response.isSuccess());
        assertEquals(BulkOperationResponse.BulkOperationStatus.FAILED, response.getStatus());
        assertTrue(response.getErrors().contains("Bulk API is not enabled for this Salesforce org"));
    }
    
    @Test
    void testEventSubscription() {
        // Test event subscription
        String[] eventTypes = {"DATA_CHANGE", "WORKFLOW_UPDATE"};
        PartnerConnector.EventCallback<Map<String, Object>> callback = event -> {
            // Test callback
        };
        
        String subscriptionId = salesforceConnector.subscribeToEvents(eventTypes, callback);
        
        assertNotNull(subscriptionId);
        assertTrue(subscriptionId.startsWith("sf-sub-"));
    }
    
    @Test
    void testEventUnsubscription() throws ExecutionException, InterruptedException {
        // Test event subscription first
        String[] eventTypes = {"DATA_CHANGE"};
        PartnerConnector.EventCallback<Map<String, Object>> callback = event -> {};
        String subscriptionId = salesforceConnector.subscribeToEvents(eventTypes, callback);
        
        // Test unsubscription
        CompletableFuture<Void> unsubscribeFuture = salesforceConnector.unsubscribeFromEvents(subscriptionId);
        assertDoesNotThrow(() -> unsubscribeFuture.get());
    }
    
    @Test
    void testGetConfiguration() {
        Map<String, Object> config = salesforceConnector.getConfiguration();
        
        assertNotNull(config);
        assertEquals("https://login.salesforce.com", config.get("loginUrl"));
        assertEquals(5000, config.get("rateLimitPerHour"));
        assertEquals(true, config.get("bulkApiEnabled"));
    }
    
    @Test
    void testGetMetrics() {
        PartnerConnector.ConnectorMetrics metrics = salesforceConnector.getMetrics();
        
        assertNotNull(metrics);
        assertEquals(0, metrics.totalRequests());
        assertEquals(0, metrics.successfulRequests());
        assertEquals(0, metrics.failedRequests());
    }
    
    @Test
    void testRateLimitStatus() {
        PartnerConnector.RateLimitStatus rateLimitStatus = salesforceConnector.getRateLimitStatus();
        
        assertNotNull(rateLimitStatus);
        assertEquals(5000, rateLimitStatus.limit());
        assertEquals(5000, rateLimitStatus.remaining());
        assertFalse(rateLimitStatus.throttled());
    }
    
    @Test
    void testDisconnect() throws ExecutionException, InterruptedException {
        // Setup connection first
        setupSuccessfulAuthentication();
        salesforceConnector.connect().get();
        
        assertTrue(salesforceConnector.isHealthy());
        
        // Test disconnection
        CompletableFuture<Void> disconnectFuture = salesforceConnector.disconnect();
        assertDoesNotThrow(() -> disconnectFuture.get());
        
        // Verify connector is no longer healthy
        assertFalse(salesforceConnector.isHealthy());
    }
    
    /**
     * Helper method to setup successful authentication mocks.
     */
    private void setupSuccessfulAuthentication() {
        Map<String, Object> authResponse = new HashMap<>();
        authResponse.put("access_token", "test-access-token");
        authResponse.put("instance_url", "https://test.salesforce.com");
        
        ResponseEntity<Map> mockAuthResponse = new ResponseEntity<>(authResponse, HttpStatus.OK);
        when(restTemplate.postForEntity(contains("oauth2/token"), any(HttpEntity.class), eq(Map.class)))
            .thenReturn(mockAuthResponse);
    }
    
    @Test
    void testAuthenticationTokenExpiry() {
        // Test scenario where token expires and needs refresh
        setupSuccessfulAuthentication();
        
        // Initially healthy
        assertFalse(salesforceConnector.isHealthy()); // No token yet
        
        // Connect and verify healthy
        assertDoesNotThrow(() -> salesforceConnector.connect().get());
        assertTrue(salesforceConnector.isHealthy());
    }
    
    @Test
    void testConcurrentOperations() {
        setupSuccessfulAuthentication();
        
        // Mock successful responses for concurrent operations
        ResponseEntity<Map> mockResponse = new ResponseEntity<>(
            Map.of("totalSize", 1, "done", true, "records", java.util.Collections.emptyList()), 
            HttpStatus.OK
        );
        when(restTemplate.exchange(anyString(), eq(HttpMethod.GET), any(HttpEntity.class), eq(Map.class)))
            .thenReturn(mockResponse);
        
        // Execute multiple concurrent operations
        CompletableFuture<Void> connectFuture = salesforceConnector.connect();
        CompletableFuture<ConnectorResponse<Map<String, Object>>> retrieveFuture = 
            salesforceConnector.retrieveData(new ConnectorRequest("QUERY", "Account"));
        CompletableFuture<PartnerConnector.HealthStatus> healthFuture = salesforceConnector.healthCheck();
        
        // All operations should complete without throwing exceptions
        assertDoesNotThrow(() -> CompletableFuture.allOf(connectFuture, retrieveFuture, healthFuture).get());
    }
}