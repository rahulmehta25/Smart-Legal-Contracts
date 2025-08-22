package com.enterprise.integrations.orchestration;

import com.enterprise.integrations.connectors.PartnerConnector;
import com.enterprise.integrations.connectors.ConnectorResponse;
import com.enterprise.integrations.transformers.DataTransformer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Comprehensive tests for PartnerSyncWorkflow.
 * 
 * Tests workflow orchestration functionality including:
 * - Workflow execution and step processing
 * - Error handling and retry logic
 * - Validation and configuration
 * - Workflow state management
 * - Metrics and monitoring
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@ExtendWith(MockitoExtension.class)
class PartnerSyncWorkflowTest {
    
    @Mock
    private PartnerConnector<Map<String, Object>> sourceConnector;
    
    @Mock
    private PartnerConnector<Map<String, Object>> targetConnector;
    
    @Mock
    private DataTransformer<Map<String, Object>, Map<String, Object>> dataTransformer;
    
    private PartnerSyncWorkflow workflow;
    private Map<String, PartnerConnector<?>> connectors;
    private Map<String, DataTransformer<?, ?>> transformers;
    
    @BeforeEach
    void setUp() {
        // Setup connectors
        connectors = new HashMap<>();
        connectors.put("salesforce", sourceConnector);
        connectors.put("microsoft365", targetConnector);
        
        // Setup transformers
        transformers = new HashMap<>();
        transformers.put("salesforce-to-microsoft365", dataTransformer);
        transformers.put("generic-transformer", dataTransformer);
        
        workflow = new PartnerSyncWorkflow(connectors, transformers);
    }
    
    @Test
    void testGetWorkflowId() {
        assertEquals("partner-sync-workflow", workflow.getWorkflowId());
    }
    
    @Test
    void testGetDisplayName() {
        assertEquals("Partner Data Synchronization Workflow", workflow.getDisplayName());
    }
    
    @Test
    void testGetVersion() {
        assertEquals("1.0.0", workflow.getVersion());
    }
    
    @Test
    void testGetDefinition() {
        IntegrationWorkflow.WorkflowDefinition definition = workflow.getDefinition();
        
        assertNotNull(definition);
        assertEquals("partner-sync-workflow", definition.workflowId());
        assertEquals("Partner Data Synchronization Workflow", definition.name());
        assertFalse(definition.steps().isEmpty());
        assertEquals(8, definition.steps().size());
        assertEquals(60, definition.timeoutMinutes());
        assertFalse(definition.allowParallelExecution());
        
        // Verify step sequence
        IntegrationWorkflow.WorkflowStep firstStep = definition.steps().get(0);
        assertEquals("validate-input", firstStep.stepId());
        assertEquals("validation", firstStep.stepType());
        assertFalse(firstStep.optional());
        
        IntegrationWorkflow.WorkflowStep lastStep = definition.steps().get(definition.steps().size() - 1);
        assertEquals("update-sync-status", lastStep.stepId());
        assertEquals("status-update", lastStep.stepType());
    }
    
    @Test
    void testValidateInputSuccess() {
        // Prepare valid input data
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("sourcePartner", "salesforce");
        inputData.put("targetPartner", "microsoft365");
        inputData.put("syncType", "incremental");
        inputData.put("resourceType", "contacts");
        
        IntegrationWorkflow.WorkflowContext context = new IntegrationWorkflow.WorkflowContext(
            "test-execution-123",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        IntegrationWorkflow.WorkflowValidationResult result = workflow.validate(context);
        
        assertTrue(result.valid());
        assertTrue(result.errors().isEmpty());
    }
    
    @Test
    void testValidateInputMissingRequiredFields() {
        // Prepare invalid input data - missing required fields
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("syncType", "incremental");
        
        IntegrationWorkflow.WorkflowContext context = new IntegrationWorkflow.WorkflowContext(
            "test-execution-123",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        IntegrationWorkflow.WorkflowValidationResult result = workflow.validate(context);
        
        assertFalse(result.valid());
        assertFalse(result.errors().isEmpty());
        assertTrue(result.errors().stream().anyMatch(error -> error.contains("Source partner")));
        assertTrue(result.errors().stream().anyMatch(error -> error.contains("Target partner")));
    }
    
    @Test
    void testValidateInputUnavailableConnectors() {
        // Prepare input data with unavailable connectors
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("sourcePartner", "unavailable-partner");
        inputData.put("targetPartner", "another-unavailable-partner");
        inputData.put("syncType", "full");
        
        IntegrationWorkflow.WorkflowContext context = new IntegrationWorkflow.WorkflowContext(
            "test-execution-123",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        IntegrationWorkflow.WorkflowValidationResult result = workflow.validate(context);
        
        assertFalse(result.valid());
        assertTrue(result.errors().stream().anyMatch(error -> error.contains("Source partner connector not available")));
        assertTrue(result.errors().stream().anyMatch(error -> error.contains("Target partner connector not available")));
    }
    
    @Test
    void testSuccessfulWorkflowExecution() throws ExecutionException, InterruptedException {
        // Setup mock connector behaviors
        when(sourceConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        when(targetConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        when(sourceConnector.isHealthy()).thenReturn(true);
        when(targetConnector.isHealthy()).thenReturn(true);
        
        // Mock data retrieval
        Map<String, Object> sourceData = Map.of(
            "Id", "001000000000001",
            "Name", "Test Account",
            "Email", "test@example.com"
        );
        ConnectorResponse<Map<String, Object>> retrieveResponse = ConnectorResponse.success(sourceData);
        when(sourceConnector.retrieveData(any())).thenReturn(CompletableFuture.completedFuture(retrieveResponse));
        
        // Mock data transformation
        Map<String, Object> transformedData = Map.of(
            "id", "001000000000001",
            "displayName", "Test Account",
            "mail", "test@example.com"
        );
        when(dataTransformer.transform(eq(sourceData), any())).thenReturn(transformedData);
        
        // Mock data synchronization
        ConnectorResponse<Map<String, Object>> syncResponse = ConnectorResponse.success(
            Map.of("id", "new-target-id", "success", true)
        );
        when(targetConnector.sendData(any())).thenReturn(CompletableFuture.completedFuture(syncResponse));
        
        // Prepare workflow context
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("sourcePartner", "salesforce");
        inputData.put("targetPartner", "microsoft365");
        inputData.put("syncType", "incremental");
        inputData.put("resourceType", "contacts");
        
        IntegrationWorkflow.WorkflowContext context = new IntegrationWorkflow.WorkflowContext(
            "test-execution-123",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        // Execute workflow
        CompletableFuture<IntegrationWorkflow.WorkflowExecutionResult> resultFuture = workflow.execute(context);
        IntegrationWorkflow.WorkflowExecutionResult result = resultFuture.get();
        
        // Verify successful execution
        assertTrue(result.isSuccessful());
        assertEquals(IntegrationWorkflow.WorkflowStatus.COMPLETED, result.status());
        assertNotNull(result.outputData());
        assertEquals(8, result.stepResults().size());
        assertTrue(result.executionTimeMs() > 0);
        
        // Verify all steps completed successfully
        for (IntegrationWorkflow.WorkflowStepResult stepResult : result.stepResults()) {
            assertEquals(IntegrationWorkflow.WorkflowStepStatus.COMPLETED, stepResult.status());
        }
        
        // Verify connector interactions
        verify(sourceConnector).connect();
        verify(targetConnector).connect();
        verify(sourceConnector).retrieveData(any());
        verify(targetConnector).sendData(any());
    }
    
    @Test
    void testWorkflowExecutionWithConnectionFailure() throws ExecutionException, InterruptedException {
        // Setup connection failure
        when(sourceConnector.connect()).thenReturn(
            CompletableFuture.failedFuture(new RuntimeException("Connection failed"))
        );
        when(targetConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        
        // Prepare workflow context
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("sourcePartner", "salesforce");
        inputData.put("targetPartner", "microsoft365");
        inputData.put("syncType", "full");
        inputData.put("resourceType", "accounts");
        
        IntegrationWorkflow.WorkflowContext context = new IntegrationWorkflow.WorkflowContext(
            "test-execution-456",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        // Execute workflow
        CompletableFuture<IntegrationWorkflow.WorkflowExecutionResult> resultFuture = workflow.execute(context);
        IntegrationWorkflow.WorkflowExecutionResult result = resultFuture.get();
        
        // Verify failed execution
        assertTrue(result.hasFailed());
        assertEquals(IntegrationWorkflow.WorkflowStatus.FAILED, result.status());
        assertNotNull(result.errorMessage());
        
        // Verify failure occurred at connection step
        boolean connectionStepFailed = result.stepResults().stream()
            .anyMatch(step -> "connect-partners".equals(step.stepId()) && 
                     step.status() == IntegrationWorkflow.WorkflowStepStatus.FAILED);
        assertTrue(connectionStepFailed);
    }
    
    @Test
    void testWorkflowExecutionWithDataRetrievalFailure() throws ExecutionException, InterruptedException {
        // Setup successful connections but failed data retrieval
        when(sourceConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        when(targetConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        when(sourceConnector.isHealthy()).thenReturn(true);
        when(targetConnector.isHealthy()).thenReturn(true);
        
        // Mock failed data retrieval
        ConnectorResponse<Map<String, Object>> failedResponse = 
            ConnectorResponse.error("DATA_FETCH_FAILED", "Failed to retrieve data");
        when(sourceConnector.retrieveData(any())).thenReturn(
            CompletableFuture.completedFuture(failedResponse)
        );
        
        // Prepare workflow context
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("sourcePartner", "salesforce");
        inputData.put("targetPartner", "microsoft365");
        inputData.put("syncType", "incremental");
        inputData.put("resourceType", "opportunities");
        
        IntegrationWorkflow.WorkflowContext context = new IntegrationWorkflow.WorkflowContext(
            "test-execution-789",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        // Execute workflow
        CompletableFuture<IntegrationWorkflow.WorkflowExecutionResult> resultFuture = workflow.execute(context);
        IntegrationWorkflow.WorkflowExecutionResult result = resultFuture.get();
        
        // Verify failed execution
        assertTrue(result.hasFailed());
        assertEquals(IntegrationWorkflow.WorkflowStatus.FAILED, result.status());
        
        // Verify failure occurred at data retrieval step
        boolean dataRetrievalFailed = result.stepResults().stream()
            .anyMatch(step -> "fetch-source-data".equals(step.stepId()) && 
                     step.status() == IntegrationWorkflow.WorkflowStepStatus.FAILED);
        assertTrue(dataRetrievalFailed);
    }
    
    @Test
    void testWorkflowExecutionWithTransformationFailure() throws ExecutionException, InterruptedException {
        // Setup successful connections and data retrieval
        when(sourceConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        when(targetConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        when(sourceConnector.isHealthy()).thenReturn(true);
        when(targetConnector.isHealthy()).thenReturn(true);
        
        Map<String, Object> sourceData = Map.of("Id", "001", "Name", "Test");
        ConnectorResponse<Map<String, Object>> retrieveResponse = ConnectorResponse.success(sourceData);
        when(sourceConnector.retrieveData(any())).thenReturn(CompletableFuture.completedFuture(retrieveResponse));
        
        // Mock transformation failure
        when(dataTransformer.transform(any(), any())).thenThrow(
            new RuntimeException("Transformation failed")
        );
        
        // Prepare workflow context
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("sourcePartner", "salesforce");
        inputData.put("targetPartner", "microsoft365");
        inputData.put("syncType", "full");
        inputData.put("resourceType", "contacts");
        
        IntegrationWorkflow.WorkflowContext context = new IntegrationWorkflow.WorkflowContext(
            "test-execution-999",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        // Execute workflow
        CompletableFuture<IntegrationWorkflow.WorkflowExecutionResult> resultFuture = workflow.execute(context);
        IntegrationWorkflow.WorkflowExecutionResult result = resultFuture.get();
        
        // Verify failed execution
        assertTrue(result.hasFailed());
        assertEquals(IntegrationWorkflow.WorkflowStatus.FAILED, result.status());
        
        // Verify failure occurred at transformation step
        boolean transformationFailed = result.stepResults().stream()
            .anyMatch(step -> "transform-data".equals(step.stepId()) && 
                     step.status() == IntegrationWorkflow.WorkflowStepStatus.FAILED);
        assertTrue(transformationFailed);
    }
    
    @Test
    void testGetExecutionStatus() throws ExecutionException, InterruptedException {
        CompletableFuture<IntegrationWorkflow.WorkflowExecutionStatus> statusFuture = 
            workflow.getExecutionStatus("test-execution-123");
        IntegrationWorkflow.WorkflowExecutionStatus status = statusFuture.get();
        
        assertNotNull(status);
        assertEquals("test-execution-123", status.executionId());
        assertEquals(IntegrationWorkflow.WorkflowStatus.COMPLETED, status.status());
        assertEquals(8, status.totalSteps());
        assertEquals(8, status.completedSteps());
    }
    
    @Test
    void testCancelExecution() throws ExecutionException, InterruptedException {
        CompletableFuture<Void> cancelFuture = workflow.cancelExecution("test-execution-123");
        assertDoesNotThrow(() -> cancelFuture.get());
    }
    
    @Test
    void testResumeExecution() throws ExecutionException, InterruptedException {
        // Setup mock behaviors for resume
        when(sourceConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        when(targetConnector.connect()).thenReturn(CompletableFuture.completedFuture(null));
        when(sourceConnector.isHealthy()).thenReturn(true);
        when(targetConnector.isHealthy()).thenReturn(true);
        
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("sourcePartner", "salesforce");
        inputData.put("targetPartner", "microsoft365");
        inputData.put("syncType", "incremental");
        
        IntegrationWorkflow.WorkflowContext resumeContext = new IntegrationWorkflow.WorkflowContext(
            "resume-execution-123",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        CompletableFuture<IntegrationWorkflow.WorkflowExecutionResult> resumeFuture = 
            workflow.resumeExecution("original-execution-123", resumeContext);
        
        assertDoesNotThrow(() -> resumeFuture.get());
    }
    
    @Test
    void testGetExecutionHistory() throws ExecutionException, InterruptedException {
        CompletableFuture<IntegrationWorkflow.WorkflowExecutionHistory> historyFuture = 
            workflow.getExecutionHistory("test-execution-123");
        IntegrationWorkflow.WorkflowExecutionHistory history = historyFuture.get();
        
        assertNotNull(history);
        assertEquals("test-execution-123", history.executionId());
        assertNotNull(history.stepHistory());
        assertNotNull(history.events());
        assertNotNull(history.executionData());
    }
    
    @Test
    void testGetMetrics() {
        IntegrationWorkflow.WorkflowMetrics metrics = workflow.getMetrics();
        
        assertNotNull(metrics);
        assertTrue(metrics.totalExecutions() >= 0);
        assertTrue(metrics.successfulExecutions() >= 0);
        assertTrue(metrics.failedExecutions() >= 0);
        assertTrue(metrics.successRate() >= 0 && metrics.successRate() <= 100);
        assertTrue(metrics.averageExecutionTime() >= 0);
        assertNotNull(metrics.stepMetrics());
    }
    
    @Test
    void testWorkflowValidationWarnings() {
        // Prepare input with missing optional field to generate warnings
        Map<String, Object> inputData = new HashMap<>();
        inputData.put("sourcePartner", "salesforce");
        inputData.put("targetPartner", "microsoft365");
        // Missing syncType should generate a warning
        
        IntegrationWorkflow.WorkflowContext context = new IntegrationWorkflow.WorkflowContext(
            "test-execution-warnings",
            "test-user",
            inputData,
            new HashMap<>(),
            new HashMap<>(),
            LocalDateTime.now()
        );
        
        IntegrationWorkflow.WorkflowValidationResult result = workflow.validate(context);
        
        assertTrue(result.valid());
        assertFalse(result.warnings().isEmpty());
        assertTrue(result.warnings().stream().anyMatch(warning -> warning.contains("syncType")));
    }
    
    @Test
    void testWorkflowDefinitionStepDependencies() {
        IntegrationWorkflow.WorkflowDefinition definition = workflow.getDefinition();
        
        // Verify step dependencies
        IntegrationWorkflow.WorkflowStep connectStep = definition.steps().stream()
            .filter(step -> "connect-partners".equals(step.stepId()))
            .findFirst()
            .orElse(null);
        
        assertNotNull(connectStep);
        assertTrue(connectStep.dependsOn().contains("validate-input"));
        
        IntegrationWorkflow.WorkflowStep fetchStep = definition.steps().stream()
            .filter(step -> "fetch-source-data".equals(step.stepId()))
            .findFirst()
            .orElse(null);
        
        assertNotNull(fetchStep);
        assertTrue(fetchStep.dependsOn().contains("connect-partners"));
    }
    
    @Test
    void testOptionalStepHandling() {
        IntegrationWorkflow.WorkflowDefinition definition = workflow.getDefinition();
        
        // Verify some steps are marked as optional
        boolean hasOptionalSteps = definition.steps().stream()
            .anyMatch(IntegrationWorkflow.WorkflowStep::optional);
        
        assertTrue(hasOptionalSteps);
        
        // Verify optional steps (like conflict detection and verification)
        IntegrationWorkflow.WorkflowStep conflictStep = definition.steps().stream()
            .filter(step -> "detect-conflicts".equals(step.stepId()))
            .findFirst()
            .orElse(null);
        
        assertNotNull(conflictStep);
        assertTrue(conflictStep.optional());
    }
}