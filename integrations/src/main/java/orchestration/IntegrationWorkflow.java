package com.enterprise.integrations.orchestration;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Base interface for integration workflow orchestration.
 * 
 * Defines the contract for orchestrating complex integration workflows
 * that involve multiple partners, data transformations, and business rules.
 * Supports:
 * - Workflow definition and execution
 * - Step-by-step orchestration
 * - Error handling and compensation
 * - Parallel and sequential processing
 * - Conditional logic and routing
 * - State management and persistence
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
public interface IntegrationWorkflow {

    /**
     * Gets the unique identifier for this workflow.
     * 
     * @return workflow identifier
     */
    String getWorkflowId();

    /**
     * Gets the display name for this workflow.
     * 
     * @return human-readable workflow name
     */
    String getDisplayName();

    /**
     * Gets the workflow version.
     * 
     * @return workflow version
     */
    String getVersion();

    /**
     * Gets the workflow definition and configuration.
     * 
     * @return workflow definition
     */
    WorkflowDefinition getDefinition();

    /**
     * Executes the workflow with the given context.
     * 
     * @param context workflow execution context
     * @return workflow execution result
     * @throws WorkflowException if workflow execution fails
     */
    CompletableFuture<WorkflowExecutionResult> execute(WorkflowContext context) 
            throws WorkflowException;

    /**
     * Validates workflow input and configuration.
     * 
     * @param context workflow context
     * @return validation result
     */
    WorkflowValidationResult validate(WorkflowContext context);

    /**
     * Gets the current status of a workflow execution.
     * 
     * @param executionId workflow execution identifier
     * @return execution status
     */
    CompletableFuture<WorkflowExecutionStatus> getExecutionStatus(String executionId);

    /**
     * Cancels a running workflow execution.
     * 
     * @param executionId workflow execution identifier
     * @return cancellation result
     */
    CompletableFuture<Void> cancelExecution(String executionId);

    /**
     * Resumes a paused or failed workflow execution.
     * 
     * @param executionId workflow execution identifier
     * @param resumeContext optional resume context
     * @return resume result
     */
    CompletableFuture<WorkflowExecutionResult> resumeExecution(String executionId, 
                                                              WorkflowContext resumeContext);

    /**
     * Gets workflow execution history and logs.
     * 
     * @param executionId workflow execution identifier
     * @return execution history
     */
    CompletableFuture<WorkflowExecutionHistory> getExecutionHistory(String executionId);

    /**
     * Gets workflow metrics and performance data.
     * 
     * @return workflow metrics
     */
    WorkflowMetrics getMetrics();

    /**
     * Workflow execution context containing input data and configuration.
     */
    record WorkflowContext(
        String executionId,
        String initiatedBy,
        Map<String, Object> inputData,
        Map<String, Object> parameters,
        Map<String, Object> metadata,
        LocalDateTime startTime
    ) {
        public WorkflowContext withParameter(String key, Object value) {
            Map<String, Object> newParams = new java.util.HashMap<>(parameters);
            newParams.put(key, value);
            return new WorkflowContext(executionId, initiatedBy, inputData, 
                                     newParams, metadata, startTime);
        }
        
        public WorkflowContext withMetadata(String key, Object value) {
            Map<String, Object> newMeta = new java.util.HashMap<>(metadata);
            newMeta.put(key, value);
            return new WorkflowContext(executionId, initiatedBy, inputData, 
                                     parameters, newMeta, startTime);
        }
    }

    /**
     * Workflow definition containing steps and configuration.
     */
    record WorkflowDefinition(
        String workflowId,
        String name,
        String description,
        java.util.List<WorkflowStep> steps,
        Map<String, Object> configuration,
        java.util.List<String> requiredPartners,
        int timeoutMinutes,
        boolean allowParallelExecution
    ) {}

    /**
     * Individual step in a workflow.
     */
    record WorkflowStep(
        String stepId,
        String stepType,
        String description,
        Map<String, Object> configuration,
        java.util.List<String> dependsOn,
        boolean optional,
        int retryAttempts,
        int timeoutSeconds
    ) {}

    /**
     * Result of workflow execution.
     */
    record WorkflowExecutionResult(
        String executionId,
        WorkflowStatus status,
        Map<String, Object> outputData,
        java.util.List<WorkflowStepResult> stepResults,
        LocalDateTime startTime,
        LocalDateTime endTime,
        long executionTimeMs,
        String errorMessage,
        Map<String, Object> metadata
    ) {
        public boolean isSuccessful() {
            return status == WorkflowStatus.COMPLETED;
        }
        
        public boolean hasFailed() {
            return status == WorkflowStatus.FAILED;
        }
    }

    /**
     * Result of individual workflow step execution.
     */
    record WorkflowStepResult(
        String stepId,
        WorkflowStepStatus status,
        Object outputData,
        LocalDateTime startTime,
        LocalDateTime endTime,
        long executionTimeMs,
        String errorMessage,
        Map<String, Object> metadata
    ) {}

    /**
     * Validation result for workflow input and configuration.
     */
    record WorkflowValidationResult(
        boolean valid,
        java.util.List<String> errors,
        java.util.List<String> warnings,
        Map<String, Object> validationDetails
    ) {
        public static WorkflowValidationResult success() {
            return new WorkflowValidationResult(true,
                java.util.Collections.emptyList(),
                java.util.Collections.emptyList(),
                java.util.Collections.emptyMap());
        }
        
        public static WorkflowValidationResult failure(java.util.List<String> errors) {
            return new WorkflowValidationResult(false, errors,
                java.util.Collections.emptyList(),
                java.util.Collections.emptyMap());
        }
    }

    /**
     * Current status of workflow execution.
     */
    record WorkflowExecutionStatus(
        String executionId,
        WorkflowStatus status,
        String currentStepId,
        int completedSteps,
        int totalSteps,
        LocalDateTime lastUpdate,
        String statusMessage,
        Map<String, Object> statusDetails
    ) {}

    /**
     * Workflow execution history.
     */
    record WorkflowExecutionHistory(
        String executionId,
        java.util.List<WorkflowStepResult> stepHistory,
        java.util.List<WorkflowEvent> events,
        Map<String, Object> executionData
    ) {}

    /**
     * Workflow event for auditing and monitoring.
     */
    record WorkflowEvent(
        String eventId,
        String eventType,
        String stepId,
        LocalDateTime timestamp,
        String message,
        Map<String, Object> eventData
    ) {}

    /**
     * Workflow performance metrics.
     */
    record WorkflowMetrics(
        long totalExecutions,
        long successfulExecutions,
        long failedExecutions,
        double averageExecutionTime,
        double successRate,
        LocalDateTime lastExecution,
        Map<String, Object> stepMetrics
    ) {}

    /**
     * Workflow execution status enumeration.
     */
    enum WorkflowStatus {
        PENDING,
        RUNNING,
        PAUSED,
        COMPLETED,
        FAILED,
        CANCELLED,
        TIMEOUT
    }

    /**
     * Workflow step execution status enumeration.
     */
    enum WorkflowStepStatus {
        PENDING,
        RUNNING,
        COMPLETED,
        FAILED,
        SKIPPED,
        CANCELLED,
        TIMEOUT
    }
}