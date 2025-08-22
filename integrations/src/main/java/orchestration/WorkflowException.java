package com.enterprise.integrations.orchestration;

/**
 * Exception thrown during workflow execution operations.
 * 
 * Provides detailed error information including error codes,
 * workflow context, and recovery information.
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
public class WorkflowException extends RuntimeException {
    
    private final String errorCode;
    private final String workflowId;
    private final String executionId;
    private final String stepId;
    private final boolean retryable;
    private final WorkflowFailureReason failureReason;
    private final Object workflowContext;
    
    /**
     * Creates a workflow exception.
     * 
     * @param message error message
     */
    public WorkflowException(String message) {
        super(message);
        this.errorCode = "WORKFLOW_ERROR";
        this.workflowId = null;
        this.executionId = null;
        this.stepId = null;
        this.retryable = false;
        this.failureReason = WorkflowFailureReason.UNKNOWN;
        this.workflowContext = null;
    }
    
    /**
     * Creates a workflow exception with error code.
     * 
     * @param errorCode error code
     * @param message error message
     */
    public WorkflowException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
        this.workflowId = null;
        this.executionId = null;
        this.stepId = null;
        this.retryable = false;
        this.failureReason = WorkflowFailureReason.UNKNOWN;
        this.workflowContext = null;
    }
    
    /**
     * Creates a workflow exception with full details.
     * 
     * @param errorCode error code
     * @param message error message
     * @param workflowId workflow identifier
     * @param executionId execution identifier
     * @param stepId step identifier
     * @param retryable whether the operation can be retried
     * @param failureReason reason for failure
     * @param workflowContext workflow context data
     */
    public WorkflowException(String errorCode, String message, String workflowId,
                           String executionId, String stepId, boolean retryable,
                           WorkflowFailureReason failureReason, Object workflowContext) {
        super(message);
        this.errorCode = errorCode;
        this.workflowId = workflowId;
        this.executionId = executionId;
        this.stepId = stepId;
        this.retryable = retryable;
        this.failureReason = failureReason;
        this.workflowContext = workflowContext;
    }
    
    /**
     * Creates a workflow exception with cause.
     * 
     * @param errorCode error code
     * @param message error message
     * @param cause the underlying cause
     * @param workflowId workflow identifier
     * @param executionId execution identifier
     * @param retryable whether the operation can be retried
     */
    public WorkflowException(String errorCode, String message, Throwable cause,
                           String workflowId, String executionId, boolean retryable) {
        super(message, cause);
        this.errorCode = errorCode;
        this.workflowId = workflowId;
        this.executionId = executionId;
        this.stepId = null;
        this.retryable = retryable;
        this.failureReason = WorkflowFailureReason.SYSTEM_ERROR;
        this.workflowContext = null;
    }
    
    /**
     * Creates a step execution exception.
     * 
     * @param errorCode error code
     * @param message error message
     * @param workflowId workflow identifier
     * @param executionId execution identifier
     * @param stepId step identifier
     * @param failureReason reason for failure
     * @return workflow exception for step failure
     */
    public static WorkflowException stepFailure(String errorCode, String message,
                                              String workflowId, String executionId,
                                              String stepId, WorkflowFailureReason failureReason) {
        return new WorkflowException(errorCode, message, workflowId, executionId,
                                   stepId, true, failureReason, null);
    }
    
    /**
     * Creates a workflow timeout exception.
     * 
     * @param workflowId workflow identifier
     * @param executionId execution identifier
     * @param timeoutMinutes timeout duration
     * @return workflow exception for timeout
     */
    public static WorkflowException timeout(String workflowId, String executionId, int timeoutMinutes) {
        return new WorkflowException("WORKFLOW_TIMEOUT",
                                   String.format("Workflow timed out after %d minutes", timeoutMinutes),
                                   workflowId, executionId, null, false,
                                   WorkflowFailureReason.TIMEOUT, null);
    }
    
    /**
     * Creates a validation exception.
     * 
     * @param workflowId workflow identifier
     * @param validationErrors validation errors
     * @return workflow exception for validation failure
     */
    public static WorkflowException validationFailure(String workflowId, 
                                                    java.util.List<String> validationErrors) {
        String message = "Workflow validation failed: " + String.join(", ", validationErrors);
        return new WorkflowException("WORKFLOW_VALIDATION_FAILED", message,
                                   workflowId, null, null, false,
                                   WorkflowFailureReason.VALIDATION_ERROR, validationErrors);
    }
    
    // Getters
    
    public String getErrorCode() {
        return errorCode;
    }
    
    public String getWorkflowId() {
        return workflowId;
    }
    
    public String getExecutionId() {
        return executionId;
    }
    
    public String getStepId() {
        return stepId;
    }
    
    public boolean isRetryable() {
        return retryable;
    }
    
    public WorkflowFailureReason getFailureReason() {
        return failureReason;
    }
    
    public Object getWorkflowContext() {
        return workflowContext;
    }
    
    /**
     * Reasons for workflow failure.
     */
    public enum WorkflowFailureReason {
        UNKNOWN,
        VALIDATION_ERROR,
        CONFIGURATION_ERROR,
        PARTNER_UNAVAILABLE,
        DATA_TRANSFORMATION_ERROR,
        BUSINESS_RULE_VIOLATION,
        SYSTEM_ERROR,
        TIMEOUT,
        CANCELLED_BY_USER,
        RESOURCE_EXHAUSTED,
        DEPENDENCY_FAILURE
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("WorkflowException{");
        sb.append("errorCode='").append(errorCode).append('\'');
        if (workflowId != null) {
            sb.append(", workflowId='").append(workflowId).append('\'');
        }
        if (executionId != null) {
            sb.append(", executionId='").append(executionId).append('\'');
        }
        if (stepId != null) {
            sb.append(", stepId='").append(stepId).append('\'');
        }
        sb.append(", retryable=").append(retryable);
        sb.append(", failureReason=").append(failureReason);
        sb.append(", message='").append(getMessage()).append('\'');
        sb.append('}');
        return sb.toString();
    }
}