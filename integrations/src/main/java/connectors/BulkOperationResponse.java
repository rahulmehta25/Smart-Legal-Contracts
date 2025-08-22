package com.enterprise.integrations.connectors;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Response for bulk operations on partner systems.
 * 
 * Contains detailed results of bulk operations including
 * success/failure counts, individual record results,
 * and processing metrics.
 * 
 * @param <T> the type of data processed
 * @author Integration Platform Team
 * @version 1.0.0
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class BulkOperationResponse<T> {
    
    private String operationId;
    private boolean success;
    private BulkOperationStatus status;
    private int totalRecords;
    private int successfulRecords;
    private int failedRecords;
    private int skippedRecords;
    private List<BulkRecordResult<T>> results;
    private List<String> errors;
    private Map<String, Object> metadata;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private long processingTimeMs;
    private String jobId; // For async operations
    
    /**
     * Default constructor.
     */
    public BulkOperationResponse() {
        this.results = new ArrayList<>();
        this.errors = new ArrayList<>();
        this.status = BulkOperationStatus.PENDING;
        this.startTime = LocalDateTime.now();
    }
    
    /**
     * Constructor with operation ID.
     * 
     * @param operationId the operation identifier
     */
    public BulkOperationResponse(String operationId) {
        this();
        this.operationId = operationId;
    }
    
    /**
     * Creates a successful bulk operation response.
     * 
     * @param operationId the operation identifier
     * @param results the record results
     * @param <T> the type of data
     * @return successful bulk operation response
     */
    public static <T> BulkOperationResponse<T> success(String operationId, 
                                                      List<BulkRecordResult<T>> results) {
        BulkOperationResponse<T> response = new BulkOperationResponse<>(operationId);
        response.success = true;
        response.status = BulkOperationStatus.COMPLETED;
        response.results = results != null ? results : new ArrayList<>();
        response.endTime = LocalDateTime.now();
        
        // Calculate counts
        response.totalRecords = response.results.size();
        response.successfulRecords = (int) response.results.stream()
            .filter(BulkRecordResult::isSuccess)
            .count();
        response.failedRecords = response.totalRecords - response.successfulRecords;
        
        return response;
    }
    
    /**
     * Creates a failed bulk operation response.
     * 
     * @param operationId the operation identifier
     * @param errors the list of errors
     * @param <T> the type of data
     * @return failed bulk operation response
     */
    public static <T> BulkOperationResponse<T> failure(String operationId, List<String> errors) {
        BulkOperationResponse<T> response = new BulkOperationResponse<>(operationId);
        response.success = false;
        response.status = BulkOperationStatus.FAILED;
        response.errors = errors != null ? errors : new ArrayList<>();
        response.endTime = LocalDateTime.now();
        return response;
    }
    
    /**
     * Creates an in-progress bulk operation response.
     * 
     * @param operationId the operation identifier
     * @param jobId the job identifier for tracking
     * @param <T> the type of data
     * @return in-progress bulk operation response
     */
    public static <T> BulkOperationResponse<T> inProgress(String operationId, String jobId) {
        BulkOperationResponse<T> response = new BulkOperationResponse<>(operationId);
        response.success = true;
        response.status = BulkOperationStatus.IN_PROGRESS;
        response.jobId = jobId;
        return response;
    }
    
    /**
     * Adds a record result.
     * 
     * @param result the record result to add
     */
    public void addResult(BulkRecordResult<T> result) {
        this.results.add(result);
        this.totalRecords++;
        if (result.isSuccess()) {
            this.successfulRecords++;
        } else {
            this.failedRecords++;
        }
    }
    
    /**
     * Adds an error message.
     * 
     * @param error the error message to add
     */
    public void addError(String error) {
        this.errors.add(error);
    }
    
    /**
     * Calculates and sets the processing time.
     */
    public void calculateProcessingTime() {
        if (startTime != null && endTime != null) {
            this.processingTimeMs = java.time.Duration.between(startTime, endTime).toMillis();
        }
    }
    
    /**
     * Marks the operation as completed.
     */
    public void markCompleted() {
        this.endTime = LocalDateTime.now();
        this.status = BulkOperationStatus.COMPLETED;
        this.success = this.failedRecords == 0;
        calculateProcessingTime();
    }
    
    /**
     * Marks the operation as failed.
     * 
     * @param error the error message
     */
    public void markFailed(String error) {
        this.endTime = LocalDateTime.now();
        this.status = BulkOperationStatus.FAILED;
        this.success = false;
        addError(error);
        calculateProcessingTime();
    }
    
    // Getters and setters
    
    public String getOperationId() {
        return operationId;
    }
    
    public void setOperationId(String operationId) {
        this.operationId = operationId;
    }
    
    public boolean isSuccess() {
        return success;
    }
    
    public void setSuccess(boolean success) {
        this.success = success;
    }
    
    public BulkOperationStatus getStatus() {
        return status;
    }
    
    public void setStatus(BulkOperationStatus status) {
        this.status = status;
    }
    
    public int getTotalRecords() {
        return totalRecords;
    }
    
    public void setTotalRecords(int totalRecords) {
        this.totalRecords = totalRecords;
    }
    
    public int getSuccessfulRecords() {
        return successfulRecords;
    }
    
    public void setSuccessfulRecords(int successfulRecords) {
        this.successfulRecords = successfulRecords;
    }
    
    public int getFailedRecords() {
        return failedRecords;
    }
    
    public void setFailedRecords(int failedRecords) {
        this.failedRecords = failedRecords;
    }
    
    public int getSkippedRecords() {
        return skippedRecords;
    }
    
    public void setSkippedRecords(int skippedRecords) {
        this.skippedRecords = skippedRecords;
    }
    
    public List<BulkRecordResult<T>> getResults() {
        return results;
    }
    
    public void setResults(List<BulkRecordResult<T>> results) {
        this.results = results;
    }
    
    public List<String> getErrors() {
        return errors;
    }
    
    public void setErrors(List<String> errors) {
        this.errors = errors;
    }
    
    public Map<String, Object> getMetadata() {
        return metadata;
    }
    
    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }
    
    public LocalDateTime getStartTime() {
        return startTime;
    }
    
    public void setStartTime(LocalDateTime startTime) {
        this.startTime = startTime;
    }
    
    public LocalDateTime getEndTime() {
        return endTime;
    }
    
    public void setEndTime(LocalDateTime endTime) {
        this.endTime = endTime;
    }
    
    public long getProcessingTimeMs() {
        return processingTimeMs;
    }
    
    public void setProcessingTimeMs(long processingTimeMs) {
        this.processingTimeMs = processingTimeMs;
    }
    
    public String getJobId() {
        return jobId;
    }
    
    public void setJobId(String jobId) {
        this.jobId = jobId;
    }
    
    /**
     * Status of bulk operations.
     */
    public enum BulkOperationStatus {
        PENDING,
        IN_PROGRESS,
        COMPLETED,
        FAILED,
        CANCELLED
    }
    
    /**
     * Result of processing a single record in a bulk operation.
     * 
     * @param <T> the type of record data
     */
    public static class BulkRecordResult<T> {
        private boolean success;
        private String id;
        private T data;
        private String errorCode;
        private String errorMessage;
        private Map<String, Object> metadata;
        
        public BulkRecordResult() {}
        
        public BulkRecordResult(boolean success, String id) {
            this.success = success;
            this.id = id;
        }
        
        public BulkRecordResult(boolean success, String id, T data) {
            this(success, id);
            this.data = data;
        }
        
        public static <T> BulkRecordResult<T> success(String id, T data) {
            return new BulkRecordResult<>(true, id, data);
        }
        
        public static <T> BulkRecordResult<T> failure(String id, String errorCode, String errorMessage) {
            BulkRecordResult<T> result = new BulkRecordResult<>(false, id);
            result.setErrorCode(errorCode);
            result.setErrorMessage(errorMessage);
            return result;
        }
        
        // Getters and setters
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getId() { return id; }
        public void setId(String id) { this.id = id; }
        public T getData() { return data; }
        public void setData(T data) { this.data = data; }
        public String getErrorCode() { return errorCode; }
        public void setErrorCode(String errorCode) { this.errorCode = errorCode; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
    }
    
    @Override
    public String toString() {
        return "BulkOperationResponse{" +
                "operationId='" + operationId + '\'' +
                ", status=" + status +
                ", totalRecords=" + totalRecords +
                ", successfulRecords=" + successfulRecords +
                ", failedRecords=" + failedRecords +
                ", processingTimeMs=" + processingTimeMs +
                '}';
    }
}