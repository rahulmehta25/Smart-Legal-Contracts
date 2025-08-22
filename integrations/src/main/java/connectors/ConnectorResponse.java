package com.enterprise.integrations.connectors;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Response object for connector operations.
 * 
 * Encapsulates the result of operations with partner systems,
 * including success/failure status, data payload, metadata,
 * and error information.
 * 
 * @param <T> the type of data returned
 * @author Integration Platform Team
 * @version 1.0.0
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ConnectorResponse<T> {
    
    private boolean success;
    private T data;
    private String message;
    private String errorCode;
    private List<String> errors;
    private Map<String, Object> metadata;
    private LocalDateTime timestamp;
    private String requestId;
    private String correlationId;
    private ResponseMetrics metrics;
    private PaginationResponse pagination;
    
    /**
     * Default constructor.
     */
    public ConnectorResponse() {
        this.metadata = new HashMap<>();
        this.timestamp = LocalDateTime.now();
    }
    
    /**
     * Constructor for successful response.
     * 
     * @param data the response data
     */
    public ConnectorResponse(T data) {
        this();
        this.success = true;
        this.data = data;
        this.message = "Operation completed successfully";
    }
    
    /**
     * Constructor for error response.
     * 
     * @param errorCode the error code
     * @param message the error message
     */
    public ConnectorResponse(String errorCode, String message) {
        this();
        this.success = false;
        this.errorCode = errorCode;
        this.message = message;
    }
    
    /**
     * Constructor for error response with errors list.
     * 
     * @param errorCode the error code
     * @param message the error message
     * @param errors list of detailed errors
     */
    public ConnectorResponse(String errorCode, String message, List<String> errors) {
        this(errorCode, message);
        this.errors = errors;
    }
    
    /**
     * Creates a successful response.
     * 
     * @param data the response data
     * @param <T> the type of data
     * @return successful ConnectorResponse
     */
    public static <T> ConnectorResponse<T> success(T data) {
        return new ConnectorResponse<>(data);
    }
    
    /**
     * Creates a successful response with message.
     * 
     * @param data the response data
     * @param message success message
     * @param <T> the type of data
     * @return successful ConnectorResponse
     */
    public static <T> ConnectorResponse<T> success(T data, String message) {
        ConnectorResponse<T> response = new ConnectorResponse<>(data);
        response.setMessage(message);
        return response;
    }
    
    /**
     * Creates an error response.
     * 
     * @param errorCode the error code
     * @param message the error message
     * @param <T> the type of data
     * @return error ConnectorResponse
     */
    public static <T> ConnectorResponse<T> error(String errorCode, String message) {
        return new ConnectorResponse<>(errorCode, message);
    }
    
    /**
     * Creates an error response with detailed errors.
     * 
     * @param errorCode the error code
     * @param message the error message
     * @param errors list of detailed errors
     * @param <T> the type of data
     * @return error ConnectorResponse
     */
    public static <T> ConnectorResponse<T> error(String errorCode, String message, List<String> errors) {
        return new ConnectorResponse<>(errorCode, message, errors);
    }
    
    /**
     * Adds metadata to the response.
     * 
     * @param key metadata key
     * @param value metadata value
     * @return this response for method chaining
     */
    public ConnectorResponse<T> withMetadata(String key, Object value) {
        this.metadata.put(key, value);
        return this;
    }
    
    /**
     * Sets pagination information.
     * 
     * @param pagination pagination details
     * @return this response for method chaining
     */
    public ConnectorResponse<T> withPagination(PaginationResponse pagination) {
        this.pagination = pagination;
        return this;
    }
    
    /**
     * Sets response metrics.
     * 
     * @param metrics response metrics
     * @return this response for method chaining
     */
    public ConnectorResponse<T> withMetrics(ResponseMetrics metrics) {
        this.metrics = metrics;
        return this;
    }
    
    /**
     * Sets correlation ID for tracing.
     * 
     * @param correlationId correlation identifier
     * @return this response for method chaining
     */
    public ConnectorResponse<T> withCorrelationId(String correlationId) {
        this.correlationId = correlationId;
        return this;
    }
    
    // Getters and setters
    
    public boolean isSuccess() {
        return success;
    }
    
    public void setSuccess(boolean success) {
        this.success = success;
    }
    
    public T getData() {
        return data;
    }
    
    public void setData(T data) {
        this.data = data;
    }
    
    public String getMessage() {
        return message;
    }
    
    public void setMessage(String message) {
        this.message = message;
    }
    
    public String getErrorCode() {
        return errorCode;
    }
    
    public void setErrorCode(String errorCode) {
        this.errorCode = errorCode;
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
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public String getRequestId() {
        return requestId;
    }
    
    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }
    
    public String getCorrelationId() {
        return correlationId;
    }
    
    public void setCorrelationId(String correlationId) {
        this.correlationId = correlationId;
    }
    
    public ResponseMetrics getMetrics() {
        return metrics;
    }
    
    public void setMetrics(ResponseMetrics metrics) {
        this.metrics = metrics;
    }
    
    public PaginationResponse getPagination() {
        return pagination;
    }
    
    public void setPagination(PaginationResponse pagination) {
        this.pagination = pagination;
    }
    
    /**
     * Response metrics for performance tracking.
     */
    public static class ResponseMetrics {
        private long processingTimeMs;
        private long networkTimeMs;
        private int recordsProcessed;
        private int recordsSkipped;
        private String partnerLatency;
        
        public ResponseMetrics() {}
        
        public ResponseMetrics(long processingTimeMs, long networkTimeMs) {
            this.processingTimeMs = processingTimeMs;
            this.networkTimeMs = networkTimeMs;
        }
        
        // Getters and setters
        public long getProcessingTimeMs() { return processingTimeMs; }
        public void setProcessingTimeMs(long processingTimeMs) { this.processingTimeMs = processingTimeMs; }
        public long getNetworkTimeMs() { return networkTimeMs; }
        public void setNetworkTimeMs(long networkTimeMs) { this.networkTimeMs = networkTimeMs; }
        public int getRecordsProcessed() { return recordsProcessed; }
        public void setRecordsProcessed(int recordsProcessed) { this.recordsProcessed = recordsProcessed; }
        public int getRecordsSkipped() { return recordsSkipped; }
        public void setRecordsSkipped(int recordsSkipped) { this.recordsSkipped = recordsSkipped; }
        public String getPartnerLatency() { return partnerLatency; }
        public void setPartnerLatency(String partnerLatency) { this.partnerLatency = partnerLatency; }
    }
    
    /**
     * Pagination information in responses.
     */
    public static class PaginationResponse {
        private int page;
        private int size;
        private long totalElements;
        private int totalPages;
        private boolean hasNext;
        private boolean hasPrevious;
        private String nextCursor;
        private String previousCursor;
        
        public PaginationResponse() {}
        
        public PaginationResponse(int page, int size, long totalElements) {
            this.page = page;
            this.size = size;
            this.totalElements = totalElements;
            this.totalPages = (int) Math.ceil((double) totalElements / size);
            this.hasNext = page < totalPages;
            this.hasPrevious = page > 1;
        }
        
        // Getters and setters
        public int getPage() { return page; }
        public void setPage(int page) { this.page = page; }
        public int getSize() { return size; }
        public void setSize(int size) { this.size = size; }
        public long getTotalElements() { return totalElements; }
        public void setTotalElements(long totalElements) { this.totalElements = totalElements; }
        public int getTotalPages() { return totalPages; }
        public void setTotalPages(int totalPages) { this.totalPages = totalPages; }
        public boolean isHasNext() { return hasNext; }
        public void setHasNext(boolean hasNext) { this.hasNext = hasNext; }
        public boolean isHasPrevious() { return hasPrevious; }
        public void setHasPrevious(boolean hasPrevious) { this.hasPrevious = hasPrevious; }
        public String getNextCursor() { return nextCursor; }
        public void setNextCursor(String nextCursor) { this.nextCursor = nextCursor; }
        public String getPreviousCursor() { return previousCursor; }
        public void setPreviousCursor(String previousCursor) { this.previousCursor = previousCursor; }
    }
    
    @Override
    public String toString() {
        return "ConnectorResponse{" +
                "success=" + success +
                ", message='" + message + '\'' +
                ", errorCode='" + errorCode + '\'' +
                ", timestamp=" + timestamp +
                ", requestId='" + requestId + '\'' +
                '}';
    }
}