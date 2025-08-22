package com.enterprise.integrations.connectors;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Represents a bulk operation to be performed on partner systems.
 * 
 * Supports batch operations like bulk insert, update, delete,
 * and upsert operations for improved efficiency when handling
 * large datasets.
 * 
 * @param <T> the type of data being processed
 * @author Integration Platform Team
 * @version 1.0.0
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class BulkOperation<T> {
    
    private String operationId;
    private BulkOperationType type;
    private List<T> data;
    private Map<String, Object> options;
    private LocalDateTime timestamp;
    private int batchSize;
    private boolean continueOnError;
    private String externalIdField;
    
    /**
     * Default constructor.
     */
    public BulkOperation() {
        this.timestamp = LocalDateTime.now();
        this.operationId = generateOperationId();
        this.batchSize = 200; // Default batch size
        this.continueOnError = true;
    }
    
    /**
     * Constructor with basic parameters.
     * 
     * @param type the type of bulk operation
     * @param data the data to process
     */
    public BulkOperation(BulkOperationType type, List<T> data) {
        this();
        this.type = type;
        this.data = data;
    }
    
    /**
     * Constructor with all parameters.
     * 
     * @param type the type of bulk operation
     * @param data the data to process
     * @param batchSize the batch size for processing
     * @param continueOnError whether to continue on errors
     */
    public BulkOperation(BulkOperationType type, List<T> data, int batchSize, boolean continueOnError) {
        this(type, data);
        this.batchSize = batchSize;
        this.continueOnError = continueOnError;
    }
    
    /**
     * Creates a bulk insert operation.
     * 
     * @param data the data to insert
     * @param <T> the type of data
     * @return bulk insert operation
     */
    public static <T> BulkOperation<T> insert(List<T> data) {
        return new BulkOperation<>(BulkOperationType.INSERT, data);
    }
    
    /**
     * Creates a bulk update operation.
     * 
     * @param data the data to update
     * @param <T> the type of data
     * @return bulk update operation
     */
    public static <T> BulkOperation<T> update(List<T> data) {
        return new BulkOperation<>(BulkOperationType.UPDATE, data);
    }
    
    /**
     * Creates a bulk upsert operation.
     * 
     * @param data the data to upsert
     * @param externalIdField the field to use for matching existing records
     * @param <T> the type of data
     * @return bulk upsert operation
     */
    public static <T> BulkOperation<T> upsert(List<T> data, String externalIdField) {
        BulkOperation<T> operation = new BulkOperation<>(BulkOperationType.UPSERT, data);
        operation.setExternalIdField(externalIdField);
        return operation;
    }
    
    /**
     * Creates a bulk delete operation.
     * 
     * @param data the data containing identifiers for deletion
     * @param <T> the type of data
     * @return bulk delete operation
     */
    public static <T> BulkOperation<T> delete(List<T> data) {
        return new BulkOperation<>(BulkOperationType.DELETE, data);
    }
    
    /**
     * Sets an option for the bulk operation.
     * 
     * @param key option key
     * @param value option value
     * @return this operation for method chaining
     */
    public BulkOperation<T> withOption(String key, Object value) {
        if (this.options == null) {
            this.options = new java.util.HashMap<>();
        }
        this.options.put(key, value);
        return this;
    }
    
    /**
     * Sets the batch size for processing.
     * 
     * @param batchSize the batch size
     * @return this operation for method chaining
     */
    public BulkOperation<T> withBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }
    
    /**
     * Sets whether to continue processing on errors.
     * 
     * @param continueOnError whether to continue on errors
     * @return this operation for method chaining
     */
    public BulkOperation<T> withContinueOnError(boolean continueOnError) {
        this.continueOnError = continueOnError;
        return this;
    }
    
    /**
     * Generates a unique operation ID.
     * 
     * @return unique operation identifier
     */
    private String generateOperationId() {
        return "bulk-" + System.currentTimeMillis() + "-" + 
               Integer.toHexString(this.hashCode());
    }
    
    // Getters and setters
    
    public String getOperationId() {
        return operationId;
    }
    
    public void setOperationId(String operationId) {
        this.operationId = operationId;
    }
    
    public BulkOperationType getType() {
        return type;
    }
    
    public void setType(BulkOperationType type) {
        this.type = type;
    }
    
    public List<T> getData() {
        return data;
    }
    
    public void setData(List<T> data) {
        this.data = data;
    }
    
    public Map<String, Object> getOptions() {
        return options;
    }
    
    public void setOptions(Map<String, Object> options) {
        this.options = options;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    
    public boolean isContinueOnError() {
        return continueOnError;
    }
    
    public void setContinueOnError(boolean continueOnError) {
        this.continueOnError = continueOnError;
    }
    
    public String getExternalIdField() {
        return externalIdField;
    }
    
    public void setExternalIdField(String externalIdField) {
        this.externalIdField = externalIdField;
    }
    
    /**
     * Types of bulk operations supported.
     */
    public enum BulkOperationType {
        INSERT,   // Create new records
        UPDATE,   // Update existing records
        UPSERT,   // Insert or update based on external ID
        DELETE,   // Delete existing records
        QUERY     // Bulk query operation
    }
    
    @Override
    public String toString() {
        return "BulkOperation{" +
                "operationId='" + operationId + '\'' +
                ", type=" + type +
                ", dataCount=" + (data != null ? data.size() : 0) +
                ", batchSize=" + batchSize +
                ", continueOnError=" + continueOnError +
                ", timestamp=" + timestamp +
                '}';
    }
}