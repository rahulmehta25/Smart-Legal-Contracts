package com.enterprise.integrations.connectors;

/**
 * Exception thrown by connector operations.
 * 
 * Provides detailed error information including error codes,
 * retry hints, and partner-specific error details.
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
public class ConnectorException extends RuntimeException {
    
    private final String errorCode;
    private final boolean retryable;
    private final String partnerId;
    private final Object errorDetails;
    
    /**
     * Creates a connector exception.
     * 
     * @param message error message
     */
    public ConnectorException(String message) {
        super(message);
        this.errorCode = "CONNECTOR_ERROR";
        this.retryable = false;
        this.partnerId = null;
        this.errorDetails = null;
    }
    
    /**
     * Creates a connector exception with error code.
     * 
     * @param errorCode error code
     * @param message error message
     */
    public ConnectorException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
        this.retryable = false;
        this.partnerId = null;
        this.errorDetails = null;
    }
    
    /**
     * Creates a connector exception with full details.
     * 
     * @param errorCode error code
     * @param message error message
     * @param retryable whether the operation can be retried
     * @param partnerId the partner system identifier
     * @param errorDetails additional error details
     */
    public ConnectorException(String errorCode, String message, boolean retryable, 
                             String partnerId, Object errorDetails) {
        super(message);
        this.errorCode = errorCode;
        this.retryable = retryable;
        this.partnerId = partnerId;
        this.errorDetails = errorDetails;
    }
    
    /**
     * Creates a connector exception with cause.
     * 
     * @param errorCode error code
     * @param message error message
     * @param cause the underlying cause
     * @param retryable whether the operation can be retried
     */
    public ConnectorException(String errorCode, String message, Throwable cause, boolean retryable) {
        super(message, cause);
        this.errorCode = errorCode;
        this.retryable = retryable;
        this.partnerId = null;
        this.errorDetails = null;
    }
    
    // Getters
    
    public String getErrorCode() {
        return errorCode;
    }
    
    public boolean isRetryable() {
        return retryable;
    }
    
    public String getPartnerId() {
        return partnerId;
    }
    
    public Object getErrorDetails() {
        return errorDetails;
    }
}