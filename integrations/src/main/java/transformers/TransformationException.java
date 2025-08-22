package com.enterprise.integrations.transformers;

/**
 * Exception thrown during data transformation operations.
 * 
 * Provides detailed error information including error codes,
 * transformation stage, and recovery hints.
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
public class TransformationException extends RuntimeException {
    
    private final String errorCode;
    private final String transformerId;
    private final TransformationStage stage;
    private final boolean retryable;
    private final Object sourceData;
    private final Object partialResult;
    
    /**
     * Creates a transformation exception.
     * 
     * @param message error message
     */
    public TransformationException(String message) {
        super(message);
        this.errorCode = "TRANSFORMATION_ERROR";
        this.transformerId = null;
        this.stage = TransformationStage.UNKNOWN;
        this.retryable = false;
        this.sourceData = null;
        this.partialResult = null;
    }
    
    /**
     * Creates a transformation exception with error code.
     * 
     * @param errorCode error code
     * @param message error message
     */
    public TransformationException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
        this.transformerId = null;
        this.stage = TransformationStage.UNKNOWN;
        this.retryable = false;
        this.sourceData = null;
        this.partialResult = null;
    }
    
    /**
     * Creates a transformation exception with full details.
     * 
     * @param errorCode error code
     * @param message error message
     * @param transformerId transformer identifier
     * @param stage transformation stage where error occurred
     * @param retryable whether the transformation can be retried
     * @param sourceData the original source data
     * @param partialResult any partial transformation result
     */
    public TransformationException(String errorCode, String message, String transformerId,
                                 TransformationStage stage, boolean retryable,
                                 Object sourceData, Object partialResult) {
        super(message);
        this.errorCode = errorCode;
        this.transformerId = transformerId;
        this.stage = stage;
        this.retryable = retryable;
        this.sourceData = sourceData;
        this.partialResult = partialResult;
    }
    
    /**
     * Creates a transformation exception with cause.
     * 
     * @param errorCode error code
     * @param message error message
     * @param cause the underlying cause
     * @param transformerId transformer identifier
     * @param stage transformation stage where error occurred
     * @param retryable whether the transformation can be retried
     */
    public TransformationException(String errorCode, String message, Throwable cause,
                                 String transformerId, TransformationStage stage, boolean retryable) {
        super(message, cause);
        this.errorCode = errorCode;
        this.transformerId = transformerId;
        this.stage = stage;
        this.retryable = retryable;
        this.sourceData = null;
        this.partialResult = null;
    }
    
    // Getters
    
    public String getErrorCode() {
        return errorCode;
    }
    
    public String getTransformerId() {
        return transformerId;
    }
    
    public TransformationStage getStage() {
        return stage;
    }
    
    public boolean isRetryable() {
        return retryable;
    }
    
    public Object getSourceData() {
        return sourceData;
    }
    
    public Object getPartialResult() {
        return partialResult;
    }
    
    /**
     * Stages in the transformation process.
     */
    public enum TransformationStage {
        UNKNOWN,
        VALIDATION,
        PREPROCESSING,
        MAPPING,
        CONVERSION,
        POSTPROCESSING,
        SERIALIZATION
    }
}