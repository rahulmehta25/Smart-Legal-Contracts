package com.enterprise.integrations.transformers;

import java.util.Map;

/**
 * Base interface for data transformation operations.
 * 
 * Defines the contract for transforming data between different
 * partner systems and internal formats. Supports:
 * - Schema mapping and conversion
 * - Data validation and cleansing
 * - Format transformation (XML, JSON, CSV)
 * - Business rule application
 * - Error handling and logging
 * 
 * @param <S> source data type
 * @param <T> target data type
 * @author Integration Platform Team
 * @version 1.0.0
 */
public interface DataTransformer<S, T> {

    /**
     * Gets the unique identifier for this transformer.
     * 
     * @return transformer identifier
     */
    String getTransformerId();

    /**
     * Gets the display name for this transformer.
     * 
     * @return human-readable transformer name
     */
    String getDisplayName();

    /**
     * Gets the source data format/type.
     * 
     * @return source format identifier
     */
    String getSourceFormat();

    /**
     * Gets the target data format/type.
     * 
     * @return target format identifier
     */
    String getTargetFormat();

    /**
     * Transforms data from source format to target format.
     * 
     * @param source the source data to transform
     * @param context transformation context and parameters
     * @return transformed data
     * @throws TransformationException if transformation fails
     */
    T transform(S source, TransformationContext context) throws TransformationException;

    /**
     * Transforms data in reverse direction (target to source).
     * 
     * @param target the target data to reverse transform
     * @param context transformation context and parameters
     * @return reverse transformed data
     * @throws TransformationException if reverse transformation fails
     */
    S reverseTransform(T target, TransformationContext context) throws TransformationException;

    /**
     * Validates source data before transformation.
     * 
     * @param source the source data to validate
     * @param context transformation context
     * @return validation result
     */
    ValidationResult validateSource(S source, TransformationContext context);

    /**
     * Validates target data after transformation.
     * 
     * @param target the target data to validate
     * @param context transformation context
     * @return validation result
     */
    ValidationResult validateTarget(T target, TransformationContext context);

    /**
     * Gets transformation schema or mapping configuration.
     * 
     * @return transformation schema/mapping
     */
    Map<String, Object> getTransformationSchema();

    /**
     * Checks if this transformer supports the given source and target formats.
     * 
     * @param sourceFormat source data format
     * @param targetFormat target data format
     * @return true if transformation is supported
     */
    boolean supports(String sourceFormat, String targetFormat);

    /**
     * Gets transformation metrics and statistics.
     * 
     * @return transformation metrics
     */
    TransformationMetrics getMetrics();

    /**
     * Transformation context containing metadata and parameters.
     */
    record TransformationContext(
        String partnerId,
        String operationType,
        Map<String, Object> parameters,
        Map<String, Object> metadata
    ) {
        public TransformationContext withParameter(String key, Object value) {
            Map<String, Object> newParams = new java.util.HashMap<>(parameters);
            newParams.put(key, value);
            return new TransformationContext(partnerId, operationType, newParams, metadata);
        }
        
        public TransformationContext withMetadata(String key, Object value) {
            Map<String, Object> newMeta = new java.util.HashMap<>(metadata);
            newMeta.put(key, value);
            return new TransformationContext(partnerId, operationType, parameters, newMeta);
        }
    }

    /**
     * Validation result for data validation operations.
     */
    record ValidationResult(
        boolean valid,
        java.util.List<String> errors,
        java.util.List<String> warnings,
        Map<String, Object> details
    ) {
        public static ValidationResult success() {
            return new ValidationResult(true, 
                java.util.Collections.emptyList(),
                java.util.Collections.emptyList(),
                java.util.Collections.emptyMap());
        }
        
        public static ValidationResult failure(java.util.List<String> errors) {
            return new ValidationResult(false, errors,
                java.util.Collections.emptyList(),
                java.util.Collections.emptyMap());
        }
        
        public static ValidationResult withWarnings(java.util.List<String> warnings) {
            return new ValidationResult(true,
                java.util.Collections.emptyList(),
                warnings,
                java.util.Collections.emptyMap());
        }
    }

    /**
     * Transformation metrics for monitoring and performance tracking.
     */
    record TransformationMetrics(
        long totalTransformations,
        long successfulTransformations,
        long failedTransformations,
        double averageTransformationTime,
        java.time.LocalDateTime lastActivity
    ) {}
}