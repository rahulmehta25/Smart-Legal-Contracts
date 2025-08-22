package com.enterprise.integrations.transformers;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Salesforce data transformer for converting between Salesforce and internal formats.
 * 
 * Handles transformation of Salesforce SObject data including:
 * - Field mapping and data type conversion
 * - Date/time format standardization
 * - Custom field handling
 * - Relationship data flattening/expansion
 * - Validation according to Salesforce rules
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Component
public class SalesforceTransformer implements DataTransformer<Map<String, Object>, Map<String, Object>> {
    
    private static final Logger logger = LoggerFactory.getLogger(SalesforceTransformer.class);
    
    private final ObjectMapper objectMapper;
    private final Map<String, Map<String, String>> fieldMappings;
    private final AtomicLong totalTransformations = new AtomicLong(0);
    private final AtomicLong successfulTransformations = new AtomicLong(0);
    private final AtomicLong failedTransformations = new AtomicLong(0);
    private volatile LocalDateTime lastActivity;
    
    /**
     * Constructor.
     * 
     * @param objectMapper JSON object mapper
     */
    public SalesforceTransformer(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
        this.fieldMappings = initializeFieldMappings();
        this.lastActivity = LocalDateTime.now();
    }
    
    @Override
    public String getTransformerId() {
        return "salesforce-transformer";
    }
    
    @Override
    public String getDisplayName() {
        return "Salesforce Data Transformer";
    }
    
    @Override
    public String getSourceFormat() {
        return "salesforce-sobject";
    }
    
    @Override
    public String getTargetFormat() {
        return "internal-entity";
    }
    
    @Override
    public Map<String, Object> transform(Map<String, Object> source, TransformationContext context) 
            throws TransformationException {
        long startTime = System.currentTimeMillis();
        totalTransformations.incrementAndGet();
        lastActivity = LocalDateTime.now();
        
        try {
            logger.debug("Transforming Salesforce data for partner: {}", context.partnerId());
            
            // Validate source data
            ValidationResult validation = validateSource(source, context);
            if (!validation.valid()) {
                throw new TransformationException("VALIDATION_FAILED", 
                    "Source validation failed: " + String.join(", ", validation.errors()),
                    getTransformerId(), TransformationException.TransformationStage.VALIDATION, 
                    false, source, null);
            }
            
            Map<String, Object> transformed = new HashMap<>();
            String objectType = (String) source.get("attributes");
            if (objectType != null) {
                Map<String, Object> attributes = (Map<String, Object>) source.get("attributes");
                objectType = (String) attributes.get("type");
            }
            
            // Get field mappings for the object type
            Map<String, String> mappings = fieldMappings.getOrDefault(objectType, Collections.emptyMap());
            
            for (Map.Entry<String, Object> entry : source.entrySet()) {
                String fieldName = entry.getKey();
                Object fieldValue = entry.getValue();
                
                // Skip system fields
                if (isSystemField(fieldName)) {
                    continue;
                }
                
                // Apply field mapping
                String targetField = mappings.getOrDefault(fieldName, fieldName);
                Object transformedValue = transformFieldValue(fieldName, fieldValue, objectType, context);
                
                if (transformedValue != null) {
                    transformed.put(targetField, transformedValue);
                }
            }
            
            // Add metadata
            transformed.put("_sourceSystem", "salesforce");
            transformed.put("_objectType", objectType);
            transformed.put("_transformedAt", LocalDateTime.now());
            transformed.put("_transformerId", getTransformerId());
            
            // Post-process based on context
            Map<String, Object> postProcessed = postProcessTransformation(transformed, context);
            
            successfulTransformations.incrementAndGet();
            logger.debug("Successfully transformed Salesforce data in {}ms", 
                        System.currentTimeMillis() - startTime);
            
            return postProcessed;
            
        } catch (TransformationException e) {
            failedTransformations.incrementAndGet();
            throw e;
        } catch (Exception e) {
            failedTransformations.incrementAndGet();
            logger.error("Error transforming Salesforce data", e);
            throw new TransformationException("TRANSFORMATION_ERROR", 
                "Error during transformation: " + e.getMessage(), e,
                getTransformerId(), TransformationException.TransformationStage.MAPPING, true);
        }
    }
    
    @Override
    public Map<String, Object> reverseTransform(Map<String, Object> target, TransformationContext context) 
            throws TransformationException {
        long startTime = System.currentTimeMillis();
        totalTransformations.incrementAndGet();
        lastActivity = LocalDateTime.now();
        
        try {
            logger.debug("Reverse transforming data to Salesforce format for partner: {}", context.partnerId());
            
            Map<String, Object> salesforceData = new HashMap<>();
            String objectType = (String) target.get("_objectType");
            
            if (objectType == null) {
                throw new TransformationException("MISSING_OBJECT_TYPE", 
                    "Object type is required for reverse transformation");
            }
            
            // Get reverse field mappings
            Map<String, String> mappings = fieldMappings.getOrDefault(objectType, Collections.emptyMap());
            Map<String, String> reverseMappings = new HashMap<>();
            for (Map.Entry<String, String> entry : mappings.entrySet()) {
                reverseMappings.put(entry.getValue(), entry.getKey());
            }
            
            for (Map.Entry<String, Object> entry : target.entrySet()) {
                String fieldName = entry.getKey();
                Object fieldValue = entry.getValue();
                
                // Skip metadata fields
                if (fieldName.startsWith("_")) {
                    continue;
                }
                
                // Apply reverse field mapping
                String salesforceField = reverseMappings.getOrDefault(fieldName, fieldName);
                Object transformedValue = reverseTransformFieldValue(salesforceField, fieldValue, objectType, context);
                
                if (transformedValue != null) {
                    salesforceData.put(salesforceField, transformedValue);
                }
            }
            
            successfulTransformations.incrementAndGet();
            logger.debug("Successfully reverse transformed data to Salesforce format in {}ms", 
                        System.currentTimeMillis() - startTime);
            
            return salesforceData;
            
        } catch (TransformationException e) {
            failedTransformations.incrementAndGet();
            throw e;
        } catch (Exception e) {
            failedTransformations.incrementAndGet();
            logger.error("Error reverse transforming to Salesforce format", e);
            throw new TransformationException("REVERSE_TRANSFORMATION_ERROR", 
                "Error during reverse transformation: " + e.getMessage(), e,
                getTransformerId(), TransformationException.TransformationStage.MAPPING, true);
        }
    }
    
    @Override
    public ValidationResult validateSource(Map<String, Object> source, TransformationContext context) {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();
        
        if (source == null || source.isEmpty()) {
            errors.add("Source data is null or empty");
            return ValidationResult.failure(errors);
        }
        
        // Check for required Salesforce fields
        if (!source.containsKey("Id") && !context.operationType().equals("CREATE")) {
            errors.add("Id field is required for non-CREATE operations");
        }
        
        // Validate data types
        for (Map.Entry<String, Object> entry : source.entrySet()) {
            String fieldName = entry.getKey();
            Object value = entry.getValue();
            
            if (value != null) {
                // Check date fields
                if (fieldName.endsWith("Date") || fieldName.endsWith("__c") && value instanceof String) {
                    if (!isValidSalesforceDate((String) value)) {
                        warnings.add("Invalid date format in field: " + fieldName);
                    }
                }
                
                // Check lookup relationships
                if (fieldName.endsWith("Id") && !(value instanceof String)) {
                    errors.add("Lookup field must be a string: " + fieldName);
                }
            }
        }
        
        return errors.isEmpty() ? 
            (warnings.isEmpty() ? ValidationResult.success() : ValidationResult.withWarnings(warnings)) :
            ValidationResult.failure(errors);
    }
    
    @Override
    public ValidationResult validateTarget(Map<String, Object> target, TransformationContext context) {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();
        
        if (target == null || target.isEmpty()) {
            errors.add("Target data is null or empty");
            return ValidationResult.failure(errors);
        }
        
        // Check for required metadata
        if (!target.containsKey("_sourceSystem")) {
            warnings.add("Missing _sourceSystem metadata");
        }
        
        if (!target.containsKey("_objectType")) {
            errors.add("Missing _objectType metadata");
        }
        
        return errors.isEmpty() ? 
            (warnings.isEmpty() ? ValidationResult.success() : ValidationResult.withWarnings(warnings)) :
            ValidationResult.failure(errors);
    }
    
    @Override
    public Map<String, Object> getTransformationSchema() {
        Map<String, Object> schema = new HashMap<>();
        schema.put("fieldMappings", fieldMappings);
        schema.put("supportedObjectTypes", Arrays.asList("Account", "Contact", "Lead", "Opportunity"));
        schema.put("dateFormats", Arrays.asList("yyyy-MM-dd", "yyyy-MM-dd'T'HH:mm:ss.SSSZ"));
        schema.put("systemFields", Arrays.asList("attributes", "Id", "CreatedDate", "LastModifiedDate"));
        return schema;
    }
    
    @Override
    public boolean supports(String sourceFormat, String targetFormat) {
        return ("salesforce-sobject".equals(sourceFormat) && "internal-entity".equals(targetFormat)) ||
               ("internal-entity".equals(sourceFormat) && "salesforce-sobject".equals(targetFormat));
    }
    
    @Override
    public TransformationMetrics getMetrics() {
        double averageTime = totalTransformations.get() > 0 ? 
            (double) successfulTransformations.get() / totalTransformations.get() * 100 : 0.0;
        
        return new TransformationMetrics(
            totalTransformations.get(),
            successfulTransformations.get(),
            failedTransformations.get(),
            averageTime,
            lastActivity
        );
    }
    
    /**
     * Initializes field mappings for different Salesforce objects.
     * 
     * @return field mappings by object type
     */
    private Map<String, Map<String, String>> initializeFieldMappings() {
        Map<String, Map<String, String>> mappings = new HashMap<>();
        
        // Account mappings
        Map<String, String> accountMappings = new HashMap<>();
        accountMappings.put("Name", "name");
        accountMappings.put("BillingStreet", "billingAddress.street");
        accountMappings.put("BillingCity", "billingAddress.city");
        accountMappings.put("BillingState", "billingAddress.state");
        accountMappings.put("BillingPostalCode", "billingAddress.postalCode");
        accountMappings.put("BillingCountry", "billingAddress.country");
        accountMappings.put("Phone", "phoneNumber");
        accountMappings.put("Website", "website");
        accountMappings.put("Industry", "industry");
        mappings.put("Account", accountMappings);
        
        // Contact mappings
        Map<String, String> contactMappings = new HashMap<>();
        contactMappings.put("FirstName", "firstName");
        contactMappings.put("LastName", "lastName");
        contactMappings.put("Email", "emailAddress");
        contactMappings.put("Phone", "phoneNumber");
        contactMappings.put("Title", "jobTitle");
        contactMappings.put("AccountId", "accountId");
        mappings.put("Contact", contactMappings);
        
        // Lead mappings
        Map<String, String> leadMappings = new HashMap<>();
        leadMappings.put("FirstName", "firstName");
        leadMappings.put("LastName", "lastName");
        leadMappings.put("Company", "companyName");
        leadMappings.put("Email", "emailAddress");
        leadMappings.put("Phone", "phoneNumber");
        leadMappings.put("Status", "status");
        leadMappings.put("LeadSource", "source");
        mappings.put("Lead", leadMappings);
        
        // Opportunity mappings
        Map<String, String> opportunityMappings = new HashMap<>();
        opportunityMappings.put("Name", "name");
        opportunityMappings.put("Amount", "value");
        opportunityMappings.put("CloseDate", "closeDate");
        opportunityMappings.put("StageName", "stage");
        opportunityMappings.put("AccountId", "accountId");
        opportunityMappings.put("Probability", "probability");
        mappings.put("Opportunity", opportunityMappings);
        
        return mappings;
    }
    
    /**
     * Checks if a field is a system field that should be excluded from transformation.
     * 
     * @param fieldName the field name to check
     * @return true if it's a system field
     */
    private boolean isSystemField(String fieldName) {
        return fieldName.equals("attributes") ||
               fieldName.equals("CreatedDate") ||
               fieldName.equals("LastModifiedDate") ||
               fieldName.equals("SystemModstamp") ||
               fieldName.equals("CreatedById") ||
               fieldName.equals("LastModifiedById");
    }
    
    /**
     * Transforms a field value based on field type and context.
     * 
     * @param fieldName the field name
     * @param value the field value
     * @param objectType the Salesforce object type
     * @param context the transformation context
     * @return transformed value
     */
    private Object transformFieldValue(String fieldName, Object value, String objectType, 
                                     TransformationContext context) {
        if (value == null) {
            return null;
        }
        
        // Handle date fields
        if (fieldName.endsWith("Date") && value instanceof String) {
            return transformSalesforceDate((String) value);
        }
        
        // Handle currency fields
        if (fieldName.equals("Amount") || fieldName.endsWith("__c") && fieldName.contains("Amount")) {
            return transformCurrencyField(value);
        }
        
        // Handle boolean fields
        if (value instanceof Boolean) {
            return value;
        }
        
        // Handle lookup relationships
        if (fieldName.endsWith("Id") && value instanceof String) {
            return transformLookupId((String) value, fieldName);
        }
        
        // Handle picklist values
        if (fieldName.equals("Status") || fieldName.equals("StageName") || fieldName.equals("Industry")) {
            return transformPicklistValue((String) value, fieldName);
        }
        
        return value;
    }
    
    /**
     * Reverse transforms a field value from internal format to Salesforce format.
     * 
     * @param fieldName the Salesforce field name
     * @param value the internal field value
     * @param objectType the Salesforce object type
     * @param context the transformation context
     * @return reverse transformed value
     */
    private Object reverseTransformFieldValue(String fieldName, Object value, String objectType, 
                                            TransformationContext context) {
        if (value == null) {
            return null;
        }
        
        // Handle date fields
        if (fieldName.endsWith("Date") && value instanceof LocalDateTime) {
            return ((LocalDateTime) value).format(DateTimeFormatter.ISO_INSTANT);
        }
        
        return value;
    }
    
    /**
     * Transforms Salesforce date string to LocalDateTime.
     * 
     * @param dateString the Salesforce date string
     * @return LocalDateTime object
     */
    private LocalDateTime transformSalesforceDate(String dateString) {
        try {
            // Handle different Salesforce date formats
            if (dateString.contains("T")) {
                return LocalDateTime.parse(dateString.substring(0, 19));
            } else {
                return LocalDateTime.parse(dateString + "T00:00:00");
            }
        } catch (Exception e) {
            logger.warn("Failed to parse date: {}", dateString);
            return null;
        }
    }
    
    /**
     * Validates Salesforce date format.
     * 
     * @param dateString the date string to validate
     * @return true if valid Salesforce date format
     */
    private boolean isValidSalesforceDate(String dateString) {
        try {
            transformSalesforceDate(dateString);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Transforms currency field values.
     * 
     * @param value the currency value
     * @return transformed currency value
     */
    private Object transformCurrencyField(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        if (value instanceof String) {
            try {
                return Double.parseDouble((String) value);
            } catch (NumberFormatException e) {
                logger.warn("Failed to parse currency value: {}", value);
                return null;
            }
        }
        return value;
    }
    
    /**
     * Transforms lookup ID values.
     * 
     * @param id the lookup ID
     * @param fieldName the field name
     * @return transformed lookup ID
     */
    private String transformLookupId(String id, String fieldName) {
        // Validate Salesforce ID format (15 or 18 characters)
        if (id.length() != 15 && id.length() != 18) {
            logger.warn("Invalid Salesforce ID format: {} for field: {}", id, fieldName);
        }
        return id;
    }
    
    /**
     * Transforms picklist values.
     * 
     * @param value the picklist value
     * @param fieldName the field name
     * @return transformed picklist value
     */
    private String transformPicklistValue(String value, String fieldName) {
        // Apply any picklist value transformations
        return value;
    }
    
    /**
     * Post-processes the transformation based on context.
     * 
     * @param transformed the transformed data
     * @param context the transformation context
     * @return post-processed data
     */
    private Map<String, Object> postProcessTransformation(Map<String, Object> transformed, 
                                                        TransformationContext context) {
        // Apply any business rules or additional processing
        String operationType = context.operationType();
        
        if ("CREATE".equals(operationType)) {
            // Remove Id field for create operations
            transformed.remove("Id");
        }
        
        // Add audit fields
        transformed.put("_processedAt", LocalDateTime.now());
        transformed.put("_operationType", operationType);
        
        return transformed;
    }
}