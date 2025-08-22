package com.enterprise.integrations.transformers;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for SalesforceTransformer.
 * 
 * Tests data transformation functionality including:
 * - Field mapping and conversion
 * - Data validation
 * - Error handling
 * - Date/time transformations
 * - Custom field processing
 * - Reverse transformations
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@ExtendWith(MockitoExtension.class)
class SalesforceTransformerTest {
    
    @Mock
    private ObjectMapper objectMapper;
    
    private SalesforceTransformer transformer;
    
    @BeforeEach
    void setUp() {
        transformer = new SalesforceTransformer(objectMapper);
    }
    
    @Test
    void testGetTransformerId() {
        assertEquals("salesforce-transformer", transformer.getTransformerId());
    }
    
    @Test
    void testGetDisplayName() {
        assertEquals("Salesforce Data Transformer", transformer.getDisplayName());
    }
    
    @Test
    void testGetSourceFormat() {
        assertEquals("salesforce-sobject", transformer.getSourceFormat());
    }
    
    @Test
    void testGetTargetFormat() {
        assertEquals("internal-entity", transformer.getTargetFormat());
    }
    
    @Test
    void testSupportsCorrectFormats() {
        assertTrue(transformer.supports("salesforce-sobject", "internal-entity"));
        assertTrue(transformer.supports("internal-entity", "salesforce-sobject"));
        assertFalse(transformer.supports("other-format", "internal-entity"));
        assertFalse(transformer.supports("salesforce-sobject", "other-format"));
    }
    
    @Test
    void testTransformAccountData() {
        // Prepare Salesforce Account data
        Map<String, Object> salesforceAccount = new HashMap<>();
        salesforceAccount.put("Id", "001000000000001");
        salesforceAccount.put("Name", "Test Company");
        salesforceAccount.put("BillingStreet", "123 Main Street");
        salesforceAccount.put("BillingCity", "San Francisco");
        salesforceAccount.put("BillingState", "CA");
        salesforceAccount.put("BillingPostalCode", "94105");
        salesforceAccount.put("BillingCountry", "USA");
        salesforceAccount.put("Phone", "(555) 123-4567");
        salesforceAccount.put("Website", "https://testcompany.com");
        salesforceAccount.put("Industry", "Technology");
        
        // Add attributes (Salesforce metadata)
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("type", "Account");
        attributes.put("url", "/services/data/v58.0/sobjects/Account/001000000000001");
        salesforceAccount.put("attributes", attributes);
        
        // Create transformation context
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Transform data
        Map<String, Object> transformed = transformer.transform(salesforceAccount, context);
        
        // Verify transformations
        assertNotNull(transformed);
        assertEquals("Test Company", transformed.get("name"));
        assertEquals("123 Main Street", transformed.get("billingAddress.street"));
        assertEquals("San Francisco", transformed.get("billingAddress.city"));
        assertEquals("CA", transformed.get("billingAddress.state"));
        assertEquals("94105", transformed.get("billingAddress.postalCode"));
        assertEquals("USA", transformed.get("billingAddress.country"));
        assertEquals("(555) 123-4567", transformed.get("phoneNumber"));
        assertEquals("https://testcompany.com", transformed.get("website"));
        assertEquals("Technology", transformed.get("industry"));
        
        // Verify metadata
        assertEquals("salesforce", transformed.get("_sourceSystem"));
        assertEquals("Account", transformed.get("_objectType"));
        assertEquals("salesforce-transformer", transformed.get("_transformerId"));
        assertNotNull(transformed.get("_transformedAt"));
    }
    
    @Test
    void testTransformContactData() {
        // Prepare Salesforce Contact data
        Map<String, Object> salesforceContact = new HashMap<>();
        salesforceContact.put("Id", "003000000000001");
        salesforceContact.put("FirstName", "John");
        salesforceContact.put("LastName", "Doe");
        salesforceContact.put("Email", "john.doe@example.com");
        salesforceContact.put("Phone", "(555) 987-6543");
        salesforceContact.put("Title", "Senior Developer");
        salesforceContact.put("AccountId", "001000000000001");
        
        // Add attributes
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("type", "Contact");
        salesforceContact.put("attributes", attributes);
        
        // Create transformation context
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Transform data
        Map<String, Object> transformed = transformer.transform(salesforceContact, context);
        
        // Verify transformations
        assertNotNull(transformed);
        assertEquals("John", transformed.get("firstName"));
        assertEquals("Doe", transformed.get("lastName"));
        assertEquals("john.doe@example.com", transformed.get("emailAddress"));
        assertEquals("(555) 987-6543", transformed.get("phoneNumber"));
        assertEquals("Senior Developer", transformed.get("jobTitle"));
        assertEquals("001000000000001", transformed.get("accountId"));
        
        // Verify metadata
        assertEquals("salesforce", transformed.get("_sourceSystem"));
        assertEquals("Contact", transformed.get("_objectType"));
    }
    
    @Test
    void testTransformWithDateFields() {
        // Prepare Salesforce data with date fields
        Map<String, Object> salesforceData = new HashMap<>();
        salesforceData.put("Id", "006000000000001");
        salesforceData.put("CloseDate", "2024-12-31");
        salesforceData.put("CreatedDate", "2024-01-15T10:30:00.000Z");
        
        // Add attributes
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("type", "Opportunity");
        salesforceData.put("attributes", attributes);
        
        // Create transformation context
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Transform data
        Map<String, Object> transformed = transformer.transform(salesforceData, context);
        
        // Verify date transformations
        assertNotNull(transformed);
        assertInstanceOf(LocalDateTime.class, transformed.get("closeDate"));
    }
    
    @Test
    void testTransformWithCurrencyFields() {
        // Prepare Salesforce data with currency fields
        Map<String, Object> salesforceData = new HashMap<>();
        salesforceData.put("Id", "006000000000001");
        salesforceData.put("Amount", "50000.00");
        
        // Add attributes
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("type", "Opportunity");
        salesforceData.put("attributes", attributes);
        
        // Create transformation context
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Transform data
        Map<String, Object> transformed = transformer.transform(salesforceData, context);
        
        // Verify currency transformation
        assertNotNull(transformed);
        assertEquals(50000.0, transformed.get("value"));
    }
    
    @Test
    void testReverseTransformation() {
        // Prepare internal entity data
        Map<String, Object> internalData = new HashMap<>();
        internalData.put("name", "Test Company");
        internalData.put("phoneNumber", "(555) 123-4567");
        internalData.put("website", "https://testcompany.com");
        internalData.put("industry", "Technology");
        internalData.put("_objectType", "Account");
        internalData.put("_sourceSystem", "salesforce");
        internalData.put("_transformedAt", LocalDateTime.now());
        
        // Create transformation context
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "WRITE",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Reverse transform data
        Map<String, Object> salesforceData = transformer.reverseTransform(internalData, context);
        
        // Verify reverse transformations
        assertNotNull(salesforceData);
        assertEquals("Test Company", salesforceData.get("Name"));
        assertEquals("(555) 123-4567", salesforceData.get("Phone"));
        assertEquals("https://testcompany.com", salesforceData.get("Website"));
        assertEquals("Technology", salesforceData.get("Industry"));
        
        // Verify metadata fields are excluded
        assertFalse(salesforceData.containsKey("_objectType"));
        assertFalse(salesforceData.containsKey("_sourceSystem"));
        assertFalse(salesforceData.containsKey("_transformedAt"));
    }
    
    @Test
    void testReverseTransformationWithoutObjectType() {
        // Prepare internal data without object type
        Map<String, Object> internalData = new HashMap<>();
        internalData.put("name", "Test Company");
        
        // Create transformation context
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "WRITE",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Reverse transformation should fail
        TransformationException exception = assertThrows(TransformationException.class, 
            () -> transformer.reverseTransform(internalData, context));
        
        assertEquals("MISSING_OBJECT_TYPE", exception.getErrorCode());
    }
    
    @Test
    void testValidateSourceDataSuccess() {
        // Prepare valid Salesforce data
        Map<String, Object> salesforceData = new HashMap<>();
        salesforceData.put("Id", "001000000000001");
        salesforceData.put("Name", "Test Account");
        
        // Create transformation context for update operation
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "UPDATE",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Validate source data
        DataTransformer.ValidationResult result = transformer.validateSource(salesforceData, context);
        
        assertTrue(result.valid());
        assertTrue(result.errors().isEmpty());
    }
    
    @Test
    void testValidateSourceDataMissingId() {
        // Prepare Salesforce data without Id for update operation
        Map<String, Object> salesforceData = new HashMap<>();
        salesforceData.put("Name", "Test Account");
        
        // Create transformation context for update operation
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "UPDATE",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Validate source data
        DataTransformer.ValidationResult result = transformer.validateSource(salesforceData, context);
        
        assertFalse(result.valid());
        assertFalse(result.errors().isEmpty());
        assertTrue(result.errors().get(0).contains("Id field is required"));
    }
    
    @Test
    void testValidateSourceDataEmpty() {
        // Test with empty data
        Map<String, Object> emptyData = new HashMap<>();
        
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "CREATE",
            new HashMap<>(),
            new HashMap<>()
        );
        
        DataTransformer.ValidationResult result = transformer.validateSource(emptyData, context);
        
        assertFalse(result.valid());
        assertTrue(result.errors().get(0).contains("null or empty"));
    }
    
    @Test
    void testValidateTargetDataSuccess() {
        // Prepare valid transformed data
        Map<String, Object> transformedData = new HashMap<>();
        transformedData.put("name", "Test Account");
        transformedData.put("_sourceSystem", "salesforce");
        transformedData.put("_objectType", "Account");
        
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        DataTransformer.ValidationResult result = transformer.validateTarget(transformedData, context);
        
        assertTrue(result.valid());
    }
    
    @Test
    void testValidateTargetDataMissingMetadata() {
        // Prepare transformed data missing metadata
        Map<String, Object> transformedData = new HashMap<>();
        transformedData.put("name", "Test Account");
        
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        DataTransformer.ValidationResult result = transformer.validateTarget(transformedData, context);
        
        assertFalse(result.valid());
        assertTrue(result.errors().get(0).contains("_objectType"));
    }
    
    @Test
    void testGetTransformationSchema() {
        Map<String, Object> schema = transformer.getTransformationSchema();
        
        assertNotNull(schema);
        assertTrue(schema.containsKey("fieldMappings"));
        assertTrue(schema.containsKey("supportedObjectTypes"));
        assertTrue(schema.containsKey("dateFormats"));
        assertTrue(schema.containsKey("systemFields"));
        
        // Verify supported object types
        @SuppressWarnings("unchecked")
        java.util.List<String> supportedTypes = (java.util.List<String>) schema.get("supportedObjectTypes");
        assertTrue(supportedTypes.contains("Account"));
        assertTrue(supportedTypes.contains("Contact"));
        assertTrue(supportedTypes.contains("Lead"));
        assertTrue(supportedTypes.contains("Opportunity"));
    }
    
    @Test
    void testTransformationMetrics() {
        // Perform some transformations to generate metrics
        Map<String, Object> testData = new HashMap<>();
        testData.put("Id", "001000000000001");
        testData.put("Name", "Test");
        
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("type", "Account");
        testData.put("attributes", attributes);
        
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Perform transformation
        transformer.transform(testData, context);
        
        // Get metrics
        DataTransformer.TransformationMetrics metrics = transformer.getMetrics();
        
        assertNotNull(metrics);
        assertTrue(metrics.totalTransformations() > 0);
        assertTrue(metrics.successfulTransformations() > 0);
        assertEquals(0, metrics.failedTransformations());
        assertNotNull(metrics.lastActivity());
    }
    
    @Test
    void testTransformationFailure() {
        // Prepare invalid data that will cause transformation to fail
        Map<String, Object> invalidData = null;
        
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        // Transformation should fail
        TransformationException exception = assertThrows(TransformationException.class, 
            () -> transformer.transform(invalidData, context));
        
        assertEquals("VALIDATION_FAILED", exception.getErrorCode());
        assertEquals("salesforce-transformer", exception.getTransformerId());
        assertEquals(TransformationException.TransformationStage.VALIDATION, exception.getStage());
    }
    
    @Test
    void testSystemFieldsExclusion() {
        // Prepare Salesforce data with system fields
        Map<String, Object> salesforceData = new HashMap<>();
        salesforceData.put("Id", "001000000000001");
        salesforceData.put("Name", "Test Account");
        salesforceData.put("CreatedDate", "2024-01-01T00:00:00.000Z");
        salesforceData.put("LastModifiedDate", "2024-01-02T00:00:00.000Z");
        salesforceData.put("SystemModstamp", "2024-01-02T00:00:00.000Z");
        salesforceData.put("CreatedById", "005000000000001");
        salesforceData.put("LastModifiedById", "005000000000001");
        
        // Add attributes
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("type", "Account");
        salesforceData.put("attributes", attributes);
        
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "READ",
            new HashMap<>(),
            new HashMap<>()
        );
        
        Map<String, Object> transformed = transformer.transform(salesforceData, context);
        
        // Verify system fields are excluded
        assertFalse(transformed.containsKey("CreatedDate"));
        assertFalse(transformed.containsKey("LastModifiedDate"));
        assertFalse(transformed.containsKey("SystemModstamp"));
        assertFalse(transformed.containsKey("CreatedById"));
        assertFalse(transformed.containsKey("LastModifiedById"));
        assertFalse(transformed.containsKey("attributes"));
        
        // Verify regular fields are included
        assertTrue(transformed.containsKey("name"));
    }
    
    @Test
    void testCreateOperationFieldRemoval() {
        // Prepare data for CREATE operation
        Map<String, Object> internalData = new HashMap<>();
        internalData.put("Id", "001000000000001"); // Should be removed for CREATE
        internalData.put("name", "Test Account");
        internalData.put("_objectType", "Account");
        internalData.put("_sourceSystem", "salesforce");
        
        DataTransformer.TransformationContext context = new DataTransformer.TransformationContext(
            "salesforce",
            "CREATE",
            new HashMap<>(),
            new HashMap<>()
        );
        
        Map<String, Object> transformed = transformer.transform(internalData, context);
        
        // Verify Id is removed for CREATE operations
        assertFalse(transformed.containsKey("Id"));
        assertTrue(transformed.containsKey("name"));
        assertEquals("CREATE", transformed.get("_operationType"));
    }
}