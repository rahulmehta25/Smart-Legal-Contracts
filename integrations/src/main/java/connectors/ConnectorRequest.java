package com.enterprise.integrations.connectors;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Request object for connector operations.
 * 
 * Encapsulates all parameters needed for data retrieval, updates,
 * and other operations with partner systems.
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ConnectorRequest {
    
    private String operation;
    private String resourceType;
    private String resourceId;
    private Map<String, Object> parameters;
    private Map<String, String> headers;
    private Object payload;
    private LocalDateTime timestamp;
    private String requestId;
    private PaginationRequest pagination;
    private FilterRequest filter;
    
    /**
     * Default constructor.
     */
    public ConnectorRequest() {
        this.parameters = new HashMap<>();
        this.headers = new HashMap<>();
        this.timestamp = LocalDateTime.now();
        this.requestId = generateRequestId();
    }
    
    /**
     * Constructor with basic parameters.
     * 
     * @param operation the operation to perform
     * @param resourceType the type of resource
     */
    public ConnectorRequest(String operation, String resourceType) {
        this();
        this.operation = operation;
        this.resourceType = resourceType;
    }
    
    /**
     * Constructor with full parameters.
     * 
     * @param operation the operation to perform
     * @param resourceType the type of resource
     * @param resourceId the resource identifier
     * @param payload the request payload
     */
    public ConnectorRequest(String operation, String resourceType, String resourceId, Object payload) {
        this(operation, resourceType);
        this.resourceId = resourceId;
        this.payload = payload;
    }
    
    /**
     * Adds a parameter to the request.
     * 
     * @param key parameter key
     * @param value parameter value
     * @return this request for method chaining
     */
    public ConnectorRequest withParameter(String key, Object value) {
        this.parameters.put(key, value);
        return this;
    }
    
    /**
     * Adds a header to the request.
     * 
     * @param key header key
     * @param value header value
     * @return this request for method chaining
     */
    public ConnectorRequest withHeader(String key, String value) {
        this.headers.put(key, value);
        return this;
    }
    
    /**
     * Sets pagination for the request.
     * 
     * @param pagination pagination parameters
     * @return this request for method chaining
     */
    public ConnectorRequest withPagination(PaginationRequest pagination) {
        this.pagination = pagination;
        return this;
    }
    
    /**
     * Sets filtering for the request.
     * 
     * @param filter filter parameters
     * @return this request for method chaining
     */
    public ConnectorRequest withFilter(FilterRequest filter) {
        this.filter = filter;
        return this;
    }
    
    /**
     * Generates a unique request ID.
     * 
     * @return unique request identifier
     */
    private String generateRequestId() {
        return "req-" + System.currentTimeMillis() + "-" + 
               Integer.toHexString(this.hashCode());
    }
    
    // Getters and setters
    
    public String getOperation() {
        return operation;
    }
    
    public void setOperation(String operation) {
        this.operation = operation;
    }
    
    public String getResourceType() {
        return resourceType;
    }
    
    public void setResourceType(String resourceType) {
        this.resourceType = resourceType;
    }
    
    public String getResourceId() {
        return resourceId;
    }
    
    public void setResourceId(String resourceId) {
        this.resourceId = resourceId;
    }
    
    public Map<String, Object> getParameters() {
        return parameters;
    }
    
    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters;
    }
    
    public Map<String, String> getHeaders() {
        return headers;
    }
    
    public void setHeaders(Map<String, String> headers) {
        this.headers = headers;
    }
    
    public Object getPayload() {
        return payload;
    }
    
    public void setPayload(Object payload) {
        this.payload = payload;
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
    
    public PaginationRequest getPagination() {
        return pagination;
    }
    
    public void setPagination(PaginationRequest pagination) {
        this.pagination = pagination;
    }
    
    public FilterRequest getFilter() {
        return filter;
    }
    
    public void setFilter(FilterRequest filter) {
        this.filter = filter;
    }
    
    /**
     * Pagination parameters for requests.
     */
    public static class PaginationRequest {
        private int page = 1;
        private int size = 50;
        private String cursor;
        private String sortBy;
        private SortDirection sortDirection = SortDirection.ASC;
        
        public PaginationRequest() {}
        
        public PaginationRequest(int page, int size) {
            this.page = page;
            this.size = size;
        }
        
        public PaginationRequest(String cursor, int size) {
            this.cursor = cursor;
            this.size = size;
        }
        
        // Getters and setters
        public int getPage() { return page; }
        public void setPage(int page) { this.page = page; }
        public int getSize() { return size; }
        public void setSize(int size) { this.size = size; }
        public String getCursor() { return cursor; }
        public void setCursor(String cursor) { this.cursor = cursor; }
        public String getSortBy() { return sortBy; }
        public void setSortBy(String sortBy) { this.sortBy = sortBy; }
        public SortDirection getSortDirection() { return sortDirection; }
        public void setSortDirection(SortDirection sortDirection) { this.sortDirection = sortDirection; }
    }
    
    /**
     * Filter parameters for requests.
     */
    public static class FilterRequest {
        private Map<String, Object> criteria;
        private String query;
        private LocalDateTime fromDate;
        private LocalDateTime toDate;
        
        public FilterRequest() {
            this.criteria = new HashMap<>();
        }
        
        public FilterRequest(Map<String, Object> criteria) {
            this.criteria = criteria != null ? criteria : new HashMap<>();
        }
        
        public FilterRequest withCriteria(String key, Object value) {
            this.criteria.put(key, value);
            return this;
        }
        
        public FilterRequest withDateRange(LocalDateTime from, LocalDateTime to) {
            this.fromDate = from;
            this.toDate = to;
            return this;
        }
        
        // Getters and setters
        public Map<String, Object> getCriteria() { return criteria; }
        public void setCriteria(Map<String, Object> criteria) { this.criteria = criteria; }
        public String getQuery() { return query; }
        public void setQuery(String query) { this.query = query; }
        public LocalDateTime getFromDate() { return fromDate; }
        public void setFromDate(LocalDateTime fromDate) { this.fromDate = fromDate; }
        public LocalDateTime getToDate() { return toDate; }
        public void setToDate(LocalDateTime toDate) { this.toDate = toDate; }
    }
    
    /**
     * Sort direction enumeration.
     */
    public enum SortDirection {
        ASC, DESC
    }
    
    @Override
    public String toString() {
        return "ConnectorRequest{" +
                "operation='" + operation + '\'' +
                ", resourceType='" + resourceType + '\'' +
                ", resourceId='" + resourceId + '\'' +
                ", requestId='" + requestId + '\'' +
                ", timestamp=" + timestamp +
                '}';
    }
}