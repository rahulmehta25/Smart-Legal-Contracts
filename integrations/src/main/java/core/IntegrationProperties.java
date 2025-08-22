package com.enterprise.integrations.core;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * Configuration properties for the integration platform.
 * 
 * This class centralizes all configuration settings for:
 * - Partner connectors
 * - API rate limits
 * - Circuit breaker settings
 * - Retry policies
 * - Monitoring thresholds
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Configuration
@ConfigurationProperties(prefix = "integration.platform")
public class IntegrationProperties {

    private Partners partners = new Partners();
    private Gateway gateway = new Gateway();
    private Monitoring monitoring = new Monitoring();
    private Kafka kafka = new Kafka();
    private Retry retry = new Retry();

    /**
     * Partner-specific configuration settings.
     */
    public static class Partners {
        private Salesforce salesforce = new Salesforce();
        private Microsoft microsoft = new Microsoft();
        private Google google = new Google();
        private DocuSign docusign = new DocuSign();
        private Slack slack = new Slack();
        private Box box = new Box();
        private Sap sap = new Sap();
        private NetSuite netsuite = new NetSuite();

        // Getters and setters
        public Salesforce getSalesforce() { return salesforce; }
        public void setSalesforce(Salesforce salesforce) { this.salesforce = salesforce; }
        public Microsoft getMicrosoft() { return microsoft; }
        public void setMicrosoft(Microsoft microsoft) { this.microsoft = microsoft; }
        public Google getGoogle() { return google; }
        public void setGoogle(Google google) { this.google = google; }
        public DocuSign getDocusign() { return docusign; }
        public void setDocusign(DocuSign docusign) { this.docusign = docusign; }
        public Slack getSlack() { return slack; }
        public void setSlack(Slack slack) { this.slack = slack; }
        public Box getBox() { return box; }
        public void setBox(Box box) { this.box = box; }
        public Sap getSap() { return sap; }
        public void setSap(Sap sap) { this.sap = sap; }
        public NetSuite getNetsuite() { return netsuite; }
        public void setNetsuite(NetSuite netsuite) { this.netsuite = netsuite; }
    }

    /**
     * Salesforce-specific configuration.
     */
    public static class Salesforce {
        private String clientId;
        private String clientSecret;
        private String username;
        private String password;
        private String securityToken;
        private String loginUrl = "https://login.salesforce.com";
        private int rateLimitPerHour = 5000;
        private boolean bulkApiEnabled = true;

        // Getters and setters
        public String getClientId() { return clientId; }
        public void setClientId(String clientId) { this.clientId = clientId; }
        public String getClientSecret() { return clientSecret; }
        public void setClientSecret(String clientSecret) { this.clientSecret = clientSecret; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getSecurityToken() { return securityToken; }
        public void setSecurityToken(String securityToken) { this.securityToken = securityToken; }
        public String getLoginUrl() { return loginUrl; }
        public void setLoginUrl(String loginUrl) { this.loginUrl = loginUrl; }
        public int getRateLimitPerHour() { return rateLimitPerHour; }
        public void setRateLimitPerHour(int rateLimitPerHour) { this.rateLimitPerHour = rateLimitPerHour; }
        public boolean isBulkApiEnabled() { return bulkApiEnabled; }
        public void setBulkApiEnabled(boolean bulkApiEnabled) { this.bulkApiEnabled = bulkApiEnabled; }
    }

    /**
     * Microsoft 365 configuration.
     */
    public static class Microsoft {
        private String clientId;
        private String clientSecret;
        private String tenantId;
        private String scope = "https://graph.microsoft.com/.default";
        private String graphUrl = "https://graph.microsoft.com/v1.0";
        private int rateLimitPerMinute = 600;

        // Getters and setters
        public String getClientId() { return clientId; }
        public void setClientId(String clientId) { this.clientId = clientId; }
        public String getClientSecret() { return clientSecret; }
        public void setClientSecret(String clientSecret) { this.clientSecret = clientSecret; }
        public String getTenantId() { return tenantId; }
        public void setTenantId(String tenantId) { this.tenantId = tenantId; }
        public String getScope() { return scope; }
        public void setScope(String scope) { this.scope = scope; }
        public String getGraphUrl() { return graphUrl; }
        public void setGraphUrl(String graphUrl) { this.graphUrl = graphUrl; }
        public int getRateLimitPerMinute() { return rateLimitPerMinute; }
        public void setRateLimitPerMinute(int rateLimitPerMinute) { this.rateLimitPerMinute = rateLimitPerMinute; }
    }

    /**
     * Google Workspace configuration.
     */
    public static class Google {
        private String clientId;
        private String clientSecret;
        private String serviceAccountKeyPath;
        private String domain;
        private int rateLimitPerSecond = 100;
        private String[] scopes = {
            "https://www.googleapis.com/auth/admin.directory.user",
            "https://www.googleapis.com/auth/admin.directory.group"
        };

        // Getters and setters
        public String getClientId() { return clientId; }
        public void setClientId(String clientId) { this.clientId = clientId; }
        public String getClientSecret() { return clientSecret; }
        public void setClientSecret(String clientSecret) { this.clientSecret = clientSecret; }
        public String getServiceAccountKeyPath() { return serviceAccountKeyPath; }
        public void setServiceAccountKeyPath(String serviceAccountKeyPath) { this.serviceAccountKeyPath = serviceAccountKeyPath; }
        public String getDomain() { return domain; }
        public void setDomain(String domain) { this.domain = domain; }
        public int getRateLimitPerSecond() { return rateLimitPerSecond; }
        public void setRateLimitPerSecond(int rateLimitPerSecond) { this.rateLimitPerSecond = rateLimitPerSecond; }
        public String[] getScopes() { return scopes; }
        public void setScopes(String[] scopes) { this.scopes = scopes; }
    }

    /**
     * DocuSign configuration.
     */
    public static class DocuSign {
        private String integrationKey;
        private String userId;
        private String accountId;
        private String privateKeyPath;
        private String basePath = "https://demo.docusign.net/restapi";
        private int rateLimitPerHour = 1000;

        // Getters and setters
        public String getIntegrationKey() { return integrationKey; }
        public void setIntegrationKey(String integrationKey) { this.integrationKey = integrationKey; }
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        public String getAccountId() { return accountId; }
        public void setAccountId(String accountId) { this.accountId = accountId; }
        public String getPrivateKeyPath() { return privateKeyPath; }
        public void setPrivateKeyPath(String privateKeyPath) { this.privateKeyPath = privateKeyPath; }
        public String getBasePath() { return basePath; }
        public void setBasePath(String basePath) { this.basePath = basePath; }
        public int getRateLimitPerHour() { return rateLimitPerHour; }
        public void setRateLimitPerHour(int rateLimitPerHour) { this.rateLimitPerHour = rateLimitPerHour; }
    }

    /**
     * Slack configuration.
     */
    public static class Slack {
        private String botToken;
        private String appToken;
        private String signingSecret;
        private int rateLimitTier = 1; // Tier 1: 1+ per minute, Tier 2: 20+ per minute, etc.

        // Getters and setters
        public String getBotToken() { return botToken; }
        public void setBotToken(String botToken) { this.botToken = botToken; }
        public String getAppToken() { return appToken; }
        public void setAppToken(String appToken) { this.appToken = appToken; }
        public String getSigningSecret() { return signingSecret; }
        public void setSigningSecret(String signingSecret) { this.signingSecret = signingSecret; }
        public int getRateLimitTier() { return rateLimitTier; }
        public void setRateLimitTier(int rateLimitTier) { this.rateLimitTier = rateLimitTier; }
    }

    /**
     * Box configuration.
     */
    public static class Box {
        private String clientId;
        private String clientSecret;
        private String enterpriseId;
        private String publicKeyId;
        private String privateKeyPath;
        private String passphrase;
        private int rateLimitPerSecond = 10;

        // Getters and setters
        public String getClientId() { return clientId; }
        public void setClientId(String clientId) { this.clientId = clientId; }
        public String getClientSecret() { return clientSecret; }
        public void setClientSecret(String clientSecret) { this.clientSecret = clientSecret; }
        public String getEnterpriseId() { return enterpriseId; }
        public void setEnterpriseId(String enterpriseId) { this.enterpriseId = enterpriseId; }
        public String getPublicKeyId() { return publicKeyId; }
        public void setPublicKeyId(String publicKeyId) { this.publicKeyId = publicKeyId; }
        public String getPrivateKeyPath() { return privateKeyPath; }
        public void setPrivateKeyPath(String privateKeyPath) { this.privateKeyPath = privateKeyPath; }
        public String getPassphrase() { return passphrase; }
        public void setPassphrase(String passphrase) { this.passphrase = passphrase; }
        public int getRateLimitPerSecond() { return rateLimitPerSecond; }
        public void setRateLimitPerSecond(int rateLimitPerSecond) { this.rateLimitPerSecond = rateLimitPerSecond; }
    }

    /**
     * SAP configuration.
     */
    public static class Sap {
        private String baseUrl;
        private String username;
        private String password;
        private String client = "100";
        private String language = "EN";
        private int maxConnections = 10;

        // Getters and setters
        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getClient() { return client; }
        public void setClient(String client) { this.client = client; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public int getMaxConnections() { return maxConnections; }
        public void setMaxConnections(int maxConnections) { this.maxConnections = maxConnections; }
    }

    /**
     * NetSuite configuration.
     */
    public static class NetSuite {
        private String accountId;
        private String consumerKey;
        private String consumerSecret;
        private String tokenId;
        private String tokenSecret;
        private String baseUrl;
        private int rateLimitPerMinute = 100;

        // Getters and setters
        public String getAccountId() { return accountId; }
        public void setAccountId(String accountId) { this.accountId = accountId; }
        public String getConsumerKey() { return consumerKey; }
        public void setConsumerKey(String consumerKey) { this.consumerKey = consumerKey; }
        public String getConsumerSecret() { return consumerSecret; }
        public void setConsumerSecret(String consumerSecret) { this.consumerSecret = consumerSecret; }
        public String getTokenId() { return tokenId; }
        public void setTokenId(String tokenId) { this.tokenId = tokenId; }
        public String getTokenSecret() { return tokenSecret; }
        public void setTokenSecret(String tokenSecret) { this.tokenSecret = tokenSecret; }
        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
        public int getRateLimitPerMinute() { return rateLimitPerMinute; }
        public void setRateLimitPerMinute(int rateLimitPerMinute) { this.rateLimitPerMinute = rateLimitPerMinute; }
    }

    /**
     * API Gateway configuration.
     */
    public static class Gateway {
        private int globalRateLimitPerSecond = 1000;
        private Duration circuitBreakerTimeout = Duration.ofSeconds(10);
        private int circuitBreakerFailureThreshold = 5;
        private Duration circuitBreakerRecoveryTime = Duration.ofMinutes(1);
        private Map<String, Integer> partnerRateLimits = new HashMap<>();

        // Getters and setters
        public int getGlobalRateLimitPerSecond() { return globalRateLimitPerSecond; }
        public void setGlobalRateLimitPerSecond(int globalRateLimitPerSecond) { this.globalRateLimitPerSecond = globalRateLimitPerSecond; }
        public Duration getCircuitBreakerTimeout() { return circuitBreakerTimeout; }
        public void setCircuitBreakerTimeout(Duration circuitBreakerTimeout) { this.circuitBreakerTimeout = circuitBreakerTimeout; }
        public int getCircuitBreakerFailureThreshold() { return circuitBreakerFailureThreshold; }
        public void setCircuitBreakerFailureThreshold(int circuitBreakerFailureThreshold) { this.circuitBreakerFailureThreshold = circuitBreakerFailureThreshold; }
        public Duration getCircuitBreakerRecoveryTime() { return circuitBreakerRecoveryTime; }
        public void setCircuitBreakerRecoveryTime(Duration circuitBreakerRecoveryTime) { this.circuitBreakerRecoveryTime = circuitBreakerRecoveryTime; }
        public Map<String, Integer> getPartnerRateLimits() { return partnerRateLimits; }
        public void setPartnerRateLimits(Map<String, Integer> partnerRateLimits) { this.partnerRateLimits = partnerRateLimits; }
    }

    /**
     * Monitoring configuration.
     */
    public static class Monitoring {
        private Duration healthCheckInterval = Duration.ofMinutes(1);
        private Duration metricsCollectionInterval = Duration.ofSeconds(30);
        private int errorAlertThreshold = 10;
        private Duration slowOperationThreshold = Duration.ofSeconds(5);

        // Getters and setters
        public Duration getHealthCheckInterval() { return healthCheckInterval; }
        public void setHealthCheckInterval(Duration healthCheckInterval) { this.healthCheckInterval = healthCheckInterval; }
        public Duration getMetricsCollectionInterval() { return metricsCollectionInterval; }
        public void setMetricsCollectionInterval(Duration metricsCollectionInterval) { this.metricsCollectionInterval = metricsCollectionInterval; }
        public int getErrorAlertThreshold() { return errorAlertThreshold; }
        public void setErrorAlertThreshold(int errorAlertThreshold) { this.errorAlertThreshold = errorAlertThreshold; }
        public Duration getSlowOperationThreshold() { return slowOperationThreshold; }
        public void setSlowOperationThreshold(Duration slowOperationThreshold) { this.slowOperationThreshold = slowOperationThreshold; }
    }

    /**
     * Kafka configuration.
     */
    public static class Kafka {
        private String integrationEventsTopic = "integration.events";
        private String errorsTopic = "integration.errors";
        private String metricsTopic = "integration.metrics";
        private String auditTopic = "integration.audit";
        private int partitions = 3;
        private short replicationFactor = 3;

        // Getters and setters
        public String getIntegrationEventsTopic() { return integrationEventsTopic; }
        public void setIntegrationEventsTopic(String integrationEventsTopic) { this.integrationEventsTopic = integrationEventsTopic; }
        public String getErrorsTopic() { return errorsTopic; }
        public void setErrorsTopic(String errorsTopic) { this.errorsTopic = errorsTopic; }
        public String getMetricsTopic() { return metricsTopic; }
        public void setMetricsTopic(String metricsTopic) { this.metricsTopic = metricsTopic; }
        public String getAuditTopic() { return auditTopic; }
        public void setAuditTopic(String auditTopic) { this.auditTopic = auditTopic; }
        public int getPartitions() { return partitions; }
        public void setPartitions(int partitions) { this.partitions = partitions; }
        public short getReplicationFactor() { return replicationFactor; }
        public void setReplicationFactor(short replicationFactor) { this.replicationFactor = replicationFactor; }
    }

    /**
     * Retry configuration.
     */
    public static class Retry {
        private int maxAttempts = 3;
        private Duration initialDelay = Duration.ofSeconds(1);
        private double multiplier = 2.0;
        private Duration maxDelay = Duration.ofMinutes(1);

        // Getters and setters
        public int getMaxAttempts() { return maxAttempts; }
        public void setMaxAttempts(int maxAttempts) { this.maxAttempts = maxAttempts; }
        public Duration getInitialDelay() { return initialDelay; }
        public void setInitialDelay(Duration initialDelay) { this.initialDelay = initialDelay; }
        public double getMultiplier() { return multiplier; }
        public void setMultiplier(double multiplier) { this.multiplier = multiplier; }
        public Duration getMaxDelay() { return maxDelay; }
        public void setMaxDelay(Duration maxDelay) { this.maxDelay = maxDelay; }
    }

    // Main getters and setters
    public Partners getPartners() { return partners; }
    public void setPartners(Partners partners) { this.partners = partners; }
    public Gateway getGateway() { return gateway; }
    public void setGateway(Gateway gateway) { this.gateway = gateway; }
    public Monitoring getMonitoring() { return monitoring; }
    public void setMonitoring(Monitoring monitoring) { this.monitoring = monitoring; }
    public Kafka getKafka() { return kafka; }
    public void setKafka(Kafka kafka) { this.kafka = kafka; }
    public Retry getRetry() { return retry; }
    public void setRetry(Retry retry) { this.retry = retry; }
}