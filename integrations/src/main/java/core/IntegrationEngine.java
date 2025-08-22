package com.enterprise.integrations.core;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.gateway.config.GatewayProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.transaction.annotation.EnableTransactionManagement;

/**
 * Core Integration Engine - Main application class for the partner integration platform.
 * 
 * This engine orchestrates partner integrations using:
 * - Spring Boot for application framework
 * - Apache Camel for data transformation and routing
 * - Apache Kafka for event streaming
 * - Spring Cloud Gateway for API gateway functionality
 * 
 * Key Features:
 * - Partner connector management
 * - Data transformation pipelines
 * - Event-driven architecture
 * - Circuit breaker patterns
 * - Comprehensive monitoring
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 * @since 2025-08-22
 */
@SpringBootApplication
@EnableKafka
@EnableAsync
@EnableScheduling
@EnableTransactionManagement
@ComponentScan(basePackages = {
    "com.enterprise.integrations.core",
    "com.enterprise.integrations.connectors",
    "com.enterprise.integrations.transformers",
    "com.enterprise.integrations.orchestration",
    "com.enterprise.integrations.monitoring",
    "com.enterprise.integrations.gateway",
    "com.enterprise.integrations.streaming"
})
public class IntegrationEngine {

    /**
     * Main entry point for the Integration Engine application.
     * 
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        SpringApplication.run(IntegrationEngine.class, args);
    }

    /**
     * Gateway properties configuration for Spring Cloud Gateway.
     * 
     * @return configured gateway properties
     */
    @Bean
    public GatewayProperties gatewayProperties() {
        return new GatewayProperties();
    }
}