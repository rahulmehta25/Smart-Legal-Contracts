package com.enterprise.integrations.core;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.apache.camel.CamelContext;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.spring.boot.CamelContextConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.kafka.core.DefaultKafkaConsumerFactory;
import org.springframework.kafka.core.DefaultKafkaProducerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.core.ProducerFactory;
import org.springframework.retry.annotation.EnableRetry;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executor;

/**
 * Core configuration for the integration platform.
 * 
 * This configuration class sets up:
 * - JSON serialization/deserialization
 * - Apache Camel context configuration
 * - Kafka producer and consumer configurations
 * - Thread pool executors for async operations
 * - Retry mechanisms for resilience
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Configuration
@EnableRetry
public class IntegrationConfig {

    /**
     * Primary ObjectMapper bean for JSON processing.
     * Configured to handle Java 8 time types and proper formatting.
     * 
     * @return configured ObjectMapper instance
     */
    @Bean
    @Primary
    public ObjectMapper objectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        mapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        return mapper;
    }

    /**
     * Camel context configuration for route management and error handling.
     * 
     * @return CamelContextConfiguration with custom settings
     */
    @Bean
    public CamelContextConfiguration camelContextConfiguration() {
        return new CamelContextConfiguration() {
            @Override
            public void beforeApplicationStart(CamelContext camelContext) {
                // Enable JMX management
                camelContext.setUseMDCLogging(true);
                camelContext.setUseDataType(true);
                
                // Configure global error handling
                camelContext.setErrorHandlerBuilder(new RouteBuilder() {
                    @Override
                    public void configure() throws Exception {
                        errorHandler(
                            deadLetterChannel("kafka:integration.errors")
                                .maximumRedeliveries(3)
                                .redeliveryDelay(5000)
                                .exponentialBackOff()
                        );
                    }
                }.getErrorHandlerBuilder());
            }

            @Override
            public void afterApplicationStart(CamelContext camelContext) {
                // Post-startup configuration if needed
            }
        };
    }

    /**
     * Kafka producer configuration for outbound messaging.
     * 
     * @return ProducerFactory with optimized settings
     */
    @Bean
    public ProducerFactory<String, Object> producerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put("bootstrap.servers", "${spring.kafka.bootstrap-servers:localhost:9092}");
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.springframework.kafka.support.serializer.JsonSerializer");
        props.put("acks", "all");
        props.put("retries", 3);
        props.put("batch.size", 16384);
        props.put("linger.ms", 10);
        props.put("buffer.memory", 33554432);
        props.put("compression.type", "snappy");
        props.put("enable.idempotence", true);
        
        return new DefaultKafkaProducerFactory<>(props);
    }

    /**
     * Kafka template for message publishing.
     * 
     * @return configured KafkaTemplate
     */
    @Bean
    public KafkaTemplate<String, Object> kafkaTemplate() {
        return new KafkaTemplate<>(producerFactory());
    }

    /**
     * Kafka consumer factory configuration.
     * 
     * @return DefaultKafkaConsumerFactory with optimized settings
     */
    @Bean
    public DefaultKafkaConsumerFactory<String, Object> consumerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put("bootstrap.servers", "${spring.kafka.bootstrap-servers:localhost:9092}");
        props.put("group.id", "${spring.kafka.consumer.group-id:integration-platform}");
        props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("value.deserializer", "org.springframework.kafka.support.serializer.JsonDeserializer");
        props.put("auto.offset.reset", "earliest");
        props.put("enable.auto.commit", false);
        props.put("max.poll.records", 100);
        props.put("session.timeout.ms", 30000);
        props.put("heartbeat.interval.ms", 10000);
        
        return new DefaultKafkaConsumerFactory<>(props);
    }

    /**
     * Kafka listener container factory configuration.
     * 
     * @return configured container factory
     */
    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, Object> kafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, Object> factory = 
            new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory());
        factory.setConcurrency(3);
        factory.getContainerProperties().setAckOnError(false);
        factory.setCommonErrorHandler(null); // Will be configured per listener
        return factory;
    }

    /**
     * Thread pool executor for async integration operations.
     * 
     * @return configured ThreadPoolTaskExecutor
     */
    @Bean(name = "integrationTaskExecutor")
    public Executor integrationTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(10);
        executor.setMaxPoolSize(50);
        executor.setQueueCapacity(100);
        executor.setThreadNamePrefix("Integration-");
        executor.setWaitForTasksToCompleteOnShutdown(true);
        executor.setAwaitTerminationSeconds(30);
        executor.initialize();
        return executor;
    }

    /**
     * Thread pool executor for connector operations.
     * 
     * @return configured ThreadPoolTaskExecutor for connectors
     */
    @Bean(name = "connectorTaskExecutor")
    public Executor connectorTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);
        executor.setMaxPoolSize(20);
        executor.setQueueCapacity(50);
        executor.setThreadNamePrefix("Connector-");
        executor.setWaitForTasksToCompleteOnShutdown(true);
        executor.setAwaitTerminationSeconds(30);
        executor.initialize();
        return executor;
    }
}