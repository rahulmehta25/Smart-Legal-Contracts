package com.enterprise.integrations.gateway;

import com.enterprise.integrations.core.IntegrationProperties;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.timelimiter.TimeLimiterConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.circuitbreaker.resilience4j.ReactiveResilience4JCircuitBreakerFactory;
import org.springframework.cloud.circuitbreaker.resilience4j.Resilience4JConfigBuilder;
import org.springframework.cloud.client.circuitbreaker.Customizer;
import org.springframework.cloud.gateway.filter.GatewayFilter;
import org.springframework.cloud.gateway.filter.factory.AbstractGatewayFilterFactory;
import org.springframework.cloud.gateway.filter.ratelimit.KeyResolver;
import org.springframework.cloud.gateway.filter.ratelimit.RedisRateLimiter;
import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Arrays;
import java.util.List;

/**
 * Spring Cloud Gateway configuration for the integration platform.
 * 
 * Provides API gateway functionality including:
 * - Route configuration and load balancing
 * - Rate limiting per partner
 * - Circuit breaker patterns
 * - Request/response transformation
 * - Authentication and authorization
 * - Monitoring and logging
 * - API versioning support
 * 
 * @author Integration Platform Team
 * @version 1.0.0
 */
@Configuration
public class GatewayConfig {
    
    private final IntegrationProperties properties;
    
    /**
     * Constructor with dependency injection.
     * 
     * @param properties integration properties
     */
    @Autowired
    public GatewayConfig(IntegrationProperties properties) {
        this.properties = properties;
    }
    
    /**
     * Configures API gateway routes.
     * 
     * @param builder route locator builder
     * @return configured route locator
     */
    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
            // Health check route
            .route("health-check", r -> r
                .path("/health/**")
                .uri("lb://integration-platform-health"))
            
            // Salesforce integration routes
            .route("salesforce-api", r -> r
                .path("/api/v1/salesforce/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(partnerKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("salesforce-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Partner-Id", "salesforce")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://salesforce-service"))
            
            // Microsoft 365 integration routes
            .route("microsoft-api", r -> r
                .path("/api/v1/microsoft/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(partnerKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("microsoft-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Partner-Id", "microsoft365")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://microsoft-service"))
            
            // Google Workspace integration routes
            .route("google-api", r -> r
                .path("/api/v1/google/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(partnerKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("google-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Partner-Id", "google")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://google-service"))
            
            // DocuSign integration routes
            .route("docusign-api", r -> r
                .path("/api/v1/docusign/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(partnerKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("docusign-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Partner-Id", "docusign")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://docusign-service"))
            
            // Slack integration routes
            .route("slack-api", r -> r
                .path("/api/v1/slack/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(partnerKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("slack-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Partner-Id", "slack")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://slack-service"))
            
            // Box integration routes
            .route("box-api", r -> r
                .path("/api/v1/box/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(partnerKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("box-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Partner-Id", "box")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://box-service"))
            
            // SAP integration routes
            .route("sap-api", r -> r
                .path("/api/v1/sap/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(partnerKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("sap-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Partner-Id", "sap")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://sap-service"))
            
            // NetSuite integration routes
            .route("netsuite-api", r -> r
                .path("/api/v1/netsuite/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(partnerKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("netsuite-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Partner-Id", "netsuite")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://netsuite-service"))
            
            // Workflow orchestration routes
            .route("workflow-api", r -> r
                .path("/api/v1/workflows/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .requestRateLimiter(config -> {
                        config.setRateLimiter(redisRateLimiter());
                        config.setKeyResolver(userKeyResolver());
                    })
                    .circuitBreaker(config -> config.setName("workflow-cb"))
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Service-Type", "workflow")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://workflow-service"))
            
            // Monitoring and metrics routes
            .route("metrics-api", r -> r
                .path("/api/v1/metrics/**")
                .filters(f -> f
                    .stripPrefix(3)
                    .filter(authenticationFilter())
                    .filter(loggingFilter())
                    .addRequestHeader("X-Service-Type", "metrics")
                    .addResponseHeader("X-Gateway-Version", "1.0.0"))
                .uri("lb://metrics-service"))
            
            // WebSocket routes for real-time events
            .route("websocket-events", r -> r
                .path("/ws/**")
                .filters(f -> f
                    .filter(authenticationFilter())
                    .addRequestHeader("X-Service-Type", "websocket"))
                .uri("lb://event-service"))
            
            .build();
    }
    
    /**
     * Configures Redis rate limiter.
     * 
     * @return configured Redis rate limiter
     */
    @Bean
    public RedisRateLimiter redisRateLimiter() {
        return new RedisRateLimiter(
            properties.getGateway().getGlobalRateLimitPerSecond(), // replenish rate
            properties.getGateway().getGlobalRateLimitPerSecond() * 2, // burst capacity
            1 // requested tokens
        );
    }
    
    /**
     * Key resolver for partner-based rate limiting.
     * 
     * @return partner-based key resolver
     */
    @Bean
    public KeyResolver partnerKeyResolver() {
        return exchange -> {
            String partnerId = exchange.getRequest().getHeaders().getFirst("X-Partner-Id");
            String clientId = exchange.getRequest().getHeaders().getFirst("X-Client-Id");
            
            if (partnerId != null && clientId != null) {
                return Mono.just(partnerId + ":" + clientId);
            } else if (partnerId != null) {
                return Mono.just(partnerId);
            } else {
                return Mono.just("anonymous");
            }
        };
    }
    
    /**
     * Key resolver for user-based rate limiting.
     * 
     * @return user-based key resolver
     */
    @Bean
    public KeyResolver userKeyResolver() {
        return exchange -> {
            String userId = exchange.getRequest().getHeaders().getFirst("X-User-Id");
            return Mono.just(userId != null ? userId : "anonymous");
        };
    }
    
    /**
     * Circuit breaker configuration customizer.
     * 
     * @return circuit breaker customizer
     */
    @Bean
    public Customizer<ReactiveResilience4JCircuitBreakerFactory> defaultCustomizer() {
        return factory -> factory.configureDefault(id -> new Resilience4JConfigBuilder(id)
            .circuitBreakerConfig(CircuitBreakerConfig.custom()
                .slidingWindowSize(properties.getGateway().getCircuitBreakerFailureThreshold())
                .failureRateThreshold(50.0f)
                .waitDurationInOpenState(properties.getGateway().getCircuitBreakerRecoveryTime())
                .slowCallRateThreshold(50.0f)
                .slowCallDurationThreshold(properties.getGateway().getCircuitBreakerTimeout())
                .permittedNumberOfCallsInHalfOpenState(3)
                .minimumNumberOfCalls(5)
                .build())
            .timeLimiterConfig(TimeLimiterConfig.custom()
                .timeoutDuration(properties.getGateway().getCircuitBreakerTimeout())
                .build())
            .build());
    }
    
    /**
     * Authentication filter for API security.
     * 
     * @return authentication gateway filter
     */
    @Bean
    public GatewayFilter authenticationFilter() {
        return new AuthenticationGatewayFilterFactory().apply(new AuthenticationGatewayFilterFactory.Config());
    }
    
    /**
     * Logging filter for request/response monitoring.
     * 
     * @return logging gateway filter
     */
    @Bean
    public GatewayFilter loggingFilter() {
        return new LoggingGatewayFilterFactory().apply(new LoggingGatewayFilterFactory.Config());
    }
    
    /**
     * Custom authentication gateway filter factory.
     */
    public static class AuthenticationGatewayFilterFactory 
            extends AbstractGatewayFilterFactory<AuthenticationGatewayFilterFactory.Config> {
        
        public AuthenticationGatewayFilterFactory() {
            super(Config.class);
        }
        
        @Override
        public GatewayFilter apply(Config config) {
            return (exchange, chain) -> {
                ServerWebExchange.Builder builder = exchange.mutate();
                
                // Extract and validate authentication token
                String authHeader = exchange.getRequest().getHeaders().getFirst("Authorization");
                if (authHeader == null || !authHeader.startsWith("Bearer ")) {
                    exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
                    return exchange.getResponse().setComplete();
                }
                
                String token = authHeader.substring(7);
                
                // Validate token (simplified validation)
                if (!isValidToken(token)) {
                    exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
                    return exchange.getResponse().setComplete();
                }
                
                // Add user information to request headers
                String userId = extractUserIdFromToken(token);
                builder.request(r -> r.header("X-User-Id", userId));
                
                return chain.filter(builder.build());
            };
        }
        
        private boolean isValidToken(String token) {
            // In production, this would validate against OAuth2 server or JWT
            return token != null && token.length() > 10;
        }
        
        private String extractUserIdFromToken(String token) {
            // In production, this would extract from JWT claims
            return "user-" + token.hashCode();
        }
        
        public static class Config {
            // Configuration properties if needed
        }
    }
    
    /**
     * Custom logging gateway filter factory.
     */
    public static class LoggingGatewayFilterFactory 
            extends AbstractGatewayFilterFactory<LoggingGatewayFilterFactory.Config> {
        
        private static final org.slf4j.Logger logger = 
            org.slf4j.LoggerFactory.getLogger(LoggingGatewayFilterFactory.class);
        
        public LoggingGatewayFilterFactory() {
            super(Config.class);
        }
        
        @Override
        public GatewayFilter apply(Config config) {
            return (exchange, chain) -> {
                long startTime = System.currentTimeMillis();
                String requestId = java.util.UUID.randomUUID().toString();
                
                ServerWebExchange.Builder builder = exchange.mutate();
                builder.request(r -> r.header("X-Request-Id", requestId));
                
                logger.info("Gateway Request: {} {} {} - Request ID: {}", 
                           exchange.getRequest().getMethod(),
                           exchange.getRequest().getPath(),
                           exchange.getRequest().getRemoteAddress(),
                           requestId);
                
                return chain.filter(builder.build()).then(Mono.fromRunnable(() -> {
                    long duration = System.currentTimeMillis() - startTime;
                    logger.info("Gateway Response: {} - Duration: {}ms - Request ID: {}", 
                               exchange.getResponse().getStatusCode(),
                               duration,
                               requestId);
                }));
            };
        }
        
        public static class Config {
            // Configuration properties if needed
        }
    }
    
    /**
     * Global exception handler for gateway errors.
     * 
     * @return global error web exception handler
     */
    @Bean
    public org.springframework.boot.web.reactive.error.ErrorWebExceptionHandler errorWebExceptionHandler() {
        return new CustomErrorWebExceptionHandler();
    }
    
    /**
     * Custom error handler for gateway exceptions.
     */
    public static class CustomErrorWebExceptionHandler 
            implements org.springframework.boot.web.reactive.error.ErrorWebExceptionHandler {
        
        private static final org.slf4j.Logger logger = 
            org.slf4j.LoggerFactory.getLogger(CustomErrorWebExceptionHandler.class);
        
        @Override
        public Mono<Void> handle(ServerWebExchange exchange, Throwable ex) {
            logger.error("Gateway error: {}", ex.getMessage(), ex);
            
            exchange.getResponse().setStatusCode(HttpStatus.INTERNAL_SERVER_ERROR);
            exchange.getResponse().getHeaders().add("Content-Type", "application/json");
            
            String errorResponse = """
                {
                    "error": "Gateway Error",
                    "message": "%s",
                    "timestamp": "%s",
                    "path": "%s"
                }
                """.formatted(
                ex.getMessage(),
                java.time.LocalDateTime.now(),
                exchange.getRequest().getPath()
            );
            
            org.springframework.core.io.buffer.DataBuffer buffer = 
                exchange.getResponse().bufferFactory().wrap(errorResponse.getBytes());
            
            return exchange.getResponse().writeWith(Mono.just(buffer));
        }
    }
    
    /**
     * CORS configuration for cross-origin requests.
     * 
     * @return CORS configuration source
     */
    @Bean
    public org.springframework.web.cors.reactive.CorsConfigurationSource corsConfigurationSource() {
        org.springframework.web.cors.CorsConfiguration configuration = 
            new org.springframework.web.cors.CorsConfiguration();
        
        configuration.setAllowedOriginPatterns(Arrays.asList("*"));
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        configuration.setAllowCredentials(true);
        configuration.setMaxAge(3600L);
        
        org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource source = 
            new org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        
        return source;
    }
    
    /**
     * WebClient configuration for downstream service calls.
     * 
     * @return configured WebClient
     */
    @Bean
    public org.springframework.web.reactive.function.client.WebClient webClient() {
        return org.springframework.web.reactive.function.client.WebClient.builder()
            .defaultHeader("User-Agent", "Integration-Platform-Gateway/1.0.0")
            .defaultHeader("Accept", "application/json")
            .codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(10 * 1024 * 1024))
            .build();
    }
}