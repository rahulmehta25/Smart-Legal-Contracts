# Multi-stage Docker build for Partner Integration Platform

# Build stage
FROM openjdk:21-jdk-slim AS builder

# Set working directory
WORKDIR /app

# Copy Maven wrapper and pom.xml
COPY mvnw .
COPY .mvn .mvn
COPY pom.xml .

# Download dependencies
RUN ./mvnw dependency:go-offline -B

# Copy source code
COPY src src

# Build application
RUN ./mvnw clean package -DskipTests

# Runtime stage
FROM openjdk:21-jre-slim AS runtime

# Create non-root user
RUN groupadd -r integration && useradd -r -g integration integration

# Set working directory
WORKDIR /app

# Install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Copy application jar
COPY --from=builder /app/target/partner-integration-platform-*.jar app.jar

# Create logs directory
RUN mkdir -p /var/log/integration-platform && \
    chown -R integration:integration /var/log/integration-platform /app

# Switch to non-root user
USER integration

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/integration-platform/actuator/health || exit 1

# Expose port
EXPOSE 8080

# Set JVM options
ENV JAVA_OPTS="-Xms512m -Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+UseStringDeduplication"

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start application
CMD ["sh", "-c", "java $JAVA_OPTS -Djava.security.egd=file:/dev/./urandom -jar app.jar"]