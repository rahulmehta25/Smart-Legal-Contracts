# Data Lakehouse Platform Docker Image
FROM openjdk:11-jre-slim

# Set environment variables
ENV SPARK_VERSION=3.4.0
ENV HADOOP_VERSION=3
ENV DELTA_VERSION=2.4.0
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-*-src.zip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    procps \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Spark
RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} $SPARK_HOME \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Install Delta Lake
RUN wget https://repo1.maven.org/maven2/io/delta/delta-core_2.12/${DELTA_VERSION}/delta-core_2.12-${DELTA_VERSION}.jar -P $SPARK_HOME/jars/ \
    && wget https://repo1.maven.org/maven2/io/delta/delta-storage/${DELTA_VERSION}/delta-storage-${DELTA_VERSION}.jar -P $SPARK_HOME/jars/

# Create app directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/lakehouse/{data,checkpoints,metadata,feature_store,models} \
    && mkdir -p /var/log/lakehouse

# Set up non-root user
RUN groupadd -r lakehouse && useradd -r -g lakehouse lakehouse \
    && chown -R lakehouse:lakehouse /app /tmp/lakehouse /var/log/lakehouse

USER lakehouse

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -m data_lakehouse.main --status || exit 1

# Expose ports
EXPOSE 4040 8080 8081 9090

# Default command
CMD ["python3", "-m", "data_lakehouse.main", "--environment", "production"]