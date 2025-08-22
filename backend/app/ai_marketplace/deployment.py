"""
Model Deployment Pipeline

Handles model deployment, serving, scaling, and lifecycle management.
"""

import os
import json
import yaml
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import docker
import kubernetes
from kubernetes import client, config
import requests
import subprocess
import tempfile
from pathlib import Path
import logging
import torch
import tensorflow as tf
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import onnxruntime as ort
import tritonclient.http as httpclient
import mlflow
from ray import serve
import bentoml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import numpy as np

logger = logging.getLogger(__name__)


class DeploymentTarget(Enum):
    """Deployment target environments"""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    SERVERLESS = "serverless"
    EDGE = "edge"
    CLOUD_RUN = "cloud_run"
    LAMBDA = "lambda"
    AZURE_FUNCTIONS = "azure_functions"
    SAGEMAKER = "sagemaker"
    VERTEX_AI = "vertex_ai"


class ServingFramework(Enum):
    """Model serving frameworks"""
    FASTAPI = "fastapi"
    TRITON = "triton"
    TORCHSERVE = "torchserve"
    TENSORFLOW_SERVING = "tensorflow_serving"
    MLFLOW = "mlflow"
    BENTOML = "bentoml"
    RAY_SERVE = "ray_serve"
    SELDON = "seldon"
    KSERVE = "kserve"


class AutoScalingPolicy(Enum):
    """Auto-scaling policies"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    LATENCY_BASED = "latency_based"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    model_id: str
    model_name: str
    version: str
    target: DeploymentTarget
    framework: ServingFramework
    replicas: int = 1
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    gpu_enabled: bool = False
    gpu_count: int = 0
    autoscaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    scaling_policy: AutoScalingPolicy = AutoScalingPolicy.CPU_BASED
    scaling_threshold: float = 0.7
    health_check_path: str = "/health"
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    canary_enabled: bool = False
    canary_weight: int = 10
    timeout_seconds: int = 60
    environment_vars: Dict[str, str] = None
    secrets: Dict[str, str] = None


@dataclass
class DeploymentStatus:
    """Deployment status information"""
    deployment_id: str
    status: str
    endpoint: str
    replicas_ready: int
    replicas_total: int
    version: str
    created_at: datetime
    updated_at: datetime
    metrics: Dict[str, Any]
    health: str
    errors: List[str]


# Metrics for monitoring
deployment_counter = Counter('model_deployments_total', 'Total model deployments')
deployment_duration = Histogram('model_deployment_duration_seconds', 'Model deployment duration')
active_deployments = Gauge('model_deployments_active', 'Active model deployments')
inference_counter = Counter('model_inference_total', 'Total model inferences')
inference_duration = Histogram('model_inference_duration_seconds', 'Model inference duration')


class ModelDeployment:
    """
    Comprehensive model deployment system
    """
    
    def __init__(self,
                 registry_url: str = "http://localhost:8000/registry",
                 docker_registry: str = "localhost:5000",
                 k8s_namespace: str = "ai-marketplace"):
        
        self.registry_url = registry_url
        self.docker_registry = docker_registry
        self.k8s_namespace = k8s_namespace
        
        # Initialize Docker client
        self.docker_client = docker.from_env()
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_apps = client.AppsV1Api()
        self.k8s_core = client.CoreV1Api()
        self.k8s_autoscaling = client.AutoscalingV2Api()
        
        # Initialize cache
        self.cache = redis.Redis(host='localhost', port=6379, db=1)
        
        # Active deployments tracker
        self.active_deployments = {}
    
    async def deploy_model(self, config: DeploymentConfig) -> DeploymentStatus:
        """
        Deploy model to target environment
        
        Args:
            config: Deployment configuration
        
        Returns:
            Deployment status
        """
        deployment_counter.inc()
        
        with deployment_duration.time():
            try:
                # Download model from registry
                model_path = await self._download_model(config.model_id, config.version)
                
                # Build deployment package
                package_path = await self._build_deployment_package(model_path, config)
                
                # Deploy based on target
                if config.target == DeploymentTarget.LOCAL:
                    status = await self._deploy_local(package_path, config)
                elif config.target == DeploymentTarget.DOCKER:
                    status = await self._deploy_docker(package_path, config)
                elif config.target == DeploymentTarget.KUBERNETES:
                    status = await self._deploy_kubernetes(package_path, config)
                elif config.target == DeploymentTarget.SERVERLESS:
                    status = await self._deploy_serverless(package_path, config)
                elif config.target == DeploymentTarget.EDGE:
                    status = await self._deploy_edge(package_path, config)
                else:
                    raise ValueError(f"Unsupported deployment target: {config.target}")
                
                # Register deployment
                self._register_deployment(status)
                
                # Setup monitoring
                if config.monitoring_enabled:
                    await self._setup_monitoring(status)
                
                # Setup autoscaling
                if config.autoscaling:
                    await self._setup_autoscaling(config, status)
                
                active_deployments.inc()
                logger.info(f"Model {config.model_name} deployed successfully")
                return status
                
            except Exception as e:
                logger.error(f"Deployment failed: {e}")
                raise
    
    async def _deploy_local(self, package_path: str, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy model locally using FastAPI"""
        
        # Create FastAPI application
        app = FastAPI(title=f"{config.model_name} API")
        
        # Load model
        model = self._load_model(package_path, config.framework)
        
        class PredictionRequest(BaseModel):
            data: Dict[str, Any]
        
        class PredictionResponse(BaseModel):
            predictions: List[Any]
            model_version: str
            inference_time_ms: float
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            start_time = datetime.now()
            
            try:
                # Preprocess input
                processed_input = self._preprocess_input(request.data, model)
                
                # Run inference
                with inference_duration.time():
                    predictions = model.predict(processed_input)
                
                inference_counter.inc()
                
                # Postprocess output
                output = self._postprocess_output(predictions)
                
                inference_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return PredictionResponse(
                    predictions=output,
                    model_version=config.version,
                    inference_time_ms=inference_time
                )
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": config.model_name, "version": config.version}
        
        @app.get("/metrics")
        async def metrics():
            return prometheus_client.generate_latest()
        
        # Start server in background
        import threading
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": 8080}
        )
        server_thread.daemon = True
        server_thread.start()
        
        return DeploymentStatus(
            deployment_id=f"local-{config.model_name}-{config.version}",
            status="running",
            endpoint="http://localhost:8080",
            replicas_ready=1,
            replicas_total=1,
            version=config.version,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metrics={},
            health="healthy",
            errors=[]
        )
    
    async def _deploy_docker(self, package_path: str, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy model as Docker container"""
        
        # Build Docker image
        image_name = f"{self.docker_registry}/{config.model_name}:{config.version}"
        
        # Create Dockerfile
        dockerfile_content = f"""
        FROM python:3.9-slim
        
        WORKDIR /app
        
        # Install dependencies
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        
        # Copy model and code
        COPY model/ ./model/
        COPY server.py .
        
        # Expose port
        EXPOSE 8080
        
        # Set environment variables
        ENV MODEL_NAME={config.model_name}
        ENV MODEL_VERSION={config.version}
        
        # Run server
        CMD ["python", "server.py"]
        """
        
        dockerfile_path = Path(package_path) / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Build image
        image, logs = self.docker_client.images.build(
            path=package_path,
            tag=image_name,
            rm=True
        )
        
        # Push to registry
        self.docker_client.images.push(image_name)
        
        # Run container
        container = self.docker_client.containers.run(
            image_name,
            name=f"{config.model_name}-{config.version}",
            ports={'8080/tcp': 8080},
            detach=True,
            environment=config.environment_vars or {},
            mem_limit=config.memory_limit,
            cpu_quota=int(float(config.cpu_limit) * 100000),
            restart_policy={"Name": "unless-stopped"}
        )
        
        return DeploymentStatus(
            deployment_id=container.id,
            status="running",
            endpoint=f"http://localhost:8080",
            replicas_ready=1,
            replicas_total=config.replicas,
            version=config.version,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metrics={},
            health="healthy",
            errors=[]
        )
    
    async def _deploy_kubernetes(self, package_path: str, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy model to Kubernetes cluster"""
        
        # Build and push Docker image
        image_name = f"{self.docker_registry}/{config.model_name}:{config.version}"
        await self._build_and_push_image(package_path, image_name)
        
        # Create deployment
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=f"{config.model_name}-{config.version}",
                namespace=self.k8s_namespace,
                labels={
                    "app": config.model_name,
                    "version": config.version
                }
            ),
            spec=client.V1DeploymentSpec(
                replicas=config.replicas,
                selector=client.V1LabelSelector(
                    match_labels={
                        "app": config.model_name,
                        "version": config.version
                    }
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": config.model_name,
                            "version": config.version
                        }
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="model-server",
                                image=image_name,
                                ports=[client.V1ContainerPort(container_port=8080)],
                                resources=client.V1ResourceRequirements(
                                    limits={
                                        "cpu": config.cpu_limit,
                                        "memory": config.memory_limit
                                    },
                                    requests={
                                        "cpu": str(float(config.cpu_limit) / 2),
                                        "memory": "2Gi"
                                    }
                                ),
                                env=[
                                    client.V1EnvVar(name=k, value=v)
                                    for k, v in (config.environment_vars or {}).items()
                                ],
                                liveness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path=config.health_check_path,
                                        port=8080
                                    ),
                                    initial_delay_seconds=30,
                                    period_seconds=10
                                ),
                                readiness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path=config.health_check_path,
                                        port=8080
                                    ),
                                    initial_delay_seconds=10,
                                    period_seconds=5
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        # Add GPU resources if enabled
        if config.gpu_enabled:
            deployment.spec.template.spec.containers[0].resources.limits["nvidia.com/gpu"] = str(config.gpu_count)
        
        # Create deployment
        self.k8s_apps.create_namespaced_deployment(
            namespace=self.k8s_namespace,
            body=deployment
        )
        
        # Create service
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=f"{config.model_name}-service",
                namespace=self.k8s_namespace
            ),
            spec=client.V1ServiceSpec(
                selector={
                    "app": config.model_name,
                    "version": config.version
                },
                ports=[
                    client.V1ServicePort(
                        port=80,
                        target_port=8080
                    )
                ],
                type="LoadBalancer" if config.target == DeploymentTarget.KUBERNETES else "ClusterIP"
            )
        )
        
        self.k8s_core.create_namespaced_service(
            namespace=self.k8s_namespace,
            body=service
        )
        
        # Wait for deployment to be ready
        await self._wait_for_deployment(f"{config.model_name}-{config.version}")
        
        # Get service endpoint
        svc = self.k8s_core.read_namespaced_service(
            name=f"{config.model_name}-service",
            namespace=self.k8s_namespace
        )
        
        endpoint = f"http://{svc.status.load_balancer.ingress[0].ip}" if svc.status.load_balancer.ingress else f"http://{svc.spec.cluster_ip}"
        
        return DeploymentStatus(
            deployment_id=f"k8s-{config.model_name}-{config.version}",
            status="running",
            endpoint=endpoint,
            replicas_ready=config.replicas,
            replicas_total=config.replicas,
            version=config.version,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metrics={},
            health="healthy",
            errors=[]
        )
    
    async def _deploy_serverless(self, package_path: str, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy model to serverless platform"""
        
        if config.target == DeploymentTarget.LAMBDA:
            return await self._deploy_lambda(package_path, config)
        elif config.target == DeploymentTarget.CLOUD_RUN:
            return await self._deploy_cloud_run(package_path, config)
        elif config.target == DeploymentTarget.AZURE_FUNCTIONS:
            return await self._deploy_azure_functions(package_path, config)
        else:
            raise ValueError(f"Unsupported serverless platform")
    
    async def _deploy_edge(self, package_path: str, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy model to edge devices"""
        
        # Convert model to edge-optimized format
        optimized_model = await self._optimize_for_edge(package_path)
        
        # Package for edge deployment
        edge_package = await self._create_edge_package(optimized_model, config)
        
        # Deploy to edge devices
        deployment_id = f"edge-{config.model_name}-{config.version}"
        
        # Register edge deployment
        self.cache.setex(
            f"edge_deployment:{deployment_id}",
            86400,
            json.dumps({
                "model": config.model_name,
                "version": config.version,
                "package": edge_package,
                "devices": []
            })
        )
        
        return DeploymentStatus(
            deployment_id=deployment_id,
            status="ready",
            endpoint=f"edge://{deployment_id}",
            replicas_ready=0,
            replicas_total=0,
            version=config.version,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metrics={},
            health="healthy",
            errors=[]
        )
    
    async def scale_deployment(self, 
                              deployment_id: str,
                              replicas: int) -> bool:
        """
        Scale deployment to specified replicas
        
        Args:
            deployment_id: Deployment ID
            replicas: Target number of replicas
        
        Returns:
            Success status
        """
        try:
            if deployment_id.startswith("k8s-"):
                # Scale Kubernetes deployment
                name = deployment_id.replace("k8s-", "")
                
                # Update deployment
                body = {"spec": {"replicas": replicas}}
                self.k8s_apps.patch_namespaced_deployment_scale(
                    name=name,
                    namespace=self.k8s_namespace,
                    body=body
                )
                
                logger.info(f"Scaled deployment {deployment_id} to {replicas} replicas")
                return True
                
            elif deployment_id.startswith("docker-"):
                # Scale Docker Swarm service
                service_name = deployment_id.replace("docker-", "")
                subprocess.run([
                    "docker", "service", "scale",
                    f"{service_name}={replicas}"
                ], check=True)
                
                return True
            
            else:
                logger.warning(f"Scaling not supported for deployment {deployment_id}")
                return False
                
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return False
    
    async def update_deployment(self,
                               deployment_id: str,
                               new_version: str) -> DeploymentStatus:
        """
        Update deployment to new model version
        
        Args:
            deployment_id: Deployment ID
            new_version: New model version
        
        Returns:
            Updated deployment status
        """
        try:
            # Get current deployment
            deployment = self.active_deployments.get(deployment_id)
            if not deployment:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            # Download new model version
            model_path = await self._download_model(
                deployment['model_id'],
                new_version
            )
            
            if deployment_id.startswith("k8s-"):
                # Update Kubernetes deployment
                name = deployment_id.replace("k8s-", "")
                
                # Build new image
                image_name = f"{self.docker_registry}/{deployment['model_name']}:{new_version}"
                await self._build_and_push_image(model_path, image_name)
                
                # Update deployment image
                body = {
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "model-server",
                                    "image": image_name
                                }]
                            }
                        }
                    }
                }
                
                self.k8s_apps.patch_namespaced_deployment(
                    name=name,
                    namespace=self.k8s_namespace,
                    body=body
                )
                
                # Wait for rollout
                await self._wait_for_deployment(name)
                
                return DeploymentStatus(
                    deployment_id=deployment_id,
                    status="updated",
                    endpoint=deployment['endpoint'],
                    replicas_ready=deployment['replicas'],
                    replicas_total=deployment['replicas'],
                    version=new_version,
                    created_at=deployment['created_at'],
                    updated_at=datetime.now(),
                    metrics={},
                    health="healthy",
                    errors=[]
                )
            
            else:
                raise ValueError(f"Update not supported for deployment type")
                
        except Exception as e:
            logger.error(f"Deployment update failed: {e}")
            raise
    
    async def rollback_deployment(self,
                                 deployment_id: str,
                                 target_version: Optional[str] = None) -> bool:
        """
        Rollback deployment to previous version
        
        Args:
            deployment_id: Deployment ID
            target_version: Optional target version (previous if not specified)
        
        Returns:
            Success status
        """
        try:
            if deployment_id.startswith("k8s-"):
                name = deployment_id.replace("k8s-", "")
                
                if target_version:
                    # Rollback to specific version
                    return await self.update_deployment(deployment_id, target_version)
                else:
                    # Rollback to previous version
                    subprocess.run([
                        "kubectl", "rollout", "undo",
                        f"deployment/{name}",
                        "-n", self.k8s_namespace
                    ], check=True)
                
                logger.info(f"Rolled back deployment {deployment_id}")
                return True
            
            else:
                logger.warning(f"Rollback not supported for deployment {deployment_id}")
                return False
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def delete_deployment(self, deployment_id: str) -> bool:
        """
        Delete deployment
        
        Args:
            deployment_id: Deployment ID
        
        Returns:
            Success status
        """
        try:
            if deployment_id.startswith("k8s-"):
                name = deployment_id.replace("k8s-", "")
                
                # Delete deployment
                self.k8s_apps.delete_namespaced_deployment(
                    name=name,
                    namespace=self.k8s_namespace
                )
                
                # Delete service
                self.k8s_core.delete_namespaced_service(
                    name=f"{name}-service",
                    namespace=self.k8s_namespace
                )
                
            elif deployment_id.startswith("docker-"):
                container_id = deployment_id.replace("docker-", "")
                container = self.docker_client.containers.get(container_id)
                container.stop()
                container.remove()
            
            # Remove from active deployments
            self.active_deployments.pop(deployment_id, None)
            active_deployments.dec()
            
            logger.info(f"Deleted deployment {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            return False
    
    async def get_deployment_status(self, deployment_id: str) -> DeploymentStatus:
        """
        Get deployment status
        
        Args:
            deployment_id: Deployment ID
        
        Returns:
            Deployment status
        """
        try:
            if deployment_id.startswith("k8s-"):
                name = deployment_id.replace("k8s-", "")
                
                # Get deployment status
                deployment = self.k8s_apps.read_namespaced_deployment_status(
                    name=name,
                    namespace=self.k8s_namespace
                )
                
                # Get service
                service = self.k8s_core.read_namespaced_service(
                    name=f"{name}-service",
                    namespace=self.k8s_namespace
                )
                
                endpoint = f"http://{service.status.load_balancer.ingress[0].ip}" if service.status.load_balancer.ingress else f"http://{service.spec.cluster_ip}"
                
                return DeploymentStatus(
                    deployment_id=deployment_id,
                    status="running" if deployment.status.ready_replicas == deployment.spec.replicas else "updating",
                    endpoint=endpoint,
                    replicas_ready=deployment.status.ready_replicas or 0,
                    replicas_total=deployment.spec.replicas,
                    version=deployment.metadata.labels.get('version', 'unknown'),
                    created_at=deployment.metadata.creation_timestamp,
                    updated_at=datetime.now(),
                    metrics=await self._get_deployment_metrics(deployment_id),
                    health="healthy" if deployment.status.ready_replicas else "unhealthy",
                    errors=[]
                )
            
            elif deployment_id.startswith("docker-"):
                container_id = deployment_id.replace("docker-", "")
                container = self.docker_client.containers.get(container_id)
                
                return DeploymentStatus(
                    deployment_id=deployment_id,
                    status=container.status,
                    endpoint="http://localhost:8080",
                    replicas_ready=1 if container.status == "running" else 0,
                    replicas_total=1,
                    version="unknown",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    metrics={},
                    health="healthy" if container.status == "running" else "unhealthy",
                    errors=[]
                )
            
            else:
                raise ValueError(f"Unknown deployment type: {deployment_id}")
                
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            raise
    
    async def _download_model(self, model_id: str, version: str) -> str:
        """Download model from registry"""
        response = requests.get(
            f"{self.registry_url}/models/{model_id}/download",
            params={"version": version}
        )
        response.raise_for_status()
        
        # Save to temporary directory
        temp_dir = tempfile.mkdtemp()
        model_path = Path(temp_dir) / "model"
        
        with open(model_path, 'wb') as f:
            f.write(response.content)
        
        return str(model_path)
    
    async def _build_deployment_package(self, model_path: str, config: DeploymentConfig) -> str:
        """Build deployment package with model and serving code"""
        
        package_dir = tempfile.mkdtemp()
        package_path = Path(package_dir)
        
        # Copy model
        model_dir = package_path / "model"
        model_dir.mkdir()
        shutil.copy2(model_path, model_dir / "model.pkl")
        
        # Create server code based on framework
        if config.framework == ServingFramework.FASTAPI:
            server_code = self._generate_fastapi_server(config)
        elif config.framework == ServingFramework.TRITON:
            server_code = self._generate_triton_config(config)
        else:
            server_code = self._generate_generic_server(config)
        
        with open(package_path / "server.py", 'w') as f:
            f.write(server_code)
        
        # Create requirements file
        requirements = [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "numpy>=1.21.0",
            "scikit-learn>=0.24.0",
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "transformers>=4.10.0",
            "redis>=3.5.0",
            "prometheus-client>=0.11.0"
        ]
        
        with open(package_path / "requirements.txt", 'w') as f:
            f.write("\n".join(requirements))
        
        return str(package_path)
    
    async def _setup_monitoring(self, status: DeploymentStatus):
        """Setup monitoring for deployment"""
        
        # Configure Prometheus metrics
        metrics_config = {
            "deployment_id": status.deployment_id,
            "model_name": status.deployment_id.split("-")[1],
            "version": status.version,
            "metrics": [
                "inference_count",
                "inference_latency",
                "error_rate",
                "cpu_usage",
                "memory_usage",
                "gpu_usage"
            ]
        }
        
        self.cache.setex(
            f"monitoring:{status.deployment_id}",
            86400,
            json.dumps(metrics_config)
        )
    
    async def _setup_autoscaling(self, config: DeploymentConfig, status: DeploymentStatus):
        """Setup autoscaling for deployment"""
        
        if status.deployment_id.startswith("k8s-"):
            name = status.deployment_id.replace("k8s-", "")
            
            # Create HorizontalPodAutoscaler
            if config.scaling_policy == AutoScalingPolicy.CPU_BASED:
                target_metric = client.V2MetricSpec(
                    type="Resource",
                    resource=client.V2ResourceMetricSource(
                        name="cpu",
                        target=client.V2MetricTarget(
                            type="Utilization",
                            average_utilization=int(config.scaling_threshold * 100)
                        )
                    )
                )
            elif config.scaling_policy == AutoScalingPolicy.MEMORY_BASED:
                target_metric = client.V2MetricSpec(
                    type="Resource",
                    resource=client.V2ResourceMetricSource(
                        name="memory",
                        target=client.V2MetricTarget(
                            type="Utilization",
                            average_utilization=int(config.scaling_threshold * 100)
                        )
                    )
                )
            else:
                # Custom metric
                target_metric = client.V2MetricSpec(
                    type="Pods",
                    pods=client.V2PodsMetricSource(
                        metric=client.V2MetricIdentifier(name="custom_metric"),
                        target=client.V2MetricTarget(
                            type="AverageValue",
                            average_value=str(config.scaling_threshold)
                        )
                    )
                )
            
            hpa = client.V2HorizontalPodAutoscaler(
                api_version="autoscaling/v2",
                kind="HorizontalPodAutoscaler",
                metadata=client.V1ObjectMeta(
                    name=f"{name}-hpa",
                    namespace=self.k8s_namespace
                ),
                spec=client.V2HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V2CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=name
                    ),
                    min_replicas=config.min_replicas,
                    max_replicas=config.max_replicas,
                    metrics=[target_metric]
                )
            )
            
            self.k8s_autoscaling.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.k8s_namespace,
                body=hpa
            )
    
    def _register_deployment(self, status: DeploymentStatus):
        """Register deployment in tracking system"""
        self.active_deployments[status.deployment_id] = {
            "status": status,
            "created_at": status.created_at,
            "endpoint": status.endpoint
        }
    
    def _load_model(self, model_path: str, framework: ServingFramework):
        """Load model based on framework"""
        # Implementation depends on model format
        pass
    
    def _preprocess_input(self, data: Dict[str, Any], model: Any) -> Any:
        """Preprocess input data for model"""
        # Implementation depends on model requirements
        pass
    
    def _postprocess_output(self, predictions: Any) -> List[Any]:
        """Postprocess model output"""
        # Implementation depends on model output format
        pass
    
    async def _wait_for_deployment(self, name: str, timeout: int = 300):
        """Wait for Kubernetes deployment to be ready"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            deployment = self.k8s_apps.read_namespaced_deployment_status(
                name=name,
                namespace=self.k8s_namespace
            )
            
            if deployment.status.ready_replicas == deployment.spec.replicas:
                return
            
            await asyncio.sleep(5)
        
        raise TimeoutError(f"Deployment {name} not ready after {timeout} seconds")
    
    async def _build_and_push_image(self, package_path: str, image_name: str):
        """Build and push Docker image"""
        # Build image
        self.docker_client.images.build(
            path=package_path,
            tag=image_name,
            rm=True
        )
        
        # Push to registry
        self.docker_client.images.push(image_name)
    
    async def _get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment metrics from monitoring system"""
        # Query Prometheus or other monitoring system
        return {
            "inference_count": 0,
            "avg_latency_ms": 0,
            "error_rate": 0,
            "cpu_usage": 0,
            "memory_usage": 0
        }
    
    def _generate_fastapi_server(self, config: DeploymentConfig) -> str:
        """Generate FastAPI server code"""
        return """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from datetime import datetime

app = FastAPI()

# Load model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    data: dict

class PredictionResponse(BaseModel):
    predictions: list
    model_version: str
    inference_time_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = datetime.now()
    
    try:
        # Process input
        input_data = np.array(request.data['features']).reshape(1, -1)
        
        # Make prediction
        predictions = model.predict(input_data).tolist()
        
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            predictions=predictions,
            model_version=os.environ.get('MODEL_VERSION', 'unknown'),
            inference_time_ms=inference_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""
    
    def _generate_triton_config(self, config: DeploymentConfig) -> str:
        """Generate Triton Inference Server configuration"""
        return """
name: "model"
platform: "python"
max_batch_size: 32
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [-1]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
"""
    
    def _generate_generic_server(self, config: DeploymentConfig) -> str:
        """Generate generic model server code"""
        return self._generate_fastapi_server(config)
    
    async def _optimize_for_edge(self, model_path: str) -> str:
        """Optimize model for edge deployment"""
        # Convert to ONNX or TensorFlow Lite
        # Quantize model
        # Reduce model size
        return model_path
    
    async def _create_edge_package(self, model_path: str, config: DeploymentConfig) -> str:
        """Create edge deployment package"""
        # Package model with edge runtime
        return model_path
    
    async def _deploy_lambda(self, package_path: str, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy to AWS Lambda"""
        # Implementation for Lambda deployment
        pass
    
    async def _deploy_cloud_run(self, package_path: str, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy to Google Cloud Run"""
        # Implementation for Cloud Run deployment
        pass
    
    async def _deploy_azure_functions(self, package_path: str, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy to Azure Functions"""
        # Implementation for Azure Functions deployment
        pass