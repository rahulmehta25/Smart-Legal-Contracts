"""
Saga Orchestrator Implementation

Manages distributed transactions using the saga pattern to ensure
data consistency across microservices without requiring distributed locks.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SagaStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    TIMEOUT = "timeout"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    """Represents a single step in a saga"""
    step_id: str
    name: str
    service_name: str
    action: str
    compensation_action: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "service_name": self.service_name,
            "action": self.action,
            "compensation_action": self.compensation_action,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class SagaInstance:
    """Represents a running saga instance"""
    saga_id: str
    saga_type: str
    status: SagaStatus
    steps: List[SagaStep]
    current_step_index: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    timeout_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @property
    def current_step(self) -> Optional[SagaStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    @property
    def completed_steps(self) -> List[SagaStep]:
        return [step for step in self.steps if step.status == StepStatus.COMPLETED]
    
    @property
    def failed_steps(self) -> List[SagaStep]:
        return [step for step in self.steps if step.status == StepStatus.FAILED]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "saga_id": self.saga_id,
            "saga_type": self.saga_type,
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "current_step_index": self.current_step_index,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout_at": self.timeout_at.isoformat() if self.timeout_at else None,
            "context": self.context,
            "error_message": self.error_message
        }


class SagaDefinition(ABC):
    """Abstract base class for saga definitions"""
    
    @abstractmethod
    def get_saga_type(self) -> str:
        """Return the saga type identifier"""
        pass
    
    @abstractmethod  
    def define_steps(self, context: Dict[str, Any]) -> List[SagaStep]:
        """Define the steps for this saga"""
        pass
    
    @abstractmethod
    def get_timeout_seconds(self) -> int:
        """Return the saga timeout in seconds"""
        pass


class SagaOrchestrator:
    """
    Orchestrates saga execution and manages distributed transactions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.running_sagas: Dict[str, SagaInstance] = {}
        self.saga_definitions: Dict[str, SagaDefinition] = {}
        self.step_handlers: Dict[str, Callable] = {}
        self.compensation_handlers: Dict[str, Callable] = {}
        
        # Event integration
        self.event_bus = None  # Will be injected
        self.service_discovery = None  # Will be injected
        
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        
        # Statistics
        self.stats = {
            "sagas_started": 0,
            "sagas_completed": 0,
            "sagas_failed": 0,
            "sagas_compensated": 0,
            "steps_executed": 0,
            "compensations_executed": 0
        }
    
    async def start(self):
        """Start the saga orchestrator"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting Saga Orchestrator")
        
        # Start background tasks
        execution_task = asyncio.create_task(self._saga_execution_loop())
        timeout_task = asyncio.create_task(self._timeout_monitoring_loop())
        
        self._tasks.update([execution_task, timeout_task])
        
        # Register built-in saga definitions
        await self._register_builtin_sagas()
        
        logger.info("Saga Orchestrator started successfully")
    
    async def stop(self):
        """Stop the saga orchestrator"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping Saga Orchestrator")
        
        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        
        # Attempt to complete running sagas gracefully
        for saga in list(self.running_sagas.values()):
            if saga.status == SagaStatus.RUNNING:
                saga.status = SagaStatus.TIMEOUT
                saga.error_message = "System shutdown"
                await self._trigger_compensation(saga)
        
        logger.info("Saga Orchestrator stopped")
    
    def register_saga_definition(self, saga_definition: SagaDefinition):
        """Register a saga definition"""
        saga_type = saga_definition.get_saga_type()
        self.saga_definitions[saga_type] = saga_definition
        logger.info(f"Registered saga definition: {saga_type}")
    
    def register_step_handler(self, action: str, handler: Callable):
        """Register a step handler for an action"""
        self.step_handlers[action] = handler
        logger.info(f"Registered step handler: {action}")
    
    def register_compensation_handler(self, action: str, handler: Callable):
        """Register a compensation handler for an action"""
        self.compensation_handlers[action] = handler
        logger.info(f"Registered compensation handler: {action}")
    
    async def start_saga(
        self, 
        saga_type: str, 
        context: Dict[str, Any],
        saga_id: Optional[str] = None
    ) -> str:
        """Start a new saga instance"""
        try:
            if saga_type not in self.saga_definitions:
                raise ValueError(f"Unknown saga type: {saga_type}")
            
            saga_id = saga_id or str(uuid.uuid4())
            saga_definition = self.saga_definitions[saga_type]
            
            # Define saga steps
            steps = saga_definition.define_steps(context)
            
            # Create saga instance
            saga = SagaInstance(
                saga_id=saga_id,
                saga_type=saga_type,
                status=SagaStatus.PENDING,
                steps=steps,
                context=context,
                timeout_at=datetime.now() + timedelta(seconds=saga_definition.get_timeout_seconds())
            )
            
            # Store saga instance
            self.running_sagas[saga_id] = saga
            self.stats["sagas_started"] += 1
            
            logger.info(f"Started saga {saga_id} of type {saga_type} with {len(steps)} steps")
            
            # Trigger immediate execution
            asyncio.create_task(self._execute_saga_step(saga))
            
            return saga_id
            
        except Exception as e:
            logger.error(f"Failed to start saga {saga_type}: {e}")
            raise
    
    async def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a saga"""
        saga = self.running_sagas.get(saga_id)
        return saga.to_dict() if saga else None
    
    async def cancel_saga(self, saga_id: str) -> bool:
        """Cancel a running saga and trigger compensation"""
        saga = self.running_sagas.get(saga_id)
        if not saga:
            logger.warning(f"Saga {saga_id} not found for cancellation")
            return False
        
        if saga.status in [SagaStatus.COMPLETED, SagaStatus.COMPENSATED, SagaStatus.FAILED]:
            logger.warning(f"Saga {saga_id} cannot be cancelled in status {saga.status.value}")
            return False
        
        saga.status = SagaStatus.COMPENSATING
        saga.error_message = "Saga cancelled by user"
        
        await self._trigger_compensation(saga)
        logger.info(f"Cancelled saga {saga_id}")
        return True
    
    async def _saga_execution_loop(self):
        """Background loop for executing saga steps"""
        while self._running:
            try:
                # Find sagas ready for execution
                ready_sagas = [
                    saga for saga in self.running_sagas.values()
                    if saga.status in [SagaStatus.PENDING, SagaStatus.RUNNING]
                ]
                
                # Execute saga steps concurrently
                execution_tasks = []
                for saga in ready_sagas:
                    if saga.status == SagaStatus.PENDING:
                        saga.status = SagaStatus.RUNNING
                    
                    task = asyncio.create_task(self._execute_saga_step(saga))
                    execution_tasks.append(task)
                
                if execution_tasks:
                    await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                await asyncio.sleep(1)  # Short delay between execution cycles
                
            except Exception as e:
                logger.error(f"Error in saga execution loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_saga_step(self, saga: SagaInstance):
        """Execute the current step of a saga"""
        if saga.status not in [SagaStatus.RUNNING]:
            return
        
        current_step = saga.current_step
        if not current_step:
            # All steps completed
            await self._complete_saga(saga)
            return
        
        if current_step.status != StepStatus.PENDING:
            return  # Step already processed or in progress
        
        try:
            current_step.status = StepStatus.RUNNING
            current_step.started_at = datetime.now()
            saga.updated_at = datetime.now()
            
            # Find and execute step handler
            handler = self.step_handlers.get(current_step.action)
            if not handler:
                raise Exception(f"No handler registered for action: {current_step.action}")
            
            logger.info(f"Executing step {current_step.name} in saga {saga.saga_id}")
            
            # Execute step with timeout
            try:
                result = await asyncio.wait_for(
                    handler(current_step.input_data, saga.context),
                    timeout=current_step.timeout_seconds
                )
                
                # Step completed successfully
                current_step.output_data = result or {}
                current_step.status = StepStatus.COMPLETED
                current_step.completed_at = datetime.now()
                
                # Move to next step
                saga.current_step_index += 1
                saga.updated_at = datetime.now()
                
                self.stats["steps_executed"] += 1
                
                logger.info(f"Completed step {current_step.name} in saga {saga.saga_id}")
                
                # Continue with next step
                await asyncio.sleep(0.1)  # Small delay before next step
                await self._execute_saga_step(saga)
                
            except asyncio.TimeoutError:
                current_step.error_message = f"Step timed out after {current_step.timeout_seconds} seconds"
                await self._handle_step_failure(saga, current_step)
                
            except Exception as step_error:
                current_step.error_message = str(step_error)
                await self._handle_step_failure(saga, current_step)
        
        except Exception as e:
            logger.error(f"Error executing saga step: {e}")
            current_step.error_message = str(e)
            await self._handle_step_failure(saga, current_step)
    
    async def _handle_step_failure(self, saga: SagaInstance, failed_step: SagaStep):
        """Handle failure of a saga step"""
        failed_step.status = StepStatus.FAILED
        failed_step.retry_count += 1
        
        # Check if we should retry
        if failed_step.retry_count <= failed_step.max_retries:
            logger.info(f"Retrying step {failed_step.name} (attempt {failed_step.retry_count})")
            
            # Reset step for retry
            failed_step.status = StepStatus.PENDING
            failed_step.started_at = None
            
            # Exponential backoff
            delay = min(2 ** failed_step.retry_count, 60)
            await asyncio.sleep(delay)
            
            await self._execute_saga_step(saga)
        else:
            # Max retries exceeded, trigger compensation
            logger.error(f"Step {failed_step.name} failed after {failed_step.retry_count} attempts")
            saga.status = SagaStatus.COMPENSATING
            saga.error_message = f"Step {failed_step.name} failed: {failed_step.error_message}"
            
            await self._trigger_compensation(saga)
    
    async def _complete_saga(self, saga: SagaInstance):
        """Mark saga as completed"""
        saga.status = SagaStatus.COMPLETED
        saga.completed_at = datetime.now()
        saga.updated_at = datetime.now()
        
        self.stats["sagas_completed"] += 1
        
        # Remove from running sagas (optionally keep for audit trail)
        # del self.running_sagas[saga.saga_id]
        
        logger.info(f"Saga {saga.saga_id} completed successfully")
        
        # Publish completion event
        if self.event_bus:
            await self.event_bus.publish({
                "type": "saga.completed",
                "source": "saga-orchestrator",
                "data": {
                    "saga_id": saga.saga_id,
                    "saga_type": saga.saga_type,
                    "completed_at": saga.completed_at.isoformat()
                }
            })
    
    async def _trigger_compensation(self, saga: SagaInstance):
        """Trigger compensation for completed steps in reverse order"""
        saga.status = SagaStatus.COMPENSATING
        saga.updated_at = datetime.now()
        
        logger.info(f"Starting compensation for saga {saga.saga_id}")
        
        # Get completed steps in reverse order
        completed_steps = [
            step for step in reversed(saga.steps)
            if step.status == StepStatus.COMPLETED and step.compensation_action
        ]
        
        compensation_successful = True
        
        for step in completed_steps:
            try:
                compensation_handler = self.compensation_handlers.get(step.compensation_action)
                if not compensation_handler:
                    logger.warning(f"No compensation handler for {step.compensation_action}")
                    continue
                
                logger.info(f"Compensating step {step.name} in saga {saga.saga_id}")
                
                await compensation_handler(step.output_data, saga.context)
                step.status = StepStatus.COMPENSATED
                
                self.stats["compensations_executed"] += 1
                
            except Exception as e:
                logger.error(f"Compensation failed for step {step.name}: {e}")
                compensation_successful = False
                break
        
        # Update saga status
        if compensation_successful:
            saga.status = SagaStatus.COMPENSATED
            self.stats["sagas_compensated"] += 1
        else:
            saga.status = SagaStatus.FAILED
            self.stats["sagas_failed"] += 1
        
        saga.completed_at = datetime.now()
        saga.updated_at = datetime.now()
        
        logger.info(f"Compensation completed for saga {saga.saga_id} with status {saga.status.value}")
    
    async def _timeout_monitoring_loop(self):
        """Monitor sagas for timeout conditions"""
        while self._running:
            try:
                current_time = datetime.now()
                
                for saga in list(self.running_sagas.values()):
                    if (saga.timeout_at and current_time > saga.timeout_at and 
                        saga.status in [SagaStatus.PENDING, SagaStatus.RUNNING]):
                        
                        logger.warning(f"Saga {saga.saga_id} timed out")
                        saga.status = SagaStatus.TIMEOUT
                        saga.error_message = "Saga execution timed out"
                        
                        await self._trigger_compensation(saga)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in timeout monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _register_builtin_sagas(self):
        """Register built-in saga definitions for new features"""
        # Document Analysis Saga
        self.register_saga_definition(DocumentAnalysisSaga())
        
        # User Onboarding Saga
        self.register_saga_definition(UserOnboardingSaga())
        
        # Compliance Check Saga
        self.register_saga_definition(ComplianceCheckSaga())
        
        # Payment Processing Saga
        self.register_saga_definition(PaymentProcessingSaga())
        
        # Register corresponding step handlers
        await self._register_step_handlers()
    
    async def _register_step_handlers(self):
        """Register step handlers for built-in sagas"""
        # Document analysis handlers
        self.register_step_handler("upload_document", self._handle_upload_document)
        self.register_step_handler("extract_text", self._handle_extract_text)
        self.register_step_handler("analyze_clauses", self._handle_analyze_clauses)
        self.register_step_handler("generate_report", self._handle_generate_report)
        
        # User onboarding handlers
        self.register_step_handler("create_user", self._handle_create_user)
        self.register_step_handler("setup_tenant", self._handle_setup_tenant)
        self.register_step_handler("send_welcome_email", self._handle_send_welcome_email)
        self.register_step_handler("configure_defaults", self._handle_configure_defaults)
        
        # Compliance handlers
        self.register_step_handler("check_jurisdiction", self._handle_check_jurisdiction)
        self.register_step_handler("validate_clauses", self._handle_validate_clauses)
        self.register_step_handler("generate_compliance_report", self._handle_generate_compliance_report)
        
        # Compensation handlers
        self.register_compensation_handler("delete_document", self._handle_delete_document)
        self.register_compensation_handler("delete_user", self._handle_delete_user)
        self.register_compensation_handler("delete_tenant", self._handle_delete_tenant)
    
    # Step handler implementations (simplified)
    async def _handle_upload_document(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Handling document upload: {input_data.get('document_name')}")
        return {"document_id": str(uuid.uuid4()), "status": "uploaded"}
    
    async def _handle_extract_text(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Extracting text from document: {input_data.get('document_id')}")
        return {"text_extracted": True, "text_length": 1000}
    
    async def _handle_analyze_clauses(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Analyzing clauses for document: {input_data.get('document_id')}")
        return {"arbitration_found": True, "confidence": 0.95}
    
    async def _handle_generate_report(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Generating analysis report")
        return {"report_id": str(uuid.uuid4()), "report_url": "/reports/analysis.pdf"}
    
    async def _handle_create_user(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Creating user: {input_data.get('email')}")
        return {"user_id": str(uuid.uuid4()), "status": "created"}
    
    async def _handle_setup_tenant(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Setting up tenant for user: {input_data.get('user_id')}")
        return {"tenant_id": str(uuid.uuid4()), "status": "configured"}
    
    async def _handle_send_welcome_email(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Sending welcome email to: {input_data.get('email')}")
        return {"email_sent": True, "email_id": str(uuid.uuid4())}
    
    async def _handle_configure_defaults(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Configuring defaults for tenant: {input_data.get('tenant_id')}")
        return {"defaults_configured": True}
    
    async def _handle_check_jurisdiction(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Checking jurisdiction: {input_data.get('jurisdiction')}")
        return {"jurisdiction_valid": True, "rules_loaded": True}
    
    async def _handle_validate_clauses(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Validating clauses for compliance")
        return {"compliant": True, "violations": []}
    
    async def _handle_generate_compliance_report(self, input_data: Dict, context: Dict) -> Dict:
        logger.info(f"Generating compliance report")
        return {"report_id": str(uuid.uuid4()), "compliance_score": 95}
    
    # Compensation handlers
    async def _handle_delete_document(self, input_data: Dict, context: Dict):
        logger.info(f"Compensating: deleting document {input_data.get('document_id')}")
    
    async def _handle_delete_user(self, input_data: Dict, context: Dict):
        logger.info(f"Compensating: deleting user {input_data.get('user_id')}")
    
    async def _handle_delete_tenant(self, input_data: Dict, context: Dict):
        logger.info(f"Compensating: deleting tenant {input_data.get('tenant_id')}")
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            **self.stats,
            "running_sagas": len(self.running_sagas),
            "registered_saga_types": len(self.saga_definitions),
            "registered_step_handlers": len(self.step_handlers),
            "registered_compensation_handlers": len(self.compensation_handlers),
            "saga_status_breakdown": {
                status.value: len([s for s in self.running_sagas.values() if s.status == status])
                for status in SagaStatus
            }
        }


# Built-in Saga Definitions
class DocumentAnalysisSaga(SagaDefinition):
    def get_saga_type(self) -> str:
        return "document_analysis"
    
    def define_steps(self, context: Dict[str, Any]) -> List[SagaStep]:
        return [
            SagaStep(
                step_id="upload",
                name="Upload Document", 
                service_name="document-service",
                action="upload_document",
                compensation_action="delete_document",
                input_data={"document_name": context.get("document_name")}
            ),
            SagaStep(
                step_id="extract",
                name="Extract Text",
                service_name="ml-service", 
                action="extract_text",
                input_data={}
            ),
            SagaStep(
                step_id="analyze",
                name="Analyze Clauses",
                service_name="analysis-service",
                action="analyze_clauses", 
                input_data={}
            ),
            SagaStep(
                step_id="report",
                name="Generate Report",
                service_name="document-service",
                action="generate_report",
                input_data={}
            )
        ]
    
    def get_timeout_seconds(self) -> int:
        return 1800  # 30 minutes


class UserOnboardingSaga(SagaDefinition):
    def get_saga_type(self) -> str:
        return "user_onboarding"
    
    def define_steps(self, context: Dict[str, Any]) -> List[SagaStep]:
        return [
            SagaStep(
                step_id="create_user",
                name="Create User Account",
                service_name="user-service",
                action="create_user",
                compensation_action="delete_user",
                input_data={"email": context.get("email")}
            ),
            SagaStep(
                step_id="setup_tenant",
                name="Setup Tenant",
                service_name="whitelabel-service",
                action="setup_tenant",
                compensation_action="delete_tenant",
                input_data={}
            ),
            SagaStep(
                step_id="send_email",
                name="Send Welcome Email",
                service_name="notification-service",
                action="send_welcome_email",
                input_data={"email": context.get("email")}
            ),
            SagaStep(
                step_id="configure",
                name="Configure Defaults",
                service_name="user-service",
                action="configure_defaults",
                input_data={}
            )
        ]
    
    def get_timeout_seconds(self) -> int:
        return 600  # 10 minutes


class ComplianceCheckSaga(SagaDefinition):
    def get_saga_type(self) -> str:
        return "compliance_check"
    
    def define_steps(self, context: Dict[str, Any]) -> List[SagaStep]:
        return [
            SagaStep(
                step_id="check_jurisdiction",
                name="Check Jurisdiction",
                service_name="legal-service",
                action="check_jurisdiction",
                input_data={"jurisdiction": context.get("jurisdiction")}
            ),
            SagaStep(
                step_id="validate_clauses",
                name="Validate Clauses",
                service_name="compliance-automation",
                action="validate_clauses",
                input_data={"document_id": context.get("document_id")}
            ),
            SagaStep(
                step_id="generate_report",
                name="Generate Compliance Report",
                service_name="compliance-automation",
                action="generate_compliance_report",
                input_data={}
            )
        ]
    
    def get_timeout_seconds(self) -> int:
        return 900  # 15 minutes


class PaymentProcessingSaga(SagaDefinition):
    def get_saga_type(self) -> str:
        return "payment_processing"
    
    def define_steps(self, context: Dict[str, Any]) -> List[SagaStep]:
        return [
            SagaStep(
                step_id="validate_payment",
                name="Validate Payment Method",
                service_name="payment-service",
                action="validate_payment_method",
                input_data={"payment_method": context.get("payment_method")}
            ),
            SagaStep(
                step_id="charge_payment",
                name="Charge Payment",
                service_name="payment-service", 
                action="charge_payment",
                compensation_action="refund_payment",
                input_data={"amount": context.get("amount")}
            ),
            SagaStep(
                step_id="update_subscription",
                name="Update Subscription",
                service_name="user-service",
                action="update_subscription",
                compensation_action="revert_subscription",
                input_data={"plan": context.get("plan")}
            )
        ]
    
    def get_timeout_seconds(self) -> int:
        return 300  # 5 minutes