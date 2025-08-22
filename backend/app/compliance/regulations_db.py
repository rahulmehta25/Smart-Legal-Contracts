from enum import Enum, auto
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

class RegulationType(Enum):
    GDPR = auto()
    CCPA = auto()
    HIPAA = auto()
    SOX = auto()
    ISO27001 = auto()

@dataclass
class Regulation:
    name: RegulationType
    description: str
    jurisdiction: str
    requirements: List[str]
    penalties: Dict[str, float]
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

class RegulationsDatabase:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_regulations()
        return cls._instance
    
    def _init_regulations(self):
        self.regulations: Dict[RegulationType, Regulation] = {
            RegulationType.GDPR: Regulation(
                name=RegulationType.GDPR,
                description="General Data Protection Regulation",
                jurisdiction="European Union",
                requirements=[
                    "Consent management",
                    "Right to erasure",
                    "Data portability",
                    "Privacy by design",
                    "Breach notification within 72 hours"
                ],
                penalties={
                    "minor_breach": 10000000,
                    "major_breach": 20000000
                }
            ),
            RegulationType.CCPA: Regulation(
                name=RegulationType.CCPA,
                description="California Consumer Privacy Act",
                jurisdiction="California, USA",
                requirements=[
                    "Consumer right to know",
                    "Opt-out of data sale",
                    "Right to delete personal information",
                    "Non-discrimination provisions"
                ],
                penalties={
                    "per_violation": 7500,
                    "annual_cap": 7500000
                }
            ),
            # Add other regulations similarly
        }
    
    def get_regulation(self, reg_type: RegulationType) -> Regulation:
        return self.regulations.get(reg_type)
    
    def update_regulation(self, reg_type: RegulationType, updates: Dict[str, Any]):
        """Update regulation details dynamically"""
        if reg_type in self.regulations:
            for key, value in updates.items():
                setattr(self.regulations[reg_type], key, value)
            self.regulations[reg_type].last_updated = datetime.now()
    
    def check_regulation_status(self, reg_type: RegulationType) -> bool:
        """Check if regulation is current and not outdated"""
        regulation = self.get_regulation(reg_type)
        return regulation is not None and (
            datetime.now() - regulation.last_updated < timedelta(days=365)
        )

# Singleton instance for global access
regulations_db = RegulationsDatabase()