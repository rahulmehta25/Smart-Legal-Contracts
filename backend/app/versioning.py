from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional
from datetime import datetime, timedelta

class VersionStatus(Enum):
    ACTIVE = auto()
    DEPRECATED = auto()
    RETIRED = auto()

@dataclass
class APIVersion:
    version: str
    status: VersionStatus
    release_date: datetime
    deprecation_date: Optional[datetime] = None
    retirement_date: Optional[datetime] = None
    migration_guide: Optional[str] = None

class VersionManager:
    def __init__(self):
        self._versions: List[APIVersion] = []

    def register_version(
        self, 
        version: str, 
        status: VersionStatus = VersionStatus.ACTIVE,
        deprecation_period: int = 180,
        retirement_period: int = 365
    ) -> APIVersion:
        """
        Register a new API version with automatic deprecation and retirement dates
        
        :param version: Version string
        :param status: Initial version status
        :param deprecation_period: Days until version is deprecated
        :param retirement_period: Days until version is retired
        :return: Registered APIVersion instance
        """
        now = datetime.now()
        
        new_version = APIVersion(
            version=version,
            status=status,
            release_date=now,
            deprecation_date=now + timedelta(days=deprecation_period),
            retirement_date=now + timedelta(days=retirement_period)
        )
        
        self._versions.append(new_version)
        return new_version

    def get_version(self, version: str) -> Optional[APIVersion]:
        """
        Retrieve a specific version's details
        
        :param version: Version string to retrieve
        :return: APIVersion or None if not found
        """
        return next((v for v in self._versions if v.version == version), None)

    def list_versions(
        self, 
        status: Optional[VersionStatus] = None
    ) -> List[APIVersion]:
        """
        List versions with optional status filtering
        
        :param status: Optional status to filter versions
        :return: List of matching versions
        """
        if status:
            return [v for v in self._versions if v.status == status]
        return self._versions

    def update_version_status(
        self, 
        version: str, 
        new_status: VersionStatus
    ) -> Optional[APIVersion]:
        """
        Update the status of a specific version
        
        :param version: Version to update
        :param new_status: New status for the version
        :return: Updated APIVersion or None if not found
        """
        existing_version = self.get_version(version)
        if existing_version:
            existing_version.status = new_status
            return existing_version
        return None

def main():
    # Example usage
    version_manager = VersionManager()
    
    # Register versions
    v1 = version_manager.register_version('1.0.0')
    v2 = version_manager.register_version('2.0.0')
    
    # List active versions
    active_versions = version_manager.list_versions(VersionStatus.ACTIVE)
    for version in active_versions:
        print(f"Active Version: {version.version}")

if __name__ == '__main__':
    main()