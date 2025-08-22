import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum, auto

class APICategory(Enum):
    FINANCE = auto()
    COMMUNICATION = auto()
    PRODUCTIVITY = auto()
    SOCIAL_MEDIA = auto()
    UTILITIES = auto()
    MACHINE_LEARNING = auto()
    OTHER = auto()

class PricingTier(Enum):
    FREE = 'free'
    BASIC = 'basic'
    PRO = 'pro'
    ENTERPRISE = 'enterprise'

@dataclass
class Endpoint:
    path: str
    method: str
    description: Optional[str] = None

@dataclass
class API:
    id: str
    name: str
    description: Optional[str] = None
    version: str = '1.0.0'
    category: APICategory = APICategory.OTHER
    endpoints: List[Endpoint] = None
    pricing_tier: PricingTier = PricingTier.FREE

class APIRegistry:
    def __init__(self):
        self._apis: Dict[str, API] = {}

    def register_api(self, api: API) -> str:
        """
        Register a new API in the marketplace
        
        :param api: API object to register
        :return: Registered API's unique ID
        """
        if not api.id:
            api.id = str(uuid.uuid4())
        
        # Validate API before registration
        self._validate_api(api)
        
        self._apis[api.id] = api
        return api.id

    def get_api(self, api_id: str) -> Optional[API]:
        """
        Retrieve an API by its ID
        
        :param api_id: Unique identifier of the API
        :return: API object or None if not found
        """
        return self._apis.get(api_id)

    def list_apis(
        self, 
        category: Optional[APICategory] = None, 
        pricing_tier: Optional[PricingTier] = None
    ) -> List[API]:
        """
        List APIs with optional filtering
        
        :param category: Filter by API category
        :param pricing_tier: Filter by pricing tier
        :return: List of matching APIs
        """
        apis = list(self._apis.values())
        
        if category:
            apis = [api for api in apis if api.category == category]
        
        if pricing_tier:
            apis = [api for api in apis if api.pricing_tier == pricing_tier]
        
        return apis

    def update_api(self, api_id: str, updated_api: API) -> Optional[API]:
        """
        Update an existing API
        
        :param api_id: ID of the API to update
        :param updated_api: Updated API details
        :return: Updated API or None if not found
        """
        if api_id not in self._apis:
            return None
        
        self._validate_api(updated_api)
        updated_api.id = api_id
        self._apis[api_id] = updated_api
        return updated_api

    def delete_api(self, api_id: str) -> bool:
        """
        Delete an API from the registry
        
        :param api_id: ID of the API to delete
        :return: True if deleted, False if not found
        """
        return self._apis.pop(api_id, None) is not None

    def _validate_api(self, api: API):
        """
        Validate API before registration or update
        
        :param api: API to validate
        :raises ValueError: If API is invalid
        """
        if not api.name:
            raise ValueError("API must have a name")
        
        if not api.endpoints:
            raise ValueError("API must have at least one endpoint")

def main():
    # Example usage
    registry = APIRegistry()
    
    example_api = API(
        id='',
        name='Weather API',
        description='Get real-time weather data',
        category=APICategory.UTILITIES,
        endpoints=[
            Endpoint(
                path='/current-weather', 
                method='GET', 
                description='Get current weather'
            )
        ],
        pricing_tier=PricingTier.BASIC
    )
    
    api_id = registry.register_api(example_api)
    print(f"Registered API with ID: {api_id}")

if __name__ == '__main__':
    main()