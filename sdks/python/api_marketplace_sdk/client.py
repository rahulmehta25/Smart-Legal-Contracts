import requests
from typing import Dict, Any, Optional
import json

class APIMarketplaceClient:
    """
    Official Python client for API Marketplace
    """
    BASE_URL = 'https://api.example.com/v1'

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the client with authentication credentials

        :param api_key: API Key for authentication
        :param api_secret: API Secret for authentication
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = self._authenticate()

    def _authenticate(self) -> str:
        """
        Authenticate and retrieve access token

        :return: Access token for API requests
        """
        auth_endpoint = f'{self.BASE_URL}/oauth/token'
        response = requests.post(auth_endpoint, json={
            'grant_type': 'client_credentials',
            'client_id': self._api_key,
            'client_secret': self._api_secret
        })
        response.raise_for_status()
        return response.json()['access_token']

    def list_apis(
        self, 
        category: Optional[str] = None, 
        search_term: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List available APIs in the marketplace

        :param category: Optional API category filter
        :param search_term: Optional search term
        :return: List of APIs matching the criteria
        """
        endpoint = f'{self.BASE_URL}/apis'
        params = {}
        if category:
            params['category'] = category
        if search_term:
            params['search'] = search_term

        headers = {
            'Authorization': f'Bearer {self._access_token}',
            'Content-Type': 'application/json'
        }

        response = requests.get(endpoint, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    def register_api(self, api_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new API in the marketplace

        :param api_details: Detailed information about the API
        :return: Registered API details
        """
        endpoint = f'{self.BASE_URL}/marketplace/register'
        headers = {
            'Authorization': f'Bearer {self._access_token}',
            'Content-Type': 'application/json'
        }

        response = requests.post(endpoint, json=api_details, headers=headers)
        response.raise_for_status()
        return response.json()

def main():
    # Example usage
    client = APIMarketplaceClient(
        api_key='your_api_key', 
        api_secret='your_api_secret'
    )
    
    # List APIs
    apis = client.list_apis(category='FINANCE')
    print(json.dumps(apis, indent=2))

    # Register a new API
    new_api = {
        'name': 'Stock Price API',
        'description': 'Real-time stock price data',
        'category': 'FINANCE',
        'version': '1.0.0'
    }
    registered_api = client.register_api(new_api)
    print(json.dumps(registered_api, indent=2))

if __name__ == '__main__':
    main()