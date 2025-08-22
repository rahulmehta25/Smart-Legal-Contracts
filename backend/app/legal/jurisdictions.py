from enum import Enum, auto
from typing import List, Dict, Optional
import pycountry
import ipaddress
import geoip2.database

class LegalSystem(Enum):
    COMMON_LAW = auto()
    CIVIL_LAW = auto()
    MIXED_SYSTEM = auto()

class JurisdictionDetector:
    def __init__(self, geoip_database_path: str = None):
        """
        Initialize jurisdiction detection with optional GeoIP database
        
        :param geoip_database_path: Path to MaxMind GeoIP database
        """
        self.geoip_reader = None
        if geoip_database_path:
            self.geoip_reader = geoip2.database.Reader(geoip_database_path)

    def detect_jurisdiction_by_ip(self, ip_address: str) -> Optional[str]:
        """
        Detect jurisdiction based on IP address
        
        :param ip_address: IP address to check
        :return: Country code or None
        """
        try:
            # Validate IP address
            ipaddress.ip_address(ip_address)
            
            if self.geoip_reader:
                response = self.geoip_reader.country(ip_address)
                return response.country.iso_code
        except Exception:
            return None

    def get_jurisdiction_details(self, country_code: str) -> Dict:
        """
        Get detailed jurisdiction information
        
        :param country_code: ISO country code
        :return: Dictionary of jurisdiction details
        """
        try:
            country = pycountry.countries.get(alpha_2=country_code)
            return {
                'name': country.name,
                'official_name': getattr(country, 'official_name', country.name),
                'legal_system': self._determine_legal_system(country_code),
                'continent': self._get_continent(country_code)
            }
        except Exception:
            return {}

    def _determine_legal_system(self, country_code: str) -> LegalSystem:
        """
        Determine the legal system for a given country
        
        :param country_code: ISO country code
        :return: LegalSystem enum
        """
        common_law_countries = ['US', 'UK', 'CA', 'AU', 'NZ', 'IN']
        civil_law_countries = ['FR', 'DE', 'IT', 'ES', 'BR', 'JP']
        
        if country_code in common_law_countries:
            return LegalSystem.COMMON_LAW
        elif country_code in civil_law_countries:
            return LegalSystem.CIVIL_LAW
        else:
            return LegalSystem.MIXED_SYSTEM

    def _get_continent(self, country_code: str) -> str:
        """
        Get the continent for a given country
        
        :param country_code: ISO country code
        :return: Continent name
        """
        try:
            country = pycountry.countries.get(alpha_2=country_code)
            continent = pycountry.countries.get(alpha_2=country_code).continent
            return continent
        except Exception:
            return 'Unknown'

    def list_supported_jurisdictions(self) -> List[Dict]:
        """
        List all supported jurisdictions
        
        :return: List of jurisdiction details
        """
        supported_codes = ['US', 'CA', 'GB', 'AU', 'EU', 'BR', 'IN', 'JP', 'FR', 'DE']
        return [self.get_jurisdiction_details(code) for code in supported_codes]

# Disclaimer and usage guidance
__doc__ = """
Jurisdiction Detection Module

This module provides tools for detecting and analyzing legal jurisdictions.
It supports IP-based detection, legal system classification, and jurisdiction metadata.

Dependencies:
- pycountry
- geoip2
- ipaddress

Recommended Usage:
1. Initialize with GeoIP database path
2. Use detect_jurisdiction_by_ip() for automatic detection
3. Use get_jurisdiction_details() for in-depth information

Note: Always consult local legal experts for definitive jurisdiction interpretation.
"""