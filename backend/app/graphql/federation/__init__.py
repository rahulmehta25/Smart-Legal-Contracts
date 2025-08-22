"""
Apollo Federation support for GraphQL schema
"""

from .schema import create_federated_schema
from .entities import EntityResolver
from .directives import federation_directives

__all__ = ["create_federated_schema", "EntityResolver", "federation_directives"]