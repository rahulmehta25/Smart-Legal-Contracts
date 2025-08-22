"""
Pagination utilities for GraphQL Relay-style connections
"""

import base64
import json
from typing import List, Optional, Any, Dict
from sqlalchemy.orm import Query


def encode_cursor(value: Any) -> str:
    """Encode cursor value to base64 string"""
    try:
        cursor_data = {"value": str(value)}
        cursor_json = json.dumps(cursor_data)
        return base64.b64encode(cursor_json.encode()).decode()
    except Exception:
        return base64.b64encode(str(value).encode()).decode()


def decode_cursor(cursor: str) -> Any:
    """Decode cursor from base64 string"""
    try:
        cursor_json = base64.b64decode(cursor.encode()).decode()
        cursor_data = json.loads(cursor_json)
        return cursor_data.get("value")
    except Exception:
        # Fallback for simple cursors
        try:
            return base64.b64decode(cursor.encode()).decode()
        except Exception:
            return None


async def create_connection(
    query: Query,
    first: Optional[int] = None,
    after: Optional[str] = None,
    last: Optional[int] = None,
    before: Optional[str] = None,
    loader = None,
    cursor_field: str = "id"
):
    """Create Relay-style connection from SQLAlchemy query"""
    
    # Validate pagination arguments
    if first is not None and first < 0:
        raise ValueError("first must be non-negative")
    if last is not None and last < 0:
        raise ValueError("last must be non-negative")
    if first is not None and last is not None:
        raise ValueError("Cannot specify both first and last")
    
    # Apply cursor filtering
    if after:
        after_value = decode_cursor(after)
        if after_value:
            query = query.filter(getattr(query.column_descriptions[0]['type'], cursor_field) > after_value)
    
    if before:
        before_value = decode_cursor(before)
        if before_value:
            query = query.filter(getattr(query.column_descriptions[0]['type'], cursor_field) < before_value)
    
    # Determine limit and offset
    limit = first or last or 20
    
    # Add buffer for has_next_page/has_previous_page calculation
    items = query.limit(limit + 1).all()
    
    # Check if there are more items
    has_next_page = len(items) > limit
    has_previous_page = after is not None
    
    # Trim to actual limit
    if has_next_page:
        items = items[:limit]
    
    # Get total count (expensive operation, consider caching)
    total_count = query.count()
    
    # Create edges
    edges = []
    for item in items:
        cursor_value = getattr(item, cursor_field)
        cursor = encode_cursor(cursor_value)
        
        # Convert item using loader if provided
        if loader:
            node = await loader._convert_to_graphql_type(item)
        else:
            node = item
        
        edges.append({
            "node": node,
            "cursor": cursor
        })
    
    # Create page info
    page_info = {
        "has_next_page": has_next_page,
        "has_previous_page": has_previous_page,
        "start_cursor": edges[0]["cursor"] if edges else None,
        "end_cursor": edges[-1]["cursor"] if edges else None
    }
    
    return {
        "edges": edges,
        "page_info": page_info,
        "total_count": total_count
    }