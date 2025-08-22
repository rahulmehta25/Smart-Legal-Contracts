"""
Dimensional Modeling System

Provides comprehensive dimensional modeling capabilities for data warehousing
including star schema, snowflake schema, and slowly changing dimensions.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import text, MetaData, Table, Column, Integer, String, DateTime, Boolean, Numeric, ForeignKey
from sqlalchemy.dialects import postgresql, mysql, snowflake
from sqlalchemy.schema import CreateTable


class DimensionType(Enum):
    """Types of dimensions"""
    STANDARD = "standard"
    TIME = "time"
    SLOWLY_CHANGING_TYPE_1 = "scd_type_1"
    SLOWLY_CHANGING_TYPE_2 = "scd_type_2"
    SLOWLY_CHANGING_TYPE_3 = "scd_type_3"
    JUNK = "junk"
    DEGENERATE = "degenerate"
    ROLE_PLAYING = "role_playing"


class SchemaType(Enum):
    """Data warehouse schema types"""
    STAR = "star"
    SNOWFLAKE = "snowflake"
    GALAXY = "galaxy"
    DATA_VAULT = "data_vault"


class SCDType(Enum):
    """Slowly Changing Dimension types"""
    TYPE_0 = 0  # No changes allowed
    TYPE_1 = 1  # Overwrite
    TYPE_2 = 2  # Add new record with versioning
    TYPE_3 = 3  # Add new attribute
    TYPE_4 = 4  # History table
    TYPE_6 = 6  # Hybrid approach (Type 1 + Type 2 + Type 3)


@dataclass
class DimensionAttribute:
    """Dimension attribute definition"""
    name: str
    data_type: str
    is_key: bool = False
    is_natural_key: bool = False
    scd_type: SCDType = SCDType.TYPE_1
    description: Optional[str] = None
    default_value: Optional[Any] = None
    nullable: bool = True
    max_length: Optional[int] = None


@dataclass
class DimensionDefinition:
    """Dimension table definition"""
    name: str
    dimension_type: DimensionType
    attributes: List[DimensionAttribute]
    description: Optional[str] = None
    source_table: Optional[str] = None
    natural_key_columns: List[str] = None
    parent_dimension: Optional[str] = None  # For snowflake schema
    
    def __post_init__(self):
        if self.natural_key_columns is None:
            self.natural_key_columns = []


@dataclass
class FactAttribute:
    """Fact table attribute definition"""
    name: str
    data_type: str
    is_measure: bool = True
    is_additive: bool = True
    aggregation_function: str = "SUM"
    description: Optional[str] = None
    nullable: bool = True


@dataclass
class FactDefinition:
    """Fact table definition"""
    name: str
    grain: str  # Description of the fact table grain
    measures: List[FactAttribute]
    dimensions: List[str]  # List of dimension names
    description: Optional[str] = None
    source_table: Optional[str] = None
    partition_column: Optional[str] = None


@dataclass
class SchemaDefinition:
    """Data warehouse schema definition"""
    name: str
    schema_type: SchemaType
    dimensions: List[DimensionDefinition]
    facts: List[FactDefinition]
    description: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class DimensionalModeler:
    """
    Comprehensive dimensional modeling system
    """
    
    def __init__(self, database_engine, metadata: MetaData):
        self.db_engine = database_engine
        self.metadata = metadata
        self.schemas: Dict[str, SchemaDefinition] = {}
        self.table_registry: Dict[str, Table] = {}
        
        # Standard dimension templates
        self.dimension_templates = {
            'date': self._create_date_dimension_template(),
            'time': self._create_time_dimension_template(),
            'geography': self._create_geography_dimension_template(),
            'customer': self._create_customer_dimension_template(),
            'product': self._create_product_dimension_template()
        }
    
    async def create_star_schema(self, 
                                schema_name: str,
                                fact_definition: FactDefinition,
                                dimension_definitions: List[DimensionDefinition]) -> SchemaDefinition:
        """
        Create star schema with fact table and dimensions
        
        Args:
            schema_name: Name of the schema
            fact_definition: Fact table definition
            dimension_definitions: List of dimension definitions
        
        Returns:
            Created schema definition
        """
        schema_def = SchemaDefinition(
            name=schema_name,
            schema_type=SchemaType.STAR,
            dimensions=dimension_definitions,
            facts=[fact_definition],
            description=f"Star schema for {fact_definition.name}"
        )
        
        # Validate schema
        await self._validate_schema(schema_def)
        
        # Create dimension tables
        for dim_def in dimension_definitions:
            await self._create_dimension_table(dim_def)
        
        # Create fact table
        await self._create_fact_table(fact_definition, dimension_definitions)
        
        # Store schema
        self.schemas[schema_name] = schema_def
        
        return schema_def
    
    async def create_snowflake_schema(self, 
                                    schema_name: str,
                                    fact_definition: FactDefinition,
                                    dimension_hierarchy: Dict[str, List[str]]) -> SchemaDefinition:
        """
        Create snowflake schema with normalized dimensions
        
        Args:
            schema_name: Name of the schema
            fact_definition: Fact table definition
            dimension_hierarchy: Dictionary mapping parent dimensions to child dimensions
        
        Returns:
            Created schema definition
        """
        # Build dimension definitions with hierarchy
        all_dimensions = []
        
        for parent_dim, child_dims in dimension_hierarchy.items():
            # Create parent dimension
            parent_def = self._create_parent_dimension(parent_dim)
            all_dimensions.append(parent_def)
            
            # Create child dimensions
            for child_dim in child_dims:
                child_def = self._create_child_dimension(child_dim, parent_dim)
                all_dimensions.append(child_def)
        
        schema_def = SchemaDefinition(
            name=schema_name,
            schema_type=SchemaType.SNOWFLAKE,
            dimensions=all_dimensions,
            facts=[fact_definition],
            description=f"Snowflake schema for {fact_definition.name}"
        )
        
        # Validate and create
        await self._validate_schema(schema_def)
        
        # Create tables in dependency order
        await self._create_snowflake_tables(schema_def)
        
        self.schemas[schema_name] = schema_def
        
        return schema_def
    
    async def create_dimension_table(self, dimension_def: DimensionDefinition) -> Table:
        """
        Create dimension table based on definition
        
        Args:
            dimension_def: Dimension definition
        
        Returns:
            Created SQLAlchemy Table object
        """
        return await self._create_dimension_table(dimension_def)
    
    async def implement_scd_type_2(self, 
                                  dimension_name: str,
                                  source_data: pd.DataFrame,
                                  natural_key_columns: List[str]) -> Dict[str, Any]:
        """
        Implement Type 2 Slowly Changing Dimension
        
        Args:
            dimension_name: Name of dimension table
            source_data: Source data to process
            natural_key_columns: Natural key columns for tracking changes
        
        Returns:
            Processing result with statistics
        """
        result = {
            'processed_rows': 0,
            'new_records': 0,
            'updated_records': 0,
            'unchanged_records': 0,
            'start_time': datetime.utcnow()
        }
        
        # Get current dimension data
        current_query = f"SELECT * FROM {dimension_name} WHERE is_current = true"
        current_df = pd.read_sql(current_query, self.db_engine)
        
        # Add SCD Type 2 columns to source data if not present
        if 'effective_date' not in source_data.columns:
            source_data['effective_date'] = datetime.utcnow().date()
        if 'end_date' not in source_data.columns:
            source_data['end_date'] = date(9999, 12, 31)
        if 'is_current' not in source_data.columns:
            source_data['is_current'] = True
        if 'version' not in source_data.columns:
            source_data['version'] = 1
        
        # Process each record
        for _, source_record in source_data.iterrows():
            result['processed_rows'] += 1
            
            # Find current record by natural key
            natural_key_filter = ' AND '.join([
                f"{col} = '{source_record[col]}'" for col in natural_key_columns
            ])
            current_records = current_df.query(natural_key_filter) if not current_df.empty else pd.DataFrame()
            
            if current_records.empty:
                # New record
                await self._insert_new_scd_record(dimension_name, source_record)
                result['new_records'] += 1
            else:
                current_record = current_records.iloc[0]
                
                # Check if record has changed
                if self._has_record_changed(current_record, source_record, natural_key_columns):
                    # Close current record
                    await self._close_current_scd_record(dimension_name, current_record['surrogate_key'])
                    
                    # Insert new version
                    source_record['version'] = current_record['version'] + 1
                    await self._insert_new_scd_record(dimension_name, source_record)
                    result['updated_records'] += 1
                else:
                    result['unchanged_records'] += 1
        
        result['end_time'] = datetime.utcnow()
        result['duration_seconds'] = (result['end_time'] - result['start_time']).total_seconds()
        
        return result
    
    async def create_aggregate_table(self, 
                                   base_fact_table: str,
                                   aggregation_config: Dict[str, Any]) -> str:
        """
        Create aggregate table for improved query performance
        
        Args:
            base_fact_table: Base fact table name
            aggregation_config: Aggregation configuration
        
        Returns:
            Name of created aggregate table
        """
        agg_table_name = f"{base_fact_table}_agg_{aggregation_config['grain']}"
        
        # Build aggregation query
        group_by_columns = aggregation_config['group_by_columns']
        measures = aggregation_config['measures']
        
        group_by_clause = ', '.join(group_by_columns)
        measure_clauses = []
        
        for measure_name, measure_config in measures.items():
            agg_function = measure_config.get('function', 'SUM')
            source_column = measure_config.get('source_column', measure_name)
            measure_clauses.append(f"{agg_function}({source_column}) as {measure_name}")
        
        measure_clause = ', '.join(measure_clauses)
        
        # Create aggregate table
        create_query = f"""
        CREATE TABLE {agg_table_name} AS
        SELECT 
            {group_by_clause},
            {measure_clause},
            COUNT(*) as record_count,
            CURRENT_TIMESTAMP as created_at
        FROM {base_fact_table}
        GROUP BY {group_by_clause}
        """
        
        async with self.db_engine.begin() as conn:
            await conn.execute(text(create_query))
        
        # Create indexes for performance
        await self._create_aggregate_indexes(agg_table_name, group_by_columns)
        
        return agg_table_name
    
    async def create_time_dimension(self, 
                                  start_date: date,
                                  end_date: date,
                                  table_name: str = "dim_time") -> Table:
        """
        Create comprehensive time dimension table
        
        Args:
            start_date: Start date for time dimension
            end_date: End date for time dimension
            table_name: Name of time dimension table
        
        Returns:
            Created time dimension table
        """
        # Generate time dimension data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        time_data = []
        for date_val in date_range:
            time_record = {
                'date_key': int(date_val.strftime('%Y%m%d')),
                'full_date': date_val.date(),
                'day_of_week': date_val.day_name(),
                'day_of_month': date_val.day,
                'day_of_year': date_val.dayofyear,
                'week_of_year': date_val.isocalendar()[1],
                'month': date_val.month,
                'month_name': date_val.month_name(),
                'quarter': f"Q{(date_val.month - 1) // 3 + 1}",
                'year': date_val.year,
                'is_weekend': date_val.weekday() >= 5,
                'is_holiday': self._is_holiday(date_val),
                'fiscal_year': self._get_fiscal_year(date_val),
                'fiscal_quarter': self._get_fiscal_quarter(date_val)
            }
            time_data.append(time_record)
        
        # Create table and insert data
        time_df = pd.DataFrame(time_data)
        time_df.to_sql(table_name, self.db_engine, if_exists='replace', index=False)
        
        # Create indexes
        await self._create_time_dimension_indexes(table_name)
        
        return self.metadata.tables[table_name]
    
    def _create_date_dimension_template(self) -> DimensionDefinition:
        """Create standard date dimension template"""
        attributes = [
            DimensionAttribute("date_key", "INTEGER", is_key=True),
            DimensionAttribute("full_date", "DATE", is_natural_key=True),
            DimensionAttribute("day_of_week", "VARCHAR(10)"),
            DimensionAttribute("day_of_month", "INTEGER"),
            DimensionAttribute("day_of_year", "INTEGER"),
            DimensionAttribute("week_of_year", "INTEGER"),
            DimensionAttribute("month", "INTEGER"),
            DimensionAttribute("month_name", "VARCHAR(10)"),
            DimensionAttribute("quarter", "VARCHAR(2)"),
            DimensionAttribute("year", "INTEGER"),
            DimensionAttribute("is_weekend", "BOOLEAN"),
            DimensionAttribute("is_holiday", "BOOLEAN"),
            DimensionAttribute("fiscal_year", "INTEGER"),
            DimensionAttribute("fiscal_quarter", "VARCHAR(2)")
        ]
        
        return DimensionDefinition(
            name="dim_date",
            dimension_type=DimensionType.TIME,
            attributes=attributes,
            description="Standard date dimension",
            natural_key_columns=["full_date"]
        )
    
    def _create_time_dimension_template(self) -> DimensionDefinition:
        """Create standard time dimension template"""
        attributes = [
            DimensionAttribute("time_key", "INTEGER", is_key=True),
            DimensionAttribute("hour", "INTEGER"),
            DimensionAttribute("minute", "INTEGER"),
            DimensionAttribute("second", "INTEGER"),
            DimensionAttribute("time_of_day", "VARCHAR(20)"),
            DimensionAttribute("hour_12", "INTEGER"),
            DimensionAttribute("am_pm", "VARCHAR(2)"),
            DimensionAttribute("time_period", "VARCHAR(20)")
        ]
        
        return DimensionDefinition(
            name="dim_time",
            dimension_type=DimensionType.TIME,
            attributes=attributes,
            description="Standard time dimension",
            natural_key_columns=["hour", "minute", "second"]
        )
    
    def _create_geography_dimension_template(self) -> DimensionDefinition:
        """Create standard geography dimension template"""
        attributes = [
            DimensionAttribute("geography_key", "INTEGER", is_key=True),
            DimensionAttribute("country_code", "VARCHAR(3)", is_natural_key=True),
            DimensionAttribute("country_name", "VARCHAR(100)"),
            DimensionAttribute("region", "VARCHAR(50)"),
            DimensionAttribute("state_province", "VARCHAR(100)"),
            DimensionAttribute("city", "VARCHAR(100)"),
            DimensionAttribute("postal_code", "VARCHAR(20)"),
            DimensionAttribute("latitude", "DECIMAL(10,8)"),
            DimensionAttribute("longitude", "DECIMAL(11,8)"),
            DimensionAttribute("time_zone", "VARCHAR(50)")
        ]
        
        return DimensionDefinition(
            name="dim_geography",
            dimension_type=DimensionType.STANDARD,
            attributes=attributes,
            description="Standard geography dimension",
            natural_key_columns=["country_code", "state_province", "city"]
        )
    
    def _create_customer_dimension_template(self) -> DimensionDefinition:
        """Create customer dimension template with SCD Type 2"""
        attributes = [
            DimensionAttribute("customer_key", "INTEGER", is_key=True),
            DimensionAttribute("customer_id", "VARCHAR(50)", is_natural_key=True),
            DimensionAttribute("customer_name", "VARCHAR(200)", scd_type=SCDType.TYPE_2),
            DimensionAttribute("customer_type", "VARCHAR(50)", scd_type=SCDType.TYPE_2),
            DimensionAttribute("industry", "VARCHAR(100)", scd_type=SCDType.TYPE_2),
            DimensionAttribute("registration_date", "DATE", scd_type=SCDType.TYPE_0),
            DimensionAttribute("effective_date", "DATE"),
            DimensionAttribute("end_date", "DATE"),
            DimensionAttribute("is_current", "BOOLEAN"),
            DimensionAttribute("version", "INTEGER")
        ]
        
        return DimensionDefinition(
            name="dim_customer",
            dimension_type=DimensionType.SLOWLY_CHANGING_TYPE_2,
            attributes=attributes,
            description="Customer dimension with SCD Type 2",
            natural_key_columns=["customer_id"]
        )
    
    def _create_product_dimension_template(self) -> DimensionDefinition:
        """Create product dimension template"""
        attributes = [
            DimensionAttribute("product_key", "INTEGER", is_key=True),
            DimensionAttribute("product_id", "VARCHAR(50)", is_natural_key=True),
            DimensionAttribute("product_name", "VARCHAR(200)"),
            DimensionAttribute("category", "VARCHAR(100)"),
            DimensionAttribute("subcategory", "VARCHAR(100)"),
            DimensionAttribute("brand", "VARCHAR(100)"),
            DimensionAttribute("unit_price", "DECIMAL(10,2)"),
            DimensionAttribute("launch_date", "DATE"),
            DimensionAttribute("status", "VARCHAR(20)")
        ]
        
        return DimensionDefinition(
            name="dim_product",
            dimension_type=DimensionType.STANDARD,
            attributes=attributes,
            description="Standard product dimension",
            natural_key_columns=["product_id"]
        )
    
    async def _create_dimension_table(self, dimension_def: DimensionDefinition) -> Table:
        """Create dimension table from definition"""
        columns = []
        
        for attr in dimension_def.attributes:
            # Map data types
            if attr.data_type.upper() == "INTEGER":
                col_type = Integer
            elif attr.data_type.upper().startswith("VARCHAR"):
                length = int(attr.data_type.split('(')[1].rstrip(')')) if '(' in attr.data_type else 255
                col_type = String(length)
            elif attr.data_type.upper() == "BOOLEAN":
                col_type = Boolean
            elif attr.data_type.upper().startswith("DECIMAL"):
                col_type = Numeric
            elif attr.data_type.upper() == "DATE":
                col_type = DateTime
            else:
                col_type = String(255)
            
            column = Column(
                attr.name,
                col_type,
                primary_key=attr.is_key,
                nullable=attr.nullable,
                default=attr.default_value
            )
            columns.append(column)
        
        table = Table(dimension_def.name, self.metadata, *columns)
        
        # Create table
        table.create(self.db_engine, checkfirst=True)
        
        # Store in registry
        self.table_registry[dimension_def.name] = table
        
        return table
    
    async def _create_fact_table(self, 
                                fact_def: FactDefinition,
                                dimension_defs: List[DimensionDefinition]) -> Table:
        """Create fact table with foreign keys to dimensions"""
        columns = []
        
        # Add dimension foreign keys
        for dim_def in dimension_defs:
            # Find the primary key column
            pk_column = next((attr for attr in dim_def.attributes if attr.is_key), None)
            if pk_column:
                fk_column = Column(
                    f"{dim_def.name}_key",
                    Integer,
                    ForeignKey(f"{dim_def.name}.{pk_column.name}"),
                    nullable=False
                )
                columns.append(fk_column)
        
        # Add measures
        for measure in fact_def.measures:
            if measure.data_type.upper() == "INTEGER":
                col_type = Integer
            elif measure.data_type.upper().startswith("DECIMAL"):
                col_type = Numeric
            else:
                col_type = Numeric
            
            column = Column(
                measure.name,
                col_type,
                nullable=measure.nullable
            )
            columns.append(column)
        
        # Add audit columns
        columns.extend([
            Column("created_at", DateTime, default=datetime.utcnow),
            Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        ])
        
        table = Table(fact_def.name, self.metadata, *columns)
        
        # Create table
        table.create(self.db_engine, checkfirst=True)
        
        # Create partition if specified
        if fact_def.partition_column:
            await self._create_fact_table_partitions(fact_def.name, fact_def.partition_column)
        
        self.table_registry[fact_def.name] = table
        
        return table
    
    async def _validate_schema(self, schema_def: SchemaDefinition):
        """Validate schema definition"""
        # Check that all fact dimensions exist
        for fact_def in schema_def.facts:
            for dim_name in fact_def.dimensions:
                if not any(dim.name == dim_name for dim in schema_def.dimensions):
                    raise ValueError(f"Dimension {dim_name} referenced in fact {fact_def.name} not found")
        
        # Validate dimension attributes
        for dim_def in schema_def.dimensions:
            # Check for primary key
            if not any(attr.is_key for attr in dim_def.attributes):
                raise ValueError(f"Dimension {dim_def.name} must have a primary key")
            
            # Validate SCD Type 2 requirements
            if dim_def.dimension_type == DimensionType.SLOWLY_CHANGING_TYPE_2:
                required_scd_columns = ["effective_date", "end_date", "is_current", "version"]
                existing_columns = [attr.name for attr in dim_def.attributes]
                missing_columns = [col for col in required_scd_columns if col not in existing_columns]
                if missing_columns:
                    raise ValueError(f"SCD Type 2 dimension {dim_def.name} missing columns: {missing_columns}")
    
    def _is_holiday(self, date_val: pd.Timestamp) -> bool:
        """Check if date is a holiday (basic implementation)"""
        # This would typically connect to a holiday service or database
        # For now, just check common holidays
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25)  # Christmas
        ]
        return (date_val.month, date_val.day) in holidays
    
    def _get_fiscal_year(self, date_val: pd.Timestamp) -> int:
        """Get fiscal year (assuming fiscal year starts in October)"""
        if date_val.month >= 10:
            return date_val.year + 1
        else:
            return date_val.year
    
    def _get_fiscal_quarter(self, date_val: pd.Timestamp) -> str:
        """Get fiscal quarter"""
        fiscal_month = (date_val.month - 10) % 12 + 1
        fiscal_quarter = (fiscal_month - 1) // 3 + 1
        return f"FQ{fiscal_quarter}"
    
    async def _insert_new_scd_record(self, table_name: str, record: pd.Series):
        """Insert new SCD Type 2 record"""
        # Generate surrogate key
        max_key_query = f"SELECT COALESCE(MAX(surrogate_key), 0) + 1 as next_key FROM {table_name}"
        
        async with self.db_engine.begin() as conn:
            result = await conn.execute(text(max_key_query))
            next_key = result.scalar()
        
        # Prepare insert data
        insert_data = record.to_dict()
        insert_data['surrogate_key'] = next_key
        
        # Insert record
        columns = ', '.join(insert_data.keys())
        placeholders = ', '.join([f":{key}" for key in insert_data.keys()])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        async with self.db_engine.begin() as conn:
            await conn.execute(text(insert_query), insert_data)
    
    async def _close_current_scd_record(self, table_name: str, surrogate_key: int):
        """Close current SCD Type 2 record"""
        update_query = f"""
        UPDATE {table_name} 
        SET end_date = CURRENT_DATE - 1, is_current = false 
        WHERE surrogate_key = :surrogate_key
        """
        
        async with self.db_engine.begin() as conn:
            await conn.execute(text(update_query), {"surrogate_key": surrogate_key})
    
    def _has_record_changed(self, 
                           current_record: pd.Series,
                           source_record: pd.Series,
                           natural_key_columns: List[str]) -> bool:
        """Check if record has changed (excluding natural keys and SCD columns)"""
        scd_columns = ["effective_date", "end_date", "is_current", "version", "surrogate_key"]
        exclude_columns = natural_key_columns + scd_columns
        
        for column in source_record.index:
            if column not in exclude_columns:
                if column in current_record.index:
                    if current_record[column] != source_record[column]:
                        return True
        
        return False
    
    async def _create_aggregate_indexes(self, table_name: str, columns: List[str]):
        """Create indexes for aggregate table"""
        for column in columns:
            index_name = f"idx_{table_name}_{column}"
            create_index_query = f"CREATE INDEX {index_name} ON {table_name} ({column})"
            
            async with self.db_engine.begin() as conn:
                await conn.execute(text(create_index_query))
    
    async def _create_time_dimension_indexes(self, table_name: str):
        """Create indexes for time dimension"""
        indexes = [
            ("idx_time_date_key", "date_key"),
            ("idx_time_full_date", "full_date"),
            ("idx_time_year_month", "year, month"),
            ("idx_time_year_quarter", "year, quarter")
        ]
        
        for index_name, columns in indexes:
            create_index_query = f"CREATE INDEX {index_name} ON {table_name} ({columns})"
            
            async with self.db_engine.begin() as conn:
                await conn.execute(text(create_index_query))
    
    async def _create_fact_table_partitions(self, table_name: str, partition_column: str):
        """Create partitions for fact table (PostgreSQL example)"""
        # This would be database-specific
        # Example for PostgreSQL
        partition_queries = [
            f"CREATE TABLE {table_name}_2023 PARTITION OF {table_name} FOR VALUES FROM ('2023-01-01') TO ('2024-01-01')",
            f"CREATE TABLE {table_name}_2024 PARTITION OF {table_name} FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')"
        ]
        
        async with self.db_engine.begin() as conn:
            for query in partition_queries:
                try:
                    await conn.execute(text(query))
                except Exception as e:
                    # Partition might already exist
                    pass
    
    def get_schema_definition(self, schema_name: str) -> Optional[SchemaDefinition]:
        """Get schema definition by name"""
        return self.schemas.get(schema_name)
    
    def list_schemas(self) -> List[str]:
        """List all schema names"""
        return list(self.schemas.keys())
    
    def get_dimension_template(self, template_name: str) -> Optional[DimensionDefinition]:
        """Get dimension template by name"""
        return self.dimension_templates.get(template_name)