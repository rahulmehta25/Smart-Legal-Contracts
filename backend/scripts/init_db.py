#!/usr/bin/env python3
"""
Database initialization script for the arbitration detection system.
Creates all tables, indexes, and inserts initial data.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import bcrypt

# Import models after path setup
try:
    from backend.app.models import (
        BaseModel, User, Organization, Document, Chunk, 
        Analysis, Detection, Pattern, configure_mappers
    )
    from backend.app.db.database import get_database_url, create_database_engine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Database initialization and setup."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database setup.
        
        Args:
            database_url: Optional database URL (defaults to environment)
        """
        self.database_url = database_url or get_database_url()
        self.engine = None
        self.session_factory = None
    
    def create_engine(self) -> None:
        """Create database engine and session factory."""
        logger.info("Creating database engine...")
        self.engine = create_database_engine(self.database_url)
        self.session_factory = sessionmaker(bind=self.engine)
        logger.info("Database engine created successfully")
    
    def create_tables(self) -> None:
        """Create all database tables."""
        logger.info("Creating database tables...")
        
        try:
            # Configure all mappers first
            configure_mappers()
            
            # Create all tables
            BaseModel.metadata.create_all(self.engine)
            logger.info("All tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def create_extensions(self) -> None:
        """Create required PostgreSQL extensions."""
        logger.info("Creating PostgreSQL extensions...")
        
        extensions = [
            'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"',
            'CREATE EXTENSION IF NOT EXISTS "pg_trgm"',
            'CREATE EXTENSION IF NOT EXISTS "btree_gin"',
            # Note: vector extension requires separate installation
            # 'CREATE EXTENSION IF NOT EXISTS "vector"'
        ]
        
        with self.engine.connect() as conn:
            for ext in extensions:
                try:
                    conn.execute(text(ext))
                    conn.commit()
                    logger.info(f"Created extension: {ext}")
                except Exception as e:
                    logger.warning(f"Could not create extension {ext}: {e}")
    
    def insert_sample_data(self) -> None:
        """Insert sample data for development and testing."""
        logger.info("Inserting sample data...")
        
        with self.session_factory() as session:
            try:
                # Create demo organization
                demo_org = self._create_demo_organization(session)
                
                # Create demo users
                admin_user, demo_user = self._create_demo_users(session, demo_org.id)
                
                # Create sample patterns
                self._create_sample_patterns(session, admin_user.id, demo_org.id)
                
                # Create sample documents and analyses
                self._create_sample_documents(session, demo_user.id, demo_org.id)
                
                session.commit()
                logger.info("Sample data inserted successfully")
                
            except IntegrityError as e:
                session.rollback()
                logger.warning(f"Sample data may already exist: {e}")
            except Exception as e:
                session.rollback()
                logger.error(f"Error inserting sample data: {e}")
                raise
    
    def _create_demo_organization(self, session) -> Organization:
        """Create demo organization."""
        logger.info("Creating demo organization...")
        
        org = Organization(
            name="Demo Organization",
            slug="demo-org",
            description="Default organization for demo and testing purposes",
            subscription_tier="enterprise",
            max_documents=1000,
            max_users=100,
            max_storage_gb=100
        )
        
        session.add(org)
        session.flush()
        return org
    
    def _create_demo_users(self, session, org_id: str) -> tuple[User, User]:
        """Create demo users."""
        logger.info("Creating demo users...")
        
        # Admin user
        admin_user = User(
            email="admin@demo.com",
            username="admin",
            first_name="Demo",
            last_name="Admin",
            role="admin",
            organization_id=org_id,
            is_active=True,
            is_verified=True
        )
        admin_user.set_password("admin123")
        session.add(admin_user)
        
        # Regular user
        demo_user = User(
            email="user@demo.com", 
            username="demo_user",
            first_name="Demo",
            last_name="User",
            role="user",
            organization_id=org_id,
            is_active=True,
            is_verified=True
        )
        demo_user.set_password("demo123")
        session.add(demo_user)
        
        session.flush()
        return admin_user, demo_user
    
    def _create_sample_patterns(self, session, creator_id: str, org_id: str) -> None:
        """Create sample arbitration detection patterns."""
        logger.info("Creating sample patterns...")
        
        patterns_data = [
            {
                'pattern_name': 'Mandatory Arbitration Clause',
                'pattern_text': r'any\s+(claim|dispute|controversy).*arbitration.*binding',
                'pattern_type': 'regex',
                'category': 'mandatory_arbitration',
                'subcategory': 'general',
                'effectiveness_score': 0.9,
                'accuracy_score': 0.85,
                'description': 'Detects mandatory arbitration clauses in legal documents',
                'examples': [
                    'Any claim or dispute shall be resolved by binding arbitration',
                    'All controversies must be submitted to binding arbitration'
                ]
            },
            {
                'pattern_name': 'Class Action Waiver',
                'pattern_text': r'waive.*class\s+action|no\s+class\s+action|class\s+action.*waived',
                'pattern_type': 'regex',
                'category': 'class_waiver',
                'subcategory': 'general',
                'effectiveness_score': 0.8,
                'accuracy_score': 0.82,
                'description': 'Identifies class action waiver provisions',
                'examples': [
                    'You waive any right to participate in a class action',
                    'No class action lawsuits are permitted'
                ]
            },
            {
                'pattern_name': 'JAMS Arbitration Provider',
                'pattern_text': r'JAMS.*arbitration|arbitration.*JAMS|Judicial\s+Arbitration.*Mediation',
                'pattern_type': 'regex',
                'category': 'arbitration_provider',
                'subcategory': 'jams',
                'effectiveness_score': 0.85,
                'accuracy_score': 0.90,
                'description': 'Identifies JAMS as the arbitration service provider',
                'examples': [
                    'Arbitration will be conducted under JAMS rules',
                    'Disputes administered by JAMS'
                ]
            },
            {
                'pattern_name': 'American Arbitration Association',
                'pattern_text': r'American\s+Arbitration\s+Association|AAA.*arbitration|arbitration.*AAA',
                'pattern_type': 'regex',
                'category': 'arbitration_provider',
                'subcategory': 'aaa',
                'effectiveness_score': 0.85,
                'accuracy_score': 0.88,
                'description': 'Identifies AAA as the arbitration service provider',
                'examples': [
                    'Under American Arbitration Association Commercial Rules',
                    'AAA arbitration procedures shall apply'
                ]
            },
            {
                'pattern_name': 'Jury Trial Waiver',
                'pattern_text': r'waive.*jury\s+trial|jury\s+trial.*waived|right\s+to.*jury.*waived',
                'pattern_type': 'regex',
                'category': 'jury_waiver',
                'subcategory': 'general',
                'effectiveness_score': 0.85,
                'accuracy_score': 0.80,
                'description': 'Detects jury trial waiver clauses',
                'examples': [
                    'Both parties waive their right to jury trial',
                    'Jury trial rights are hereby waived'
                ]
            }
        ]
        
        for pattern_data in patterns_data:
            pattern = Pattern(
                created_by=creator_id,
                organization_id=None,  # System-wide patterns
                **pattern_data
            )
            session.add(pattern)
        
        session.flush()
    
    def _create_sample_documents(self, session, user_id: str, org_id: str) -> None:
        """Create sample documents for testing."""
        logger.info("Creating sample documents...")
        
        # Sample document content with arbitration clauses
        sample_content = """
        TERMS OF SERVICE AGREEMENT
        
        1. DISPUTE RESOLUTION
        Any claim, dispute, or controversy arising under or relating to this Agreement
        shall be resolved exclusively through binding arbitration administered by the
        American Arbitration Association (AAA) under its Commercial Arbitration Rules.
        
        2. CLASS ACTION WAIVER  
        You agree that you will resolve any claim you have with us only on an
        individual basis and waive any right to participate in any class action lawsuit
        or class-wide arbitration.
        
        3. JURY TRIAL WAIVER
        Both parties hereby waive their right to trial by jury in any proceeding
        arising out of or related to this Agreement.
        """
        
        # Create sample document
        doc = Document(
            user_id=user_id,
            organization_id=org_id,
            filename="sample_terms_of_service.txt",
            original_filename="Terms of Service.txt",
            file_path="/tmp/sample_terms.txt",
            file_type="text/plain",
            file_size=len(sample_content),
            content_hash="sample_hash_123",
            processing_status="completed",
            document_type="terms_of_service",
            language="en",
            tags=["sample", "terms", "arbitration"]
        )
        session.add(doc)
        session.flush()
        
        # Create sample chunk
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            content=sample_content,
            content_length=len(sample_content),
            chunk_hash="chunk_hash_123",
            page_number=1,
            section_title="Terms of Service",
            language="en",
            tokens=200
        )
        session.add(chunk)
        session.flush()
        
        # Create sample analysis
        analysis = Analysis(
            document_id=doc.id,
            user_id=user_id,
            organization_id=org_id,
            status="completed",
            overall_score=0.9,
            risk_level="high",
            summary="Document contains multiple arbitration clauses including mandatory arbitration, class action waiver, and jury trial waiver.",
            recommendations="Review arbitration provisions carefully as they may limit legal remedies available to users."
        )
        session.add(analysis)
        session.flush()
        
        # Create sample detections
        detections_data = [
            {
                'detection_type': 'mandatory_arbitration',
                'confidence_score': 0.95,
                'severity': 'high',
                'matched_text': 'Any claim, dispute, or controversy arising under or relating to this Agreement shall be resolved exclusively through binding arbitration',
                'detection_method': 'regex'
            },
            {
                'detection_type': 'class_waiver',
                'confidence_score': 0.88,
                'severity': 'high', 
                'matched_text': 'waive any right to participate in any class action lawsuit or class-wide arbitration',
                'detection_method': 'regex'
            },
            {
                'detection_type': 'jury_waiver',
                'confidence_score': 0.92,
                'severity': 'medium',
                'matched_text': 'Both parties hereby waive their right to trial by jury',
                'detection_method': 'regex'
            }
        ]
        
        for det_data in detections_data:
            detection = Detection(
                analysis_id=analysis.id,
                chunk_id=chunk.id,
                document_id=doc.id,
                **det_data
            )
            session.add(detection)
        
        session.flush()
    
    def run_sql_schema(self) -> None:
        """Execute the SQL schema file for additional setup."""
        schema_path = project_root / "backend" / "app" / "db" / "schema.sql"
        
        if not schema_path.exists():
            logger.warning(f"Schema file not found: {schema_path}")
            return
        
        logger.info("Executing SQL schema file...")
        
        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Split by statements (simple approach)
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            with self.engine.connect() as conn:
                for stmt in statements:
                    if stmt and not stmt.startswith('--'):
                        try:
                            conn.execute(text(stmt))
                            conn.commit()
                        except Exception as e:
                            logger.warning(f"Could not execute statement: {e}")
                            continue
            
            logger.info("SQL schema executed successfully")
            
        except Exception as e:
            logger.error(f"Error executing SQL schema: {e}")
    
    def initialize_database(self, include_sample_data: bool = True) -> None:
        """
        Full database initialization.
        
        Args:
            include_sample_data: Whether to insert sample data
        """
        logger.info("Starting database initialization...")
        
        try:
            # Create engine
            self.create_engine()
            
            # Create extensions
            self.create_extensions()
            
            # Create tables
            self.create_tables()
            
            # Execute additional SQL schema
            # self.run_sql_schema()  # Commented out as it may conflict with SQLAlchemy
            
            # Insert sample data
            if include_sample_data:
                self.insert_sample_data()
            
            logger.info("Database initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        finally:
            if self.engine:
                self.engine.dispose()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize arbitration detection database')
    parser.add_argument('--no-sample-data', action='store_true',
                       help='Skip inserting sample data')
    parser.add_argument('--database-url', 
                       help='Database URL (defaults to environment)')
    args = parser.parse_args()
    
    # Initialize database
    initializer = DatabaseInitializer(args.database_url)
    initializer.initialize_database(include_sample_data=not args.no_sample_data)


if __name__ == '__main__':
    main()