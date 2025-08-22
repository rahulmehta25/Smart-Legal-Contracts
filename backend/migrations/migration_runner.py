#!/usr/bin/env python3
"""
Database Migration Runner for Arbitration Detection RAG System
Handles schema migrations with rollback support and integrity checks
"""

import os
import sys
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Custom exception for migration errors"""
    pass


class DatabaseMigrator:
    """Database migration runner with rollback support"""
    
    def __init__(self, database_path: str, migrations_dir: str = None):
        self.database_path = database_path
        self.migrations_dir = migrations_dir or os.path.dirname(__file__)
        self.connection = None
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        
        logger.info(f"Initialized migrator for database: {database_path}")
        logger.info(f"Migrations directory: {self.migrations_dir}")
    
    def connect(self) -> sqlite3.Connection:
        """Create database connection with optimizations"""
        if self.connection is None:
            self.connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                timeout=30.0
            )
            
            # Enable optimizations
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")
            self.connection.execute("PRAGMA cache_size = 10000")
            self.connection.execute("PRAGMA temp_store = memory")
            
            logger.info("Database connection established with optimizations")
        
        return self.connection
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def _ensure_migrations_table(self):
        """Ensure schema_migrations table exists"""
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                checksum VARCHAR(64),
                execution_time_ms INTEGER
            )
        """)
        self.connection.commit()
        logger.debug("Migration table ensured")
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate MD5 checksum of migration content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_applied_migrations(self) -> List[int]:
        """Get list of applied migration versions"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [row[0] for row in cursor.fetchall()]
    
    def _get_available_migrations(self) -> List[Tuple[int, str, str]]:
        """Get list of available migration files"""
        migrations = []
        migrations_path = Path(self.migrations_dir)
        
        for file_path in migrations_path.glob("*.sql"):
            if file_path.name.startswith(('.', '_')):
                continue
                
            # Extract version number from filename (e.g., 001_initial_schema.sql)
            try:
                version_str = file_path.stem.split('_')[0]
                version = int(version_str)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                migrations.append((version, str(file_path), content))
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid migration file {file_path}: {e}")
        
        # Sort by version number
        migrations.sort(key=lambda x: x[0])
        return migrations
    
    def _validate_migration_integrity(self, version: int, content: str) -> bool:
        """Validate migration integrity by checking stored checksum"""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT checksum FROM schema_migrations WHERE version = ?",
            (version,)
        )
        
        result = cursor.fetchone()
        if result:
            stored_checksum = result[0]
            current_checksum = self._calculate_checksum(content)
            
            if stored_checksum != current_checksum:
                logger.error(f"Migration {version} checksum mismatch!")
                logger.error(f"Stored: {stored_checksum}, Current: {current_checksum}")
                return False
        
        return True
    
    def _execute_migration(self, version: int, content: str, description: str = "") -> bool:
        """Execute a single migration"""
        logger.info(f"Applying migration {version}: {description}")
        
        start_time = datetime.now()
        cursor = self.connection.cursor()
        
        try:
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Split migration into individual statements
            statements = [stmt.strip() for stmt in content.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement:
                    cursor.execute(statement)
            
            # Record migration
            checksum = self._calculate_checksum(content)
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            cursor.execute("""
                INSERT OR REPLACE INTO schema_migrations 
                (version, description, checksum, execution_time_ms)
                VALUES (?, ?, ?, ?)
            """, (version, description, checksum, execution_time))
            
            # Commit transaction
            self.connection.commit()
            
            logger.info(f"Migration {version} applied successfully in {execution_time}ms")
            return True
            
        except Exception as e:
            # Rollback on error
            self.connection.rollback()
            logger.error(f"Failed to apply migration {version}: {e}")
            raise MigrationError(f"Migration {version} failed: {e}")
    
    def _rollback_migration(self, version: int) -> bool:
        """Rollback a migration (basic implementation)"""
        logger.warning(f"Rolling back migration {version}")
        
        cursor = self.connection.cursor()
        
        try:
            # Remove migration record
            cursor.execute("DELETE FROM schema_migrations WHERE version = ?", (version,))
            self.connection.commit()
            
            logger.warning(f"Migration {version} rollback completed")
            logger.warning("Note: Data changes are not automatically reverted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            return False
    
    def migrate(self, target_version: Optional[int] = None, dry_run: bool = False) -> bool:
        """Run database migrations"""
        try:
            self._ensure_migrations_table()
            
            applied_migrations = self._get_applied_migrations()
            available_migrations = self._get_available_migrations()
            
            if not available_migrations:
                logger.info("No migration files found")
                return True
            
            logger.info(f"Applied migrations: {applied_migrations}")
            logger.info(f"Available migrations: {[m[0] for m in available_migrations]}")
            
            # Validate applied migrations
            for version, file_path, content in available_migrations:
                if version in applied_migrations:
                    if not self._validate_migration_integrity(version, content):
                        raise MigrationError(f"Migration {version} integrity check failed")
            
            # Determine migrations to run
            migrations_to_run = []
            for version, file_path, content in available_migrations:
                if version not in applied_migrations:
                    if target_version is None or version <= target_version:
                        # Extract description from first comment line
                        lines = content.split('\n')
                        description = ""
                        for line in lines:
                            if line.strip().startswith('-- Description:'):
                                description = line.split(':', 1)[1].strip()
                                break
                        
                        migrations_to_run.append((version, content, description))
            
            if not migrations_to_run:
                logger.info("No pending migrations to run")
                return True
            
            logger.info(f"Migrations to run: {[m[0] for m in migrations_to_run]}")
            
            if dry_run:
                logger.info("DRY RUN - No actual migrations will be executed")
                for version, content, description in migrations_to_run:
                    logger.info(f"Would apply migration {version}: {description}")
                return True
            
            # Execute migrations
            for version, content, description in migrations_to_run:
                if not self._execute_migration(version, content, description):
                    return False
            
            logger.info("All migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def rollback(self, target_version: int) -> bool:
        """Rollback to a specific migration version"""
        try:
            self._ensure_migrations_table()
            
            applied_migrations = self._get_applied_migrations()
            
            # Find migrations to rollback (in reverse order)
            to_rollback = [v for v in applied_migrations if v > target_version]
            to_rollback.sort(reverse=True)
            
            if not to_rollback:
                logger.info(f"Already at or below version {target_version}")
                return True
            
            logger.warning(f"Rolling back migrations: {to_rollback}")
            
            for version in to_rollback:
                if not self._rollback_migration(version):
                    return False
            
            logger.info(f"Rollback to version {target_version} completed")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def status(self) -> Dict[str, Any]:
        """Get migration status"""
        try:
            self._ensure_migrations_table()
            
            applied_migrations = self._get_applied_migrations()
            available_migrations = self._get_available_migrations()
            
            pending_migrations = []
            for version, file_path, content in available_migrations:
                if version not in applied_migrations:
                    pending_migrations.append(version)
            
            # Get migration details
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT version, applied_at, description, execution_time_ms 
                FROM schema_migrations 
                ORDER BY version
            """)
            
            migration_details = []
            for row in cursor.fetchall():
                migration_details.append({
                    'version': row[0],
                    'applied_at': row[1],
                    'description': row[2],
                    'execution_time_ms': row[3]
                })
            
            return {
                'current_version': max(applied_migrations) if applied_migrations else 0,
                'latest_available': max([m[0] for m in available_migrations]) if available_migrations else 0,
                'applied_migrations': applied_migrations,
                'pending_migrations': pending_migrations,
                'migration_details': migration_details
            }
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {}
    
    def reset(self, confirm: bool = False) -> bool:
        """Reset database (drop all tables)"""
        if not confirm:
            logger.error("Reset requires explicit confirmation")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Get all table names
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            logger.warning(f"Dropping {len(tables)} tables")
            
            # Drop all tables
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            
            self.connection.commit()
            
            logger.warning("Database reset completed")
            return True
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False


def main():
    """Command line interface for migration runner"""
    parser = argparse.ArgumentParser(description="Database Migration Runner")
    parser.add_argument("--database", "-d", required=True, help="Database file path")
    parser.add_argument("--migrations-dir", "-m", help="Migrations directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument("--target", "-t", type=int, help="Target migration version")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("target", type=int, help="Target migration version")
    
    # Status command
    subparsers.add_parser("status", help="Show migration status")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset database")
    reset_parser.add_argument("--confirm", action="store_true", help="Confirm reset")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        with DatabaseMigrator(args.database, args.migrations_dir) as migrator:
            if args.command == "migrate":
                success = migrator.migrate(args.target, args.dry_run)
            elif args.command == "rollback":
                success = migrator.rollback(args.target)
            elif args.command == "status":
                status = migrator.status()
                print("\nMigration Status:")
                print(f"Current Version: {status.get('current_version', 0)}")
                print(f"Latest Available: {status.get('latest_available', 0)}")
                print(f"Applied: {status.get('applied_migrations', [])}")
                print(f"Pending: {status.get('pending_migrations', [])}")
                success = True
            elif args.command == "reset":
                success = migrator.reset(args.confirm)
            else:
                print(f"Unknown command: {args.command}")
                return 1
            
            return 0 if success else 1
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())