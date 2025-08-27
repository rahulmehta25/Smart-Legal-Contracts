# PostgreSQL Database Setup Guide

This guide covers the complete setup and usage of the production-ready PostgreSQL database system for the arbitration detection platform.

## Quick Start

### 1. Start Database Services

```bash
# Start PostgreSQL, Redis, and Adminer
cd demo/
docker-compose -f docker-compose.db.yml up -d

# Check service status
docker-compose -f docker-compose.db.yml ps
```

### 2. Initialize Database

```bash
# Initialize database with sample data
cd ../backend/
python scripts/init_db.py

# Or without sample data
python scripts/init_db.py --no-sample-data
```

### 3. Access Database

**Adminer Web Interface:**
- URL: http://localhost:8080
- System: PostgreSQL
- Server: postgres
- Username: arbitration_user
- Password: arbitration_secure_pass_2024
- Database: arbitration_db

**Direct PostgreSQL Connection:**
```bash
psql -h localhost -p 5432 -U arbitration_user -d arbitration_db
# Password: arbitration_secure_pass_2024
```

## Architecture Overview

### Database Components

1. **PostgreSQL 15** - Primary database with advanced features
2. **Redis** - Caching layer for performance optimization
3. **Adminer** - Web-based database administration
4. **Backup Service** - Automated database backups

### Schema Design

The database uses a multi-tenant architecture with the following core tables:

- **users** - User authentication and profiles
- **organizations** - Multi-tenant organization management  
- **documents** - File storage and metadata
- **chunks** - Document chunks with vector embeddings
- **analyses** - Analysis results and findings
- **detections** - Individual arbitration clause detections
- **patterns** - Detection patterns and rules
- **audit_logs** - Comprehensive audit trail
- **user_sessions** - Session management

## Configuration

### Environment Variables

```bash
# Database Connection
DATABASE_URL=postgresql://arbitration_user:arbitration_secure_pass_2024@localhost:5432/arbitration_db

# Connection Pool Settings
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Performance Settings  
DB_STATEMENT_TIMEOUT=30000
DB_LOCK_TIMEOUT=10000
DB_ECHO=false

# Redis Settings
REDIS_URL=redis://:redis_secure_pass_2024@localhost:6379/0
```

### Production Settings

For production deployment, update the passwords and configure SSL:

```yaml
# In docker-compose.db.yml
environment:
  POSTGRES_PASSWORD: your_secure_password_here
  POSTGRES_DB: your_db_name
```

## Database Models

### User Model Features

- Secure password hashing with bcrypt
- Multi-factor authentication support
- API key management
- Role-based access control (admin, user, analyst, viewer)
- Account lockout protection
- User preferences and settings

### Organization Model Features

- Multi-tenant isolation
- Subscription tier management
- Resource limits and quotas
- Usage tracking and analytics
- Custom settings per organization

### Document Model Features

- File metadata and storage tracking
- Version control and history
- Processing status management
- Content hash deduplication
- Multi-language support
- Retention policies

### Analysis Model Features

- Comprehensive analysis results
- Risk level assessment
- Processing status tracking
- Detailed findings storage
- Performance metrics

## Database Operations

### Initialization

```bash
# Full initialization with sample data
python scripts/init_db.py

# Production initialization (no sample data)
python scripts/init_db.py --no-sample-data

# Custom database URL
python scripts/init_db.py --database-url postgresql://user:pass@host:port/db
```

### Connection Management

```python
from backend.app.db.database import get_db, transaction_scope, health_check

# Using FastAPI dependency
from fastapi import Depends
from sqlalchemy.orm import Session

async def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

# Using transaction scope
with transaction_scope() as db:
    user = User.create(db, email="test@example.com", username="test")

# Health check
status = health_check()
print(f"Database status: {status['status']}")
```

### Multi-tenant Operations

```python
from backend.app.db.database import get_db_with_org

# Session with organization context
with transaction_scope(organization_id="org-uuid") as db:
    # All queries will be filtered by organization
    documents = db.query(Document).all()
```

## Security Features

### Row-Level Security (RLS)

The database uses PostgreSQL's RLS for multi-tenant data isolation:

```sql
-- Documents are automatically filtered by organization
SELECT * FROM documents; -- Only shows current org's documents
```

### Audit Logging

All sensitive operations are automatically logged:

```python
# Automatic audit logging for document operations
doc = Document.create(db, filename="test.pdf", user_id=user.id)
# Audit log entry created automatically
```

### Password Security

- bcrypt hashing with salt
- Configurable password complexity
- Account lockout after failed attempts
- Password history tracking

## Performance Optimization

### Indexing Strategy

The database includes 50+ optimized indexes:

- B-tree indexes for equality and range queries
- GIN indexes for JSON and array operations
- Full-text search indexes for content search
- Vector indexes for similarity search
- Composite indexes for common query patterns

### Connection Pooling

Production-ready connection pooling with:

- Configurable pool size and overflow
- Connection health checks
- Automatic connection recycling
- Pool statistics monitoring

### Caching Layer

Redis integration for:

- Query result caching
- Session storage
- Rate limiting
- Real-time features

## Monitoring and Maintenance

### Health Checks

```python
from backend.app.db.database import health_check, get_pool_statistics

# Database health
health = health_check()

# Connection pool stats
stats = get_pool_statistics()
```

### Backup and Recovery

Automated daily backups with 7-day retention:

```bash
# Manual backup
docker exec arbitration_postgres pg_dump -U arbitration_user arbitration_db > backup.sql

# Restore from backup
docker exec -i arbitration_postgres psql -U arbitration_user arbitration_db < backup.sql
```

### Log Monitoring

Monitor logs for:
- Connection pool exhaustion
- Long-running queries
- Authentication failures
- Error patterns

## API Integration

### FastAPI Dependencies

```python
from backend.app.db.database import get_db
from backend.app.models import User, Document, Analysis

@app.get("/users")
async def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.post("/documents")
async def create_document(
    doc_data: DocumentCreate,
    db: Session = Depends(get_db)
):
    return Document.create(db, **doc_data.dict())
```

### Error Handling

```python
from sqlalchemy.exc import IntegrityError

try:
    with transaction_scope() as db:
        user = User.create(db, email="duplicate@example.com")
except IntegrityError as e:
    # Handle duplicate email error
    raise HTTPException(400, "Email already exists")
```

## Troubleshooting

### Common Issues

**Connection refused:**
```bash
# Check if containers are running
docker-compose -f docker-compose.db.yml ps

# Check logs
docker-compose -f docker-compose.db.yml logs postgres
```

**Performance issues:**
```bash
# Monitor slow queries
docker exec arbitration_postgres psql -U arbitration_user -d arbitration_db -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

**Authentication errors:**
```bash
# Reset user password
docker exec arbitration_postgres psql -U arbitration_user -d arbitration_db -c "ALTER USER arbitration_user PASSWORD 'new_password';"
```

### Development vs Production

**Development:**
- Use provided passwords
- Enable SQL echo for debugging
- Use sample data for testing

**Production:**
- Change all default passwords
- Enable SSL/TLS connections
- Configure proper firewall rules
- Set up monitoring and alerting
- Regular backup testing

## Migration Guide

When upgrading the database schema:

1. Create migration script in `backend/migrations/`
2. Test migration on development copy
3. Backup production database
4. Apply migration during maintenance window
5. Verify data integrity

Example migration structure:
```python
def upgrade():
    """Apply migration changes."""
    pass

def downgrade():
    """Rollback migration changes."""
    pass
```

## Support and Documentation

For additional support:
- Check application logs in `logs/` directory
- Review PostgreSQL documentation
- Monitor system resources (CPU, memory, disk)
- Use database profiling tools for performance analysis

The database system is designed for enterprise-grade performance, security, and scalability. All components are production-ready and follow industry best practices.