# Contributing to Smart Legal Contracts

Thank you for your interest in contributing to Smart Legal Contracts! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Git Workflow](#git-workflow)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Architecture Guidelines](#architecture-guidelines)

## Development Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 18+
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/smart-legal-contracts.git
   cd smart-legal-contracts
   ```

2. **Start infrastructure services**
   ```bash
   docker-compose up -d postgres redis qdrant
   ```

3. **Backend setup**
   ```bash
   cd backend

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Run database migrations
   alembic upgrade head

   # Start development server
   uvicorn app.main:app --reload --port 8000
   ```

4. **Frontend setup**
   ```bash
   cd frontend

   # Install dependencies
   npm install

   # Start development server
   npm run dev
   ```

### Docker Development

For a fully containerized development experience:

```bash
docker-compose up -d
```

This starts all services with hot reload enabled for both backend and frontend.

## Code Style

### Python (Backend)

We follow PEP 8 and use the following tools:

- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking
- **isort** for import sorting

```bash
# Format code
black .

# Check linting
ruff check .

# Type checking
mypy app/

# Sort imports
isort .
```

Configuration is in `pyproject.toml`.

### TypeScript/JavaScript (Frontend)

- **ESLint** for linting
- **Prettier** for formatting

```bash
# Lint
npm run lint

# Format
npm run format

# Type check
npm run type-check
```

### General Guidelines

1. **Use meaningful names** - Variables, functions, and classes should have descriptive names
2. **Write self-documenting code** - Prefer clear code over comments
3. **Keep functions small** - Each function should do one thing well
4. **Avoid deep nesting** - Maximum 3 levels of indentation
5. **Use type hints** - All Python functions should have type annotations

## Git Workflow

### Branch Naming

```
feature/short-description
fix/issue-number-description
docs/what-you-documented
refactor/what-you-refactored
```

### Commit Messages

Use conventional commits format:

```
type: short description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat: add document batch upload endpoint
fix: resolve memory leak in embedding generation
docs: update API documentation for analysis endpoints
refactor: extract document processing into service class
```

### Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Write/update tests
4. Ensure all tests pass
5. Submit a pull request

## Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_documents.py

# Run with verbose output
pytest -v
```

### Frontend Tests

```bash
cd frontend

# Unit tests
npm test

# E2E tests
npm run cypress:open  # Interactive
npm run cypress:run   # Headless
```

### Test Guidelines

1. **Write tests for new features** - All new code should have tests
2. **Maintain test coverage** - Aim for >80% coverage
3. **Test edge cases** - Include tests for error conditions
4. **Use fixtures** - Reuse test data with pytest fixtures
5. **Mock external services** - Don't call real APIs in tests

## Pull Request Process

1. **Create a descriptive PR title** following conventional commits
2. **Fill out the PR template** with:
   - Summary of changes
   - Related issues
   - Testing performed
   - Screenshots (for UI changes)
3. **Ensure CI passes** - All checks must be green
4. **Request review** from at least one team member
5. **Address review feedback** promptly
6. **Squash and merge** when approved

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated if needed
- [ ] No secrets or sensitive data committed
- [ ] Breaking changes documented
- [ ] Database migrations included (if applicable)

## Architecture Guidelines

### Backend

- **API endpoints** go in `app/api/`
- **Business logic** goes in `app/services/`
- **Database models** go in `app/db/models.py`
- **AI/ML code** goes in `app/ai/`
- **Background tasks** go in `app/worker/tasks.py`
- **Caching logic** goes in `app/cache/`

### Frontend

- **Pages** go in `app/` (Next.js App Router)
- **Components** go in `src/components/`
- **Hooks** go in `src/hooks/`
- **API calls** go in `src/services/`
- **Types** go in `src/types/`

### Database

- Use Alembic for migrations
- Write idempotent migrations
- Add appropriate indexes
- Use transactions for multi-step operations

### Caching

- Cache expensive AI operations
- Set appropriate TTLs
- Include cache invalidation logic
- Use correlation IDs for tracing

### Async Processing

- Use Celery for long-running tasks
- Implement proper error handling and retries
- Log task progress
- Return meaningful results

## Questions?

If you have questions:

1. Check existing documentation
2. Search closed issues
3. Ask in the team Slack channel
4. Open a discussion on GitHub

Thank you for contributing!
