# Makefile for Smart Legal Contracts
# Usage: make <target>

.PHONY: help install install-dev test test-unit test-integration test-coverage \
        lint format type-check security-check build run clean docker-build \
        docker-run docker-test pre-commit

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
DOCKER_COMPOSE := docker compose
BACKEND_DIR := backend
FRONTEND_DIR := frontend

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#==============================================================================
# Help
#==============================================================================

help: ## Show this help message
	@echo "$(BLUE)Smart Legal Contracts - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Usage:$(NC) make [target]"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

#==============================================================================
# Installation
#==============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	cd $(FRONTEND_DIR) && npm install

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	cd $(BACKEND_DIR) && $(PIP) install -r requirements-test.txt
	cd $(FRONTEND_DIR) && npm install
	$(PIP) install pre-commit
	pre-commit install

#==============================================================================
# Testing
#==============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	cd $(BACKEND_DIR) && $(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	cd $(BACKEND_DIR) && $(PYTEST) tests/ \
		-v \
		--tb=short \
		-m "not integration and not slow and not e2e" \
		--ignore=tests/integration \
		--ignore=tests/load_tests

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	cd $(BACKEND_DIR) && $(PYTEST) tests/integration/ -v --tb=short -m "integration"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	cd $(BACKEND_DIR) && $(PYTEST) tests/ \
		--cov=app \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=30 \
		-m "not integration and not slow" \
		--ignore=tests/integration \
		--ignore=tests/load_tests
	@echo "$(GREEN)Coverage report generated at $(BACKEND_DIR)/htmlcov/index.html$(NC)"

test-fast: ## Run fast tests only (no slow, integration, or benchmark)
	@echo "$(BLUE)Running fast tests...$(NC)"
	cd $(BACKEND_DIR) && $(PYTEST) tests/ \
		-v \
		--tb=short \
		-m "not slow and not integration and not benchmark and not e2e" \
		--ignore=tests/integration \
		--ignore=tests/load_tests \
		-x

test-benchmark: ## Run benchmark tests
	@echo "$(BLUE)Running benchmark tests...$(NC)"
	cd $(BACKEND_DIR) && $(PYTEST) tests/ -v -m "benchmark"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	cd $(BACKEND_DIR) && ptw -- -v --tb=short -m "not slow and not integration"

#==============================================================================
# Linting and Formatting
#==============================================================================

lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(NC)"
	cd $(BACKEND_DIR) && ruff check . || true
	cd $(BACKEND_DIR) && flake8 app --max-line-length=120 --ignore=E501,W503,E203 || true
	cd $(FRONTEND_DIR) && npm run lint || true

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	cd $(BACKEND_DIR) && black app tests
	cd $(BACKEND_DIR) && isort app tests
	cd $(BACKEND_DIR) && ruff check . --fix || true

format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	cd $(BACKEND_DIR) && black --check --diff app tests
	cd $(BACKEND_DIR) && isort --check-only --diff app tests

type-check: ## Run type checking
	@echo "$(BLUE)Running type checker...$(NC)"
	cd $(BACKEND_DIR) && mypy app --ignore-missing-imports || true

#==============================================================================
# Security
#==============================================================================

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	cd $(BACKEND_DIR) && bandit -r app -ll || true
	cd $(BACKEND_DIR) && safety check -r requirements.txt || true

#==============================================================================
# Build
#==============================================================================

build: ## Build the project
	@echo "$(BLUE)Building project...$(NC)"
	cd $(FRONTEND_DIR) && npm run build
	@echo "$(GREEN)Build complete!$(NC)"

build-backend: ## Build backend Docker image
	@echo "$(BLUE)Building backend Docker image...$(NC)"
	cd $(BACKEND_DIR) && docker build -t smart-legal-contracts-backend:latest .

build-frontend: ## Build frontend
	@echo "$(BLUE)Building frontend...$(NC)"
	cd $(FRONTEND_DIR) && npm run build

#==============================================================================
# Run
#==============================================================================

run: ## Run the application locally
	@echo "$(BLUE)Starting application...$(NC)"
	cd $(BACKEND_DIR) && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-backend: ## Run backend only
	@echo "$(BLUE)Starting backend server...$(NC)"
	cd $(BACKEND_DIR) && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend: ## Run frontend only
	@echo "$(BLUE)Starting frontend server...$(NC)"
	cd $(FRONTEND_DIR) && npm run dev

#==============================================================================
# Docker
#==============================================================================

docker-build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

docker-run: ## Run with Docker Compose
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	$(DOCKER_COMPOSE) up -d

docker-stop: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	$(DOCKER_COMPOSE) down

docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-test: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.test.yml up --build --abort-on-container-exit

docker-clean: ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

#==============================================================================
# Database
#==============================================================================

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	cd $(BACKEND_DIR) && alembic upgrade head

db-rollback: ## Rollback last migration
	@echo "$(BLUE)Rolling back last migration...$(NC)"
	cd $(BACKEND_DIR) && alembic downgrade -1

db-reset: ## Reset database (dangerous!)
	@echo "$(RED)WARNING: This will reset the database!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	cd $(BACKEND_DIR) && alembic downgrade base && alembic upgrade head

#==============================================================================
# Pre-commit
#==============================================================================

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install

pre-commit-update: ## Update pre-commit hooks
	@echo "$(BLUE)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate

#==============================================================================
# Clean
#==============================================================================

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	rm -rf $(BACKEND_DIR)/htmlcov 2>/dev/null || true
	rm -rf $(BACKEND_DIR)/.coverage 2>/dev/null || true
	rm -rf $(FRONTEND_DIR)/node_modules/.cache 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean ## Clean everything including node_modules
	@echo "$(BLUE)Deep cleaning...$(NC)"
	rm -rf $(FRONTEND_DIR)/node_modules
	rm -rf $(FRONTEND_DIR)/dist
	rm -rf $(FRONTEND_DIR)/build
	@echo "$(GREEN)Deep clean complete!$(NC)"

#==============================================================================
# Documentation
#==============================================================================

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	cd $(BACKEND_DIR) && pdoc --html app -o docs/api

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd $(BACKEND_DIR)/docs && python -m http.server 8080

#==============================================================================
# Development Utilities
#==============================================================================

check: lint type-check test-fast ## Run all checks (lint, type-check, fast tests)
	@echo "$(GREEN)All checks passed!$(NC)"

ci: lint type-check test-coverage security-check ## Run full CI pipeline locally
	@echo "$(GREEN)CI pipeline complete!$(NC)"

setup: install-dev pre-commit-install ## Full development setup
	@echo "$(GREEN)Development environment setup complete!$(NC)"
