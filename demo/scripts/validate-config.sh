#!/bin/bash

# Configuration Validation Script
# Validates environment configuration files for completeness and security

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validation results
ERRORS=0
WARNINGS=0
PASSED=0

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASSED=$((PASSED + 1))
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [ENVIRONMENT_FILE]

Validate environment configuration files for completeness and security.

OPTIONS:
    -e, --environment ENV    Environment to validate (production, staging, development)
    -s, --security-only      Run only security checks
    -c, --completeness-only  Run only completeness checks
    -v, --verbose            Verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                                    # Validate all environment files
    $0 -e production                     # Validate production environment
    $0 --security-only .env.production   # Security checks only
    $0 -v .env.staging                   # Verbose validation

EOF
}

# Parse command line arguments
ENVIRONMENT=""
SECURITY_ONLY=false
COMPLETENESS_ONLY=false
VERBOSE=false
SPECIFIC_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -s|--security-only)
            SECURITY_ONLY=true
            shift
            ;;
        -c|--completeness-only)
            COMPLETENESS_ONLY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            SPECIFIC_FILE="$1"
            shift
            ;;
    esac
done

# Required variables for each environment
declare -A REQUIRED_VARS
REQUIRED_VARS[common]="ENVIRONMENT NODE_ENV POSTGRES_PASSWORD REDIS_PASSWORD JWT_SECRET"
REQUIRED_VARS[production]="DOMAIN_URL FRONTEND_URL BACKEND_URL OPENAI_API_KEY GRAFANA_PASSWORD"
REQUIRED_VARS[staging]="DOMAIN_URL FRONTEND_URL BACKEND_URL"

# Security-sensitive variables that should not have default values
SECURITY_VARS=(
    "POSTGRES_PASSWORD"
    "REDIS_PASSWORD"
    "JWT_SECRET"
    "SESSION_SECRET"
    "ENCRYPTION_KEY"
    "OPENAI_API_KEY"
    "GRAFANA_PASSWORD"
)

# Variables that should be strong passwords/keys
PASSWORD_VARS=(
    "POSTGRES_PASSWORD"
    "REDIS_PASSWORD"
    "JWT_SECRET"
    "SESSION_SECRET"
    "ENCRYPTION_KEY"
    "GRAFANA_PASSWORD"
)

# Check if variable exists and is not empty
check_variable_exists() {
    local var_name="$1"
    local env_file="$2"
    
    if grep -q "^${var_name}=" "$env_file"; then
        local value=$(grep "^${var_name}=" "$env_file" | cut -d'=' -f2- | sed 's/^["'\'']*//;s/["'\'']*$//')
        if [[ -n "$value" ]]; then
            [[ "$VERBOSE" == "true" ]] && print_pass "Variable $var_name is set"
            return 0
        else
            print_error "Variable $var_name is empty in $env_file"
            return 1
        fi
    else
        print_error "Variable $var_name is missing from $env_file"
        return 1
    fi
}

# Check if variable has a placeholder value
check_placeholder_value() {
    local var_name="$1"
    local env_file="$2"
    
    if grep -q "^${var_name}=" "$env_file"; then
        local value=$(grep "^${var_name}=" "$env_file" | cut -d'=' -f2- | sed 's/^["'\'']*//;s/["'\'']*$//')
        
        # Check for common placeholder patterns
        if [[ "$value" =~ ^(REPLACE_WITH_|your_.*_here|change_me|placeholder|example|test123|password123|secret123).*$ ]]; then
            print_error "Variable $var_name has placeholder value in $env_file: $value"
            return 1
        elif [[ "$value" =~ ^(admin|password|secret|123456|qwerty)$ ]]; then
            print_error "Variable $var_name has weak/default value in $env_file: $value"
            return 1
        fi
    fi
    
    return 0
}

# Check password strength
check_password_strength() {
    local var_name="$1"
    local env_file="$2"
    local min_length=12
    
    if grep -q "^${var_name}=" "$env_file"; then
        local value=$(grep "^${var_name}=" "$env_file" | cut -d'=' -f2- | sed 's/^["'\'']*//;s/["'\'']*$//')
        
        if [[ ${#value} -lt $min_length ]]; then
            print_warning "Variable $var_name is shorter than $min_length characters in $env_file"
            return 1
        fi
        
        # Check for complexity (at least 3 different character types)
        local complexity=0
        [[ "$value" =~ [a-z] ]] && complexity=$((complexity + 1))
        [[ "$value" =~ [A-Z] ]] && complexity=$((complexity + 1))
        [[ "$value" =~ [0-9] ]] && complexity=$((complexity + 1))
        [[ "$value" =~ [^a-zA-Z0-9] ]] && complexity=$((complexity + 1))
        
        if [[ $complexity -lt 3 ]]; then
            print_warning "Variable $var_name lacks complexity (should include uppercase, lowercase, numbers, special chars) in $env_file"
            return 1
        fi
        
        [[ "$VERBOSE" == "true" ]] && print_pass "Variable $var_name has good strength"
    fi
    
    return 0
}

# Check URL format
check_url_format() {
    local var_name="$1"
    local env_file="$2"
    
    if grep -q "^${var_name}=" "$env_file"; then
        local value=$(grep "^${var_name}=" "$env_file" | cut -d'=' -f2- | sed 's/^["'\'']*//;s/["'\'']*$//')
        
        if [[ ! "$value" =~ ^https?://[a-zA-Z0-9.-]+[a-zA-Z0-9](:[0-9]+)?(/.*)?$ ]]; then
            print_error "Variable $var_name has invalid URL format in $env_file: $value"
            return 1
        fi
        
        # Warn about HTTP in production
        if [[ "$value" =~ ^http:// && "$env_file" =~ production ]]; then
            print_warning "Variable $var_name uses HTTP instead of HTTPS in production: $value"
            return 1
        fi
        
        [[ "$VERBOSE" == "true" ]] && print_pass "Variable $var_name has valid URL format"
    fi
    
    return 0
}

# Check database URL format
check_database_url() {
    local var_name="$1"
    local env_file="$2"
    
    if grep -q "^${var_name}=" "$env_file"; then
        local value=$(grep "^${var_name}=" "$env_file" | cut -d'=' -f2- | sed 's/^["'\'']*//;s/["'\'']*$//')
        
        if [[ ! "$value" =~ ^postgresql://[^:]+:[^@]+@[^:]+:[0-9]+/[^/]+$ ]]; then
            print_error "Variable $var_name has invalid PostgreSQL URL format in $env_file"
            return 1
        fi
        
        [[ "$VERBOSE" == "true" ]] && print_pass "Variable $var_name has valid database URL format"
    fi
    
    return 0
}

# Check Redis URL format
check_redis_url() {
    local var_name="$1"
    local env_file="$2"
    
    if grep -q "^${var_name}=" "$env_file"; then
        local value=$(grep "^${var_name}=" "$env_file" | cut -d'=' -f2- | sed 's/^["'\'']*//;s/["'\'']*$//')
        
        if [[ ! "$value" =~ ^redis://:[^@]+@[^:]+:[0-9]+/[0-9]+$ ]]; then
            print_error "Variable $var_name has invalid Redis URL format in $env_file"
            return 1
        fi
        
        [[ "$VERBOSE" == "true" ]] && print_pass "Variable $var_name has valid Redis URL format"
    fi
    
    return 0
}

# Check for sensitive data in comments
check_sensitive_comments() {
    local env_file="$1"
    
    if grep -i "password\|secret\|key\|token" "$env_file" | grep -v "^#.*example\|^#.*placeholder\|^#.*your_" | grep "^#" > /dev/null; then
        print_warning "Found potentially sensitive data in comments in $env_file"
        if [[ "$VERBOSE" == "true" ]]; then
            grep -i "password\|secret\|key\|token" "$env_file" | grep "^#" | head -3
        fi
        return 1
    fi
    
    return 0
}

# Check file permissions
check_file_permissions() {
    local env_file="$1"
    
    local permissions=$(stat -f "%A" "$env_file" 2>/dev/null || stat -c "%a" "$env_file" 2>/dev/null)
    
    if [[ "$permissions" != "600" && "$permissions" != "644" ]]; then
        print_warning "File $env_file has permissive permissions: $permissions (should be 600 or 644)"
        return 1
    fi
    
    [[ "$VERBOSE" == "true" ]] && print_pass "File $env_file has appropriate permissions: $permissions"
    return 0
}

# Validate environment file
validate_env_file() {
    local env_file="$1"
    local env_type="$2"
    
    print_info "Validating $env_file ($env_type environment)"
    
    if [[ ! -f "$env_file" ]]; then
        print_error "Environment file not found: $env_file"
        return 1
    fi
    
    # File permissions check
    check_file_permissions "$env_file"
    
    # Completeness checks
    if [[ "$SECURITY_ONLY" != "true" ]]; then
        print_info "Running completeness checks..."
        
        # Check common required variables
        for var in ${REQUIRED_VARS[common]}; do
            check_variable_exists "$var" "$env_file"
        done
        
        # Check environment-specific variables
        if [[ -n "${REQUIRED_VARS[$env_type]:-}" ]]; then
            for var in ${REQUIRED_VARS[$env_type]}; do
                check_variable_exists "$var" "$env_file"
            done
        fi
        
        # URL format checks
        check_url_format "FRONTEND_URL" "$env_file"
        check_url_format "BACKEND_URL" "$env_file"
        check_database_url "DATABASE_URL" "$env_file"
        check_redis_url "REDIS_URL" "$env_file"
    fi
    
    # Security checks
    if [[ "$COMPLETENESS_ONLY" != "true" ]]; then
        print_info "Running security checks..."
        
        # Check for placeholder values
        for var in "${SECURITY_VARS[@]}"; do
            check_placeholder_value "$var" "$env_file"
        done
        
        # Check password strength
        for var in "${PASSWORD_VARS[@]}"; do
            check_password_strength "$var" "$env_file"
        done
        
        # Check for sensitive data in comments
        check_sensitive_comments "$env_file"
        
        # Environment-specific security checks
        case "$env_type" in
            "production")
                # Production should have DEBUG=false
                if grep -q "^DEBUG=true" "$env_file"; then
                    print_error "DEBUG is enabled in production environment"
                fi
                
                # Production should use HTTPS
                if grep -q "^FRONTEND_URL=http://" "$env_file"; then
                    print_error "Frontend URL uses HTTP in production"
                fi
                
                # Production should have secure cookies
                if ! grep -q "^SESSION_COOKIE_SECURE=true" "$env_file"; then
                    print_warning "Secure cookies not enforced in production"
                fi
                ;;
                
            "staging")
                # Staging can be more lenient but should still be secure
                if grep -q "^DEBUG=true" "$env_file"; then
                    print_warning "DEBUG is enabled in staging environment"
                fi
                ;;
        esac
    fi
    
    echo
}

# Generate security recommendations
generate_recommendations() {
    echo "=== SECURITY RECOMMENDATIONS ==="
    echo
    echo "1. Use strong, unique passwords for all services"
    echo "2. Rotate secrets regularly (every 90 days)"
    echo "3. Use environment-specific secrets (don't reuse across environments)"
    echo "4. Store secrets in a secure secret management system"
    echo "5. Limit file permissions to 600 for environment files"
    echo "6. Never commit real credentials to version control"
    echo "7. Use HTTPS for all public-facing URLs"
    echo "8. Enable security headers and secure cookie settings"
    echo "9. Regularly audit and validate your configuration"
    echo "10. Monitor for credential leaks and unauthorized access"
    echo
}

# Main validation process
main() {
    print_info "=== Environment Configuration Validation ==="
    echo
    
    # Determine which files to validate
    local files_to_validate=()
    
    if [[ -n "$SPECIFIC_FILE" ]]; then
        files_to_validate=("$SPECIFIC_FILE")
    elif [[ -n "$ENVIRONMENT" ]]; then
        case "$ENVIRONMENT" in
            "production")
                files_to_validate=("$PROJECT_ROOT/.env.production")
                ;;
            "staging")
                files_to_validate=("$PROJECT_ROOT/.env.staging")
                ;;
            "development")
                files_to_validate=("$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.development")
                ;;
            *)
                print_error "Unknown environment: $ENVIRONMENT"
                exit 1
                ;;
        esac
    else
        # Validate all found environment files
        for env_file in "$PROJECT_ROOT"/.env.production "$PROJECT_ROOT"/.env.staging "$PROJECT_ROOT"/.env; do
            [[ -f "$env_file" ]] && files_to_validate+=("$env_file")
        done
    fi
    
    # Validate each file
    for env_file in "${files_to_validate[@]}"; do
        if [[ -f "$env_file" ]]; then
            local env_type="development"
            if [[ "$env_file" =~ production ]]; then
                env_type="production"
            elif [[ "$env_file" =~ staging ]]; then
                env_type="staging"
            fi
            
            validate_env_file "$env_file" "$env_type"
        else
            print_error "Environment file not found: $env_file"
        fi
    done
    
    # Summary
    echo "=== VALIDATION SUMMARY ==="
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
    echo -e "${RED}Errors: $ERRORS${NC}"
    echo
    
    if [[ $ERRORS -gt 0 ]]; then
        echo -e "${RED}❌ Validation failed with $ERRORS error(s)${NC}"
        echo "Please fix the errors before deploying to production."
        echo
        generate_recommendations
        exit 1
    elif [[ $WARNINGS -gt 0 ]]; then
        echo -e "${YELLOW}⚠️ Validation completed with $WARNINGS warning(s)${NC}"
        echo "Consider addressing the warnings for better security."
        echo
        generate_recommendations
        exit 0
    else
        echo -e "${GREEN}✅ All validation checks passed!${NC}"
        echo
    fi
}

# Run main function
main