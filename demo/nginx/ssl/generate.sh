#!/bin/bash

# SSL Certificate Generation Script
# Generates self-signed certificates for development and production setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SSL_DIR="$SCRIPT_DIR"
DOMAIN="${1:-arbitration-detector.com}"
API_DOMAIN="api.${DOMAIN}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if OpenSSL is available
if ! command -v openssl &> /dev/null; then
    print_error "OpenSSL is not installed. Please install OpenSSL first."
    exit 1
fi

print_info "Generating SSL certificates for domain: $DOMAIN"

# Create SSL directory if it doesn't exist
mkdir -p "$SSL_DIR"

# Generate private key for CA
print_info "Generating CA private key..."
openssl genrsa -out "$SSL_DIR/ca.key" 4096

# Generate CA certificate
print_info "Generating CA certificate..."
openssl req -new -x509 -days 3650 -key "$SSL_DIR/ca.key" -out "$SSL_DIR/ca.crt" -subj "/C=US/ST=CA/L=San Francisco/O=Arbitration Detector/OU=IT Department/CN=Arbitration Detector CA"

# Generate private key for main domain
print_info "Generating private key for $DOMAIN..."
openssl genrsa -out "$SSL_DIR/app.key" 2048

# Generate private key for API domain
print_info "Generating private key for $API_DOMAIN..."
openssl genrsa -out "$SSL_DIR/api.key" 2048

# Generate private key for default/catch-all
print_info "Generating private key for default certificate..."
openssl genrsa -out "$SSL_DIR/default.key" 2048

# Create certificate signing request for main domain
print_info "Creating CSR for $DOMAIN..."
openssl req -new -key "$SSL_DIR/app.key" -out "$SSL_DIR/app.csr" -subj "/C=US/ST=CA/L=San Francisco/O=Arbitration Detector/OU=IT Department/CN=$DOMAIN"

# Create certificate signing request for API domain
print_info "Creating CSR for $API_DOMAIN..."
openssl req -new -key "$SSL_DIR/api.key" -out "$SSL_DIR/api.csr" -subj "/C=US/ST=CA/L=San Francisco/O=Arbitration Detector/OU=IT Department/CN=$API_DOMAIN"

# Create certificate signing request for default
print_info "Creating CSR for default certificate..."
openssl req -new -key "$SSL_DIR/default.key" -out "$SSL_DIR/default.csr" -subj "/C=US/ST=CA/L=San Francisco/O=Arbitration Detector/OU=IT Department/CN=localhost"

# Create extension file for main domain (with SAN)
cat > "$SSL_DIR/app.ext" << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = www.$DOMAIN
DNS.3 = app.$DOMAIN
DNS.4 = localhost
IP.1 = 127.0.0.1
EOF

# Create extension file for API domain
cat > "$SSL_DIR/api.ext" << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $API_DOMAIN
DNS.2 = api.localhost
DNS.3 = localhost
IP.1 = 127.0.0.1
EOF

# Create extension file for default certificate
cat > "$SSL_DIR/default.ext" << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
EOF

# Generate signed certificate for main domain
print_info "Generating signed certificate for $DOMAIN..."
openssl x509 -req -in "$SSL_DIR/app.csr" -CA "$SSL_DIR/ca.crt" -CAkey "$SSL_DIR/ca.key" -CAcreateserial -out "$SSL_DIR/app.crt" -days 365 -extensions v3_ext -extfile "$SSL_DIR/app.ext"

# Generate signed certificate for API domain
print_info "Generating signed certificate for $API_DOMAIN..."
openssl x509 -req -in "$SSL_DIR/api.csr" -CA "$SSL_DIR/ca.crt" -CAkey "$SSL_DIR/ca.key" -CAcreateserial -out "$SSL_DIR/api.crt" -days 365 -extensions v3_ext -extfile "$SSL_DIR/api.ext"

# Generate signed certificate for default
print_info "Generating signed certificate for default..."
openssl x509 -req -in "$SSL_DIR/default.csr" -CA "$SSL_DIR/ca.crt" -CAkey "$SSL_DIR/ca.key" -CAcreateserial -out "$SSL_DIR/default.crt" -days 365 -extensions v3_ext -extfile "$SSL_DIR/default.ext"

# Create certificate chains
print_info "Creating certificate chains..."
cat "$SSL_DIR/app.crt" "$SSL_DIR/ca.crt" > "$SSL_DIR/app-chain.crt"
cat "$SSL_DIR/api.crt" "$SSL_DIR/ca.crt" > "$SSL_DIR/api-chain.crt"

# Generate DH parameters for enhanced security
print_info "Generating Diffie-Hellman parameters (this may take a while)..."
openssl dhparam -out "$SSL_DIR/dhparam.pem" 2048

# Set appropriate permissions
chmod 600 "$SSL_DIR"/*.key
chmod 644 "$SSL_DIR"/*.crt "$SSL_DIR"/*.pem

# Clean up temporary files
rm -f "$SSL_DIR"/*.csr "$SSL_DIR"/*.ext "$SSL_DIR"/ca.srl

print_info "SSL certificates generated successfully!"
print_info "Files created:"
echo "  - CA Certificate: $SSL_DIR/ca.crt"
echo "  - App Certificate: $SSL_DIR/app.crt"
echo "  - App Private Key: $SSL_DIR/app.key"
echo "  - App Certificate Chain: $SSL_DIR/app-chain.crt"
echo "  - API Certificate: $SSL_DIR/api.crt"
echo "  - API Private Key: $SSL_DIR/api.key"
echo "  - API Certificate Chain: $SSL_DIR/api-chain.crt"
echo "  - Default Certificate: $SSL_DIR/default.crt"
echo "  - Default Private Key: $SSL_DIR/default.key"
echo "  - DH Parameters: $SSL_DIR/dhparam.pem"

print_warning "These are self-signed certificates suitable for development and testing."
print_warning "For production, please use certificates from a trusted Certificate Authority like Let's Encrypt."

print_info "To trust the CA certificate on your system:"
case "$OSTYPE" in
  linux-gnu*)
    echo "  sudo cp $SSL_DIR/ca.crt /usr/local/share/ca-certificates/arbitration-detector-ca.crt"
    echo "  sudo update-ca-certificates"
    ;;
  darwin*)
    echo "  sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain $SSL_DIR/ca.crt"
    ;;
  *)
    echo "  Please refer to your operating system documentation for installing CA certificates."
    ;;
esac

print_info "Certificate generation completed!"