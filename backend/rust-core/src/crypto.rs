//! Cryptographic operations module with secure algorithms

use crate::{CoreError, CoreResult};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use blake3::Hasher as Blake3Hasher;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM, CHACHA20_POLY1305};
use ring::digest::{digest, SHA256, SHA384, SHA512};
use ring::hkdf::{Salt, Prk};
use ring::pbkdf2::{derive, PBKDF2_HMAC_SHA256};
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{Ed25519KeyPair, KeyPair, Signature, UnparsedPublicKey, ED25519};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock as AsyncRwLock;

/// Cryptographic configuration
#[derive(Debug, Clone)]
pub struct CryptoConfig {
    pub default_algorithm: EncryptionAlgorithm,
    pub key_derivation_rounds: u32,
    pub salt_length: usize,
    pub nonce_length: usize,
    pub enable_compression: bool,
    pub secure_memory: bool,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            default_algorithm: EncryptionAlgorithm::AES256GCM,
            key_derivation_rounds: 100_000,
            salt_length: 32,
            nonce_length: 12,
            enable_compression: false,
            secure_memory: true,
        }
    }
}

/// Supported encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES256GCM,
    ChaCha20Poly1305,
    AES128GCM,
}

/// Hash algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HashAlgorithm {
    SHA256,
    SHA384,
    SHA512,
    Blake3,
    Argon2id,
}

/// Digital signature algorithms
#[derive(Debug, Clone)]
pub enum SignatureAlgorithm {
    Ed25519,
    ECDSA,
    RSA,
}

/// Encrypted data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub algorithm: EncryptionAlgorithm,
    pub ciphertext: Vec<u8>,
    pub nonce: Vec<u8>,
    pub salt: Option<Vec<u8>>,
    pub metadata: EncryptionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    pub created_at: u64,
    pub key_derivation_rounds: u32,
    pub compressed: bool,
    pub version: u8,
}

/// Key derivation result
#[derive(Debug, Clone)]
pub struct DerivedKey {
    pub key: Vec<u8>,
    pub salt: Vec<u8>,
    pub algorithm: HashAlgorithm,
    pub rounds: u32,
}

/// Digital signature structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalSignature {
    pub algorithm: SignatureAlgorithm,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub message_hash: Vec<u8>,
    pub created_at: u64,
}

/// Key pair for asymmetric cryptography
#[derive(Debug, Clone)]
pub struct KeyPair {
    pub public_key: Vec<u8>,
    pub private_key: SecureBytes,
    pub algorithm: SignatureAlgorithm,
}

/// Secure byte storage that clears on drop
pub struct SecureBytes {
    data: Vec<u8>,
}

/// High-performance cryptographic engine
pub struct CryptoEngine {
    config: CryptoConfig,
    rng: Arc<SystemRandom>,
    key_cache: Arc<AsyncRwLock<HashMap<String, CachedKey>>>,
    signature_keys: Arc<AsyncRwLock<HashMap<String, KeyPair>>>,
}

#[derive(Debug, Clone)]
struct CachedKey {
    key: SecureBytes,
    created_at: SystemTime,
    algorithm: EncryptionAlgorithm,
    usage_count: u64,
}

impl CryptoEngine {
    /// Create new cryptographic engine
    pub fn new(config: CryptoConfig) -> Self {
        Self {
            config,
            rng: Arc::new(SystemRandom::new()),
            key_cache: Arc::new(AsyncRwLock::new(HashMap::new())),
            signature_keys: Arc::new(AsyncRwLock::new(HashMap::new())),
        }
    }

    /// Encrypt data with password-based encryption
    pub async fn encrypt_with_password(&self, data: &[u8], password: &str) -> CoreResult<EncryptedData> {
        let salt = self.generate_salt()?;
        let derived_key = self.derive_key_from_password(password, &salt, self.config.key_derivation_rounds)?;
        
        self.encrypt_with_key(data, &derived_key.key, Some(salt), self.config.default_algorithm.clone()).await
    }

    /// Decrypt data with password
    pub async fn decrypt_with_password(&self, encrypted_data: &EncryptedData, password: &str) -> CoreResult<Vec<u8>> {
        let salt = encrypted_data.salt.as_ref()
            .ok_or_else(|| CoreError::CryptoError("No salt found in encrypted data".to_string()))?;
        
        let derived_key = self.derive_key_from_password(password, salt, encrypted_data.metadata.key_derivation_rounds)?;
        
        self.decrypt_with_key(encrypted_data, &derived_key.key).await
    }

    /// Encrypt data with a raw key
    pub async fn encrypt_with_key(&self, data: &[u8], key: &[u8], salt: Option<Vec<u8>>, algorithm: EncryptionAlgorithm) -> CoreResult<EncryptedData> {
        let processed_data = if self.config.enable_compression {
            self.compress_data(data)?
        } else {
            data.to_vec()
        };

        let nonce = self.generate_nonce(algorithm.nonce_size())?;
        let ciphertext = match algorithm {
            EncryptionAlgorithm::AES256GCM => {
                self.encrypt_aes256_gcm(&processed_data, key, &nonce)?
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                self.encrypt_chacha20_poly1305(&processed_data, key, &nonce)?
            }
            EncryptionAlgorithm::AES128GCM => {
                self.encrypt_aes128_gcm(&processed_data, key, &nonce)?
            }
        };

        Ok(EncryptedData {
            algorithm,
            ciphertext,
            nonce,
            salt,
            metadata: EncryptionMetadata {
                created_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                key_derivation_rounds: self.config.key_derivation_rounds,
                compressed: self.config.enable_compression,
                version: 1,
            },
        })
    }

    /// Decrypt data with a raw key
    pub async fn decrypt_with_key(&self, encrypted_data: &EncryptedData, key: &[u8]) -> CoreResult<Vec<u8>> {
        let plaintext = match encrypted_data.algorithm {
            EncryptionAlgorithm::AES256GCM => {
                self.decrypt_aes256_gcm(&encrypted_data.ciphertext, key, &encrypted_data.nonce)?
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                self.decrypt_chacha20_poly1305(&encrypted_data.ciphertext, key, &encrypted_data.nonce)?
            }
            EncryptionAlgorithm::AES128GCM => {
                self.decrypt_aes128_gcm(&encrypted_data.ciphertext, key, &encrypted_data.nonce)?
            }
        };

        if encrypted_data.metadata.compressed {
            self.decompress_data(&plaintext)
        } else {
            Ok(plaintext)
        }
    }

    /// Generate cryptographic hash
    pub fn hash(&self, data: &[u8], algorithm: HashAlgorithm) -> CoreResult<Vec<u8>> {
        match algorithm {
            HashAlgorithm::SHA256 => {
                Ok(digest(&SHA256, data).as_ref().to_vec())
            }
            HashAlgorithm::SHA384 => {
                Ok(digest(&SHA384, data).as_ref().to_vec())
            }
            HashAlgorithm::SHA512 => {
                Ok(digest(&SHA512, data).as_ref().to_vec())
            }
            HashAlgorithm::Blake3 => {
                let mut hasher = Blake3Hasher::new();
                hasher.update(data);
                Ok(hasher.finalize().as_bytes().to_vec())
            }
            HashAlgorithm::Argon2id => {
                let salt = self.generate_salt()?;
                let argon2 = Argon2::default();
                let password_hash = argon2.hash_password(data, &salt)
                    .map_err(|e| CoreError::CryptoError(format!("Argon2 hashing failed: {}", e)))?;
                Ok(password_hash.hash.unwrap().as_bytes().to_vec())
            }
        }
    }

    /// Verify hash
    pub fn verify_hash(&self, data: &[u8], hash: &[u8], algorithm: HashAlgorithm) -> CoreResult<bool> {
        let computed_hash = self.hash(data, algorithm)?;
        Ok(computed_hash == hash)
    }

    /// Generate digital signature
    pub async fn sign(&self, data: &[u8], key_id: &str, algorithm: SignatureAlgorithm) -> CoreResult<DigitalSignature> {
        let message_hash = self.hash(data, HashAlgorithm::SHA256)?;
        
        let signature_keys = self.signature_keys.read().await;
        let key_pair = signature_keys.get(key_id)
            .ok_or_else(|| CoreError::CryptoError("Signing key not found".to_string()))?;

        let signature_bytes = match algorithm {
            SignatureAlgorithm::Ed25519 => {
                self.sign_ed25519(data, &key_pair.private_key.data)?
            }
            _ => {
                return Err(CoreError::CryptoError("Unsupported signature algorithm".to_string()));
            }
        };

        Ok(DigitalSignature {
            algorithm,
            signature: signature_bytes,
            public_key: key_pair.public_key.clone(),
            message_hash,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        })
    }

    /// Verify digital signature
    pub fn verify_signature(&self, data: &[u8], signature: &DigitalSignature) -> CoreResult<bool> {
        let message_hash = self.hash(data, HashAlgorithm::SHA256)?;
        
        if message_hash != signature.message_hash {
            return Ok(false);
        }

        match signature.algorithm {
            SignatureAlgorithm::Ed25519 => {
                self.verify_ed25519(data, &signature.signature, &signature.public_key)
            }
            _ => {
                Err(CoreError::CryptoError("Unsupported signature algorithm".to_string()))
            }
        }
    }

    /// Generate key pair for digital signatures
    pub async fn generate_key_pair(&self, algorithm: SignatureAlgorithm) -> CoreResult<String> {
        let (public_key, private_key) = match algorithm {
            SignatureAlgorithm::Ed25519 => {
                let rng = &*self.rng;
                let key_pair_doc = Ed25519KeyPair::generate_pkcs8(rng)
                    .map_err(|e| CoreError::CryptoError(format!("Key generation failed: {:?}", e)))?;
                
                let key_pair = Ed25519KeyPair::from_pkcs8(key_pair_doc.as_ref())
                    .map_err(|e| CoreError::CryptoError(format!("Key parsing failed: {:?}", e)))?;
                
                (key_pair.public_key().as_ref().to_vec(), key_pair_doc.as_ref().to_vec())
            }
            _ => {
                return Err(CoreError::CryptoError("Unsupported key generation algorithm".to_string()));
            }
        };

        let key_id = uuid::Uuid::new_v4().to_string();
        let key_pair_struct = KeyPair {
            public_key,
            private_key: SecureBytes::new(private_key),
            algorithm,
        };

        let mut signature_keys = self.signature_keys.write().await;
        signature_keys.insert(key_id.clone(), key_pair_struct);

        Ok(key_id)
    }

    /// Derive key from password using PBKDF2
    pub fn derive_key_from_password(&self, password: &str, salt: &[u8], rounds: u32) -> CoreResult<DerivedKey> {
        let mut key = vec![0u8; 32]; // 256-bit key
        
        derive(
            PBKDF2_HMAC_SHA256,
            std::num::NonZeroU32::new(rounds).unwrap(),
            salt,
            password.as_bytes(),
            &mut key,
        );

        Ok(DerivedKey {
            key,
            salt: salt.to_vec(),
            algorithm: HashAlgorithm::SHA256,
            rounds,
        })
    }

    /// Derive key using HKDF
    pub fn derive_key_hkdf(&self, input_key: &[u8], salt: Option<&[u8]>, info: &[u8], length: usize) -> CoreResult<Vec<u8>> {
        let salt = Salt::new(ring::hkdf::HKDF_SHA256, salt.unwrap_or(&[]));
        let prk = salt.extract(input_key);
        let okm = prk.expand(&[info], ring::hkdf::HKDF_SHA256)
            .map_err(|e| CoreError::CryptoError(format!("HKDF expansion failed: {:?}", e)))?;
        
        let mut output = vec![0u8; length];
        okm.fill(&mut output)
            .map_err(|e| CoreError::CryptoError(format!("HKDF fill failed: {:?}", e)))?;
        
        Ok(output)
    }

    /// Generate secure random bytes
    pub fn generate_random_bytes(&self, length: usize) -> CoreResult<Vec<u8>> {
        let mut bytes = vec![0u8; length];
        self.rng.fill(&mut bytes)
            .map_err(|e| CoreError::CryptoError(format!("Random generation failed: {:?}", e)))?;
        Ok(bytes)
    }

    /// Generate salt for key derivation
    fn generate_salt(&self) -> CoreResult<Vec<u8>> {
        self.generate_random_bytes(self.config.salt_length)
    }

    /// Generate nonce for encryption
    fn generate_nonce(&self, length: usize) -> CoreResult<Vec<u8>> {
        self.generate_random_bytes(length)
    }

    /// AES-256-GCM encryption
    fn encrypt_aes256_gcm(&self, data: &[u8], key: &[u8], nonce: &[u8]) -> CoreResult<Vec<u8>> {
        let unbound_key = UnboundKey::new(&AES_256_GCM, key)
            .map_err(|e| CoreError::CryptoError(format!("Key creation failed: {:?}", e)))?;
        let less_safe_key = LessSafeKey::new(unbound_key);
        
        let nonce = Nonce::try_assume_unique_for_key(nonce)
            .map_err(|e| CoreError::CryptoError(format!("Invalid nonce: {:?}", e)))?;
        
        let mut in_out = data.to_vec();
        less_safe_key.seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
            .map_err(|e| CoreError::CryptoError(format!("Encryption failed: {:?}", e)))?;
        
        Ok(in_out)
    }

    /// AES-256-GCM decryption
    fn decrypt_aes256_gcm(&self, ciphertext: &[u8], key: &[u8], nonce: &[u8]) -> CoreResult<Vec<u8>> {
        let unbound_key = UnboundKey::new(&AES_256_GCM, key)
            .map_err(|e| CoreError::CryptoError(format!("Key creation failed: {:?}", e)))?;
        let less_safe_key = LessSafeKey::new(unbound_key);
        
        let nonce = Nonce::try_assume_unique_for_key(nonce)
            .map_err(|e| CoreError::CryptoError(format!("Invalid nonce: {:?}", e)))?;
        
        let mut in_out = ciphertext.to_vec();
        let plaintext = less_safe_key.open_in_place(nonce, Aad::empty(), &mut in_out)
            .map_err(|e| CoreError::CryptoError(format!("Decryption failed: {:?}", e)))?;
        
        Ok(plaintext.to_vec())
    }

    /// ChaCha20-Poly1305 encryption
    fn encrypt_chacha20_poly1305(&self, data: &[u8], key: &[u8], nonce: &[u8]) -> CoreResult<Vec<u8>> {
        let unbound_key = UnboundKey::new(&CHACHA20_POLY1305, key)
            .map_err(|e| CoreError::CryptoError(format!("Key creation failed: {:?}", e)))?;
        let less_safe_key = LessSafeKey::new(unbound_key);
        
        let nonce = Nonce::try_assume_unique_for_key(nonce)
            .map_err(|e| CoreError::CryptoError(format!("Invalid nonce: {:?}", e)))?;
        
        let mut in_out = data.to_vec();
        less_safe_key.seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
            .map_err(|e| CoreError::CryptoError(format!("Encryption failed: {:?}", e)))?;
        
        Ok(in_out)
    }

    /// ChaCha20-Poly1305 decryption
    fn decrypt_chacha20_poly1305(&self, ciphertext: &[u8], key: &[u8], nonce: &[u8]) -> CoreResult<Vec<u8>> {
        let unbound_key = UnboundKey::new(&CHACHA20_POLY1305, key)
            .map_err(|e| CoreError::CryptoError(format!("Key creation failed: {:?}", e)))?;
        let less_safe_key = LessSafeKey::new(unbound_key);
        
        let nonce = Nonce::try_assume_unique_for_key(nonce)
            .map_err(|e| CoreError::CryptoError(format!("Invalid nonce: {:?}", e)))?;
        
        let mut in_out = ciphertext.to_vec();
        let plaintext = less_safe_key.open_in_place(nonce, Aad::empty(), &mut in_out)
            .map_err(|e| CoreError::CryptoError(format!("Decryption failed: {:?}", e)))?;
        
        Ok(plaintext.to_vec())
    }

    /// AES-128-GCM encryption (placeholder - would use ring's AES_128_GCM)
    fn encrypt_aes128_gcm(&self, data: &[u8], key: &[u8], nonce: &[u8]) -> CoreResult<Vec<u8>> {
        // For demonstration - would implement with ring's AES_128_GCM
        self.encrypt_aes256_gcm(data, key, nonce)
    }

    /// AES-128-GCM decryption (placeholder)
    fn decrypt_aes128_gcm(&self, ciphertext: &[u8], key: &[u8], nonce: &[u8]) -> CoreResult<Vec<u8>> {
        // For demonstration - would implement with ring's AES_128_GCM
        self.decrypt_aes256_gcm(ciphertext, key, nonce)
    }

    /// Ed25519 signing
    fn sign_ed25519(&self, data: &[u8], private_key: &[u8]) -> CoreResult<Vec<u8>> {
        let key_pair = Ed25519KeyPair::from_pkcs8(private_key)
            .map_err(|e| CoreError::CryptoError(format!("Invalid private key: {:?}", e)))?;
        
        let signature = key_pair.sign(data);
        Ok(signature.as_ref().to_vec())
    }

    /// Ed25519 signature verification
    fn verify_ed25519(&self, data: &[u8], signature: &[u8], public_key: &[u8]) -> CoreResult<bool> {
        let public_key = UnparsedPublicKey::new(&ED25519, public_key);
        
        match public_key.verify(data, signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Compress data before encryption
    fn compress_data(&self, data: &[u8]) -> CoreResult<Vec<u8>> {
        // Simplified compression using flate2
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)
            .map_err(|e| CoreError::CryptoError(format!("Compression failed: {}", e)))?;
        encoder.finish()
            .map_err(|e| CoreError::CryptoError(format!("Compression finalization failed: {}", e)))
    }

    /// Decompress data after decryption
    fn decompress_data(&self, compressed_data: &[u8]) -> CoreResult<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;
        
        let mut decoder = GzDecoder::new(compressed_data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| CoreError::CryptoError(format!("Decompression failed: {}", e)))?;
        Ok(decompressed)
    }

    /// Memory-safe key derivation with timing attack resistance
    pub fn secure_key_derivation(&self, password: &str, salt: &[u8], algorithm: HashAlgorithm) -> CoreResult<DerivedKey> {
        match algorithm {
            HashAlgorithm::Argon2id => {
                let argon2 = Argon2::default();
                let password_hash = argon2.hash_password(password.as_bytes(), salt)
                    .map_err(|e| CoreError::CryptoError(format!("Argon2 derivation failed: {}", e)))?;
                
                Ok(DerivedKey {
                    key: password_hash.hash.unwrap().as_bytes().to_vec(),
                    salt: salt.to_vec(),
                    algorithm,
                    rounds: 0, // Argon2 doesn't use traditional rounds
                })
            }
            _ => {
                self.derive_key_from_password(password, salt, self.config.key_derivation_rounds)
            }
        }
    }

    /// Constant-time comparison for security
    pub fn constant_time_compare(&self, a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        let mut result = 0u8;
        for i in 0..a.len() {
            result |= a[i] ^ b[i];
        }
        result == 0
    }
}

impl EncryptionAlgorithm {
    /// Get nonce size for algorithm
    fn nonce_size(&self) -> usize {
        match self {
            EncryptionAlgorithm::AES256GCM | EncryptionAlgorithm::AES128GCM => 12,
            EncryptionAlgorithm::ChaCha20Poly1305 => 12,
        }
    }

    /// Get key size for algorithm
    fn key_size(&self) -> usize {
        match self {
            EncryptionAlgorithm::AES256GCM => 32,
            EncryptionAlgorithm::AES128GCM => 16,
            EncryptionAlgorithm::ChaCha20Poly1305 => 32,
        }
    }
}

impl SecureBytes {
    /// Create new secure bytes container
    fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Get reference to data (use carefully)
    fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

impl Drop for SecureBytes {
    /// Securely clear memory on drop
    fn drop(&mut self) {
        // Zero out memory
        for byte in &mut self.data {
            *byte = 0;
        }
    }
}

impl Clone for SecureBytes {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

/// Utility functions for common crypto operations
pub mod utils {
    use super::*;

    /// Quick encrypt with password
    pub async fn quick_encrypt(data: &[u8], password: &str) -> CoreResult<EncryptedData> {
        let engine = CryptoEngine::new(CryptoConfig::default());
        engine.encrypt_with_password(data, password).await
    }

    /// Quick decrypt with password
    pub async fn quick_decrypt(encrypted_data: &EncryptedData, password: &str) -> CoreResult<Vec<u8>> {
        let engine = CryptoEngine::new(CryptoConfig::default());
        engine.decrypt_with_password(encrypted_data, password).await
    }

    /// Quick hash
    pub fn quick_hash(data: &[u8]) -> CoreResult<Vec<u8>> {
        let engine = CryptoEngine::new(CryptoConfig::default());
        engine.hash(data, HashAlgorithm::SHA256)
    }

    /// Generate secure password
    pub fn generate_password(length: usize, include_special: bool) -> CoreResult<String> {
        let engine = CryptoEngine::new(CryptoConfig::default());
        let bytes = engine.generate_random_bytes(length * 2)?; // Extra bytes for filtering
        
        let charset = if include_special {
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*"
        } else {
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        };
        
        let password: String = bytes
            .into_iter()
            .map(|b| charset.chars().nth((b as usize) % charset.len()).unwrap())
            .take(length)
            .collect();
        
        Ok(password)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_encryption_decryption() {
        let engine = CryptoEngine::new(CryptoConfig::default());
        let data = b"Hello, World!";
        let password = "test_password";
        
        let encrypted = engine.encrypt_with_password(data, password).await.unwrap();
        let decrypted = engine.decrypt_with_password(&encrypted, password).await.unwrap();
        
        assert_eq!(data, decrypted.as_slice());
    }
    
    #[test]
    fn test_hashing() {
        let engine = CryptoEngine::new(CryptoConfig::default());
        let data = b"test data";
        
        let hash1 = engine.hash(data, HashAlgorithm::SHA256).unwrap();
        let hash2 = engine.hash(data, HashAlgorithm::SHA256).unwrap();
        
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 32); // SHA256 produces 32-byte hash
    }
    
    #[tokio::test]
    async fn test_digital_signatures() {
        let engine = CryptoEngine::new(CryptoConfig::default());
        let data = b"document to sign";
        
        let key_id = engine.generate_key_pair(SignatureAlgorithm::Ed25519).await.unwrap();
        let signature = engine.sign(data, &key_id, SignatureAlgorithm::Ed25519).await.unwrap();
        let is_valid = engine.verify_signature(data, &signature).unwrap();
        
        assert!(is_valid);
    }
    
    #[test]
    fn test_key_derivation() {
        let engine = CryptoEngine::new(CryptoConfig::default());
        let password = "test_password";
        let salt = engine.generate_random_bytes(32).unwrap();
        
        let key1 = engine.derive_key_from_password(password, &salt, 1000).unwrap();
        let key2 = engine.derive_key_from_password(password, &salt, 1000).unwrap();
        
        assert_eq!(key1.key, key2.key);
        assert_eq!(key1.key.len(), 32);
    }
    
    #[test]
    fn test_secure_bytes() {
        let data = vec![1, 2, 3, 4, 5];
        let secure = SecureBytes::new(data.clone());
        assert_eq!(secure.as_slice(), &data);
        
        // SecureBytes should zero memory on drop
        drop(secure);
    }
    
    #[test]
    fn test_constant_time_compare() {
        let engine = CryptoEngine::new(CryptoConfig::default());
        let a = b"same";
        let b = b"same";
        let c = b"diff";
        
        assert!(engine.constant_time_compare(a, b));
        assert!(!engine.constant_time_compare(a, c));
    }
}