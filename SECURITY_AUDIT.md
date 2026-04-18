# Security Audit Report

**Project:** Smart Legal Contracts
**Date:** 2026-03-13
**Auditor:** Automated Security Review
**Scope:** Document upload, AI services, vector stores, authentication, infrastructure

---

## Executive Summary

This security audit identified **4 Critical**, **8 High**, **9 Medium**, and **5 Low** severity issues across the codebase. The most urgent concerns involve hardcoded secrets in configuration files, missing authentication on sensitive endpoints, and insufficient input validation for AI services.

---

## Findings

### Critical Severity

| ID | Issue | Location | Description |
|----|-------|----------|-------------|
| C-01 | Hardcoded secrets in K8s manifests | `k8s/secrets.yaml:17-27` | Base64-encoded placeholder passwords committed to version control. Values decode to `please_change_this_password`, `please_generate_a_random_secret_key`, etc. |
| C-02 | Default secrets in Docker configs | `docker-compose.yml:84-85` | Hardcoded dev secrets: `SECRET_KEY: dev-secret-key-not-for-production`, `JWT_SECRET: dev-jwt-secret-not-for-production` |
| C-03 | Insecure default secret key | `backend/app/core/config.py:49` | Fallback secret: `secret_key: str = os.getenv("SECRET_KEY", "change-this-in-production-please")` |
| C-04 | No authentication on document endpoints | `backend/app/api/documents.py:20-235` | All document upload, retrieval, and deletion endpoints lack authentication decorators |

**Remediation:**
- C-01: Use external secret management (HashiCorp Vault, AWS Secrets Manager, K8s External Secrets)
- C-02/C-03: Remove default values; fail startup if secrets not provided
- C-04: Add `Depends(get_current_user)` to all endpoints

---

### High Severity

| ID | Issue | Location | Description |
|----|-------|----------|-------------|
| H-01 | CORS allows all origins in dev | `backend/app/core/config.py:70-74` | `allowed_origins: List[str] = ["*"]` when not in production |
| H-02 | CORS allows all headers | `backend/app/main.py:174-175` | `allow_headers=["*"]` permits any header, enabling CSRF attacks |
| H-03 | Qdrant no authentication | `backend/app/db/qdrant_store.py:91-104` | API key is optional (`api_key=self.config.api_key`), connections work without auth |
| H-04 | Redis no password in dev | `docker-compose.yml:37` | `redis-server --appendonly yes` runs without `--requirepass` |
| H-05 | Database ports exposed | `docker-compose.prod.yml:19,77,114` | PostgreSQL (5432), Redis (6379), Qdrant (6333) exposed to host in production config |
| H-06 | No file size validation | `backend/app/api/documents.py:34` | `content = await file.read()` reads entire file without size check (DoS risk) |
| H-07 | Content-type header trust | `backend/app/api/documents.py:37` | File type determined by `file.content_type` header which clients control |
| H-08 | Error message exposure | `backend/app/api/documents.py:71` | `detail=f"Error uploading document: {str(e)}"` exposes internal errors |

**Remediation:**
- H-01/H-02: Restrict CORS to specific origins and headers in all environments
- H-03: Require API key for Qdrant in production
- H-04/H-05: Enable Redis AUTH; bind services to internal networks only
- H-06: Add `settings.max_file_size` check before reading
- H-07: Validate file magic bytes, not just content-type header
- H-08: Log full errors internally; return generic messages to users

---

### Medium Severity

| ID | Issue | Location | Description |
|----|-------|----------|-------------|
| M-01 | PII logged | `backend/app/core/logging_config.py:199-201` | Client IP and user-agent logged to files |
| M-02 | No prompt injection protection | `backend/app/legal_ai/providers/openai_provider.py:131-135` | User content passed directly to LLM without sanitization |
| M-03 | Missing network policies | `k8s/deployment.yaml` | No NetworkPolicy resources defined; pods can communicate freely |
| M-04 | No pod security context | `k8s/deployment.yaml:357-360` | Backend uses `readOnlyRootFilesystem: false`; writable filesystem |
| M-05 | Prometheus admin API enabled | `docker-compose.prod.yml:374-375` | `--web.enable-admin-api` allows runtime configuration changes |
| M-06 | JWT tokens in localStorage | `frontend/lib/api-client.ts:97-98` | `localStorage.getItem("auth_token")` vulnerable to XSS |
| M-07 | Missing CSP script-src | `k8s/ingress.yaml:36` | CSP includes `'unsafe-inline' 'unsafe-eval'` for scripts |
| M-08 | No rate limiting middleware | `backend/app/main.py` | Rate limiting mentioned in docs but not implemented in middleware |
| M-09 | Grafana signup not disabled explicitly | `docker-compose.prod.yml:405` | Only `GF_USERS_ALLOW_SIGN_UP: "false"` set; consider additional hardening |

**Remediation:**
- M-01: Anonymize/hash IPs; use structured logging with PII filtering
- M-02: Implement input sanitization for LLM inputs; use prompt boundaries
- M-03/M-04: Add NetworkPolicy; enable `readOnlyRootFilesystem: true`
- M-05: Disable admin API unless needed for automation
- M-06: Use httpOnly cookies for tokens
- M-07: Remove unsafe-inline/unsafe-eval; use nonces or hashes
- M-08: Add slowapi or similar rate limiting middleware
- M-09: Set `GF_AUTH_ANONYMOUS_ENABLED: false` explicitly

---

### Low Severity

| ID | Issue | Location | Description |
|----|-------|----------|-------------|
| L-01 | Dev database credentials | `docker-compose.yml:14-15` | `POSTGRES_PASSWORD: devpassword` in development config |
| L-02 | Debug mode flag | `backend/app/core/config.py:15` | `debug: bool = False` but can be overridden |
| L-03 | OpenAPI docs in production | `backend/app/core/config.py:93` | `enable_docs` defaults to false but configurable |
| L-04 | Missing security headers | `backend/app/main.py` | No X-Frame-Options, X-Content-Type-Options middleware |
| L-05 | Chroma auth weak | `k8s/deployment.yaml:181-184` | Basic auth provider for Chroma vector DB |

**Remediation:**
- L-01: Use `.env` files not committed to git
- L-02/L-03: Ensure these stay false in production via CI checks
- L-04: Add secure-headers middleware
- L-05: Consider mTLS for service-to-service auth

---

## OWASP Top 10 Analysis

| OWASP Category | Status | Notes |
|----------------|--------|-------|
| A01:2021 Broken Access Control | **FAIL** | Missing authentication on document endpoints (C-04) |
| A02:2021 Cryptographic Failures | **WARN** | Hardcoded secrets (C-01, C-02, C-03) |
| A03:2021 Injection | **WARN** | LLM prompt injection risk (M-02) |
| A04:2021 Insecure Design | **PASS** | Architecture is reasonable |
| A05:2021 Security Misconfiguration | **FAIL** | CORS wildcards, exposed ports (H-01, H-05) |
| A06:2021 Vulnerable Components | **AUDIT** | See DEPENDENCY_AUDIT.md |
| A07:2021 Auth Failures | **WARN** | JWT in localStorage (M-06) |
| A08:2021 Software/Data Integrity | **PASS** | Using signed containers |
| A09:2021 Logging Failures | **WARN** | PII in logs (M-01) |
| A10:2021 SSRF | **PASS** | No obvious SSRF vectors |

---

## Celery/Redis Security

| Finding | Severity | Location |
|---------|----------|----------|
| Redis no TLS | Medium | `docker-compose.yml:79` - `redis://redis:6379/0` |
| Celery broker unencrypted | Medium | `docker-compose.yml:80` |
| Task results not encrypted | Low | Results stored in Redis without encryption |
| Flower dashboard unprotected | Medium | If deployed, needs auth |

**Recommendation:** Enable Redis TLS in production; use `rediss://` URLs.

---

## Data Privacy

| Issue | Impact | Location |
|-------|--------|----------|
| Document content stored unencrypted | High | Documents stored in PostgreSQL and Qdrant without encryption at rest |
| IP addresses logged | Medium | `logging_config.py:199` |
| No data retention policy | Medium | Documents never automatically purged |
| GDPR deletion not implemented | High | No `/api/v1/users/me/data` delete endpoint |

---

## Prioritized Action Plan

### Immediate (0-7 days)
1. **Remove hardcoded secrets** (C-01, C-02, C-03) - Implement external secret management
2. **Add authentication to document endpoints** (C-04) - Critical security gap
3. **Disable exposed database ports in production** (H-05)

### Short-term (1-4 weeks)
4. **Implement file upload validation** (H-06, H-07) - Size limits, magic byte checking
5. **Restrict CORS configuration** (H-01, H-02) - Remove wildcards
6. **Enable Redis/Qdrant authentication** (H-03, H-04)
7. **Add rate limiting middleware** (M-08)

### Medium-term (1-3 months)
8. **Implement prompt injection protection** (M-02)
9. **Add Kubernetes NetworkPolicies** (M-03)
10. **Switch to httpOnly cookies for auth** (M-06)
11. **Implement PII filtering in logs** (M-01)

### Long-term (3-6 months)
12. **Add encryption at rest for documents**
13. **Implement GDPR data deletion**
14. **Security scanning in CI pipeline**
15. **Regular penetration testing**

---

## Appendix: Secure Configuration Checklist

```yaml
# Production environment checklist
- [ ] All secrets from external secret manager
- [ ] Redis password enabled
- [ ] Qdrant API key required
- [ ] Database not exposed to host
- [ ] CORS restricted to production domains
- [ ] API docs disabled
- [ ] Debug mode off
- [ ] All endpoints authenticated
- [ ] Rate limiting enabled
- [ ] TLS for all internal services
- [ ] Network policies applied
- [ ] Pod security contexts enforced
```

---

*Generated by automated security analysis. Manual review recommended for business logic vulnerabilities.*
