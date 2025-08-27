# Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents in the Arbitration Detection System.

## Incident Classification

### Severity Levels

**P0 - Critical**
- Complete system outage
- Data loss or corruption
- Security breach
- Revenue-impacting issues

**P1 - High**
- Partial system outage
- Significant performance degradation
- Failed deployments
- Critical feature unavailable

**P2 - Medium**
- Minor service disruption
- Performance issues
- Non-critical feature issues
- Monitoring alerts

**P3 - Low**
- Enhancement requests
- Minor bugs
- Documentation issues

## Incident Response Team

### Primary Contacts

- **Incident Commander**: devops@company.com
- **Technical Lead**: tech-lead@company.com
- **Security Lead**: security@company.com
- **Communications Lead**: communications@company.com

### Escalation Chain

1. On-call Engineer
2. Team Lead
3. Engineering Manager
4. CTO
5. CEO (for P0 incidents)

## General Incident Response Process

### 1. Detection and Alerting

Incidents can be detected through:
- Automated monitoring alerts
- Customer reports
- Team member discovery
- Third-party service notifications

### 2. Initial Response (First 5 minutes)

1. **Acknowledge the incident**
   ```bash
   # Silence related alerts to reduce noise
   curl -X POST http://alertmanager:9093/api/v1/silences \
     -H "Content-Type: application/json" \
     -d '{
       "matchers": [{"name": "alertname", "value": "AlertName"}],
       "startsAt": "2023-12-01T00:00:00Z",
       "endsAt": "2023-12-01T01:00:00Z",
       "comment": "Investigating incident",
       "createdBy": "incident-responder"
     }'
   ```

2. **Create incident tracking**
   - Open incident ticket
   - Create Slack/Teams channel: `#incident-YYYY-MM-DD-N`
   - Document initial findings

3. **Assess severity and impact**
   - Determine affected services
   - Estimate user impact
   - Classify severity level

### 3. Investigation and Diagnosis

1. **Gather information**
   ```bash
   # Check overall system health
   ./demo/scripts/health-check.sh -v
   
   # Review recent deployments
   git log --oneline --since="1 hour ago"
   
   # Check service status
   docker-compose -f demo/production/docker-compose.prod.yml ps
   ```

2. **Review monitoring data**
   - Check Grafana dashboards
   - Review Prometheus metrics
   - Analyze logs in Loki/Grafana
   - Examine error rates and response times

3. **Identify root cause**
   - Service failures
   - Database issues
   - Infrastructure problems
   - Code defects
   - External dependencies

### 4. Mitigation and Resolution

1. **Implement immediate fixes**
   - Restart failing services
   - Scale resources if needed
   - Apply hotfixes
   - Failover to backup systems

2. **Communicate status**
   - Update status page
   - Notify stakeholders
   - Provide regular updates

### 5. Recovery Verification

1. **Test system functionality**
   - Verify all services are operational
   - Perform smoke tests
   - Monitor metrics for stability

2. **Remove temporary fixes**
   - Implement permanent solutions
   - Remove hotfixes if appropriate
   - Update configurations

### 6. Post-Incident Activities

1. **Document the incident**
   - Timeline of events
   - Impact assessment
   - Actions taken
   - Lessons learned

2. **Conduct post-mortem**
   - Schedule within 24-48 hours
   - Include all stakeholders
   - Focus on process improvement

## Service-Specific Runbooks

### Backend Service Down

**Alert**: `BackendServiceDown`

**Symptoms**:
- API returning 5xx errors
- Health check failures
- Connection timeouts

**Diagnosis**:
```bash
# Check service status
docker-compose -f demo/production/docker-compose.prod.yml ps backend

# Check container logs
docker-compose -f demo/production/docker-compose.prod.yml logs --tail=100 backend

# Check resource usage
docker stats arbitration-backend-prod

# Test database connectivity
docker-compose -f demo/production/docker-compose.prod.yml exec backend python -c "
import asyncpg
import asyncio
async def test_db():
    conn = await asyncpg.connect('${DATABASE_URL}')
    await conn.close()
    print('Database OK')
asyncio.run(test_db())
"
```

**Resolution**:
1. Restart the service:
   ```bash
   docker-compose -f demo/production/docker-compose.prod.yml restart backend
   ```

2. If restart fails, rebuild and redeploy:
   ```bash
   docker-compose -f demo/production/docker-compose.prod.yml build backend
   docker-compose -f demo/production/docker-compose.prod.yml up -d backend
   ```

3. Scale up if resource constrained:
   ```bash
   docker-compose -f demo/production/docker-compose.prod.yml up -d --scale backend=2
   ```

### Database Connection Issues

**Alert**: `DatabaseDown` or `PostgreSQLTooManyConnections`

**Symptoms**:
- Connection timeouts
- "Too many connections" errors
- Application errors

**Diagnosis**:
```bash
# Check PostgreSQL status
docker-compose -f demo/production/docker-compose.prod.yml exec postgres pg_isready

# Check connections
docker-compose -f demo/production/docker-compose.prod.yml exec postgres psql -U postgres -c "
SELECT count(*) as connections, state 
FROM pg_stat_activity 
GROUP BY state;
"

# Check for long-running queries
docker-compose -f demo/production/docker-compose.prod.yml exec postgres psql -U postgres -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
"
```

**Resolution**:
1. Kill long-running queries:
   ```bash
   # Identify and kill problematic queries
   docker-compose -f demo/production/docker-compose.prod.yml exec postgres psql -U postgres -c "
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE (now() - pg_stat_activity.query_start) > interval '10 minutes';
   "
   ```

2. Restart database if necessary:
   ```bash
   docker-compose -f demo/production/docker-compose.prod.yml restart postgres
   ```

3. Increase connection limits (temporary):
   ```bash
   docker-compose -f demo/production/docker-compose.prod.yml exec postgres psql -U postgres -c "
   ALTER SYSTEM SET max_connections = 200;
   SELECT pg_reload_conf();
   "
   ```

### High Error Rate

**Alert**: `HighErrorRate`

**Symptoms**:
- Increased 5xx response codes
- User reports of failures
- Error alerts

**Diagnosis**:
```bash
# Check error patterns in logs
docker-compose -f demo/production/docker-compose.prod.yml logs backend | grep -i error | tail -20

# Check specific endpoints with high error rates
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])" | jq .

# Check application metrics
curl -s "http://backend:8000/metrics" | grep -E "(error|exception)"
```

**Resolution**:
1. Identify error patterns and affected endpoints
2. Check for recent deployments or configuration changes
3. Roll back if recent changes are causing issues:
   ```bash
   ./demo/scripts/rollback.sh --type application --force
   ```
4. Apply hotfix if specific bug identified

### SSL Certificate Issues

**Alert**: `SSLCertificateExpiringSoon` or `SSLCertificateExpired`

**Symptoms**:
- Browser SSL warnings
- API connection failures
- Certificate validation errors

**Diagnosis**:
```bash
# Check certificate expiration
openssl x509 -in demo/nginx/ssl/app.crt -noout -dates

# Test SSL connection
openssl s_client -connect arbitration-detector.com:443 -servername arbitration-detector.com

# Check Let's Encrypt renewal status
sudo certbot certificates
```

**Resolution**:
1. Renew Let's Encrypt certificates:
   ```bash
   sudo certbot renew --force-renewal
   
   # Copy renewed certificates
   sudo cp /etc/letsencrypt/live/arbitration-detector.com/fullchain.pem demo/nginx/ssl/app.crt
   sudo cp /etc/letsencrypt/live/arbitration-detector.com/privkey.pem demo/nginx/ssl/app.key
   
   # Restart nginx
   docker-compose -f demo/production/docker-compose.prod.yml restart nginx
   ```

2. For immediate fix, generate temporary self-signed certificate:
   ```bash
   ./demo/nginx/ssl/generate.sh arbitration-detector.com
   docker-compose -f demo/production/docker-compose.prod.yml restart nginx
   ```

## Security Incident Response

### Suspected Security Breach

**Immediate Actions**:
1. **Isolate affected systems**
   ```bash
   # Block suspicious IP addresses
   sudo ufw deny from SUSPICIOUS_IP
   
   # Disable compromised user accounts
   docker-compose -f demo/production/docker-compose.prod.yml exec backend python -c "
   # Add script to disable user accounts
   "
   ```

2. **Preserve evidence**
   ```bash
   # Create system snapshot
   sudo dd if=/dev/sda of=/backup/forensic-$(date +%Y%m%d-%H%M%S).img
   
   # Export logs
   docker-compose -f demo/production/docker-compose.prod.yml logs > incident-logs-$(date +%Y%m%d-%H%M%S).log
   ```

3. **Notify security team**
   - Contact security@company.com
   - Escalate to CISO if major breach
   - Consider law enforcement if required

### DDoS Attack

**Alert**: `DDOSAttack`

**Symptoms**:
- Extremely high request rates
- Service unavailability
- Network congestion

**Resolution**:
1. **Enable rate limiting**
   ```bash
   # Update nginx configuration with stricter limits
   # Restart nginx with new configuration
   docker-compose -f demo/production/docker-compose.prod.yml restart nginx
   ```

2. **Block attacking IPs**
   ```bash
   # Identify attacking IPs from logs
   docker-compose -f demo/production/docker-compose.prod.yml logs nginx | \
     grep -E "HTTP/[0-9.]+" | \
     awk '{print $1}' | sort | uniq -c | sort -nr | head -20
   
   # Block top attacking IPs
   for ip in $(attacking_ips); do
     sudo ufw deny from $ip
   done
   ```

3. **Activate DDoS protection**
   - Enable CloudFlare DDoS protection
   - Contact hosting provider
   - Consider traffic filtering service

## Communication Templates

### Initial Incident Notification

```
Subject: [INCIDENT] P0/P1 - Brief Description

We are currently investigating an issue affecting [service/feature].

Impact: [Description of impact]
Started: [Time]
Status: Investigating

We will provide updates every 30 minutes until resolved.

Status Page: https://status.arbitration-detector.com
```

### Resolution Notification

```
Subject: [RESOLVED] P0/P1 - Brief Description

The incident affecting [service/feature] has been resolved.

Summary:
- Started: [Time]
- Resolved: [Time]
- Duration: [Duration]
- Root Cause: [Brief description]
- Impact: [Description of impact]

Next Steps:
- Post-mortem scheduled for [date/time]
- Preventive measures will be implemented

Thank you for your patience.
```

## Post-Incident Checklist

- [ ] Incident documented in tracking system
- [ ] Timeline of events recorded
- [ ] Impact assessment completed
- [ ] Root cause identified
- [ ] Stakeholders notified of resolution
- [ ] Post-mortem scheduled
- [ ] Status page updated
- [ ] Monitoring alerts resolved
- [ ] Temporary fixes documented
- [ ] Lessons learned documented

## Tools and Resources

### Monitoring Dashboards
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093

### Log Analysis
- Loki: http://localhost:3100
- Container logs: `docker-compose logs`
- System logs: `/var/log/`

### Health Check Scripts
- Overall system: `./demo/scripts/health-check.sh`
- Application health: `curl http://localhost:8000/health`

### Deployment Tools
- Deploy script: `./demo/scripts/deploy.sh`
- Rollback script: `./demo/scripts/rollback.sh`
- Backup script: `./demo/scripts/backup.sh`

## Training and Preparedness

### Regular Drills
- Monthly fire drills
- Quarterly disaster recovery tests
- Annual security incident simulations

### Documentation Updates
- Review runbooks monthly
- Update contact information quarterly
- Test procedures during maintenance windows

### Team Preparedness
- On-call rotation schedule
- Incident response training
- Tool familiarization sessions