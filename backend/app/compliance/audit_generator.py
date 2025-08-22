import json
from typing import Dict, List, Any
from datetime import datetime
from .regulations_db import RegulationType
from .auto_checker import ComplianceChecker

class AuditReportGenerator:
    def __init__(self, compliance_checker: ComplianceChecker):
        self.compliance_checker = compliance_checker
        self.audit_log: List[Dict[str, Any]] = []
    
    def generate_compliance_report(self, user_data: Dict[str, Any], regulation: RegulationType) -> Dict[str, Any]:
        """Generate a comprehensive compliance audit report"""
        compliance_check = self.compliance_checker.run_comprehensive_check(user_data, regulation)
        
        report = {
            'report_id': self._generate_report_id(),
            'timestamp': datetime.now().isoformat(),
            'organization': user_data.get('organization', 'Unnamed'),
            'regulation': regulation.name,
            'compliance_details': compliance_check,
            'recommendations': self._generate_recommendations(compliance_check)
        }
        
        self._log_audit(report)
        return report
    
    def _generate_report_id(self) -> str:
        """Generate a unique report identifier"""
        return f"AUDIT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _generate_recommendations(self, compliance_check: Dict[str, Any]) -> List[str]:
        """Generate actionable compliance improvement recommendations"""
        recommendations = []
        
        if not compliance_check['details']['data_retention']:
            recommendations.append("Review and update data retention policies")
        
        if not compliance_check['details']['consent_management']:
            recommendations.append("Improve consent management mechanisms")
        
        if not compliance_check['details']['data_portability']:
            recommendations.append("Enhance data export and portability features")
        
        if compliance_check['compliance_score'] < 70:
            recommendations.append("Conduct a comprehensive compliance review")
        
        return recommendations
    
    def _log_audit(self, report: Dict[str, Any]):
        """Log audit report for record-keeping"""
        self.audit_log.append(report)
    
    def export_audit_log(self, format: str = 'json') -> str:
        """Export audit log in specified format"""
        if format == 'json':
            return json.dumps(self.audit_log, indent=2)
        elif format == 'csv':
            # Implement CSV export logic
            return self._export_to_csv()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_csv(self) -> str:
        """Convert audit log to CSV format"""
        # Placeholder for CSV conversion logic
        return "CSV export not implemented"

# Instantiate with compliance checker
audit_generator = AuditReportGenerator(ComplianceChecker(RegulationsDatabase()))