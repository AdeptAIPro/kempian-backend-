"""
Critical Security Enhancements for AdeptAI
Addresses all identified security vulnerabilities and implements best practices
"""

import os
import re
import hashlib
import secrets
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import subprocess
import json

class SecurityEnhancements:
    """Comprehensive security enhancements and vulnerability fixes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.critical_issues = []
        self.security_fixes = []
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        print("🔒 RUNNING COMPREHENSIVE SECURITY AUDIT")
        print("=" * 60)
        
        audit_results = {
            'critical_issues': [],
            'high_issues': [],
            'medium_issues': [],
            'low_issues': [],
            'security_score': 0,
            'recommendations': []
        }
        
        # 1. Check for hardcoded secrets
        self._check_hardcoded_secrets(audit_results)
        
        # 2. Validate input sanitization
        self._check_input_sanitization(audit_results)
        
        # 3. Review authentication system
        self._check_authentication_security(audit_results)
        
        # 4. Check file upload security
        self._check_file_upload_security(audit_results)
        
        # 5. Review logging security
        self._check_logging_security(audit_results)
        
        # 6. Check dependency vulnerabilities
        self._check_dependency_vulnerabilities(audit_results)
        
        # 7. Check for code injection vulnerabilities
        self._check_code_injection_vulnerabilities(audit_results)
        
        # 8. Check for information disclosure
        self._check_information_disclosure(audit_results)
        
        # Calculate security score
        audit_results['security_score'] = self._calculate_security_score(audit_results)
        
        return audit_results
    
    def _check_hardcoded_secrets(self, audit_results: Dict[str, Any]):
        """Check for hardcoded secrets and credentials"""
        print("🔍 Checking for hardcoded secrets...")
        
        secret_patterns = {
            'api_keys': r'sk-[a-zA-Z0-9]{20,}',
            'aws_keys': r'AKIA[0-9A-Z]{16}',
            'aws_secrets': r'[A-Za-z0-9/+=]{40}',
            'jwt_secrets': r'[A-Za-z0-9+/=]{32,}',
            'database_urls': r'(postgresql|mysql|mongodb)://[^\s]+',
            'passwords': r'password\s*=\s*["\'][^"\']+["\']',
            'tokens': r'token\s*=\s*["\'][^"\']+["\']'
        }
        
        issues_found = 0
        for pattern_name, pattern in secret_patterns.items():
            for file_path in Path('.').rglob('*.py'):
                if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules']):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip if it's in documentation or comments
                        line_start = content.rfind('\n', 0, match.start()) + 1
                        line = content[line_start:match.end()]
                        
                        if not self._is_documentation_line(line):
                            issues_found += 1
                            audit_results['critical_issues'].append({
                                'type': 'hardcoded_secret',
                                'pattern': pattern_name,
                                'file': str(file_path),
                                'line': content[:match.start()].count('\n') + 1,
                                'match': match.group()[:20] + '...',
                                'severity': 'CRITICAL'
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Error scanning {file_path}: {e}")
        
        if issues_found > 0:
            print(f"❌ Found {issues_found} hardcoded secrets")
        else:
            print("✅ No hardcoded secrets found")
    
    def _check_input_sanitization(self, audit_results: Dict[str, Any]):
        """Check for input sanitization vulnerabilities"""
        print("🔍 Checking input sanitization...")
        
        # Check for SQL injection patterns
        sql_injection_files = []
        for file_path in Path('.').rglob('*.py'):
            if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for potential SQL injection patterns
                dangerous_patterns = [
                    r'f["\'].*\{.*\}.*SELECT',
                    r'f["\'].*\{.*\}.*INSERT',
                    r'f["\'].*\{.*\}.*UPDATE',
                    r'f["\'].*\{.*\}.*DELETE',
                    r'execute\(.*\+.*\)',
                    r'query.*\+.*user_input',
                    r'cursor\.execute\(.*%s.*\)'
                ]
                
                for pattern in dangerous_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        sql_injection_files.append(str(file_path))
                        break
                        
            except Exception as e:
                self.logger.warning(f"Error scanning {file_path}: {e}")
        
        if sql_injection_files:
            audit_results['high_issues'].append({
                'type': 'sql_injection_risk',
                'files': sql_injection_files,
                'severity': 'HIGH',
                'description': 'Potential SQL injection vulnerabilities detected'
            })
            print(f"❌ Found potential SQL injection risks in {len(sql_injection_files)} files")
        else:
            print("✅ No SQL injection risks detected")
    
    def _check_authentication_security(self, audit_results: Dict[str, Any]):
        """Check authentication and authorization security"""
        print("🔍 Checking authentication security...")
        
        auth_issues = []
        
        # Check for weak secret keys
        secret_key_files = []
        for file_path in Path('.').rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'SECRET_KEY' in content and 'change-me' in content.lower():
                    secret_key_files.append(str(file_path))
                    
            except Exception:
                continue
        
        if secret_key_files:
            auth_issues.append({
                'type': 'weak_secret_key',
                'files': secret_key_files,
                'severity': 'HIGH',
                'description': 'Default/weak secret keys detected'
            })
        
        # Check for missing authentication
        if not Path('app/auth').exists():
            auth_issues.append({
                'type': 'missing_auth_module',
                'severity': 'HIGH',
                'description': 'Authentication module not found'
            })
        
        if auth_issues:
            audit_results['high_issues'].extend(auth_issues)
            print(f"❌ Found {len(auth_issues)} authentication issues")
        else:
            print("✅ Authentication security looks good")
    
    def _check_file_upload_security(self, audit_results: Dict[str, Any]):
        """Check file upload security"""
        print("🔍 Checking file upload security...")
        
        upload_issues = []
        
        # Check if file upload validation exists
        validation_file = Path('app/validation_simple.py')
        if validation_file.exists():
            with open(validation_file, 'r') as f:
                content = f.read()
                
            # Check for proper file validation
            if 'validate_file_upload' in content:
                # Check if it validates file extensions
                if 'allowed_extensions' in content:
                    print("✅ File upload validation exists")
                else:
                    upload_issues.append({
                        'type': 'incomplete_file_validation',
                        'severity': 'MEDIUM',
                        'description': 'File upload validation may be incomplete'
                    })
            else:
                upload_issues.append({
                    'type': 'missing_file_validation',
                    'severity': 'HIGH',
                    'description': 'No file upload validation found'
                })
        else:
            upload_issues.append({
                'type': 'missing_file_validation',
                'severity': 'HIGH',
                'description': 'No file upload validation found'
            })
        
        if upload_issues:
            audit_results['high_issues'].extend(upload_issues)
            print(f"❌ Found {len(upload_issues)} file upload security issues")
        else:
            print("✅ File upload security looks good")
    
    def _check_logging_security(self, audit_results: Dict[str, Any]):
        """Check for sensitive data in logs"""
        print("🔍 Checking logging security...")
        
        logging_issues = []
        
        # Check log files for sensitive data
        log_dir = Path('logs')
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for API keys in logs
                    if re.search(r'sk-[a-zA-Z0-9]{20,}', content):
                        logging_issues.append({
                            'type': 'sensitive_data_in_logs',
                            'file': str(log_file),
                            'severity': 'HIGH',
                            'description': 'API keys found in log files'
                        })
                        
                except Exception:
                    continue
        
        if logging_issues:
            audit_results['high_issues'].extend(logging_issues)
            print(f"❌ Found {len(logging_issues)} logging security issues")
        else:
            print("✅ Logging security looks good")
    
    def _check_dependency_vulnerabilities(self, audit_results: Dict[str, Any]):
        """Check for vulnerable dependencies"""
        print("🔍 Checking dependency vulnerabilities...")
        
        try:
            # Check if safety is available
            result = subprocess.run(['pip', 'show', 'safety'], capture_output=True, text=True)
            if result.returncode == 0:
                # Run safety check
                result = subprocess.run(['safety', 'check', '--json'], capture_output=True, text=True)
                if result.returncode != 0:
                    vulnerabilities = json.loads(result.stdout)
                    audit_results['high_issues'].append({
                        'type': 'vulnerable_dependencies',
                        'vulnerabilities': vulnerabilities,
                        'severity': 'HIGH',
                        'description': f'Found {len(vulnerabilities)} vulnerable dependencies'
                    })
                    print(f"❌ Found {len(vulnerabilities)} vulnerable dependencies")
                else:
                    print("✅ No vulnerable dependencies found")
            else:
                print("⚠️ Safety tool not installed - cannot check dependencies")
                
        except Exception as e:
            print(f"⚠️ Could not check dependencies: {e}")
    
    def _check_code_injection_vulnerabilities(self, audit_results: Dict[str, Any]):
        """Check for code injection vulnerabilities"""
        print("🔍 Checking for code injection vulnerabilities...")
        
        injection_issues = []
        
        dangerous_functions = ['eval', 'exec', 'subprocess.run', 'os.system']
        
        for file_path in Path('.').rglob('*.py'):
            if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for func in dangerous_functions:
                    if func in content:
                        # Check if it's using user input
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if func in line and ('input(' in line or 'request.' in line):
                                injection_issues.append({
                                    'type': 'code_injection_risk',
                                    'file': str(file_path),
                                    'line': i + 1,
                                    'function': func,
                                    'severity': 'HIGH',
                                    'description': f'Potential code injection with {func}'
                                })
                                
            except Exception:
                continue
        
        if injection_issues:
            audit_results['high_issues'].extend(injection_issues)
            print(f"❌ Found {len(injection_issues)} code injection risks")
        else:
            print("✅ No code injection risks detected")
    
    def _check_information_disclosure(self, audit_results: Dict[str, Any]):
        """Check for information disclosure vulnerabilities"""
        print("🔍 Checking for information disclosure...")
        
        disclosure_issues = []
        
        # Check for debug mode in production
        for file_path in Path('.').rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'DEBUG = True' in content or 'debug=True' in content:
                    disclosure_issues.append({
                        'type': 'debug_mode_enabled',
                        'file': str(file_path),
                        'severity': 'MEDIUM',
                        'description': 'Debug mode may be enabled'
                    })
                    
            except Exception:
                continue
        
        if disclosure_issues:
            audit_results['medium_issues'].extend(disclosure_issues)
            print(f"❌ Found {len(disclosure_issues)} information disclosure issues")
        else:
            print("✅ No information disclosure issues found")
    
    def _is_documentation_line(self, line: str) -> bool:
        """Check if line appears to be documentation"""
        doc_indicators = ['example', 'template', 'placeholder', 'your_', 'replace_', 'TODO', 'FIXME']
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in doc_indicators)
    
    def _calculate_security_score(self, audit_results: Dict[str, Any]) -> int:
        """Calculate security score based on issues found"""
        score = 100
        
        # Deduct points for issues
        score -= len(audit_results['critical_issues']) * 25
        score -= len(audit_results['high_issues']) * 15
        score -= len(audit_results['medium_issues']) * 10
        score -= len(audit_results['low_issues']) * 5
        
        return max(0, min(100, score))
    
    def generate_security_report(self, audit_results: Dict[str, Any]) -> str:
        """Generate comprehensive security report"""
        report = []
        report.append("🔒 COMPREHENSIVE SECURITY AUDIT REPORT")
        report.append("=" * 60)
        report.append(f"Security Score: {audit_results['security_score']}/100")
        report.append("")
        
        # Critical Issues
        if audit_results['critical_issues']:
            report.append("🚨 CRITICAL ISSUES:")
            for issue in audit_results['critical_issues']:
                report.append(f"  - {issue['type']}: {issue.get('description', 'No description')}")
                if 'file' in issue:
                    report.append(f"    File: {issue['file']}")
                if 'line' in issue:
                    report.append(f"    Line: {issue['line']}")
            report.append("")
        
        # High Issues
        if audit_results['high_issues']:
            report.append("⚠️ HIGH PRIORITY ISSUES:")
            for issue in audit_results['high_issues']:
                report.append(f"  - {issue['type']}: {issue.get('description', 'No description')}")
                if 'file' in issue:
                    report.append(f"    File: {issue['file']}")
            report.append("")
        
        # Medium Issues
        if audit_results['medium_issues']:
            report.append("⚠️ MEDIUM PRIORITY ISSUES:")
            for issue in audit_results['medium_issues']:
                report.append(f"  - {issue['type']}: {issue.get('description', 'No description')}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("  1. Fix all critical and high priority issues immediately")
        report.append("  2. Implement proper input validation and sanitization")
        report.append("  3. Use environment variables for all secrets")
        report.append("  4. Enable security headers and HTTPS")
        report.append("  5. Regular security audits and dependency updates")
        report.append("  6. Implement proper authentication and authorization")
        report.append("  7. Use secure coding practices")
        
        return "\n".join(report)


def run_security_audit():
    """Run comprehensive security audit"""
    enhancer = SecurityEnhancements()
    audit_results = enhancer.run_comprehensive_audit()
    
    print("\n" + "=" * 60)
    print(enhancer.generate_security_report(audit_results))
    
    return audit_results


if __name__ == "__main__":
    run_security_audit()
