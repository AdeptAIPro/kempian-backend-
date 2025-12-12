"""
Security Configuration for AdeptAI
Prevents API key exposure and implements security best practices
"""

import os
import re
import logging
from typing import Dict, List, Optional
from pathlib import Path

class SecurityConfig:
    """Security configuration and validation for API keys"""
    
    # API key patterns to detect
    API_KEY_PATTERNS = {
        'openai': r'sk-[a-zA-Z0-9]{20,}',
        'anthropic': r'sk-ant-[a-zA-Z0-9]{20,}',
        'generic_sk': r'sk-[a-zA-Z0-9]{20,}',
        'generic_api': r'[a-zA-Z0-9]{32,}',
        'bearer_token': r'Bearer\s+[a-zA-Z0-9]{20,}',
        'basic_auth': r'Basic\s+[a-zA-Z0-9+/=]{20,}'
    }
    
    # Sensitive environment variables
    SENSITIVE_VARS = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'GLASSDOOR_API_KEY',
        'PAYSCALE_API_KEY',
        'LEVELS_FYI_API_KEY',
        'BLS_API_KEY',
        'CENSUS_API_KEY',
        'FRED_API_KEY',
        'INDEED_API_KEY',
        'CRUNCHBASE_API_KEY',
        'CLEARBIT_API_KEY',
        'STACKOVERFLOW_API_KEY',
        'LINKEDIN_API_KEY',
        'SALARY_COM_API_KEY'
    ]
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate that all API keys are loaded from environment variables"""
        results = {}
        
        for var in cls.SENSITIVE_VARS:
            value = os.getenv(var)
            if value:
                # Check if it's a placeholder value
                if value in ['your_api_key_here', 'your_openai_api_key_here', 
                           'your_anthropic_api_key_here', '']:
                    results[var] = False
                else:
                    # Check if it looks like a real API key
                    if cls._is_valid_api_key_format(var, value):
                        results[var] = True
                    else:
                        results[var] = False
            else:
                results[var] = False
                
        return results
    
    @classmethod
    def _is_valid_api_key_format(cls, var_name: str, value: str) -> bool:
        """Check if API key has valid format"""
        if not value or len(value) < 10:
            return False
            
        # OpenAI keys
        if 'OPENAI' in var_name.upper():
            return value.startswith('sk-') and len(value) > 20
            
        # Anthropic keys
        if 'ANTHROPIC' in var_name.upper():
            return value.startswith('sk-ant-') and len(value) > 20
            
        # Generic API keys
        return len(value) > 10 and not value.startswith('your_')
    
    @classmethod
    def scan_for_hardcoded_keys(cls, directory: str = ".") -> List[Dict[str, str]]:
        """Scan codebase for hardcoded API keys"""
        issues = []
        directory_path = Path(directory)
        
        # Files to exclude from scanning
        exclude_patterns = [
            '*.pyc', '__pycache__', '.git', 'node_modules', 
            '*.log', '*.env.example', '*.env.template'
        ]
        
        for pattern_name, pattern in cls.API_KEY_PATTERNS.items():
            for file_path in directory_path.rglob('*.py'):
                # Skip excluded files
                if any(file_path.match(exclude) for exclude in exclude_patterns):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip if it's in a comment or string that looks like documentation
                        line_start = content.rfind('\n', 0, match.start()) + 1
                        line = content[line_start:match.end()]
                        
                        if not cls._is_documentation_line(line):
                            issues.append({
                                'file': str(file_path),
                                'pattern': pattern_name,
                                'line': content[:match.start()].count('\n') + 1,
                                'match': match.group()[:20] + '...',
                                'severity': 'HIGH'
                            })
                            
                except Exception as e:
                    logging.warning(f"Error scanning {file_path}: {e}")
                    
        return issues
    
    @classmethod
    def _is_documentation_line(cls, line: str) -> bool:
        """Check if line appears to be documentation/example"""
        doc_indicators = [
            'example', 'template', 'placeholder', 'your_', 'replace_',
            'TODO', 'FIXME', 'NOTE:', 'WARNING:', 'SECURITY:'
        ]
        
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in doc_indicators)
    
    @classmethod
    def create_env_template(cls) -> str:
        """Create a secure .env.template file"""
        template = "# AdeptAI Environment Variables Template\n"
        template += "# Copy this file to .env and fill in your actual API keys\n"
        template += "# NEVER commit .env files to version control!\n\n"
        
        for var in cls.SENSITIVE_VARS:
            template += f"# {var.replace('_', ' ').title()}\n"
            template += f"{var}=your_{var.lower()}_here\n\n"
            
        return template
    
    @classmethod
    def get_security_report(cls) -> Dict[str, any]:
        """Generate comprehensive security report"""
        api_status = cls.validate_api_keys()
        hardcoded_issues = cls.scan_for_hardcoded_keys()
        
        return {
            'api_keys_configured': sum(api_status.values()),
            'api_keys_total': len(api_status),
            'api_status': api_status,
            'hardcoded_keys_found': len(hardcoded_issues),
            'hardcoded_issues': hardcoded_issues,
            'security_score': cls._calculate_security_score(api_status, hardcoded_issues)
        }
    
    @classmethod
    def _calculate_security_score(cls, api_status: Dict[str, bool], 
                                hardcoded_issues: List[Dict[str, str]]) -> int:
        """Calculate security score (0-100)"""
        score = 100
        
        # Deduct for missing API keys
        missing_keys = sum(1 for status in api_status.values() if not status)
        score -= missing_keys * 5
        
        # Deduct heavily for hardcoded keys
        score -= len(hardcoded_issues) * 20
        
        return max(0, min(100, score))


def validate_security() -> bool:
    """Main security validation function"""
    config = SecurityConfig()
    report = config.get_security_report()
    
    print("🔒 ADEPTAI SECURITY REPORT")
    print("=" * 50)
    print(f"Security Score: {report['security_score']}/100")
    print(f"API Keys Configured: {report['api_keys_configured']}/{report['api_keys_total']}")
    print(f"Hardcoded Keys Found: {report['hardcoded_keys_found']}")
    
    if report['hardcoded_keys_found'] > 0:
        print("\n🚨 CRITICAL: Hardcoded API keys found!")
        for issue in report['hardcoded_issues']:
            print(f"  - {issue['file']}:{issue['line']} ({issue['pattern']})")
        return False
    
    if report['security_score'] < 80:
        print("\n⚠️  WARNING: Security score below 80")
        return False
        
    print("\n✅ Security validation passed!")
    return True


if __name__ == "__main__":
    validate_security()
