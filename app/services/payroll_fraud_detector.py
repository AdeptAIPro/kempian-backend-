"""
Payroll-Specific Fraud Detection
Detects fraud patterns specific to payroll payments
"""
from decimal import Decimal
from datetime import datetime, timedelta
from app.models import db, PaymentTransaction, UserBankAccount, Payslip, PayRun, FraudAlert
from app.simple_logger import get_logger

logger = get_logger(__name__)


class PayrollFraudDetector:
    """Detect payroll-specific fraud patterns"""
    
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
    
    def detect_fraud(self, employee_id, amount, account_number, ifsc_code, payrun_id):
        """
        Detect fraud patterns for a payment
        
        Returns: {
            'is_fraud': bool,
            'risk_score': float (0-100),
            'flags': list of fraud flags,
            'requires_review': bool
        }
        """
        flags = []
        risk_score = 0.0
        
        # Check 1: Same bank account across multiple employees
        duplicate_account = self._check_duplicate_account(account_number, employee_id)
        if duplicate_account:
            flags.append({
                'type': 'duplicate_account',
                'severity': 'high',
                'message': f'Bank account used by {duplicate_account} other employee(s)',
                'details': {'duplicate_count': duplicate_account}
            })
            risk_score += 30
        
        # Check 2: Salary spike vs historical average
        spike_check = self._check_salary_spike(employee_id, amount)
        if spike_check['is_spike']:
            flags.append({
                'type': 'salary_spike',
                'severity': 'medium',
                'message': f'Salary {spike_check["deviation"]:.1f}% above average',
                'details': {
                    'current': float(amount),
                    'average': float(spike_check['average']),
                    'deviation': spike_check['deviation']
                }
            })
            risk_score += 20
        
        # Check 3: New bank account + high amount
        new_account_check = self._check_new_account_high_amount(employee_id, account_number, amount)
        if new_account_check['is_risk']:
            flags.append({
                'type': 'new_account_high_amount',
                'severity': 'high',
                'message': f'New bank account with high amount payment',
                'details': {
                    'account_age_days': new_account_check['age_days'],
                    'amount': float(amount)
                }
            })
            risk_score += 40
        
        # Check 4: Rapid bank changes before payroll
        rapid_change = self._check_rapid_bank_change(employee_id, account_number)
        if rapid_change:
            flags.append({
                'type': 'rapid_bank_change',
                'severity': 'high',
                'message': 'Bank account changed within 72 hours of payroll',
                'details': {'hours_since_change': rapid_change}
            })
            risk_score += 35
        
        # Check 5: Unverified bank account
        unverified = self._check_unverified_account(employee_id, account_number)
        if unverified:
            flags.append({
                'type': 'unverified_account',
                'severity': 'critical',
                'message': 'Bank account not verified by penny-drop',
                'details': {}
            })
            risk_score += 50
        
        # Check 6: Amount exceeds threshold
        if amount > Decimal('500000'):  # 5 lakhs
            flags.append({
                'type': 'high_amount',
                'severity': 'medium',
                'message': f'Payment amount exceeds ₹5 lakhs',
                'details': {'amount': float(amount)}
            })
            risk_score += 15
        
        # Determine if manual review required
        requires_review = risk_score >= 50 or any(f['severity'] == 'critical' for f in flags)
        
        return {
            'is_fraud': risk_score >= 70,
            'risk_score': min(risk_score, 100),
            'flags': flags,
            'requires_review': requires_review
        }
    
    def _check_duplicate_account(self, account_number, exclude_employee_id):
        """Check if account number is used by other employees"""
        count = UserBankAccount.query.filter(
            UserBankAccount.account_number == account_number,
            UserBankAccount.user_id != exclude_employee_id
        ).join(
            db.session.query(db.Model).filter_by(tenant_id=self.tenant_id).subquery()
        ).count()
        
        return count
    
    def _check_salary_spike(self, employee_id, current_amount):
        """Check if salary deviates significantly from historical average"""
        # Get last 6 months of payslips
        six_months_ago = datetime.now().date() - timedelta(days=180)
        
        payslips = Payslip.query.filter(
            Payslip.employee_id == employee_id,
            Payslip.net_pay.isnot(None),
            Payslip.created_at >= six_months_ago
        ).order_by(Payslip.created_at.desc()).limit(6).all()
        
        if len(payslips) < 3:  # Need at least 3 months of history
            return {'is_spike': False}
        
        # Calculate average
        amounts = [float(p.net_pay) for p in payslips]
        average = sum(amounts) / len(amounts)
        
        # Check deviation
        deviation = ((float(current_amount) - average) / average) * 100
        
        # Flag if > 30% increase
        is_spike = deviation > 30
        
        return {
            'is_spike': is_spike,
            'average': Decimal(str(average)),
            'deviation': deviation
        }
    
    def _check_new_account_high_amount(self, employee_id, account_number, amount):
        """Check if new account with high amount"""
        bank_account = UserBankAccount.query.filter_by(
            user_id=employee_id,
            account_number=account_number
        ).first()
        
        if not bank_account or not bank_account.created_at:
            return {'is_risk': True, 'age_days': 0}
        
        age_days = (datetime.utcnow() - bank_account.created_at).days
        
        # New account (< 30 days) with high amount (> 1 lakh)
        is_risk = age_days < 30 and amount > Decimal('100000')
        
        return {
            'is_risk': is_risk,
            'age_days': age_days
        }
    
    def _check_rapid_bank_change(self, employee_id, account_number):
        """Check if bank account changed recently"""
        bank_account = UserBankAccount.query.filter_by(
            user_id=employee_id,
            account_number=account_number
        ).first()
        
        if not bank_account or not bank_account.updated_at:
            return None
        
        hours_since_change = (datetime.utcnow() - bank_account.updated_at).total_seconds() / 3600
        
        # Flag if changed within 72 hours
        if hours_since_change < 72:
            return hours_since_change
        
        return None
    
    def _check_unverified_account(self, employee_id, account_number):
        """Check if account is verified"""
        bank_account = UserBankAccount.query.filter_by(
            user_id=employee_id,
            account_number=account_number
        ).first()
        
        if not bank_account:
            return True
        
        return not bank_account.verified_by_penny_drop


# FraudAlert model is defined in app/models.py

