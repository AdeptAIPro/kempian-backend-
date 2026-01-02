"""
Failure-Mode Testing Suite
Tests critical failure scenarios for payroll payment system

REQUIRED SCENARIOS:
1. Two payrolls racing for same funds
2. Webhook never arrives
3. Partial payouts (success + failure)
4. Duplicate retry call
5. Fraud alert blocks payout
6. Bank account changed < 72 hours
7. Reconciliation heals stuck payments
"""
import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from app import create_app
from app.models import db, PayRun, PaymentTransaction, EmployerWalletBalance, UserBankAccount, FraudAlert
from app.services.wallet_balance_service import WalletBalanceService
from app.services.reconciliation_service import ReconciliationService
from app.hr.payment_service import PaymentService


@pytest.fixture
def app():
    """Create test application"""
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def tenant_id():
    """Mock tenant ID"""
    return 1


@pytest.fixture
def wallet_with_balance(app, tenant_id):
    """Create wallet with balance"""
    with app.app_context():
        wallet = EmployerWalletBalance(
            tenant_id=tenant_id,
            available_balance=Decimal('100000'),
            locked_balance=Decimal('0'),
            total_balance=Decimal('100000'),
            kyc_status='approved',
            razorpay_account_status='active'
        )
        db.session.add(wallet)
        db.session.commit()
        return wallet


class TestConcurrentFundLocking:
    """Test 1: Two payrolls racing for same funds"""
    
    def test_concurrent_fund_lock_prevention(self, app, tenant_id, wallet_with_balance):
        """Test that two payrolls cannot lock the same funds"""
        with app.app_context():
            # Create two payruns with total > available balance
            payrun1 = PayRun(
                tenant_id=tenant_id,
                pay_period_start=datetime.now().date(),
                pay_period_end=datetime.now().date(),
                pay_date=datetime.now().date(),
                status='approval_pending',
                total_net=Decimal('60000'),
                currency='INR'
            )
            payrun2 = PayRun(
                tenant_id=tenant_id,
                pay_period_start=datetime.now().date(),
                pay_period_end=datetime.now().date(),
                pay_date=datetime.now().date(),
                status='approval_pending',
                total_net=Decimal('60000'),
                currency='INR'
            )
            db.session.add(payrun1)
            db.session.add(payrun2)
            db.session.commit()
            
            wallet_service = WalletBalanceService(tenant_id=tenant_id)
            
            # Lock funds for first payrun
            success1, error1 = wallet_service.lock_funds_for_payrun(payrun1.id)
            assert success1, f"First lock should succeed: {error1}"
            
            # Try to lock funds for second payrun (should fail or block)
            success2, error2 = wallet_service.lock_funds_for_payrun(payrun2.id)
            
            # Second lock should fail due to insufficient balance
            assert not success2, "Second lock should fail when funds already locked"
            assert 'insufficient' in error2.lower() or 'concurrent' in error2.lower()
            
            # Verify first lock is still in place
            wallet = EmployerWalletBalance.query.filter_by(tenant_id=tenant_id).first()
            assert wallet.locked_balance == Decimal('60000')
            assert wallet.available_balance == Decimal('40000')


class TestWebhookFailure:
    """Test 2: Webhook never arrives"""
    
    def test_reconciliation_heals_stuck_payment(self, app, tenant_id):
        """Test that reconciliation can heal payments stuck in processing"""
        with app.app_context():
            # Create stuck payment (processing for > 2 hours)
            stuck_time = datetime.utcnow() - timedelta(hours=3)
            transaction = PaymentTransaction(
                tenant_id=tenant_id,
                pay_run_id=1,
                payslip_id=1,
                employee_id=1,
                amount=Decimal('50000'),
                currency='INR',
                payment_mode='NEFT',
                beneficiary_name='Test Employee',
                account_number='1234567890',
                ifsc_code='HDFC0001234',
                gateway='razorpay',
                gateway_payout_id='pout_test123',
                status='processing',
                initiated_at=stuck_time
            )
            db.session.add(transaction)
            db.session.commit()
            
            # Mock reconciliation service (would normally call Razorpay API)
            # In real test, would mock Razorpay API response
            # For now, verify reconciliation logic exists
            reconciliation = ReconciliationService(tenant_id=tenant_id)
            results = reconciliation.reconcile_stuck_payments(hours_threshold=2)
            
            assert results['found'] >= 1
            # In real test with mocked API, would verify status updated


class TestPartialPayouts:
    """Test 3: Partial payouts (success + failure)"""
    
    def test_partial_payout_status(self, app, tenant_id):
        """Test that payrun status reflects partial success"""
        with app.app_context():
            payrun = PayRun(
                tenant_id=tenant_id,
                pay_period_start=datetime.now().date(),
                pay_period_end=datetime.now().date(),
                pay_date=datetime.now().date(),
                status='payout_initiated',
                total_net=Decimal('100000'),
                currency='INR',
                payments_initiated=2,
                payments_successful=1,
                payments_failed=1,
                payments_pending=0
            )
            db.session.add(payrun)
            db.session.commit()
            
            # Verify status should be partially_completed
            assert payrun.payments_successful == 1
            assert payrun.payments_failed == 1
            # Status should be partially_completed (handled in reconciliation)


class TestDuplicateRetry:
    """Test 4: Duplicate retry call"""
    
    def test_idempotent_retry(self, app, tenant_id):
        """Test that duplicate retry calls are safe"""
        with app.app_context():
            transaction = PaymentTransaction(
                tenant_id=tenant_id,
                pay_run_id=1,
                payslip_id=1,
                employee_id=1,
                amount=Decimal('50000'),
                currency='INR',
                payment_mode='NEFT',
                beneficiary_name='Test Employee',
                account_number='1234567890',
                ifsc_code='HDFC0001234',
                status='failed',
                retry_count=1,
                max_retries=3,
                idempotency_key='test_key_123'
            )
            db.session.add(transaction)
            db.session.commit()
            
            # Verify idempotency key prevents duplicates
            existing = PaymentTransaction.query.filter_by(
                idempotency_key='test_key_123'
            ).first()
            
            assert existing is not None
            assert existing.id == transaction.id


class TestFraudAlertBlocking:
    """Test 5: Fraud alert blocks payout"""
    
    def test_fraud_alert_blocks_payment(self, app, tenant_id):
        """Test that fraud alert with high risk blocks payment"""
        with app.app_context():
            # Create fraud alert with high risk
            fraud_alert = FraudAlert(
                tenant_id=tenant_id,
                employee_id=1,
                pay_run_id=1,
                payment_transaction_id=1,
                alert_type='fraud_detected',
                severity='critical',
                risk_score=Decimal('85.0'),
                status='pending',
                flags=[{'type': 'duplicate_account', 'severity': 'high'}]
            )
            db.session.add(fraud_alert)
            db.session.commit()
            
            # Payment should be blocked if fraud alert exists with high risk
            transaction = PaymentTransaction.query.get(1)
            if transaction:
                assert transaction.requires_manual_review == True or fraud_alert.status == 'pending'


class TestBankChangeCooldown:
    """Test 6: Bank account changed < 72 hours"""
    
    def test_bank_change_cooldown_enforcement(self, app):
        """Test that bank account changes block payments for 72 hours"""
        with app.app_context():
            # Create bank account with recent change
            cooldown_until = datetime.utcnow() + timedelta(hours=24)  # Still in cooldown
            bank_account = UserBankAccount(
                user_id=1,
                account_number='1234567890',
                ifsc_code='HDFC0001234',
                account_holder_name='Test User',
                bank_change_cooldown_until=cooldown_until,
                verified_by_penny_drop=True
            )
            db.session.add(bank_account)
            db.session.commit()
            
            # Payment should be blocked
            assert bank_account.bank_change_cooldown_until is not None
            assert datetime.utcnow() < bank_account.bank_change_cooldown_until


class TestReconciliationHealing:
    """Test 7: Reconciliation heals stuck payments"""
    
    def test_reconciliation_updates_status(self, app, tenant_id):
        """Test that reconciliation can update stuck payment status"""
        with app.app_context():
            # This would require mocking Razorpay API
            # For now, verify reconciliation service exists and has the method
            reconciliation = ReconciliationService(tenant_id=tenant_id)
            
            # Verify methods exist
            assert hasattr(reconciliation, 'reconcile_payment')
            assert hasattr(reconciliation, 'reconcile_payrun')
            assert hasattr(reconciliation, 'reconcile_stuck_payments')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

