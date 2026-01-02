"""
Payment Processing Service for Indian Bank Transfers
Supports Razorpay Payouts API for NEFT, RTGS, IMPS transfers
With comprehensive security and privacy measures
"""
import requests
import json
from decimal import Decimal
from datetime import datetime, timedelta
from app.models import db, PaymentTransaction, PayRunPayslip, UserBankAccount, EmployeeProfile, PayrollSettings
from app.simple_logger import get_logger
from app.utils.payment_security import (
    PaymentEncryption, DataMasking, WebhookVerification,
    PaymentAuditLogger, FraudDetection
)
from app.utils.razorpay_error_mapper import RazorpayErrorMapper
from app.services.payroll_fraud_detector import PayrollFraudDetector, FraudAlert

logger = get_logger(__name__)
encryption = PaymentEncryption()
audit_logger = PaymentAuditLogger()


class PaymentService:
    """Service for processing bank-to-bank transfers via payment gateways"""
    
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        self.settings = PayrollSettings.query.filter_by(tenant_id=tenant_id).first()
        if not self.settings:
            raise ValueError(f"No payroll settings found for tenant {tenant_id}")
        
        self.gateway = self.settings.payment_gateway or 'manual'
        self.payment_mode = self.settings.payment_mode or 'NEFT'
        
        if self.gateway == 'razorpay':
            # Decrypt API keys if they are encrypted
            encrypted_key_id = self.settings.razorpay_key_id
            encrypted_key_secret = self.settings.razorpay_key_secret
            
            try:
                # Try to decrypt (if encrypted) or use as-is
                if encrypted_key_id and encrypted_key_id.startswith('enc:'):
                    self.razorpay_key_id = encryption.decrypt(encrypted_key_id[4:])
                else:
                    self.razorpay_key_id = encrypted_key_id
                
                if encrypted_key_secret and encrypted_key_secret.startswith('enc:'):
                    self.razorpay_key_secret = encryption.decrypt(encrypted_key_secret[4:])
                else:
                    self.razorpay_key_secret = encrypted_key_secret
            except Exception as e:
                logger.error(f"Failed to decrypt Razorpay credentials: {str(e)}")
                # Fallback to plain text (for backward compatibility)
                self.razorpay_key_id = encrypted_key_id
                self.razorpay_key_secret = encrypted_key_secret
            
            if not self.razorpay_key_id or not self.razorpay_key_secret:
                raise ValueError("Razorpay credentials not configured")
            
            # Log masked key for audit (never log full key)
            logger.info(f"Razorpay initialized with key: {DataMasking.mask_api_key(self.razorpay_key_id)}")
            
            # Razorpay API endpoints
            self.razorpay_base_url = "https://api.razorpay.com/v1"
            if 'test' in (self.razorpay_key_id or '').lower() or 'rzp_test' in (self.razorpay_key_id or ''):
                self.razorpay_base_url = "https://api.razorpay.com/v1"  # Same URL for test and live
    
    def _get_razorpay_auth(self):
        """Get Razorpay Basic Auth header"""
        import base64
        credentials = f"{self.razorpay_key_id}:{self.razorpay_key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def _create_razorpay_contact(self, employee_id, employee_name, email, phone, bank_account=None):
        """Create or get Razorpay contact for employee (with caching)"""
        try:
            # Check if contact already cached in database
            if bank_account and bank_account.razorpay_contact_id:
                # Verify contact still exists in Razorpay
                headers = {
                    "Authorization": self._get_razorpay_auth(),
                    "Content-Type": "application/json"
                }
                try:
                    response = requests.get(
                        f"{self.razorpay_base_url}/contacts/{bank_account.razorpay_contact_id}",
                        headers=headers,
                        timeout=30
                    )
                    if response.status_code == 200:
                        logger.info(f"Using cached Razorpay contact: {bank_account.razorpay_contact_id}")
                        return response.json()
                except Exception as e:
                    logger.warning(f"Cached contact {bank_account.razorpay_contact_id} not found, creating new: {str(e)}")
                    # Contact doesn't exist, will create new one below
            
            # Create new contact
            contact_data = {
                "name": employee_name,
                "email": email,
                "contact": phone or "0000000000",  # Razorpay requires contact
                "type": "employee",
                "reference_id": f"emp_{employee_id}",
                "notes": {
                    "employee_id": str(employee_id)
                }
            }
            
            headers = {
                "Authorization": self._get_razorpay_auth(),
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.razorpay_base_url}/contacts",
                json=contact_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                contact = response.json()
                # Cache contact ID in database
                if bank_account:
                    bank_account.razorpay_contact_id = contact['id']
                    bank_account.razorpay_contact_created_at = datetime.utcnow()
                    db.session.commit()
                return contact
            elif response.status_code == 400:
                # Contact might already exist, try to fetch
                error_data = response.json()
                if 'already exists' in str(error_data).lower():
                    # Try to find existing contact
                    search_response = requests.get(
                        f"{self.razorpay_base_url}/contacts",
                        params={"email": email},
                        headers=headers,
                        timeout=30
                    )
                    if search_response.status_code == 200:
                        contacts = search_response.json()
                        if contacts.get('items'):
                            contact = contacts['items'][0]
                            # Cache contact ID
                            if bank_account:
                                bank_account.razorpay_contact_id = contact['id']
                                bank_account.razorpay_contact_created_at = datetime.utcnow()
                                db.session.commit()
                            return contact
                
                error_message = RazorpayErrorMapper.get_user_friendly_message(error_data=error_data)
                logger.error(f"Failed to create Razorpay contact: {error_data}")
                raise ValueError(f"Failed to create contact: {error_message}")
            else:
                try:
                    error_data = response.json()
                except:
                    error_data = {'error': {'description': response.text or 'Unknown error'}}
                error_message = RazorpayErrorMapper.get_user_friendly_message(error_data=error_data)
                logger.error(f"Razorpay contact creation failed: {response.status_code} - {response.text}")
                raise ValueError(f"Razorpay API error: {error_message}")
                
        except ValueError:
            # Re-raise ValueError (already user-friendly)
            raise
        except Exception as e:
            # Map other exceptions to user-friendly messages
            error_message = RazorpayErrorMapper.map_razorpay_exception(e)
            logger.error(f"Error creating Razorpay contact: {str(e)}")
            raise ValueError(error_message)
    
    def _create_razorpay_fund_account(self, contact_id, account_holder_name, account_number, ifsc_code, account_type='savings', bank_account=None):
        """Create or get Razorpay fund account for employee bank account (with caching)"""
        try:
            # Check if fund account already cached in database
            if bank_account and bank_account.razorpay_fund_account_id:
                # Verify fund account still exists in Razorpay
                headers = {
                    "Authorization": self._get_razorpay_auth(),
                    "Content-Type": "application/json"
                }
                try:
                    response = requests.get(
                        f"{self.razorpay_base_url}/fund_accounts/{bank_account.razorpay_fund_account_id}",
                        headers=headers,
                        timeout=30
                    )
                    if response.status_code == 200:
                        fund_account = response.json()
                        # Verify it matches current bank details
                        if (fund_account.get('bank_account', {}).get('account_number') == account_number and
                            fund_account.get('bank_account', {}).get('ifsc') == ifsc_code):
                            logger.info(f"Using cached Razorpay fund account: {bank_account.razorpay_fund_account_id}")
                            return fund_account
                        else:
                            logger.warning("Cached fund account details don't match, creating new one")
                except Exception as e:
                    logger.warning(f"Cached fund account {bank_account.razorpay_fund_account_id} not found, creating new: {str(e)}")
                    # Fund account doesn't exist, will create new one below
            
            # Create new fund account
            fund_account_data = {
                "contact_id": contact_id,
                "account_type": "bank_account",
                "bank_account": {
                    "name": account_holder_name,
                    "ifsc": ifsc_code,
                    "account_number": account_number
                }
            }
            
            headers = {
                "Authorization": self._get_razorpay_auth(),
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.razorpay_base_url}/fund_accounts",
                json=fund_account_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                fund_account = response.json()
                # Cache fund account ID in database
                if bank_account:
                    bank_account.razorpay_fund_account_id = fund_account['id']
                    bank_account.razorpay_fund_account_created_at = datetime.utcnow()
                    db.session.commit()
                return fund_account
            elif response.status_code == 400:
                # Fund account might already exist
                try:
                    error_data = response.json()
                except:
                    error_data = {'error': {'description': response.text or 'Unknown error'}}
                error_message = RazorpayErrorMapper.get_user_friendly_message(error_data=error_data)
                logger.warning(f"Fund account creation warning: {error_data}")
                raise ValueError(f"Fund account issue: {error_message}")
            else:
                try:
                    error_data = response.json()
                except:
                    error_data = {'error': {'description': response.text or 'Unknown error'}}
                error_message = RazorpayErrorMapper.get_user_friendly_message(error_data=error_data)
                logger.error(f"Razorpay fund account creation failed: {response.status_code} - {response.text}")
                raise ValueError(f"Razorpay API error: {error_message}")
                
        except ValueError:
            # Re-raise ValueError (already user-friendly)
            raise
        except Exception as e:
            # Map other exceptions to user-friendly messages
            error_message = RazorpayErrorMapper.map_razorpay_exception(e)
            logger.error(f"Error creating Razorpay fund account: {str(e)}")
            raise ValueError(error_message)
    
    def _validate_payment_mode(self, requested_mode, amount):
        """
        Validate and restrict payment modes based on amount and business rules
        
        Rules:
        - Default payroll: NEFT only
        - IMPS: Only for emergency/manual payouts
        - RTGS: High-value only (>₹2L)
        - UPI: Disabled for payroll (unless explicitly enabled)
        """
        amount_float = float(amount)
        
        # UPI is disabled for payroll by default
        if requested_mode == 'UPI':
            logger.warning("UPI mode requested but disabled for payroll. Defaulting to NEFT.")
            return 'NEFT'
        
        # RTGS only for high-value (>₹2L)
        if requested_mode == 'RTGS' and amount_float < 200000:
            logger.warning(f"RTGS requires minimum ₹2L. Amount: {amount_float}. Defaulting to NEFT.")
            return 'NEFT'
        
        # IMPS has limits (<₹2L)
        if requested_mode == 'IMPS' and amount_float > 200000:
            logger.warning(f"IMPS limit is ₹2L. Amount: {amount_float}. Defaulting to NEFT.")
            return 'NEFT'
        
        # Default to NEFT for regular payroll
        if requested_mode not in ['NEFT', 'RTGS', 'IMPS']:
            return 'NEFT'
        
        return requested_mode
    
    def _create_razorpay_payout(self, fund_account_id, amount, currency, mode, notes=None, idempotency_key=None):
        """Create a payout via Razorpay"""
        try:
            # Validate fund account ID
            if not self.settings.razorpay_fund_account_id:
                raise ValueError("Razorpay fund account ID not configured. Please configure in payroll settings.")
            
            payout_data = {
                "account_number": self.settings.razorpay_fund_account_id,
                "fund_account_id": fund_account_id,
                "amount": int(amount * 100),  # Convert to paise
                "currency": currency,
                "mode": mode,  # 'NEFT', 'RTGS', 'IMPS' (UPI disabled)
                "purpose": "salary",  # RBI purpose code
                "queue_if_low_balance": True,
                "reference_id": idempotency_key or f"PAYOUT_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "narration": notes or "Salary Payment"
            }
            
            headers = {
                "Authorization": self._get_razorpay_auth(),
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.razorpay_base_url}/payouts",
                json=payout_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                try:
                    error_data = response.json()
                except:
                    error_data = {'error': {'description': response.text or 'Unknown error'}}
                
                # Map error to user-friendly message
                error_message = RazorpayErrorMapper.get_user_friendly_message(error_data=error_data)
                logger.error(f"Razorpay payout creation failed: {response.status_code} - {error_data}")
                raise ValueError(error_message)
                
        except ValueError:
            # Re-raise ValueError (already user-friendly)
            raise
        except Exception as e:
            # Map other exceptions to user-friendly messages
            error_message = RazorpayErrorMapper.map_razorpay_exception(e)
            logger.error(f"Error creating Razorpay payout: {str(e)}")
            raise ValueError(error_message)
    
    def process_payment(self, pay_run_id, payslip_id, employee_id, amount, currency='INR', initiated_by=None):
        """
        Process a single payment for an employee with security checks
        
        Returns:
            dict: Payment transaction details
        """
        try:
            # Security: Validate amount
            if amount <= 0:
                raise ValueError("Payment amount must be greater than zero")
            if amount > 10000000:  # 1 crore limit
                raise ValueError("Payment amount exceeds maximum limit")
            
            # Fraud detection: Check amount threshold
            fraud_check = FraudDetection.check_amount_threshold(float(amount), currency)
            if fraud_check.get('suspicious') and fraud_check.get('severity') == 'critical':
                audit_logger.log_security_event('suspicious_amount', {
                    'amount': amount,
                    'currency': currency,
                    'employee_id': employee_id
                }, initiated_by)
                raise ValueError(f"Payment amount requires manual approval: {fraud_check.get('reason')}")
            
            # Get employee details
            from app.models import User
            employee = User.query.get(employee_id)
            if not employee:
                raise ValueError(f"Employee {employee_id} not found")
            
            employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
            if not employee_profile:
                raise ValueError(f"Employee profile not found for {employee_id}")
            
            # Get bank account details
            bank_account = UserBankAccount.query.filter_by(user_id=employee_id).first()
            if not bank_account:
                raise ValueError(f"Bank account not found for employee {employee_id}")
            
            # Security: Validate bank details
            if not bank_account.ifsc_code:
                raise ValueError(f"IFSC code not provided for employee {employee_id}")
            if not bank_account.account_number:
                raise ValueError(f"Account number not provided for employee {employee_id}")
            
            # Enhanced validation using PaymentValidators
            from app.utils.payment_validators import PaymentValidators
            validation = PaymentValidators.validate_bank_account_details(
                bank_account.account_number,
                bank_account.ifsc_code,
                account_holder_name
            )
            if not validation['valid']:
                errors = ', '.join(validation['errors'])
                raise ValueError(f"Invalid bank details: {errors}")
            
            # Validate payment amount
            amount_valid, amount_error = PaymentValidators.validate_payment_amount(amount, currency)
            if not amount_valid:
                raise ValueError(amount_error)
            
            # Fraud detection: Validate bank details format (additional check)
            fraud_validation = FraudDetection.validate_bank_details(
                bank_account.account_number,
                bank_account.ifsc_code
            )
            if not fraud_validation['valid']:
                errors = ', '.join(fraud_validation['errors'])
                raise ValueError(f"Bank details validation failed: {errors}")
            
            # Payroll-specific fraud detection
            fraud_detector = PayrollFraudDetector(tenant_id=self.tenant_id)
            fraud_result = fraud_detector.detect_fraud(
                employee_id=employee_id,
                amount=Decimal(str(amount)),
                account_number=bank_account.account_number,
                ifsc_code=bank_account.ifsc_code,
                payrun_id=pay_run_id
            )
            
            # Check bank account cooldown (72 hours)
            if bank_account.bank_change_cooldown_until:
                cooldown_until = bank_account.bank_change_cooldown_until
                if datetime.utcnow() < cooldown_until:
                    hours_remaining = (cooldown_until - datetime.utcnow).total_seconds() / 3600
                    raise ValueError(
                        f"Bank account changed recently. Cooldown period active. "
                        f"Please wait {hours_remaining:.1f} more hours."
                    )
            
            # Check if account is verified
            if not bank_account.verified_by_penny_drop:
                raise ValueError(
                    "Bank account not verified. Penny-drop verification required before processing payments."
                )
            
            # Block if fraud detected (high risk)
            if fraud_result['is_fraud']:
                # Create fraud alert
                fraud_alert = FraudAlert(
                    tenant_id=self.tenant_id,
                    pay_run_id=pay_run_id,
                    employee_id=employee_id,
                    alert_type='fraud_detected',
                    severity='critical',
                    risk_score=Decimal(str(fraud_result['risk_score'])),
                    flags=fraud_result['flags'],
                    status='pending'
                )
                db.session.add(fraud_alert)
                db.session.commit()
                
                raise ValueError(
                    f"Fraud detected. Risk score: {fraud_result['risk_score']}. "
                    f"Manual review required. Alert ID: {fraud_alert.id}"
                )
            
            # Require manual review if flagged
            if fraud_result['requires_review']:
                fraud_alert = FraudAlert(
                    tenant_id=self.tenant_id,
                    pay_run_id=pay_run_id,
                    employee_id=employee_id,
                    alert_type='requires_review',
                    severity='high' if fraud_result['risk_score'] >= 50 else 'medium',
                    risk_score=Decimal(str(fraud_result['risk_score'])),
                    flags=fraud_result['flags'],
                    status='pending'
                )
                db.session.add(fraud_alert)
                db.session.commit()
                
                # Log but don't block (admin can review)
                logger.warning(
                    f"Payment requires review: employee_id={employee_id}, "
                    f"risk_score={fraud_result['risk_score']}, alert_id={fraud_alert.id}"
                )
            
            account_holder_name = bank_account.account_holder_name or f"{employee_profile.first_name or ''} {employee_profile.last_name or ''}".strip()
            if not account_holder_name:
                account_holder_name = employee.email
            
            # Security: Mask sensitive data in logs
            masked_account = DataMasking.mask_account_number(bank_account.account_number)
            masked_ifsc = DataMasking.mask_ifsc(bank_account.ifsc_code)
            logger.info(
                f"Processing payment: employee_id={employee_id}, "
                f"amount={amount} {currency}, account={masked_account}, ifsc={masked_ifsc}"
            )
            
            # Generate idempotency key (prevent duplicate payouts)
            import uuid
            idempotency_key = f"PAYOUT_{pay_run_id}_{payslip_id}_{uuid.uuid4().hex[:16]}"
            
            # Check if payment already exists with this idempotency key
            existing_txn = PaymentTransaction.query.filter_by(
                idempotency_key=idempotency_key
            ).first()
            
            if existing_txn:
                logger.warning(f"Duplicate payment attempt detected: {idempotency_key}")
                return {
                    'success': True,
                    'transaction_id': existing_txn.id,
                    'gateway_transaction_id': existing_txn.gateway_payout_id,
                    'status': existing_txn.status,
                    'message': 'Payment already processed (idempotency check)',
                    'duplicate': True
                }
            
            # Validate payment mode restrictions
            payment_mode = self._validate_payment_mode(self.payment_mode, amount)
            
            # Create payment transaction record
            payment_txn = PaymentTransaction(
                pay_run_id=pay_run_id,
                payslip_id=payslip_id,
                employee_id=employee_id,
                tenant_id=self.tenant_id,
                amount=Decimal(str(amount)),
                currency=currency,
                payment_mode=payment_mode,
                beneficiary_name=account_holder_name,
                account_number=bank_account.account_number,
                ifsc_code=bank_account.ifsc_code,
                bank_name=bank_account.bank_name,
                gateway=self.gateway,
                status='pending',
                idempotency_key=idempotency_key,
                purpose_code='SALARY',  # RBI purpose code
                payout_category='salary',
                fraud_risk_score=Decimal(str(fraud_result['risk_score'])),
                fraud_flags=fraud_result['flags'],
                requires_manual_review=fraud_result['requires_review']
            )
            db.session.add(payment_txn)
            db.session.flush()
            
            # Audit log: Payment initiated
            audit_logger.log_payment_initiated(
                payment_txn.id,
                employee_id,
                float(amount),
                currency,
                initiated_by or 0
            )
            
            # Process based on gateway
            if self.gateway == 'razorpay':
                # Pre-flight checks before processing
                # 1. Check KYC status
                from app.services.wallet_balance_service import WalletBalanceService
                wallet_service = WalletBalanceService(tenant_id=self.tenant_id)
                kyc_status = wallet_service.wallet.kyc_status
                if not kyc_status or kyc_status not in ['approved', 'verified']:
                    status_display = kyc_status or 'not set'
                    raise ValueError(
                        f"Employer KYC not approved. Current status: {status_display}. "
                        "Please complete KYC verification in Razorpay dashboard before processing payments."
                    )
                
                # 2. Check Razorpay account status
                razorpay_account_status = wallet_service.wallet.razorpay_account_status
                if razorpay_account_status and razorpay_account_status not in ['active', 'live']:
                    raise ValueError(
                        f"Razorpay account not active. Current status: {razorpay_account_status}. "
                        "Please contact Razorpay support to activate your account."
                    )
                
                # 3. Check fund account validation
                if not self.settings.razorpay_fund_account_validated:
                    raise ValueError(
                        "Razorpay fund account not validated. Please validate your fund account in payroll settings."
                    )
                
                # 4. Validate fund account ID exists
                if not self.settings.razorpay_fund_account_id:
                    raise ValueError("Razorpay fund account ID not configured. Please configure in payroll settings.")
                
                try:
                    # Step 1: Create/get contact (with caching)
                    contact = self._create_razorpay_contact(
                        employee_id=employee_id,
                        employee_name=account_holder_name,
                        email=employee.email,
                        phone=bank_account.contact_phone or employee_profile.phone or "0000000000",
                        bank_account=bank_account
                    )
                    contact_id = contact['id']
                    
                    # Step 2: Create/get fund account (with caching)
                    fund_account = self._create_razorpay_fund_account(
                        contact_id=contact_id,
                        account_holder_name=account_holder_name,
                        account_number=bank_account.account_number,
                        ifsc_code=bank_account.ifsc_code,
                        account_type=bank_account.account_type or 'savings',
                        bank_account=bank_account
                    )
                    fund_account_id = fund_account['id']
                    
                    # Step 3: Create payout (with validated mode and idempotency)
                    validated_mode = self._validate_payment_mode(self.payment_mode, amount)
                    payout = self._create_razorpay_payout(
                        fund_account_id=fund_account_id,
                        amount=float(amount),
                        currency=currency,
                        mode=validated_mode,
                        notes=f"Salary payment for {account_holder_name} - Payslip #{payslip_id}",
                        idempotency_key=payment_txn.idempotency_key
                    )
                    
                    # Update transaction record
                    payment_txn.gateway_transaction_id = payout.get('id')
                    payment_txn.gateway_payout_id = payout.get('id')
                    payment_txn.gateway_response = payout
                    payment_txn.status = 'processing'
                    payment_txn.processed_at = datetime.utcnow()
                    
                    # Update pay run payslip
                    payrun_payslip = PayRunPayslip.query.filter_by(
                        pay_run_id=pay_run_id,
                        payslip_id=payslip_id
                    ).first()
                    if payrun_payslip:
                        payrun_payslip.payment_status = 'processing'
                        payrun_payslip.payment_reference = payout.get('id')
                        payrun_payslip.payment_method = f"{self.gateway}_{self.payment_mode.lower()}"
                    
                    db.session.commit()
                    
                    # Audit log: Payment success
                    audit_logger.log_payment_success(
                        payment_txn.id,
                        payout.get('id'),
                        float(amount)
                    )
                    
                    # Log with masked data
                    logger.info(
                        f"Payment initiated successfully: transaction_id={payment_txn.id}, "
                        f"gateway_id={payout.get('id')}, amount={amount} {currency}"
                    )
                    
                    return {
                        'success': True,
                        'transaction_id': payment_txn.id,
                        'gateway_transaction_id': payout.get('id'),
                        'status': 'processing',
                        'message': 'Payment initiated successfully'
                    }
                    
                except requests.exceptions.RequestException as e:
                    # Handle Razorpay API errors
                    error_message = str(e)
                    error_data = None
                    
                    # Try to extract error from response if available
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_data = e.response.json()
                            error_message = RazorpayErrorMapper.get_user_friendly_message(error_data=error_data)
                        except:
                            pass
                    
                    # Mark transaction as failed
                    payment_txn.status = 'failed'
                    payment_txn.failure_reason = error_message
                    payment_txn.processed_at = datetime.utcnow()
                    if error_data:
                        payment_txn.gateway_response = error_data
                    
                    payrun_payslip = PayRunPayslip.query.filter_by(
                        pay_run_id=pay_run_id,
                        payslip_id=payslip_id
                    ).first()
                    if payrun_payslip:
                        payrun_payslip.payment_status = 'failed'
                    
                    db.session.commit()
                    
                    # Audit log: Payment failure
                    audit_logger.log_payment_failure(
                        payment_txn.id,
                        error_message,
                        error_data or {}
                    )
                    
                    # Log error without exposing sensitive data
                    logger.error(
                        f"Payment processing failed: transaction_id={payment_txn.id}, "
                        f"error={error_message}"
                    )
                    raise ValueError(error_message)
                    
                except Exception as e:
                    # Mark transaction as failed
                    error_message = RazorpayErrorMapper.map_razorpay_exception(e)
                    payment_txn.status = 'failed'
                    payment_txn.failure_reason = error_message
                    payment_txn.processed_at = datetime.utcnow()
                    
                    payrun_payslip = PayRunPayslip.query.filter_by(
                        pay_run_id=pay_run_id,
                        payslip_id=payslip_id
                    ).first()
                    if payrun_payslip:
                        payrun_payslip.payment_status = 'failed'
                    
                    db.session.commit()
                    
                    # Audit log: Payment failure
                    audit_logger.log_payment_failure(
                        payment_txn.id,
                        error_message,
                        {}
                    )
                    
                    # Log error without exposing sensitive data
                    logger.error(
                        f"Payment processing failed: transaction_id={payment_txn.id}, "
                        f"error={error_message}"
                    )
                    raise ValueError(error_message)
                    
            elif self.gateway == 'manual':
                # Manual processing - just mark as pending
                payment_txn.status = 'pending'
                payment_txn.failure_reason = 'Manual processing required'
                db.session.commit()
                
                return {
                    'success': True,
                    'transaction_id': payment_txn.id,
                    'status': 'pending',
                    'message': 'Payment queued for manual processing'
                }
            else:
                raise ValueError(f"Unsupported payment gateway: {self.gateway}")
                
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing payment: {str(e)}")
            raise
    
    def process_bulk_payments(self, pay_run_id, payslip_ids, initiated_by=None):
        """
        Process multiple payments for a pay run with security checks
        
        Returns:
            dict: Summary of processed payments
        """
        # Security: Limit bulk payment size
        if len(payslip_ids) > 100:
            raise ValueError("Bulk payment limit exceeded. Maximum 100 payments per batch.")
        
        results = {
            'success': [],
            'failed': [],
            'total': len(payslip_ids)
        }
        
        for payslip_id in payslip_ids:
            try:
                # Get payslip details
                from app.models import Payslip
                payslip = Payslip.query.get(payslip_id)
                if not payslip:
                    results['failed'].append({
                        'payslip_id': payslip_id,
                        'error': 'Payslip not found'
                    })
                    continue
                
                result = self.process_payment(
                    pay_run_id=pay_run_id,
                    payslip_id=payslip_id,
                    employee_id=payslip.employee_id,
                    amount=float(payslip.net_pay or 0),
                    currency=payslip.currency or 'INR',
                    initiated_by=initiated_by
                )
                
                results['success'].append({
                    'payslip_id': payslip_id,
                    'transaction_id': result.get('transaction_id'),
                    'status': result.get('status')
                })
                
            except Exception as e:
                results['failed'].append({
                    'payslip_id': payslip_id,
                    'error': str(e)
                })
                logger.error(f"Failed to process payment for payslip {payslip_id}: {str(e)}")
        
        return results
    
    def retry_payment(self, transaction_id, initiated_by=None):
        """
        Retry a failed payment transaction
        
        Returns:
            dict: Updated transaction details
        """
        from app.models import PaymentTransaction
        
        transaction = PaymentTransaction.query.get(transaction_id)
        if not transaction:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        if transaction.status != 'failed':
            raise ValueError(f"Cannot retry transaction with status: {transaction.status}")
        
        if transaction.retry_count >= transaction.max_retries:
            raise ValueError(f"Maximum retry limit ({transaction.max_retries}) reached")
        
        # Increment retry count
        transaction.retry_count += 1
        transaction.last_retry_at = datetime.utcnow()
        transaction.status = 'pending'
        
        db.session.commit()
        
        # Process the payment again
        try:
            result = self.process_payment(
                pay_run_id=transaction.pay_run_id,
                payslip_id=transaction.payslip_id,
                employee_id=transaction.employee_id,
                amount=float(transaction.amount),
                currency=transaction.currency,
                initiated_by=initiated_by
            )
            
            # Update transaction with new gateway details
            if result.get('gateway_payout_id'):
                transaction.gateway_payout_id = result['gateway_payout_id']
                transaction.gateway_transaction_id = result.get('gateway_transaction_id')
                transaction.gateway_response = result.get('gateway_response')
            
            transaction.status = 'processing'
            db.session.commit()
            
            logger.info(f"Payment {transaction_id} retry successful (attempt {transaction.retry_count})")
            
            return transaction.to_dict()
            
        except Exception as e:
            transaction.status = 'failed'
            transaction.failure_reason = f"Retry failed: {str(e)}"
            db.session.commit()
            
            logger.error(f"Payment {transaction_id} retry failed: {str(e)}")
            raise
    
    def check_payment_status(self, transaction_id):
        """Check the status of a payment transaction"""
        try:
            payment_txn = PaymentTransaction.query.get(transaction_id)
            if not payment_txn:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            if self.gateway == 'razorpay' and payment_txn.gateway_payout_id:
                # Query Razorpay for latest status
                headers = {
                    "Authorization": self._get_razorpay_auth(),
                    "Content-Type": "application/json"
                }
                
                response = requests.get(
                    f"{self.razorpay_base_url}/payouts/{payment_txn.gateway_payout_id}",
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    payout_data = response.json()
                    
                    # Map Razorpay status to our status
                    razorpay_status = payout_data.get('status', '').lower()
                    if razorpay_status == 'processed':
                        payment_txn.status = 'success'
                        payment_txn.completed_at = datetime.utcnow()
                    elif razorpay_status == 'failed' or razorpay_status == 'reversed':
                        payment_txn.status = 'failed'
                        payment_txn.failure_reason = payout_data.get('failure_reason', 'Payment failed')
                    elif razorpay_status == 'queued' or razorpay_status == 'pending':
                        payment_txn.status = 'processing'
                    
                    payment_txn.gateway_response = payout_data
                    db.session.commit()
                    
                    return payment_txn.to_dict()
                else:
                    logger.error(f"Failed to fetch payout status: {response.status_code}")
                    return payment_txn.to_dict()
            else:
                return payment_txn.to_dict()
                
        except Exception as e:
            logger.error(f"Error checking payment status: {str(e)}")
            raise

