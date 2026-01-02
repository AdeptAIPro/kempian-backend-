"""
Pay Run Management API
Handles payroll run creation, approval, and processing
"""
from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, PayRun, PayRunPayslip, Payslip, PaymentTransaction, db
from datetime import datetime
from decimal import Decimal

hr_payruns_bp = Blueprint('hr_payruns', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@hr_payruns_bp.route('/', methods=['GET'], strict_slashes=False)
def list_payruns():
    """List pay runs for current tenant"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view pay runs'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    status = request.args.get('status')
    query = PayRun.query.filter_by(tenant_id=tenant_id)
    
    if status:
        query = query.filter_by(status=status)
    
    payruns = query.order_by(PayRun.pay_date.desc(), PayRun.created_at.desc()).all()
    
    return jsonify({'pay_runs': [pr.to_dict() for pr in payruns]}), 200


@hr_payruns_bp.route('/', methods=['POST'], strict_slashes=False)
def create_payrun():
    """Create a new pay run"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to create pay runs'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json()
    
    pay_period_start_str = data.get('pay_period_start')
    pay_period_end_str = data.get('pay_period_end')
    pay_date_str = data.get('pay_date')
    payslip_ids = data.get('payslip_ids', [])
    
    if not pay_period_start_str or not pay_period_end_str or not pay_date_str:
        return jsonify({'error': 'pay_period_start, pay_period_end, and pay_date are required'}), 400
    
    try:
        pay_period_start = datetime.strptime(pay_period_start_str, '%Y-%m-%d').date()
        pay_period_end = datetime.strptime(pay_period_end_str, '%Y-%m-%d').date()
        pay_date = datetime.strptime(pay_date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    # Get payslips if provided, otherwise find all payslips for the period
    if payslip_ids:
        payslips = Payslip.query.filter(
            Payslip.id.in_(payslip_ids),
            Payslip.pay_period_start == pay_period_start,
            Payslip.pay_period_end == pay_period_end
        ).all()
    else:
        # Get all payslips for the period that aren't already in a pay run
        existing_payrun_payslips = db.session.query(PayRunPayslip.payslip_id).subquery()
        payslips = Payslip.query.filter(
            Payslip.pay_period_start == pay_period_start,
            Payslip.pay_period_end == pay_period_end,
            ~Payslip.id.in_(db.session.query(existing_payrun_payslips))
        ).all()
    
    if not payslips:
        return jsonify({'error': 'No payslips found for this pay period'}), 400
    
    # Calculate totals
    total_gross = sum(float(p.gross_earnings or 0) for p in payslips)
    total_net = sum(float(p.net_pay or 0) for p in payslips)
    total_tax = sum(float(p.tax_deduction or 0) for p in payslips)
    total_deductions = sum(float(p.total_deductions or 0) for p in payslips)
    
    # Get currency from first payslip
    currency = payslips[0].currency if payslips else 'USD'
    
    # Create pay run
    payrun = PayRun(
        tenant_id=tenant_id,
        pay_period_start=pay_period_start,
        pay_period_end=pay_period_end,
        pay_date=pay_date,
        status='draft',
        total_gross=Decimal(str(total_gross)),
        total_net=Decimal(str(total_net)),
        total_tax=Decimal(str(total_tax)),
        total_deductions=Decimal(str(total_deductions)),
        total_employees=len(payslips),
        currency=currency,
        notes=data.get('notes'),
        created_by=db_user.id
    )
    
    db.session.add(payrun)
    db.session.flush()  # Get the payrun ID
    
    # Link payslips to pay run
    for payslip in payslips:
        payrun_payslip = PayRunPayslip(
            pay_run_id=payrun.id,
            payslip_id=payslip.id,
            payment_method=data.get('payment_method', 'direct_deposit'),
            payment_status='pending'
        )
        db.session.add(payrun_payslip)
    
    db.session.commit()
    
    return jsonify({'pay_run': payrun.to_dict()}), 201


@hr_payruns_bp.route('/<int:payrun_id>', methods=['GET'], strict_slashes=False)
def get_payrun(payrun_id):
    """Get pay run details"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    payrun = PayRun.query.get(payrun_id)
    if not payrun:
        return jsonify({'error': 'Pay run not found'}), 404
    
    # Check permissions
    if payrun.tenant_id != get_current_tenant_id() and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to view this pay run'}), 403
    
    payrun_data = payrun.to_dict()
    
    # Get payslips in this pay run
    payrun_payslips = PayRunPayslip.query.filter_by(pay_run_id=payrun_id).all()
    payrun_data['payslips'] = [pp.to_dict() for pp in payrun_payslips]
    
    return jsonify({'pay_run': payrun_data}), 200


@hr_payruns_bp.route('/<int:payrun_id>/approve', methods=['POST'], strict_slashes=False)
def approve_payrun(payrun_id):
    """Approve a pay run - moves to approval_pending state"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to approve pay runs'}), 403
    
    payrun = PayRun.query.get(payrun_id)
    if not payrun:
        return jsonify({'error': 'Pay run not found'}), 404
    
    if payrun.status != 'draft':
        return jsonify({'error': f'Can only approve draft pay runs. Current status: {payrun.status}'}), 400
    
    # Move to approval_pending (next step is funds validation)
    payrun.status = 'approval_pending'
    payrun.approved_by = db_user.id
    
    db.session.commit()
    
    return jsonify({
        'pay_run': payrun.to_dict(),
        'message': 'Pay run approved. Next: Validate funds before processing payments.'
    }), 200


@hr_payruns_bp.route('/<int:payrun_id>/process', methods=['POST'], strict_slashes=False)
def process_payrun(payrun_id):
    """Process a pay run - initiate bank transfers for all payslips"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to process pay runs'}), 403
    
    payrun = PayRun.query.get(payrun_id)
    if not payrun:
        return jsonify({'error': 'Pay run not found'}), 404
    
    # State machine validation
    valid_transitions = {
        'draft': ['approval_pending'],
        'approval_pending': ['funds_validated'],
        'funds_validated': ['payout_initiated'],
        'payout_initiated': ['partially_completed', 'completed', 'failed'],
        'partially_completed': ['completed', 'failed'],
        'completed': ['reversed'],
        'failed': ['payout_initiated'],  # Can retry
        'reversed': []
    }
    
    if payrun.status not in valid_transitions:
        return jsonify({'error': f'Invalid current status: {payrun.status}'}), 400
    
    data = request.get_json() or {}
    initiate_payments = data.get('initiate_payments', False)  # Default to False for safety
    new_status = data.get('status')
    
    # Determine next status based on current state and action
    if not new_status:
        if payrun.status == 'draft':
            new_status = 'approval_pending'
        elif payrun.status == 'approval_pending':
            new_status = 'funds_validated'
        elif payrun.status == 'funds_validated' and initiate_payments:
            new_status = 'payout_initiated'
        else:
            new_status = payrun.status
    
    # Validate state transition
    if new_status not in valid_transitions.get(payrun.status, []):
        return jsonify({
            'error': f'Invalid state transition from {payrun.status} to {new_status}',
            'valid_transitions': valid_transitions.get(payrun.status, [])
        }), 400
    
    # Handle state transitions
    if new_status == 'approval_pending':
        # Move to approval pending (no action needed, just state change)
        payrun.status = new_status
    
    elif new_status == 'funds_validated':
        # Validate funds before proceeding
        from app.services.wallet_balance_service import WalletBalanceService
        wallet_service = WalletBalanceService(tenant_id=payrun.tenant_id)
        
        # Sync balance first
        from app.models import PayrollSettings
        settings = PayrollSettings.query.filter_by(tenant_id=payrun.tenant_id).first()
        if settings and settings.razorpay_key_id and settings.razorpay_key_secret:
            wallet_service.sync_balance_from_razorpay(
                settings.razorpay_key_id,
                settings.razorpay_key_secret
            )
        
        # Validate funds
        is_valid, error, available, required = wallet_service.validate_funds_for_payrun(payrun.id)
        if not is_valid:
            return jsonify({
                'error': f'Funds validation failed: {error}',
                'available': float(available),
                'required': float(required)
            }), 400
        
        # Lock funds
        lock_success, lock_error = wallet_service.lock_funds_for_payrun(payrun.id)
        if not lock_success:
            return jsonify({'error': f'Failed to lock funds: {lock_error}'}), 400
        
        payrun.status = new_status
        payrun.funds_validated_by = db_user.id
        payrun.funds_validated_at = datetime.utcnow()
    
    elif new_status == 'payout_initiated' and initiate_payments:
        # Process bank transfers
        payrun.status = new_status
        try:
            from app.hr.payment_service import PaymentService
            
            # Initialize payment service
            payment_service = PaymentService(tenant_id=payrun.tenant_id)
            
            # Get all payslips in this pay run
            payrun_payslips = PayRunPayslip.query.filter_by(pay_run_id=payrun_id).all()
            payslip_ids = [pp.payslip_id for pp in payrun_payslips if pp.payslip]
            
            if payslip_ids:
                # Generate correlation ID for tracking
                import uuid
                correlation_id = f"PAYRUN_{payrun_id}_{uuid.uuid4().hex[:12]}"
                payrun.correlation_id = correlation_id
                
                # Process bulk payments
                payment_results = payment_service.process_bulk_payments(
                    pay_run_id=payrun_id,
                    payslip_ids=payslip_ids,
                    initiated_by=db_user.id
                )
                
                # Update payment statistics
                payrun.payments_initiated = len(payslip_ids)
                payrun.payments_successful = len(payment_results.get('success', []))
                payrun.payments_failed = len(payment_results.get('failed', []))
                payrun.payments_pending = payrun.payments_initiated - payrun.payments_successful - payrun.payments_failed
                
                # Determine final status based on results
                if payment_results['failed'] and not payment_results['success']:
                    # All failed
                    payrun.status = 'failed'
                    # Unlock funds
                    from app.services.wallet_balance_service import WalletBalanceService
                    wallet_service = WalletBalanceService(tenant_id=payrun.tenant_id)
                    wallet_service.unlock_funds_for_payrun(payrun_id)
                elif payment_results['failed']:
                    # Partial success
                    payrun.status = 'partially_completed'
                elif payment_results['success']:
                    # All succeeded (will be marked completed when webhooks arrive)
                    payrun.status = 'payout_initiated'
                
                # Update pay run status based on results
                if payment_results['failed']:
                    # Some payments failed
                    if not payment_results['success']:
                        # All failed
                        payrun.status = 'failed'
                    # Otherwise keep as processing (partial success)
                elif payment_results['success']:
                    # All succeeded - will be marked completed when payments are confirmed
                    pass
            
            payrun.processed_at = datetime.utcnow()
            
        except Exception as e:
            # Log error but don't fail the entire request
            import traceback
            from app.simple_logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error processing payments: {str(e)}\n{traceback.format_exc()}")
            
            payrun.status = 'failed'
            payment_results = {
                'error': str(e),
                'success': [],
                'failed': []
            }
    
    elif new_status == 'completed':
        # Mark as completed (after all payments confirmed)
        payrun.status = new_status
        payrun.processed_at = datetime.utcnow()
        
        # Update payslip statuses to 'paid'
        payrun_payslips = PayRunPayslip.query.filter_by(pay_run_id=payrun_id).all()
        for pp in payrun_payslips:
            if pp.payslip:
                pp.payslip.status = 'paid'
                if pp.payment_status == 'pending':
                    pp.payment_status = 'processed'
        
        # Unlock any remaining locked funds (should be zero if all succeeded)
        from app.services.wallet_balance_service import WalletBalanceService
        wallet_service = WalletBalanceService(tenant_id=payrun.tenant_id)
        # Funds are unlocked as payments complete, but ensure cleanup
        if payrun.funds_locked and payrun.payments_failed == 0:
            wallet_service.unlock_funds_for_payrun(payrun_id)
    
    elif new_status == 'failed':
        payrun.status = new_status
        # Unlock funds on failure
        from app.services.wallet_balance_service import WalletBalanceService
        wallet_service = WalletBalanceService(tenant_id=payrun.tenant_id)
        wallet_service.unlock_funds_for_payrun(payrun_id)
    
    elif new_status == 'reversed':
        payrun.status = new_status
        # Unlock funds on reversal
        from app.services.wallet_balance_service import WalletBalanceService
        wallet_service = WalletBalanceService(tenant_id=payrun.tenant_id)
        wallet_service.unlock_funds_for_payrun(payrun_id)
    
    db.session.commit()
    
    response_data = {'pay_run': payrun.to_dict()}
    if payment_results:
        response_data['payment_results'] = payment_results
    
    return jsonify(response_data), 200


@hr_payruns_bp.route('/<int:payrun_id>/payslips/<int:payslip_id>', methods=['PUT'], strict_slashes=False)
def update_payrun_payslip(payrun_id, payslip_id):
    """Update payment details for a payslip in a pay run"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to update pay run payslips'}), 403
    
    payrun_payslip = PayRunPayslip.query.filter_by(
        pay_run_id=payrun_id,
        payslip_id=payslip_id
    ).first()
    
    if not payrun_payslip:
        return jsonify({'error': 'Pay run payslip not found'}), 404
    
    data = request.get_json()
    
    if 'payment_method' in data:
        payrun_payslip.payment_method = data['payment_method']
    if 'payment_status' in data:
        payrun_payslip.payment_status = data['payment_status']
    if 'payment_reference' in data:
        payrun_payslip.payment_reference = data['payment_reference']
    
    db.session.commit()
    
    return jsonify({'payrun_payslip': payrun_payslip.to_dict()}), 200


@hr_payruns_bp.route('/<int:payrun_id>/reverse', methods=['POST'], strict_slashes=False)
def reverse_payrun(payrun_id):
    """Reverse a completed pay run"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to reverse pay runs'}), 403
    
    payrun = PayRun.query.get(payrun_id)
    if not payrun:
        return jsonify({'error': 'Pay run not found'}), 404
    
    if payrun.status != 'completed':
        return jsonify({'error': 'Can only reverse completed pay runs'}), 400
    
    payrun.status = 'reversed'
    
    # Revert payslip statuses
    payrun_payslips = PayRunPayslip.query.filter_by(pay_run_id=payrun_id).all()
    for pp in payrun_payslips:
        if pp.payslip:
            pp.payslip.status = 'generated'
            pp.payment_status = 'pending'
    
    db.session.commit()
    
    return jsonify({'pay_run': payrun.to_dict()}), 200


@hr_payruns_bp.route('/<int:payrun_id>/payments', methods=['GET'], strict_slashes=False)
def get_payrun_payments(payrun_id):
    """Get all payment transactions for a pay run"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    payrun = PayRun.query.get(payrun_id)
    if not payrun:
        return jsonify({'error': 'Pay run not found'}), 404
    
    # Check permissions
    if payrun.tenant_id != get_current_tenant_id() and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to view this pay run'}), 403
    
    # Get all payment transactions for this pay run
    from app.models import PaymentTransaction
    transactions = PaymentTransaction.query.filter_by(pay_run_id=payrun_id).all()
    
    return jsonify({
        'pay_run_id': payrun_id,
        'transactions': [txn.to_dict() for txn in transactions],
        'total': len(transactions)
        }), 200


@hr_payruns_bp.route('/<int:payrun_id>/balance', methods=['GET'], strict_slashes=False)
def get_wallet_balance(payrun_id):
    """Get wallet balance for pay run tenant"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    payrun = PayRun.query.get(payrun_id)
    if not payrun:
        return jsonify({'error': 'Pay run not found'}), 404
    
    # Check permissions
    if payrun.tenant_id != get_current_tenant_id() and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to view this pay run'}), 403
    
    from app.services.wallet_balance_service import WalletBalanceService
    wallet_service = WalletBalanceService(tenant_id=payrun.tenant_id)
    
    balance_summary = wallet_service.get_balance_summary()
    
    return jsonify({
        'pay_run_id': payrun_id,
        'pay_run_total': float(payrun.total_net or 0),
        'balance': balance_summary,
        'sufficient': balance_summary['available'] >= float(payrun.total_net or 0)
    }), 200


@hr_payruns_bp.route('/payments/<int:transaction_id>/status', methods=['GET'], strict_slashes=False)
def get_payment_status(transaction_id):
    """Check the status of a payment transaction"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    from app.models import PaymentTransaction
    transaction = PaymentTransaction.query.get(transaction_id)
    if not transaction:
        return jsonify({'error': 'Transaction not found'}), 404
    
    # Check permissions
    if transaction.tenant_id != get_current_tenant_id() and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to view this transaction'}), 403
    
    # Check status with payment gateway
    try:
        from app.hr.payment_service import PaymentService
        payment_service = PaymentService(tenant_id=transaction.tenant_id)
        transaction_data = payment_service.check_payment_status(transaction_id)
        
        return jsonify({'transaction': transaction_data}), 200
    except Exception as e:
        # Return current status even if gateway check fails
        return jsonify({
            'transaction': transaction.to_dict(),
            'error': f'Failed to check gateway status: {str(e)}'
        }), 200


@hr_payruns_bp.route('/payments/<int:transaction_id>/retry', methods=['POST'], strict_slashes=False)
def retry_payment(transaction_id):
    """Retry a failed payment transaction"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to retry payments'}), 403
    
    from app.models import PaymentTransaction
    transaction = PaymentTransaction.query.get(transaction_id)
    if not transaction:
        return jsonify({'error': 'Transaction not found'}), 404
    
    # Check permissions
    if transaction.tenant_id != get_current_tenant_id() and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to retry this transaction'}), 403
    
    # Only allow retry for failed transactions
    if transaction.status != 'failed':
        return jsonify({'error': f'Can only retry failed payments. Current status: {transaction.status}'}), 400
    
    # Check retry limit
    if transaction.retry_count >= transaction.max_retries:
        return jsonify({'error': f'Maximum retry limit ({transaction.max_retries}) reached'}), 400
    
    try:
        from app.hr.payment_service import PaymentService
        payment_service = PaymentService(tenant_id=transaction.tenant_id)
        
        # Retry the payment
        result = payment_service.retry_payment(transaction_id, initiated_by=db_user.id)
        
        return jsonify({
            'success': True,
            'transaction': result,
            'message': 'Payment retry initiated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrying payment {transaction_id}: {str(e)}")
        return jsonify({'error': f'Failed to retry payment: {str(e)}'}), 500


@hr_payruns_bp.route('/<int:payrun_id>/force-resolve', methods=['POST'], strict_slashes=False)
def force_resolve_payrun(payrun_id):
    """
    Dead Payrun Escape Hatch (ADMIN SAFETY)
    
    Allows admin to force-resolve payruns stuck in PAYOUT_INITIATED or other states.
    
    STRICT RULES:
    - Only admin/owner can use this
    - Mandatory reason required (min 20 chars)
    - Action is irreversible
    - Full audit trail required
    - Must not trigger new payouts automatically
    
    Request Body:
    {
        "resolution": "force_complete" | "force_fail" | "mark_for_manual_payout",
        "reason": "string (mandatory, min 20 chars)",
        "confirm_irreversible": true
    }
    """
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # STRICT: Only admin/owner can force-resolve
    if db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'Only administrators can force-resolve payruns'}), 403
    
    payrun = PayRun.query.get(payrun_id)
    if not payrun:
        return jsonify({'error': 'Pay run not found'}), 404
    
    # Check tenant
    tenant_id = get_current_tenant_id()
    if payrun.tenant_id != tenant_id and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to resolve this pay run'}), 403
    
    data = request.get_json() or {}
    resolution = data.get('resolution')
    reason = data.get('reason', '').strip()
    confirm_irreversible = data.get('confirm_irreversible', False)
    
    # Validation
    valid_resolutions = ['force_complete', 'force_fail', 'mark_for_manual_payout']
    if resolution not in valid_resolutions:
        return jsonify({
            'error': f'Invalid resolution. Must be one of: {", ".join(valid_resolutions)}'
        }), 400
    
    if not reason or len(reason) < 20:
        return jsonify({
            'error': 'Reason is mandatory and must be at least 20 characters long'
        }), 400
    
    if not confirm_irreversible:
        return jsonify({
            'error': 'You must confirm that this action is irreversible by setting confirm_irreversible to true'
        }), 400
    
    # Audit: Log before making changes
    from app.utils.payment_security import PaymentAuditLogger
    audit_logger = PaymentAuditLogger()
    
    old_status = payrun.status
    
    audit_logger.log_security_event('payrun_force_resolve_initiated', {
        'payrun_id': payrun_id,
        'old_status': old_status,
        'resolution': resolution,
        'reason': reason,
        'initiated_by': db_user.id
    }, db_user.id)
    
    # Apply resolution
    if resolution == 'force_complete':
        payrun.status = 'completed'
        payrun.processed_at = datetime.utcnow()
        # Update all pending payments to success (manual reconciliation)
        from app.models import PaymentTransaction
        pending_txns = PaymentTransaction.query.filter_by(
            pay_run_id=payrun_id,
            status__in=['pending', 'processing']
        ).all()
        for txn in pending_txns:
            txn.status = 'success'
            txn.completed_at = datetime.utcnow()
            txn.failure_reason = f'Force completed by admin: {reason}'
        
        # Unlock funds
        from app.services.wallet_balance_service import WalletBalanceService
        wallet_service = WalletBalanceService(tenant_id=tenant_id)
        wallet_service.unlock_funds_for_payrun(payrun_id)
        
    elif resolution == 'force_fail':
        payrun.status = 'failed'
        payrun.processed_at = datetime.utcnow()
        # Mark all pending payments as failed
        from app.models import PaymentTransaction
        pending_txns = PaymentTransaction.query.filter_by(
            pay_run_id=payrun_id,
            status__in=['pending', 'processing']
        ).all()
        for txn in pending_txns:
            txn.status = 'failed'
            txn.failure_reason = f'Force failed by admin: {reason}'
        
        # Unlock funds
        from app.services.wallet_balance_service import WalletBalanceService
        wallet_service = WalletBalanceService(tenant_id=tenant_id)
        wallet_service.unlock_funds_for_payrun(payrun_id)
        
    elif resolution == 'mark_for_manual_payout':
        payrun.status = 'completed'  # Mark as completed but note it's manual
        payrun.processed_at = datetime.utcnow()
        payrun.notes = (payrun.notes or '') + f'\n[MANUAL PAYOUT] {reason}'
        # Keep payments in current state (don't auto-update)
        # Unlock funds (manual payout means funds handled outside system)
        from app.services.wallet_balance_service import WalletBalanceService
        wallet_service = WalletBalanceService(tenant_id=tenant_id)
        wallet_service.unlock_funds_for_payrun(payrun_id)
    
    # Store resolution in notes for audit
    payrun.notes = (payrun.notes or '') + f'\n[FORCE RESOLVED] {resolution} by {db_user.email} at {datetime.utcnow().isoformat()}: {reason}'
    
    db.session.commit()
    
    # Final audit log
    audit_logger.log_security_event('payrun_force_resolved', {
        'payrun_id': payrun_id,
        'old_status': old_status,
        'new_status': payrun.status,
        'resolution': resolution,
        'reason': reason,
        'resolved_by': db_user.id
    }, db_user.id)
    
    logger.warning(
        f"Payrun {payrun_id} force-resolved by admin {db_user.id}: {old_status} -> {payrun.status} "
        f"(resolution: {resolution}, reason: {reason[:50]})"
    )
    
    return jsonify({
        'pay_run': payrun.to_dict(),
        'message': f'Pay run force-resolved to {payrun.status}',
        'irreversible': True,
        'audit_logged': True,
        'resolution': resolution
    }), 200


@hr_payruns_bp.route('/reconcile-payments', methods=['POST'], strict_slashes=False)
def reconcile_payments():
    """
    Manually trigger payment reconciliation for stuck payments
    
    Only admin/owner can trigger this
    """
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin/owner can trigger reconciliation
    if db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'Only administrators can trigger payment reconciliation'}), 403
    
    data = request.get_json() or {}
    hours_threshold = data.get('hours_threshold', 2)
    limit = min(data.get('limit', 100), 500)  # Max 500 per run
    
    try:
        from app.services.payment_reconciliation import reconcile_all_tenants
        
        results = reconcile_all_tenants(hours_threshold=hours_threshold, limit=limit)
        
        return jsonify({
            'success': True,
            'results': results,
            'message': f"Reconciliation complete: {results['total_updated']} payments updated"
        }), 200
        
    except Exception as e:
        from app.simple_logger import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error during payment reconciliation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Reconciliation failed: {str(e)}'
        }), 500

