"""
Employee-Side Payment Transparency API
Provides read-only payment information to employees
"""
from flask import Blueprint, request, jsonify
from app.models import db, PaymentTransaction, PayRun, Payslip
from app.auth_utils import get_current_user_flexible, get_current_tenant_id
from app.simple_logger import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)

employee_payments_bp = Blueprint('employee_payments', __name__)


def _auth_or_401():
    user = get_current_user_flexible()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@employee_payments_bp.route('/my-payments', methods=['GET'], strict_slashes=False)
def get_my_payments():
    """
    Get employee's payment history (read-only)
    
    Returns:
    - Last payout status
    - Amount
    - Date
    - Failure reason (if any)
    - Reference ID
    
    Rules:
    - No bank details visible
    - No retries from employee side
    - Must reduce support tickets
    """
    user, error = _auth_or_401()
    if error:
        return error
    
    from app.models import User
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only employees can view their own payments
    if db_user.role != 'employee' and db_user.user_type != 'employee':
        return jsonify({'error': 'This endpoint is only available for employees'}), 403
    
    tenant_id = get_current_tenant_id()
    
    # Get query parameters
    limit = min(request.args.get('limit', 10, type=int), 50)  # Max 50
    offset = request.args.get('offset', 0, type=int)
    
    # Get employee's payment transactions
    transactions = PaymentTransaction.query.filter_by(
        employee_id=db_user.id,
        tenant_id=tenant_id
    ).order_by(PaymentTransaction.initiated_at.desc()).offset(offset).limit(limit).all()
    
    # Get total count
    total_count = PaymentTransaction.query.filter_by(
        employee_id=db_user.id,
        tenant_id=tenant_id
    ).count()
    
    # Format response (no sensitive data)
    payment_history = []
    for txn in transactions:
        # Get pay run info
        payrun = PayRun.query.get(txn.pay_run_id)
        
        payment_data = {
            'id': txn.id,
            'pay_run_id': txn.pay_run_id,
            'pay_period': None,
            'pay_date': None,
            'amount': float(txn.amount),
            'currency': txn.currency,
            'status': txn.status,
            'payment_mode': txn.payment_mode,
            'initiated_at': txn.initiated_at.isoformat() if txn.initiated_at else None,
            'completed_at': txn.completed_at.isoformat() if txn.completed_at else None,
            'failure_reason': None,  # Only show if failed
            'reference_id': txn.gateway_payout_id or txn.idempotency_key,  # Masked reference
            'purpose_code': txn.purpose_code
        }
        
        # Add pay run period info
        if payrun:
            payment_data['pay_period'] = {
                'start': payrun.pay_period_start.isoformat() if payrun.pay_period_start else None,
                'end': payrun.pay_period_end.isoformat() if payrun.pay_period_end else None,
            }
            payment_data['pay_date'] = payrun.pay_date.isoformat() if payrun.pay_date else None
        
        # Only show failure reason if payment failed
        if txn.status == 'failed':
            # Human-readable failure reasons
            failure_reason = txn.failure_reason or 'Payment failed'
            
            # Map technical errors to user-friendly messages
            if 'insufficient' in failure_reason.lower():
                failure_reason = 'Payment could not be processed due to insufficient funds.'
            elif 'verification' in failure_reason.lower() or 'penny' in failure_reason.lower():
                failure_reason = 'Bank account verification required. Please contact HR.'
            elif 'cooldown' in failure_reason.lower():
                failure_reason = 'Bank account was recently changed. Please wait 72 hours or contact HR.'
            elif 'fraud' in failure_reason.lower():
                failure_reason = 'Payment is under review for security reasons. Please contact HR.'
            elif 'kyc' in failure_reason.lower():
                failure_reason = 'Employer account verification pending. Please contact HR.'
            
            payment_data['failure_reason'] = failure_reason
        
        # Mask reference ID for privacy
        if payment_data['reference_id']:
            ref = payment_data['reference_id']
            if len(ref) > 8:
                payment_data['reference_id'] = ref[:4] + '****' + ref[-4:]
        
        payment_history.append(payment_data)
    
    # Get latest payment summary
    latest_payment = None
    if transactions:
        latest = transactions[0]
        latest_payment = {
            'status': latest.status,
            'amount': float(latest.amount),
            'currency': latest.currency,
            'date': latest.completed_at.isoformat() if latest.completed_at else latest.initiated_at.isoformat() if latest.initiated_at else None,
            'reference_id': (latest.gateway_payout_id or latest.idempotency_key or f"TXN{latest.id}")[:8] + '****'
        }
    
    return jsonify({
        'payments': payment_history,
        'latest_payment': latest_payment,
        'total': total_count,
        'limit': limit,
        'offset': offset,
        'has_more': (offset + limit) < total_count
    }), 200


@employee_payments_bp.route('/my-payments/<int:transaction_id>', methods=['GET'], strict_slashes=False)
def get_my_payment_detail(transaction_id):
    """
    Get detailed information about a specific payment (employee view)
    
    Returns detailed payment information without sensitive bank details
    """
    user, error = _auth_or_401()
    if error:
        return error
    
    from app.models import User
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only employees can view their own payments
    if db_user.role != 'employee' and db_user.user_type != 'employee':
        return jsonify({'error': 'This endpoint is only available for employees'}), 403
    
    transaction = PaymentTransaction.query.get(transaction_id)
    if not transaction:
        return jsonify({'error': 'Payment transaction not found'}), 404
    
    # Verify ownership
    if transaction.employee_id != db_user.id:
        return jsonify({'error': 'You do not have permission to view this payment'}), 403
    
    # Get pay run info
    payrun = PayRun.query.get(transaction.pay_run_id)
    payslip = Payslip.query.get(transaction.payslip_id)
    
    payment_detail = {
        'id': transaction.id,
        'pay_run_id': transaction.pay_run_id,
        'payslip_id': transaction.payslip_id,
        'amount': float(transaction.amount),
        'currency': transaction.currency,
        'status': transaction.status,
        'payment_mode': transaction.payment_mode,
        'purpose_code': transaction.purpose_code,
        'initiated_at': transaction.initiated_at.isoformat() if transaction.initiated_at else None,
        'processed_at': transaction.processed_at.isoformat() if transaction.processed_at else None,
        'completed_at': transaction.completed_at.isoformat() if transaction.completed_at else None,
        'failure_reason': None,
        'reference_id': None,
        'pay_run': None,
        'payslip': None
    }
    
    # Add pay run details
    if payrun:
        payment_detail['pay_run'] = {
            'id': payrun.id,
            'pay_period_start': payrun.pay_period_start.isoformat() if payrun.pay_period_start else None,
            'pay_period_end': payrun.pay_period_end.isoformat() if payrun.pay_period_end else None,
            'pay_date': payrun.pay_date.isoformat() if payrun.pay_date else None,
            'status': payrun.status
        }
    
    # Add payslip summary (no sensitive data)
    if payslip:
        payment_detail['payslip'] = {
            'id': payslip.id,
            'net_pay': float(payslip.net_pay) if payslip.net_pay else None,
            'currency': payslip.currency
        }
    
    # Reference ID (masked)
    if transaction.gateway_payout_id:
        ref = transaction.gateway_payout_id
        payment_detail['reference_id'] = ref[:4] + '****' + ref[-4:] if len(ref) > 8 else '****'
    elif transaction.idempotency_key:
        ref = transaction.idempotency_key
        payment_detail['reference_id'] = ref[:4] + '****' + ref[-4:] if len(ref) > 8 else '****'
    else:
        payment_detail['reference_id'] = f"TXN{transaction.id}"
    
    # Failure reason (only if failed, human-readable)
    if transaction.status == 'failed':
        failure_reason = transaction.failure_reason or 'Payment failed'
        
        # Human-readable mapping
        if 'insufficient' in failure_reason.lower():
            failure_reason = 'Payment could not be processed due to insufficient funds. Please contact HR.'
        elif 'verification' in failure_reason.lower():
            failure_reason = 'Bank account verification required. Please contact HR to verify your account.'
        elif 'cooldown' in failure_reason.lower():
            failure_reason = 'Bank account was recently changed. Please wait 72 hours or contact HR for assistance.'
        elif 'fraud' in failure_reason.lower():
            failure_reason = 'Payment is under security review. Please contact HR for more information.'
        
        payment_detail['failure_reason'] = failure_reason
        payment_detail['support_contact'] = 'Please contact your HR department for assistance with this payment.'
    
    return jsonify({'payment': payment_detail}), 200

