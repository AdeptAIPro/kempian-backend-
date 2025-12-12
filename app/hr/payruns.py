"""
Pay Run Management API
Handles payroll run creation, approval, and processing
"""
from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, PayRun, PayRunPayslip, Payslip, db
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
    """Approve a pay run"""
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
    
    payrun.status = 'approved'
    payrun.approved_by = db_user.id
    
    db.session.commit()
    
    return jsonify({'pay_run': payrun.to_dict()}), 200


@hr_payruns_bp.route('/<int:payrun_id>/process', methods=['POST'], strict_slashes=False)
def process_payrun(payrun_id):
    """Process a pay run (mark as processing/completed)"""
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
    
    if payrun.status not in ['approved', 'processing']:
        return jsonify({'error': f'Can only process approved pay runs. Current status: {payrun.status}'}), 400
    
    data = request.get_json() or {}
    new_status = data.get('status', 'processing')
    
    if new_status not in ['processing', 'completed', 'failed']:
        return jsonify({'error': 'status must be one of: processing, completed, failed'}), 400
    
    payrun.status = new_status
    
    if new_status == 'completed':
        payrun.processed_at = datetime.utcnow()
        # Update payslip statuses to 'paid'
        payrun_payslips = PayRunPayslip.query.filter_by(pay_run_id=payrun_id).all()
        for pp in payrun_payslips:
            if pp.payslip:
                pp.payslip.status = 'paid'
                pp.payment_status = 'processed'
    
    db.session.commit()
    
    return jsonify({'pay_run': payrun.to_dict()}), 200


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

