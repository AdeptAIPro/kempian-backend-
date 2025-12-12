"""
Leave Management API
Handles leave types, balances, and requests
"""
from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, LeaveType, LeaveBalance, LeaveRequest, db
from datetime import datetime, date
from decimal import Decimal

hr_leave_bp = Blueprint('hr_leave', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@hr_leave_bp.route('/types', methods=['GET'], strict_slashes=False)
def list_leave_types():
    """List leave types for current tenant"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    is_active = request.args.get('is_active')
    query = LeaveType.query.filter_by(tenant_id=tenant_id)
    
    if is_active is not None:
        query = query.filter_by(is_active=is_active.lower() == 'true')
    
    leave_types = query.order_by(LeaveType.name).all()
    
    return jsonify({'leave_types': [lt.to_dict() for lt in leave_types]}), 200


@hr_leave_bp.route('/types', methods=['POST'], strict_slashes=False)
def create_leave_type():
    """Create a new leave type"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to create leave types'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json()
    
    name = data.get('name')
    if not name:
        return jsonify({'error': 'name is required'}), 400
    
    leave_type = LeaveType(
        tenant_id=tenant_id,
        name=name,
        description=data.get('description'),
        accrual_type=data.get('accrual_type', 'none'),
        accrual_rate=Decimal(str(data.get('accrual_rate'))) if data.get('accrual_rate') else None,
        max_balance=Decimal(str(data.get('max_balance'))) if data.get('max_balance') else None,
        carry_over_allowed=data.get('carry_over_allowed', True),
        carry_over_limit=Decimal(str(data.get('carry_over_limit'))) if data.get('carry_over_limit') else None,
        is_paid=data.get('is_paid', True),
        is_active=data.get('is_active', True)
    )
    
    db.session.add(leave_type)
    db.session.commit()
    
    return jsonify({'leave_type': leave_type.to_dict()}), 201


@hr_leave_bp.route('/employee/<int:employee_id>/balance', methods=['GET'], strict_slashes=False)
def get_employee_leave_balance(employee_id):
    """Get employee leave balances"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Employee can view their own, managers can view any
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view this employee\'s leave balance'}), 403
    
    year = request.args.get('year', type=int)
    if not year:
        year = datetime.now().year
    
    balances = LeaveBalance.query.filter_by(employee_id=employee_id, year=year).all()
    
    return jsonify({'leave_balances': [b.to_dict() for b in balances], 'year': year}), 200


@hr_leave_bp.route('/employee/<int:employee_id>/requests', methods=['GET'], strict_slashes=False)
def list_leave_requests(employee_id):
    """List leave requests for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Employee can view their own, managers can view any
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view this employee\'s leave requests'}), 403
    
    status = request.args.get('status')
    query = LeaveRequest.query.filter_by(employee_id=employee_id)
    
    if status:
        query = query.filter_by(status=status)
    
    requests = query.order_by(LeaveRequest.start_date.desc()).all()
    
    return jsonify({'leave_requests': [lr.to_dict() for lr in requests]}), 200


@hr_leave_bp.route('/employee/<int:employee_id>/requests', methods=['POST'], strict_slashes=False)
def create_leave_request(employee_id):
    """Create a leave request"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Employee can create their own, managers can create for any
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to create leave requests for this employee'}), 403
    
    data = request.get_json()
    
    leave_type_id = data.get('leave_type_id')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    hours = data.get('hours')
    
    if not leave_type_id or not start_date_str or not end_date_str or not hours:
        return jsonify({'error': 'leave_type_id, start_date, end_date, and hours are required'}), 400
    
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    if end_date < start_date:
        return jsonify({'error': 'end_date must be on or after start_date'}), 400
    
    leave_type = LeaveType.query.get(leave_type_id)
    if not leave_type:
        return jsonify({'error': 'Leave type not found'}), 404
    
    # Check leave balance if it's a paid leave type
    if leave_type.is_paid:
        year = start_date.year
        balance = LeaveBalance.query.filter_by(
            employee_id=employee_id,
            leave_type_id=leave_type_id,
            year=year
        ).first()
        
        if not balance or float(balance.balance) < float(hours):
            return jsonify({'error': 'Insufficient leave balance'}), 400
    
    leave_request = LeaveRequest(
        employee_id=employee_id,
        leave_type_id=leave_type_id,
        start_date=start_date,
        end_date=end_date,
        hours=Decimal(str(hours)),
        reason=data.get('reason'),
        status='pending'
    )
    
    db.session.add(leave_request)
    db.session.commit()
    
    return jsonify({'leave_request': leave_request.to_dict()}), 201


@hr_leave_bp.route('/requests/<int:request_id>/approve', methods=['POST'], strict_slashes=False)
def approve_leave_request(request_id):
    """Approve a leave request"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to approve leave requests'}), 403
    
    leave_request = LeaveRequest.query.get(request_id)
    if not leave_request:
        return jsonify({'error': 'Leave request not found'}), 404
    
    if leave_request.status != 'pending':
        return jsonify({'error': f'Can only approve pending requests. Current status: {leave_request.status}'}), 400
    
    leave_request.status = 'approved'
    leave_request.approved_by = db_user.id
    leave_request.approved_at = datetime.utcnow()
    
    # Deduct from balance if paid leave
    if leave_request.leave_type.is_paid:
        year = leave_request.start_date.year
        balance = LeaveBalance.query.filter_by(
            employee_id=leave_request.employee_id,
            leave_type_id=leave_request.leave_type_id,
            year=year
        ).first()
        
        if balance:
            balance.used += leave_request.hours
            balance.balance -= leave_request.hours
    
    db.session.commit()
    
    return jsonify({'leave_request': leave_request.to_dict()}), 200


@hr_leave_bp.route('/requests/<int:request_id>/reject', methods=['POST'], strict_slashes=False)
def reject_leave_request(request_id):
    """Reject a leave request"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to reject leave requests'}), 403
    
    leave_request = LeaveRequest.query.get(request_id)
    if not leave_request:
        return jsonify({'error': 'Leave request not found'}), 404
    
    if leave_request.status != 'pending':
        return jsonify({'error': f'Can only reject pending requests. Current status: {leave_request.status}'}), 400
    
    data = request.get_json() or {}
    rejection_reason = data.get('rejection_reason', '')
    
    leave_request.status = 'rejected'
    leave_request.rejection_reason = rejection_reason
    
    db.session.commit()
    
    return jsonify({'leave_request': leave_request.to_dict()}), 200
