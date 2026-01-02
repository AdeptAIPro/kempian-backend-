"""
Deductions Management API
Handles deduction types and employee deduction enrollments
"""
from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, DeductionType, EmployeeDeduction, db
from datetime import datetime
from decimal import Decimal

hr_deductions_bp = Blueprint('hr_deductions', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@hr_deductions_bp.route('/types', methods=['GET'], strict_slashes=False)
def list_deduction_types():
    """List deduction types for current tenant"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    is_active = request.args.get('is_active')
    query = DeductionType.query.filter_by(tenant_id=tenant_id)
    
    if is_active is not None:
        query = query.filter_by(is_active=is_active.lower() == 'true')
    
    deduction_types = query.order_by(DeductionType.name).all()
    
    return jsonify({'deduction_types': [dt.to_dict() for dt in deduction_types]}), 200


@hr_deductions_bp.route('/types', methods=['POST'], strict_slashes=False)
def create_deduction_type():
    """Create a new deduction type"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to create deduction types'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json()
    
    name = data.get('name')
    if not name:
        return jsonify({'error': 'name is required'}), 400
    
    calculation_method = data.get('calculation_method', 'fixed')
    if calculation_method not in ['fixed', 'percentage', 'tiered']:
        return jsonify({'error': 'calculation_method must be one of: fixed, percentage, tiered'}), 400
    
    deduction_type = DeductionType(
        tenant_id=tenant_id,
        name=name,
        description=data.get('description'),
        deduction_category=data.get('deduction_category', 'other'),
        calculation_method=calculation_method,
        default_amount=Decimal(str(data.get('default_amount'))) if data.get('default_amount') else None,
        default_percentage=Decimal(str(data.get('default_percentage'))) if data.get('default_percentage') else None,
        is_pre_tax=data.get('is_pre_tax', True),
        is_active=data.get('is_active', True)
    )
    
    db.session.add(deduction_type)
    db.session.commit()
    
    return jsonify({'deduction_type': deduction_type.to_dict()}), 201


@hr_deductions_bp.route('/types/<int:type_id>', methods=['PUT'], strict_slashes=False)
def update_deduction_type(type_id):
    """Update a deduction type"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to update deduction types'}), 403
    
    deduction_type = DeductionType.query.get(type_id)
    if not deduction_type:
        return jsonify({'error': 'Deduction type not found'}), 404
    
    data = request.get_json()
    
    if 'name' in data:
        deduction_type.name = data['name']
    if 'description' in data:
        deduction_type.description = data['description']
    if 'deduction_category' in data:
        deduction_type.deduction_category = data['deduction_category']
    if 'calculation_method' in data:
        deduction_type.calculation_method = data['calculation_method']
    if 'default_amount' in data:
        deduction_type.default_amount = Decimal(str(data['default_amount'])) if data['default_amount'] else None
    if 'default_percentage' in data:
        deduction_type.default_percentage = Decimal(str(data['default_percentage'])) if data['default_percentage'] else None
    if 'is_pre_tax' in data:
        deduction_type.is_pre_tax = data['is_pre_tax']
    if 'is_active' in data:
        deduction_type.is_active = data['is_active']
    
    db.session.commit()
    
    return jsonify({'deduction_type': deduction_type.to_dict()}), 200


@hr_deductions_bp.route('/employee/<int:employee_id>', methods=['GET'], strict_slashes=False)
def get_employee_deductions(employee_id):
    """Get employee deductions"""
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
        return jsonify({'error': 'You do not have permission to view this employee\'s deductions'}), 403
    
    is_active = request.args.get('is_active')
    query = EmployeeDeduction.query.filter_by(employee_id=employee_id)
    
    if is_active is not None:
        query = query.filter_by(is_active=is_active.lower() == 'true')
    
    deductions = query.order_by(EmployeeDeduction.effective_date.desc()).all()
    
    return jsonify({'deductions': [d.to_dict() for d in deductions]}), 200


@hr_deductions_bp.route('/employee/<int:employee_id>', methods=['POST'], strict_slashes=False)
def add_employee_deduction(employee_id):
    """Add a deduction for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Employee can add their own, managers can add for any
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to add deductions for this employee'}), 403
    
    data = request.get_json()
    
    deduction_type_id = data.get('deduction_type_id')
    if not deduction_type_id:
        return jsonify({'error': 'deduction_type_id is required'}), 400
    
    deduction_type = DeductionType.query.get(deduction_type_id)
    if not deduction_type:
        return jsonify({'error': 'Deduction type not found'}), 404
    
    # Validate that amount or percentage is provided based on calculation method
    amount = data.get('amount')
    percentage = data.get('percentage')
    
    if deduction_type.calculation_method == 'fixed' and not amount:
        return jsonify({'error': 'amount is required for fixed deduction type'}), 400
    if deduction_type.calculation_method == 'percentage' and not percentage:
        return jsonify({'error': 'percentage is required for percentage deduction type'}), 400
    
    effective_date_str = data.get('effective_date')
    if effective_date_str:
        try:
            effective_date = datetime.strptime(effective_date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid effective_date format. Use YYYY-MM-DD'}), 400
    else:
        effective_date = datetime.now().date()
    
    end_date_str = data.get('end_date')
    end_date = None
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
    
    employee_deduction = EmployeeDeduction(
        employee_id=employee_id,
        deduction_type_id=deduction_type_id,
        amount=Decimal(str(amount)) if amount else None,
        percentage=Decimal(str(percentage)) if percentage else None,
        effective_date=effective_date,
        end_date=end_date,
        is_active=data.get('is_active', True)
    )
    
    db.session.add(employee_deduction)
    db.session.commit()
    
    return jsonify({'deduction': employee_deduction.to_dict()}), 201


@hr_deductions_bp.route('/employee/<int:employee_id>/<int:deduction_id>', methods=['PUT'], strict_slashes=False)
def update_employee_deduction(employee_id, deduction_id):
    """Update an employee deduction"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    employee_deduction = EmployeeDeduction.query.get(deduction_id)
    if not employee_deduction or employee_deduction.employee_id != employee_id:
        return jsonify({'error': 'Employee deduction not found'}), 404
    
    # Employee can update their own, managers can update any
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to update this deduction'}), 403
    
    data = request.get_json()
    
    if 'amount' in data:
        employee_deduction.amount = Decimal(str(data['amount'])) if data['amount'] else None
    if 'percentage' in data:
        employee_deduction.percentage = Decimal(str(data['percentage'])) if data['percentage'] else None
    if 'effective_date' in data:
        try:
            employee_deduction.effective_date = datetime.strptime(data['effective_date'], '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid effective_date format. Use YYYY-MM-DD'}), 400
    if 'end_date' in data:
        if data['end_date']:
            try:
                employee_deduction.end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'error': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
        else:
            employee_deduction.end_date = None
    if 'is_active' in data:
        employee_deduction.is_active = data['is_active']
    
    db.session.commit()
    
    return jsonify({'deduction': employee_deduction.to_dict()}), 200


@hr_deductions_bp.route('/employee/<int:employee_id>/<int:deduction_id>', methods=['DELETE'], strict_slashes=False)
def delete_employee_deduction(employee_id, deduction_id):
    """Delete an employee deduction"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    employee_deduction = EmployeeDeduction.query.get(deduction_id)
    if not employee_deduction or employee_deduction.employee_id != employee_id:
        return jsonify({'error': 'Employee deduction not found'}), 404
    
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to delete this deduction'}), 403
    
    db.session.delete(employee_deduction)
    db.session.commit()
    
    return jsonify({'message': 'Deduction deleted successfully'}), 200


@hr_deductions_bp.route('/calculate', methods=['POST'], strict_slashes=False)
def calculate_deductions():
    """Calculate deductions for a given gross amount"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json()
    employee_id = data.get('employee_id')
    gross_amount = data.get('gross_amount')
    include_pre_tax = data.get('include_pre_tax', True)
    include_post_tax = data.get('include_post_tax', True)
    
    if not employee_id or not gross_amount:
        return jsonify({'error': 'employee_id and gross_amount are required'}), 400
    
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Get active employee deductions
    today = datetime.now().date()
    deductions = EmployeeDeduction.query.filter(
        EmployeeDeduction.employee_id == employee_id,
        EmployeeDeduction.is_active == True,
        EmployeeDeduction.effective_date <= today,
        db.or_(
            EmployeeDeduction.end_date.is_(None),
            EmployeeDeduction.end_date >= today
        )
    ).join(DeductionType).all()
    
    pre_tax_total = Decimal('0')
    post_tax_total = Decimal('0')
    breakdown = []
    
    gross = Decimal(str(gross_amount))
    
    for emp_deduction in deductions:
        deduction_type = emp_deduction.deduction_type
        
        # Filter by pre-tax/post-tax
        if deduction_type.is_pre_tax and not include_pre_tax:
            continue
        if not deduction_type.is_pre_tax and not include_post_tax:
            continue
        
        amount = Decimal('0')
        
        if emp_deduction.amount:
            amount = emp_deduction.amount
        elif emp_deduction.percentage:
            amount = gross * emp_deduction.percentage
        elif deduction_type.default_amount:
            amount = deduction_type.default_amount
        elif deduction_type.default_percentage:
            amount = gross * deduction_type.default_percentage
        
        if deduction_type.is_pre_tax:
            pre_tax_total += amount
        else:
            post_tax_total += amount
        
        breakdown.append({
            'deduction_type_id': deduction_type.id,
            'deduction_type_name': deduction_type.name,
            'amount': float(amount),
            'is_pre_tax': deduction_type.is_pre_tax
        })
    
    return jsonify({
        'gross_amount': float(gross_amount),
        'pre_tax_deductions': float(pre_tax_total),
        'post_tax_deductions': float(post_tax_total),
        'total_deductions': float(pre_tax_total + post_tax_total),
        'breakdown': breakdown
    }), 200

