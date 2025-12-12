"""
Tax Management API
Handles tax configuration and employee tax profiles
"""
from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, TaxConfiguration, EmployeeTaxProfile, db
from datetime import datetime
from decimal import Decimal

hr_tax_bp = Blueprint('hr_tax', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@hr_tax_bp.route('/config', methods=['GET'], strict_slashes=False)
def list_tax_configs():
    """List tax configurations for current tenant"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, owner, employer, recruiter can view tax configs
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view tax configurations'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    tax_type = request.args.get('tax_type')
    query = TaxConfiguration.query.filter_by(tenant_id=tenant_id, is_active=True)
    
    if tax_type:
        query = query.filter_by(tax_type=tax_type)
    
    configs = query.order_by(TaxConfiguration.effective_date.desc()).all()
    
    return jsonify({'tax_configurations': [c.to_dict() for c in configs]}), 200


@hr_tax_bp.route('/config', methods=['POST'], strict_slashes=False)
def create_tax_config():
    """Create a new tax configuration"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, owner can create tax configs
    if db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to create tax configurations'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json()
    
    tax_type = data.get('tax_type')
    if not tax_type:
        return jsonify({'error': 'tax_type is required'}), 400
    
    # Validate tax_type
    valid_types = ['federal', 'state', 'local', 'fica_social_security', 'fica_medicare']
    if tax_type not in valid_types:
        return jsonify({'error': f'tax_type must be one of: {", ".join(valid_types)}'}), 400
    
    # Check if tax_rate or tax_brackets is provided
    tax_rate = data.get('tax_rate')
    tax_brackets = data.get('tax_brackets')
    
    if not tax_rate and not tax_brackets:
        return jsonify({'error': 'Either tax_rate or tax_brackets is required'}), 400
    
    effective_date_str = data.get('effective_date')
    if effective_date_str:
        try:
            effective_date = datetime.strptime(effective_date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid effective_date format. Use YYYY-MM-DD'}), 400
    else:
        effective_date = datetime.now().date()
    
    tax_config = TaxConfiguration(
        tenant_id=tenant_id,
        tax_type=tax_type,
        jurisdiction=data.get('jurisdiction'),
        tax_rate=Decimal(str(tax_rate)) if tax_rate else None,
        tax_brackets=tax_brackets,
        wage_base_limit=Decimal(str(data.get('wage_base_limit'))) if data.get('wage_base_limit') else None,
        effective_date=effective_date,
        is_active=data.get('is_active', True)
    )
    
    db.session.add(tax_config)
    db.session.commit()
    
    return jsonify({'tax_configuration': tax_config.to_dict()}), 201


@hr_tax_bp.route('/config/<int:config_id>', methods=['PUT'], strict_slashes=False)
def update_tax_config(config_id):
    """Update a tax configuration"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to update tax configurations'}), 403
    
    tax_config = TaxConfiguration.query.get(config_id)
    if not tax_config:
        return jsonify({'error': 'Tax configuration not found'}), 404
    
    data = request.get_json()
    
    if 'tax_rate' in data:
        tax_config.tax_rate = Decimal(str(data['tax_rate'])) if data['tax_rate'] else None
    if 'tax_brackets' in data:
        tax_config.tax_brackets = data['tax_brackets']
    if 'wage_base_limit' in data:
        tax_config.wage_base_limit = Decimal(str(data['wage_base_limit'])) if data['wage_base_limit'] else None
    if 'jurisdiction' in data:
        tax_config.jurisdiction = data['jurisdiction']
    if 'effective_date' in data:
        try:
            tax_config.effective_date = datetime.strptime(data['effective_date'], '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid effective_date format. Use YYYY-MM-DD'}), 400
    if 'is_active' in data:
        tax_config.is_active = data['is_active']
    
    db.session.commit()
    
    return jsonify({'tax_configuration': tax_config.to_dict()}), 200


@hr_tax_bp.route('/employee/<int:employee_id>/profile', methods=['GET'], strict_slashes=False)
def get_employee_tax_profile(employee_id):
    """Get employee tax profile"""
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
        return jsonify({'error': 'You do not have permission to view this tax profile'}), 403
    
    tax_profile = EmployeeTaxProfile.query.filter_by(employee_id=employee_id).first()
    
    if not tax_profile:
        return jsonify({'tax_profile': None}), 200
    
    return jsonify({'tax_profile': tax_profile.to_dict()}), 200


@hr_tax_bp.route('/employee/<int:employee_id>/profile', methods=['POST', 'PUT'], strict_slashes=False)
def set_employee_tax_profile(employee_id):
    """Create or update employee tax profile"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Employee can update their own, managers can update any
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to update this tax profile'}), 403
    
    data = request.get_json()
    
    tax_profile = EmployeeTaxProfile.query.filter_by(employee_id=employee_id).first()
    
    if not tax_profile:
        tax_profile = EmployeeTaxProfile(employee_id=employee_id)
        db.session.add(tax_profile)
    
    if 'filing_status' in data:
        tax_profile.filing_status = data['filing_status']
    if 'allowances' in data:
        tax_profile.allowances = int(data['allowances'])
    if 'additional_withholding' in data:
        tax_profile.additional_withholding = Decimal(str(data['additional_withholding']))
    if 'exempt_from_federal' in data:
        tax_profile.exempt_from_federal = data['exempt_from_federal']
    if 'exempt_from_state' in data:
        tax_profile.exempt_from_state = data['exempt_from_state']
    if 'exempt_from_local' in data:
        tax_profile.exempt_from_local = data['exempt_from_local']
    if 'notes' in data:
        tax_profile.notes = data['notes']
    
    db.session.commit()
    
    return jsonify({'tax_profile': tax_profile.to_dict()}), 200


@hr_tax_bp.route('/calculate', methods=['POST'], strict_slashes=False)
def calculate_tax():
    """Calculate tax for a given gross amount"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json()
    employee_id = data.get('employee_id')
    gross_amount = data.get('gross_amount')
    tax_types = data.get('tax_types', [])  # List of tax types to calculate
    
    if not employee_id or not gross_amount:
        return jsonify({'error': 'employee_id and gross_amount are required'}), 400
    
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    tenant_id = get_current_tenant_id() or employee.tenant_id
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    # Get employee tax profile
    tax_profile = EmployeeTaxProfile.query.filter_by(employee_id=employee_id).first()
    
    # Get active tax configurations
    if not tax_types:
        tax_types = ['federal', 'state', 'local', 'fica_social_security', 'fica_medicare']
    
    tax_configs = TaxConfiguration.query.filter(
        TaxConfiguration.tenant_id == tenant_id,
        TaxConfiguration.tax_type.in_(tax_types),
        TaxConfiguration.is_active == True
    ).all()
    
    calculations = {}
    total_tax = Decimal('0')
    
    for config in tax_configs:
        # Check exemptions
        if config.tax_type == 'federal' and tax_profile and tax_profile.exempt_from_federal:
            continue
        if config.tax_type == 'state' and tax_profile and tax_profile.exempt_from_state:
            continue
        if config.tax_type == 'local' and tax_profile and tax_profile.exempt_from_local:
            continue
        
        gross = Decimal(str(gross_amount))
        tax_amount = Decimal('0')
        
        if config.tax_type in ['fica_social_security', 'fica_medicare']:
            # FICA calculations
            if config.tax_type == 'fica_social_security':
                # Social Security: 6.2% up to wage base limit
                rate = Decimal('0.062')
                if config.wage_base_limit:
                    taxable_amount = min(gross, config.wage_base_limit)
                else:
                    # Default wage base for 2024
                    taxable_amount = min(gross, Decimal('160200'))
                tax_amount = taxable_amount * rate
            elif config.tax_type == 'fica_medicare':
                # Medicare: 1.45% on all earnings, +0.9% on earnings over $200k
                rate = Decimal('0.0145')
                tax_amount = gross * rate
                # Additional Medicare tax (0.9%) on earnings over $200k
                if gross > Decimal('200000'):
                    additional_tax = (gross - Decimal('200000')) * Decimal('0.009')
                    tax_amount += additional_tax
        elif config.tax_rate:
            # Flat rate tax
            tax_amount = gross * config.tax_rate
        elif config.tax_brackets:
            # Progressive tax brackets
            tax_amount = _calculate_progressive_tax(gross, config.tax_brackets)
        
        calculations[config.tax_type] = {
            'amount': float(tax_amount),
            'rate': float(config.tax_rate) if config.tax_rate else None,
            'jurisdiction': config.jurisdiction
        }
        total_tax += tax_amount
    
    # Add additional withholding if specified
    additional_withholding = Decimal('0')
    if tax_profile and tax_profile.additional_withholding:
        additional_withholding = tax_profile.additional_withholding
        calculations['additional_withholding'] = {
            'amount': float(additional_withholding),
            'rate': None,
            'jurisdiction': None
        }
        total_tax += additional_withholding
    
    return jsonify({
        'gross_amount': float(gross_amount),
        'tax_breakdown': calculations,
        'total_tax': float(total_tax),
        'net_amount': float(Decimal(str(gross_amount)) - total_tax)
    }), 200


def _calculate_progressive_tax(income, brackets):
    """Calculate tax using progressive brackets"""
    tax = Decimal('0')
    remaining_income = income
    
    # Sort brackets by min_income
    sorted_brackets = sorted(brackets, key=lambda x: x.get('min_income', 0))
    
    for i, bracket in enumerate(sorted_brackets):
        min_income = Decimal(str(bracket.get('min_income', 0)))
        max_income = Decimal(str(bracket.get('max_income', float('inf')))) if bracket.get('max_income') else Decimal('999999999')
        rate = Decimal(str(bracket.get('rate', 0)))
        
        if remaining_income <= 0:
            break
        
        if income > min_income:
            taxable_in_bracket = min(remaining_income, max_income - min_income)
            if taxable_in_bracket > 0:
                tax += taxable_in_bracket * rate
                remaining_income -= taxable_in_bracket
    
    return tax

