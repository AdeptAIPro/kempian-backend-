"""
US Compliance API Routes
Handles all US payroll compliance endpoints
"""

from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, db
from app.hr.us_state_tax import (
    calculate_state_tax, calculate_multi_state_tax, create_state_tax_configuration,
    get_state_tax_configuration, get_reciprocal_states
)
from app.hr.us_local_tax import (
    calculate_local_tax, create_local_tax_configuration, get_local_tax_configurations
)
from app.hr.sui_calculations import (
    calculate_sui_contribution, create_sui_contribution_record,
    get_sui_contributions_by_tenant, mark_sui_quarterly_report_filed
)
from app.hr.irs_tax_tables import (
    calculate_federal_income_tax, calculate_fica_tax, calculate_total_federal_tax,
    get_tax_table_by_year
)
from app.hr.form941_generator import (
    generate_form941, generate_form941_pdf, get_form941, mark_form941_efiled
)
from app.hr.form940_generator import (
    generate_form940, generate_form940_pdf, get_form940, mark_form940_efiled
)
from app.hr.workers_compensation import (
    calculate_workers_compensation, create_workers_compensation_record,
    get_workers_compensation_by_employee, get_wc_rates_by_state
)
from app.hr.garnishments import (
    calculate_garnishment_amount, create_garnishment, get_active_garnishments,
    calculate_total_garnishments, update_garnishment_collected
)
from app.hr.ach_processing import (
    generate_nacha_file, process_ach_returns, validate_routing_number
)
from app.hr.check_printing import (
    generate_check_pdf, generate_micr_line, void_check
)
from datetime import datetime
from decimal import Decimal

us_compliance_bp = Blueprint('us_compliance', __name__)


def _auth_or_401():
    """Helper to get current user or return 401"""
    user = get_current_user_flexible()
    if not user:
        return None, (jsonify({'error': 'Unauthorized'}), 401)
    return user, None


# ============================================================================
# State Tax Routes
# ============================================================================

@us_compliance_bp.route('/state-tax/calculate', methods=['POST'])
def calculate_state_tax_endpoint():
    """Calculate state tax"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_income = data.get('gross_income')
    state_code = data.get('state_code')
    
    if not gross_income or not state_code:
        return jsonify({'error': 'Gross income and state code are required'}), 400
    
    try:
        result = calculate_state_tax(
            gross_income,
            state_code,
            data.get('filing_status', 'single'),
            data.get('allowances', 0)
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@us_compliance_bp.route('/state-tax/multi-state', methods=['POST'])
def calculate_multi_state_tax_endpoint():
    """Calculate multi-state tax"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_income = data.get('gross_income')
    primary_state = data.get('primary_state')
    secondary_states = data.get('secondary_states', [])
    
    if not gross_income or not primary_state:
        return jsonify({'error': 'Gross income and primary state are required'}), 400
    
    try:
        result = calculate_multi_state_tax(gross_income, primary_state, secondary_states)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Local Tax Routes
# ============================================================================

@us_compliance_bp.route('/local-tax/calculate', methods=['POST'])
def calculate_local_tax_endpoint():
    """Calculate local tax"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_income = data.get('gross_income')
    
    if not gross_income:
        return jsonify({'error': 'Gross income is required'}), 400
    
    try:
        result = calculate_local_tax(
            gross_income,
            data.get('city'),
            data.get('county'),
            data.get('school_district')
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# SUI Routes
# ============================================================================

@us_compliance_bp.route('/sui/calculate', methods=['POST'])
def calculate_sui_endpoint():
    """Calculate SUI contribution"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_wages = data.get('gross_wages')
    state_code = data.get('state_code')
    
    if not gross_wages or not state_code:
        return jsonify({'error': 'Gross wages and state code are required'}), 400
    
    try:
        result = calculate_sui_contribution(
            gross_wages,
            state_code,
            data.get('sui_rate'),
            data.get('wage_base')
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# IRS Tax Tables Routes
# ============================================================================

@us_compliance_bp.route('/irs/federal-tax', methods=['POST'])
def calculate_federal_tax_endpoint():
    """Calculate federal income tax"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_income = data.get('gross_income')
    
    if not gross_income:
        return jsonify({'error': 'Gross income is required'}), 400
    
    try:
        result = calculate_total_federal_tax(
            gross_income,
            data.get('filing_status', 'single'),
            data.get('allowances', 0)
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Form 941 Routes
# ============================================================================

@us_compliance_bp.route('/form941/generate', methods=['POST'])
def generate_form941_endpoint():
    """Generate Form 941"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    quarter = data.get('quarter')
    tax_year = data.get('tax_year', datetime.now().year)
    
    if not quarter:
        return jsonify({'error': 'Quarter is required'}), 400
    
    try:
        form941 = generate_form941(tenant_id, quarter, tax_year, data.get('ein'))
        return jsonify(form941.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@us_compliance_bp.route('/form941/<int:form941_id>/pdf', methods=['POST'])
def generate_form941_pdf_endpoint(form941_id):
    """Generate Form 941 PDF"""
    user, error = _auth_or_401()
    if error:
        return error
    
    try:
        pdf_path = generate_form941_pdf(form941_id)
        if pdf_path:
            return jsonify({'pdf_path': pdf_path, 'message': 'PDF generated successfully'}), 200
        else:
            return jsonify({'error': 'PDF generation failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Form 940 Routes
# ============================================================================

@us_compliance_bp.route('/form940/generate', methods=['POST'])
def generate_form940_endpoint():
    """Generate Form 940"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    tax_year = data.get('tax_year', datetime.now().year)
    
    try:
        form940 = generate_form940(tenant_id, tax_year, data.get('ein'))
        return jsonify(form940.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Workers Compensation Routes
# ============================================================================

@us_compliance_bp.route('/workers-compensation/calculate', methods=['POST'])
def calculate_wc_endpoint():
    """Calculate Workers' Compensation"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_wages = data.get('gross_wages')
    state_code = data.get('state_code')
    
    if not gross_wages or not state_code:
        return jsonify({'error': 'Gross wages and state code are required'}), 400
    
    try:
        result = calculate_workers_compensation(
            gross_wages,
            state_code,
            data.get('wc_class_code')
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Garnishment Routes
# ============================================================================

@us_compliance_bp.route('/garnishment/create', methods=['POST'])
def create_garnishment_endpoint():
    """Create garnishment"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    employee_id = data.get('employee_id')
    garnishment_type = data.get('garnishment_type')
    priority_order = data.get('priority_order')
    
    if not all([employee_id, garnishment_type, priority_order]):
        return jsonify({'error': 'Employee ID, garnishment type, and priority order are required'}), 400
    
    try:
        garnishment = create_garnishment(
            employee_id=employee_id,
            garnishment_type=garnishment_type,
            priority_order=priority_order,
            amount_per_paycheck=data.get('amount_per_paycheck'),
            percentage_of_wages=data.get('percentage_of_wages'),
            maximum_deduction_limit=data.get('maximum_deduction_limit'),
            court_order_number=data.get('court_order_number'),
            court_name=data.get('court_name'),
            start_date=datetime.strptime(data['start_date'], '%Y-%m-%d').date() if data.get('start_date') else None,
            end_date=datetime.strptime(data['end_date'], '%Y-%m-%d').date() if data.get('end_date') else None
        )
        return jsonify(garnishment.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ACH Routes
# ============================================================================

@us_compliance_bp.route('/ach/generate', methods=['POST'])
def generate_ach_endpoint():
    """Generate NACHA file"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    pay_run_id = data.get('pay_run_id')
    
    if not pay_run_id:
        return jsonify({'error': 'Pay run ID is required'}), 400
    
    try:
        nacha_path = generate_nacha_file(
            pay_run_id,
            data.get('output_path'),
            data.get('company_name'),
            data.get('company_id'),
            data.get('immediate_destination'),
            data.get('immediate_origin')
        )
        return jsonify({'nacha_file_path': nacha_path, 'message': 'NACHA file generated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Check Printing Routes
# ============================================================================

@us_compliance_bp.route('/check/generate', methods=['POST'])
def generate_check_endpoint():
    """Generate check PDF"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    payslip_id = data.get('payslip_id')
    
    if not payslip_id:
        return jsonify({'error': 'Payslip ID is required'}), 400
    
    try:
        check_path = generate_check_pdf(
            payslip_id,
            data.get('output_path'),
            data.get('company_name'),
            data.get('company_address'),
            data.get('bank_name'),
            data.get('bank_routing'),
            data.get('bank_account')
        )
        if check_path:
            return jsonify({'check_path': check_path, 'message': 'Check generated successfully'}), 200
        else:
            return jsonify({'error': 'Check generation failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

