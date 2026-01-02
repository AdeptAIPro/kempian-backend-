"""
India Compliance API Routes
Handles PF, ESI, Professional Tax, TDS, HRA, LTA calculations and management
"""

from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, EmployeeProfile, Payslip, PayRun, db
from app.hr.pf_calculations import (
    calculate_pf_contribution, create_pf_contribution_record,
    get_pf_contributions_by_employee, get_pf_contributions_by_payrun,
    update_pf_challan_info
)
from app.hr.esi_calculations import (
    calculate_esi_contribution, create_esi_contribution_record,
    get_esi_contributions_by_employee, get_esi_contributions_by_payrun,
    update_esi_challan_info, validate_esi_card_number
)
from app.hr.professional_tax import (
    calculate_professional_tax, create_professional_tax_record,
    get_professional_tax_by_employee, get_professional_tax_slabs,
    get_professional_tax_exemption_limit
)
from app.hr.tds_calculations import (
    calculate_tds, create_tds_record, get_tds_records_by_employee
)
from app.hr.hra_exemption import (
    calculate_hra_exemption, update_hra_exemption, get_hra_exemption,
    validate_rent_receipts
)
from app.hr.lta_exemption import (
    calculate_lta_exemption, update_lta_exemption, get_lta_exemption,
    validate_lta_claim
)
from app.hr.form16_generator import (
    generate_form16, get_form16, generate_form16_pdf, sign_form16
)
from app.hr.form24q_generator import (
    generate_form24q, get_form24q, generate_form24q_xml, mark_form24q_efiled
)
from app.hr.challan_generator import (
    generate_challan_281, generate_epf_challan, generate_esi_challan,
    generate_pt_challan, get_challans_by_tenant, update_challan_payment_status
)
from datetime import datetime
from decimal import Decimal

india_compliance_bp = Blueprint('india_compliance', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


# ============================================================================
# PF (Provident Fund) Routes
# ============================================================================

@india_compliance_bp.route('/pf/calculate', methods=['POST'])
def calculate_pf():
    """Calculate PF contribution"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    basic_salary = data.get('basic_salary')
    employee_id = data.get('employee_id')
    
    if not basic_salary:
        return jsonify({'error': 'Basic salary is required'}), 400
    
    try:
        result = calculate_pf_contribution(basic_salary, employee_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/pf/employee/<int:employee_id>', methods=['POST'])
def create_pf_record(employee_id):
    """Create PF contribution record"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    basic_salary = data.get('basic_salary')
    
    if not basic_salary:
        return jsonify({'error': 'Basic salary is required'}), 400
    
    try:
        record = create_pf_contribution_record(
            employee_id=employee_id,
            basic_salary=basic_salary,
            payslip_id=data.get('payslip_id'),
            pay_run_id=data.get('pay_run_id'),
            contribution_month=data.get('contribution_month'),
            contribution_year=data.get('contribution_year'),
            epf_account_number=data.get('epf_account_number'),
            uan_number=data.get('uan_number')
        )
        return jsonify(record.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/pf/employee/<int:employee_id>', methods=['GET'])
def get_pf_records(employee_id):
    """Get PF contributions for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)
    
    try:
        records = get_pf_contributions_by_employee(employee_id, year, month)
        return jsonify([r.to_dict() for r in records]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/pf/payrun/<int:pay_run_id>', methods=['GET'])
def get_pf_by_payrun(pay_run_id):
    """Get PF contributions for a pay run"""
    user, error = _auth_or_401()
    if error:
        return error
    
    try:
        records = get_pf_contributions_by_payrun(pay_run_id)
        return jsonify([r.to_dict() for r in records]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/pf/<int:pf_id>/challan', methods=['PUT'])
def update_pf_challan(pf_id):
    """Update PF challan information"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    challan_number = data.get('challan_number')
    challan_date = data.get('challan_date')
    
    if not challan_number or not challan_date:
        return jsonify({'error': 'Challan number and date are required'}), 400
    
    try:
        challan_date = datetime.strptime(challan_date, '%Y-%m-%d').date()
        update_pf_challan_info(pf_id, challan_number, challan_date)
        return jsonify({'message': 'Challan information updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ESI Routes
# ============================================================================

@india_compliance_bp.route('/esi/calculate', methods=['POST'])
def calculate_esi():
    """Calculate ESI contribution"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_salary = data.get('gross_salary')
    employee_id = data.get('employee_id')
    
    if not gross_salary:
        return jsonify({'error': 'Gross salary is required'}), 400
    
    try:
        result = calculate_esi_contribution(gross_salary, employee_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/esi/employee/<int:employee_id>', methods=['POST'])
def create_esi_record(employee_id):
    """Create ESI contribution record"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_salary = data.get('gross_salary')
    
    if not gross_salary:
        return jsonify({'error': 'Gross salary is required'}), 400
    
    try:
        record = create_esi_contribution_record(
            employee_id=employee_id,
            gross_salary=gross_salary,
            payslip_id=data.get('payslip_id'),
            pay_run_id=data.get('pay_run_id'),
            contribution_month=data.get('contribution_month'),
            contribution_year=data.get('contribution_year'),
            esi_card_number=data.get('esi_card_number'),
            esi_dispensary_code=data.get('esi_dispensary_code')
        )
        return jsonify(record.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/esi/employee/<int:employee_id>', methods=['GET'])
def get_esi_records(employee_id):
    """Get ESI contributions for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)
    
    try:
        records = get_esi_contributions_by_employee(employee_id, year, month)
        return jsonify([r.to_dict() for r in records]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/esi/validate-card', methods=['POST'])
def validate_esi_card():
    """Validate ESI card number"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    esi_card_number = data.get('esi_card_number')
    
    if not esi_card_number:
        return jsonify({'error': 'ESI card number is required'}), 400
    
    try:
        is_valid = validate_esi_card_number(esi_card_number)
        return jsonify({'is_valid': is_valid}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Professional Tax Routes
# ============================================================================

@india_compliance_bp.route('/professional-tax/calculate', methods=['POST'])
def calculate_pt():
    """Calculate Professional Tax"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_salary = data.get('gross_salary')
    state_code = data.get('state_code')
    
    if not gross_salary or not state_code:
        return jsonify({'error': 'Gross salary and state code are required'}), 400
    
    try:
        result = calculate_professional_tax(gross_salary, state_code)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/professional-tax/employee/<int:employee_id>', methods=['POST'])
def create_pt_record(employee_id):
    """Create Professional Tax record"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_salary = data.get('gross_salary')
    state_code = data.get('state_code')
    
    if not gross_salary or not state_code:
        return jsonify({'error': 'Gross salary and state code are required'}), 400
    
    try:
        record = create_professional_tax_record(
            employee_id=employee_id,
            gross_salary=gross_salary,
            state_code=state_code,
            payslip_id=data.get('payslip_id'),
            deduction_month=data.get('deduction_month'),
            deduction_year=data.get('deduction_year'),
            pt_certificate_number=data.get('pt_certificate_number')
        )
        return jsonify(record.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/professional-tax/slabs/<state_code>', methods=['GET'])
def get_pt_slabs(state_code):
    """Get Professional Tax slabs for a state"""
    user, error = _auth_or_401()
    if error:
        return error
    
    try:
        slabs = get_professional_tax_slabs(state_code)
        exemption_limit = get_professional_tax_exemption_limit(state_code)
        return jsonify({
            'state_code': state_code,
            'slabs': slabs,
            'exemption_limit': float(exemption_limit)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TDS Routes
# ============================================================================

@india_compliance_bp.route('/tds/calculate', methods=['POST'])
def calculate_tds_endpoint():
    """Calculate TDS"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_salary = data.get('gross_salary')
    employee_id = data.get('employee_id')
    regime = data.get('regime', 'old')
    
    if not gross_salary or not employee_id:
        return jsonify({'error': 'Gross salary and employee ID are required'}), 400
    
    try:
        result = calculate_tds(gross_salary, employee_id, regime)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/tds/employee/<int:employee_id>', methods=['POST'])
def create_tds_record_endpoint(employee_id):
    """Create TDS record"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    gross_salary = data.get('gross_salary')
    
    if not gross_salary:
        return jsonify({'error': 'Gross salary is required'}), 400
    
    try:
        record = create_tds_record(
            employee_id=employee_id,
            gross_salary=gross_salary,
            payslip_id=data.get('payslip_id'),
            pay_run_id=data.get('pay_run_id'),
            tds_month=data.get('tds_month'),
            tds_year=data.get('tds_year'),
            tan_number=data.get('tan_number')
        )
        return jsonify(record.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/tds/employee/<int:employee_id>', methods=['GET'])
def get_tds_records(employee_id):
    """Get TDS records for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)
    
    try:
        records = get_tds_records_by_employee(employee_id, year, month)
        return jsonify([r.to_dict() for r in records]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HRA Exemption Routes
# ============================================================================

@india_compliance_bp.route('/hra/calculate', methods=['POST'])
def calculate_hra():
    """Calculate HRA exemption"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    hra_received = data.get('hra_received')
    basic_salary = data.get('basic_salary')
    rent_paid = data.get('rent_paid')
    is_metro_city = data.get('is_metro_city', False)
    
    if not all([hra_received, basic_salary, rent_paid]):
        return jsonify({'error': 'HRA received, basic salary, and rent paid are required'}), 400
    
    try:
        result = calculate_hra_exemption(hra_received, basic_salary, rent_paid, is_metro_city)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/hra/employee/<int:employee_id>', methods=['PUT'])
def update_hra(employee_id):
    """Update HRA exemption for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    hra_received = data.get('hra_received')
    basic_salary = data.get('basic_salary')
    rent_paid = data.get('rent_paid')
    
    if not all([hra_received, basic_salary, rent_paid]):
        return jsonify({'error': 'HRA received, basic salary, and rent paid are required'}), 400
    
    try:
        exemption = update_hra_exemption(
            employee_id=employee_id,
            hra_received=hra_received,
            basic_salary=basic_salary,
            rent_paid=rent_paid,
            is_metro_city=data.get('is_metro_city', False),
            rent_receipts_uploaded=data.get('rent_receipts_uploaded', False),
            financial_year=data.get('financial_year')
        )
        return jsonify(exemption.to_dict()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/hra/employee/<int:employee_id>', methods=['GET'])
def get_hra(employee_id):
    """Get HRA exemption for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    try:
        result = get_hra_exemption(employee_id)
        if result:
            return jsonify(result), 200
        else:
            return jsonify({'message': 'No HRA exemption found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# LTA Exemption Routes
# ============================================================================

@india_compliance_bp.route('/lta/calculate', methods=['POST'])
def calculate_lta():
    """Calculate LTA exemption"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    lta_received = data.get('lta_received')
    travel_cost = data.get('travel_cost')
    block_year = data.get('block_year')
    
    if not all([lta_received, travel_cost, block_year]):
        return jsonify({'error': 'LTA received, travel cost, and block year are required'}), 400
    
    try:
        result = calculate_lta_exemption(
            lta_received, travel_cost, block_year,
            data.get('travel_dates'), data.get('family_travel', False)
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/lta/employee/<int:employee_id>', methods=['PUT'])
def update_lta(employee_id):
    """Update LTA exemption for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    lta_received = data.get('lta_received')
    travel_cost = data.get('travel_cost')
    block_year = data.get('block_year')
    
    if not all([lta_received, travel_cost, block_year]):
        return jsonify({'error': 'LTA received, travel cost, and block year are required'}), 400
    
    try:
        exemption = update_lta_exemption(
            employee_id=employee_id,
            lta_received=lta_received,
            travel_cost=travel_cost,
            block_year=block_year,
            travel_dates=data.get('travel_dates'),
            family_travel=data.get('family_travel', False),
            financial_year=data.get('financial_year')
        )
        return jsonify(exemption.to_dict()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/lta/employee/<int:employee_id>', methods=['GET'])
def get_lta(employee_id):
    """Get LTA exemption for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    try:
        result = get_lta_exemption(employee_id)
        if result:
            return jsonify(result), 200
        else:
            return jsonify({'message': 'No LTA exemption found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Form 16 Routes
# ============================================================================

@india_compliance_bp.route('/form16/generate', methods=['POST'])
def generate_form16_endpoint():
    """Generate Form 16 certificate"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    employee_id = data.get('employee_id')
    financial_year = data.get('financial_year')
    
    if not employee_id or not financial_year:
        return jsonify({'error': 'Employee ID and financial year are required'}), 400
    
    try:
        form16 = generate_form16(
            employee_id=employee_id,
            financial_year=financial_year,
            tenant_id=data.get('tenant_id'),
            tan_number=data.get('tan_number')
        )
        return jsonify(form16.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/form16/employee/<int:employee_id>/<int:financial_year>', methods=['GET'])
def get_form16_endpoint(employee_id, financial_year):
    """Get Form 16 for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    try:
        form16 = get_form16(employee_id, financial_year)
        if form16:
            return jsonify(form16.to_dict()), 200
        else:
            return jsonify({'message': 'Form 16 not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/form16/<int:form16_id>/pdf', methods=['POST'])
def generate_form16_pdf_endpoint(form16_id):
    """Generate PDF for Form 16"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    output_path = data.get('output_path')
    
    try:
        pdf_path = generate_form16_pdf(form16_id, output_path)
        if pdf_path:
            return jsonify({'pdf_path': pdf_path, 'message': 'PDF generated successfully'}), 200
        else:
            return jsonify({'error': 'PDF generation failed. reportlab library may not be installed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/form16/<int:form16_id>/sign', methods=['POST'])
def sign_form16_endpoint(form16_id):
    """Sign Form 16 with digital signature"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json() or {}
    digital_signature = data.get('digital_signature')
    
    try:
        sign_form16(form16_id, db_user.id, digital_signature)
        return jsonify({'message': 'Form 16 signed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Form 24Q Routes
# ============================================================================

@india_compliance_bp.route('/form24q/generate', methods=['POST'])
def generate_form24q_endpoint():
    """Generate Form 24Q quarterly return"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    quarter = data.get('quarter')
    financial_year = data.get('financial_year')
    
    if not quarter or not financial_year:
        return jsonify({'error': 'Quarter and financial year are required'}), 400
    
    try:
        form24q = generate_form24q(
            tenant_id=tenant_id,
            quarter=quarter,
            financial_year=financial_year,
            tan_number=data.get('tan_number')
        )
        return jsonify(form24q.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/form24q/<int:form24q_id>/xml', methods=['POST'])
def generate_form24q_xml_endpoint(form24q_id):
    """Generate XML for Form 24Q e-filing"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    output_path = data.get('output_path')
    
    try:
        xml_path = generate_form24q_xml(form24q_id, output_path)
        return jsonify({'xml_path': xml_path, 'message': 'XML generated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/form24q/<int:form24q_id>/efile', methods=['POST'])
def mark_form24q_efiled_endpoint(form24q_id):
    """Mark Form 24Q as e-filed"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json() or {}
    acknowledgment_number = data.get('acknowledgment_number')
    
    if not acknowledgment_number:
        return jsonify({'error': 'Acknowledgment number is required'}), 400
    
    try:
        mark_form24q_efiled(form24q_id, acknowledgment_number, db_user.id)
        return jsonify({'message': 'Form 24Q marked as e-filed'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Challan Routes
# ============================================================================

@india_compliance_bp.route('/challan/281', methods=['POST'])
def generate_challan_281_endpoint():
    """Generate Challan 281 for TDS payment"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    payment_month = data.get('payment_month')
    payment_year = data.get('payment_year')
    
    if not payment_month or not payment_year:
        return jsonify({'error': 'Payment month and year are required'}), 400
    
    try:
        challan = generate_challan_281(
            tenant_id=tenant_id,
            payment_month=payment_month,
            payment_year=payment_year,
            tan_number=data.get('tan_number')
        )
        return jsonify(challan.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/challan/epf', methods=['POST'])
def generate_epf_challan_endpoint():
    """Generate EPF challan"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    payment_month = data.get('payment_month')
    payment_year = data.get('payment_year')
    
    if not payment_month or not payment_year:
        return jsonify({'error': 'Payment month and year are required'}), 400
    
    try:
        challan = generate_epf_challan(
            tenant_id=tenant_id,
            payment_month=payment_month,
            payment_year=payment_year,
            pay_run_id=data.get('pay_run_id')
        )
        return jsonify(challan.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/challan/esi', methods=['POST'])
def generate_esi_challan_endpoint():
    """Generate ESI challan"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    payment_month = data.get('payment_month')
    payment_year = data.get('payment_year')
    
    if not payment_month or not payment_year:
        return jsonify({'error': 'Payment month and year are required'}), 400
    
    try:
        challan = generate_esi_challan(
            tenant_id=tenant_id,
            payment_month=payment_month,
            payment_year=payment_year,
            pay_run_id=data.get('pay_run_id')
        )
        return jsonify(challan.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/challan/pt', methods=['POST'])
def generate_pt_challan_endpoint():
    """Generate Professional Tax challan"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    payment_month = data.get('payment_month')
    payment_year = data.get('payment_year')
    state_code = data.get('state_code')
    
    if not all([payment_month, payment_year, state_code]):
        return jsonify({'error': 'Payment month, year, and state code are required'}), 400
    
    try:
        challan = generate_pt_challan(
            tenant_id=tenant_id,
            payment_month=payment_month,
            payment_year=payment_year,
            state_code=state_code
        )
        return jsonify(challan.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/challan', methods=['GET'])
def get_challans():
    """Get challans for current tenant"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    challan_type = request.args.get('challan_type')
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)
    
    try:
        challans = get_challans_by_tenant(tenant_id, challan_type, year, month)
        return jsonify([c.to_dict() for c in challans]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@india_compliance_bp.route('/challan/<int:challan_id>/payment-status', methods=['PUT'])
def update_challan_payment_status_endpoint(challan_id):
    """Update challan payment status"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    payment_status = data.get('payment_status')
    
    if not payment_status:
        return jsonify({'error': 'Payment status is required'}), 400
    
    if payment_status not in ['pending', 'paid', 'failed']:
        return jsonify({'error': 'Payment status must be pending, paid, or failed'}), 400
    
    try:
        update_challan_payment_status(
            challan_id,
            payment_status,
            data.get('payment_reference')
        )
        return jsonify({'message': 'Payment status updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

