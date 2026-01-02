"""
Compliance Reports API
Handles W-2, 1099, quarterly, and annual payroll reports
"""
from flask import Blueprint, jsonify, request, send_file
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, Payslip, Timesheet, EmployeeProfile, db
from datetime import datetime
from decimal import Decimal
import csv
import io
import calendar

hr_compliance_bp = Blueprint('hr_compliance', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@hr_compliance_bp.route('/w2/<int:year>', methods=['GET'], strict_slashes=False)
def generate_w2(year):
    """Generate W-2 forms for a given year"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to generate W-2 forms'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    employee_id = request.args.get('employee_id', type=int)
    
    # Get payslips for the year
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()
    
    query = Payslip.query.filter(
        Payslip.pay_period_start >= start_date,
        Payslip.pay_period_end <= end_date,
        Payslip.status == 'paid'
    )
    
    if employee_id:
        query = query.filter_by(employee_id=employee_id)
    else:
        # Get all employees in tenant
        employee_ids = [u.id for u in User.query.filter_by(tenant_id=tenant_id).all()]
        query = query.filter(Payslip.employee_id.in_(employee_ids))
    
    payslips = query.all()
    
    # Group by employee
    employee_data = {}
    for payslip in payslips:
        emp_id = payslip.employee_id
        if emp_id not in employee_data:
            employee_data[emp_id] = {
                'payslips': [],
                'total_wages': Decimal('0'),
                'total_federal_tax': Decimal('0'),
                'total_social_security_wages': Decimal('0'),
                'total_social_security_tax': Decimal('0'),
                'total_medicare_wages': Decimal('0'),
                'total_medicare_tax': Decimal('0')
            }
        
        employee_data[emp_id]['payslips'].append(payslip)
        employee_data[emp_id]['total_wages'] += payslip.gross_earnings or Decimal('0')
        employee_data[emp_id]['total_federal_tax'] += payslip.tax_deduction or Decimal('0')
        # Note: For accurate W-2, you'd need to track Social Security and Medicare separately
        # This is a simplified version
        employee_data[emp_id]['total_social_security_wages'] += payslip.gross_earnings or Decimal('0')
        employee_data[emp_id]['total_medicare_wages'] += payslip.gross_earnings or Decimal('0')
    
    # Build W-2 data
    w2_forms = []
    for emp_id, data in employee_data.items():
        employee = User.query.get(emp_id)
        employee_profile = EmployeeProfile.query.filter_by(user_id=emp_id).first()
        
        if not employee or not employee_profile:
            continue
        
        w2_form = {
            'employee_id': emp_id,
            'employee_name': f"{employee_profile.first_name or ''} {employee_profile.last_name or ''}".strip(),
            'employee_ssn': '***-**-****',  # Should be stored securely
            'employee_address': employee_profile.location or '',
            'wages_tips_other_compensation': float(data['total_wages']),
            'federal_income_tax_withheld': float(data['total_federal_tax']),
            'social_security_wages': float(data['total_social_security_wages']),
            'social_security_tax_withheld': float(data['total_social_security_wages'] * Decimal('0.062')),
            'medicare_wages': float(data['total_medicare_wages']),
            'medicare_tax_withheld': float(data['total_medicare_wages'] * Decimal('0.0145')),
            'year': year
        }
        w2_forms.append(w2_form)
    
    # Return as CSV if requested
    format_type = request.args.get('format', 'json')
    if format_type == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            'Employee ID', 'Employee Name', 'SSN', 'Wages', 'Federal Tax',
            'Social Security Wages', 'Social Security Tax', 'Medicare Wages', 'Medicare Tax', 'Year'
        ])
        
        for w2 in w2_forms:
            writer.writerow([
                w2['employee_id'],
                w2['employee_name'],
                w2['employee_ssn'],
                w2['wages_tips_other_compensation'],
                w2['federal_income_tax_withheld'],
                w2['social_security_wages'],
                w2['social_security_tax_withheld'],
                w2['medicare_wages'],
                w2['medicare_tax_withheld'],
                w2['year']
            ])
        
        output.seek(0)
        response = send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'w2_forms_{year}.csv'
        )
        return response
    
    return jsonify({'w2_forms': w2_forms, 'year': year, 'count': len(w2_forms)}), 200


@hr_compliance_bp.route('/1099/<int:year>', methods=['GET'], strict_slashes=False)
def generate_1099(year):
    """Generate 1099 forms for contractors for a given year"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to generate 1099 forms'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    # Get contractors (employees with employment_type = 'contractor')
    contractors = User.query.join(EmployeeProfile).filter(
        User.tenant_id == tenant_id,
        (User.user_type == 'employee') | (User.role == 'employee'),
        EmployeeProfile.employment_type == 'contractor'
    ).all()
    
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()
    
    contractor_data = []
    for contractor in contractors:
        employee_profile = EmployeeProfile.query.filter_by(user_id=contractor.id).first()
        if not employee_profile:
            continue
        
        # Get payslips for this contractor
        payslips = Payslip.query.filter(
            Payslip.employee_id == contractor.id,
            Payslip.pay_period_start >= start_date,
            Payslip.pay_period_end <= end_date,
            Payslip.status == 'paid'
        ).all()
        
        if not payslips:
            continue
        
        total_compensation = sum(float(p.gross_earnings or 0) for p in payslips)
        
        if total_compensation >= 600:  # 1099 threshold
            contractor_data.append({
                'contractor_id': contractor.id,
                'contractor_name': f"{employee_profile.first_name or ''} {employee_profile.last_name or ''}".strip(),
                'contractor_tin': '***-**-****',  # Should be stored securely
                'contractor_address': employee_profile.location or '',
                'nonemployee_compensation': total_compensation,
                'year': year
            })
    
    format_type = request.args.get('format', 'json')
    if format_type == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            'Contractor ID', 'Contractor Name', 'TIN', 'Nonemployee Compensation', 'Year'
        ])
        
        for data in contractor_data:
            writer.writerow([
                data['contractor_id'],
                data['contractor_name'],
                data['contractor_tin'],
                data['nonemployee_compensation'],
                data['year']
            ])
        
        output.seek(0)
        response = send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'1099_forms_{year}.csv'
        )
        return response
    
    return jsonify({'1099_forms': contractor_data, 'year': year, 'count': len(contractor_data)}), 200


@hr_compliance_bp.route('/quarterly', methods=['GET'], strict_slashes=False)
def get_quarterly_report():
    """Get quarterly payroll report"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view quarterly reports'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    year = request.args.get('year', type=int) or datetime.now().year
    quarter = request.args.get('quarter', type=int) or ((datetime.now().month - 1) // 3 + 1)
    
    if quarter < 1 or quarter > 4:
        return jsonify({'error': 'Quarter must be between 1 and 4'}), 400
    
    # Calculate quarter dates
    quarter_start_month = (quarter - 1) * 3 + 1
    quarter_end_month = quarter * 3
    start_date = datetime(year, quarter_start_month, 1).date()
    last_day = calendar.monthrange(year, quarter_end_month)[1]
    end_date = datetime(year, quarter_end_month, last_day).date()
    
    # Get payslips for the quarter
    employee_ids = [u.id for u in User.query.filter_by(tenant_id=tenant_id).all()]
    payslips = Payslip.query.filter(
        Payslip.employee_id.in_(employee_ids),
        Payslip.pay_period_start >= start_date,
        Payslip.pay_period_end <= end_date,
        Payslip.status == 'paid'
    ).all()
    
    # Calculate totals
    total_wages = sum(float(p.gross_earnings or 0) for p in payslips)
    total_tax_withheld = sum(float(p.tax_deduction or 0) for p in payslips)
    total_social_security_wages = sum(float(p.gross_earnings or 0) for p in payslips)
    total_social_security_tax = total_social_security_wages * 0.062
    total_medicare_wages = sum(float(p.gross_earnings or 0) for p in payslips)
    total_medicare_tax = total_medicare_wages * 0.0145
    total_employees = len(set(p.employee_id for p in payslips))
    
    report = {
        'quarter': quarter,
        'year': year,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_wages': round(total_wages, 2),
        'total_tax_withheld': round(total_tax_withheld, 2),
        'total_social_security_wages': round(total_social_security_wages, 2),
        'total_social_security_tax': round(total_social_security_tax, 2),
        'total_medicare_wages': round(total_medicare_wages, 2),
        'total_medicare_tax': round(total_medicare_tax, 2),
        'total_employees': total_employees,
        'payslip_count': len(payslips)
    }
    
    return jsonify({'quarterly_report': report}), 200


@hr_compliance_bp.route('/annual', methods=['GET'], strict_slashes=False)
def get_annual_report():
    """Get annual payroll summary"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view annual reports'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    year = request.args.get('year', type=int) or datetime.now().year
    
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()
    
    # Get payslips for the year
    employee_ids = [u.id for u in User.query.filter_by(tenant_id=tenant_id).all()]
    payslips = Payslip.query.filter(
        Payslip.employee_id.in_(employee_ids),
        Payslip.pay_period_start >= start_date,
        Payslip.pay_period_end <= end_date,
        Payslip.status == 'paid'
    ).all()
    
    # Calculate totals
    total_wages = sum(float(p.gross_earnings or 0) for p in payslips)
    total_tax_withheld = sum(float(p.tax_deduction or 0) for p in payslips)
    total_deductions = sum(float(p.total_deductions or 0) for p in payslips)
    total_net_pay = sum(float(p.net_pay or 0) for p in payslips)
    total_employees = len(set(p.employee_id for p in payslips))
    
    # Quarterly breakdown
    quarters = {}
    for q in range(1, 5):
        q_start_month = (q - 1) * 3 + 1
        q_end_month = q * 3
        q_start = datetime(year, q_start_month, 1).date()
        q_last_day = calendar.monthrange(year, q_end_month)[1]
        q_end = datetime(year, q_end_month, q_last_day).date()
        
        q_payslips = [p for p in payslips if q_start <= p.pay_period_start <= q_end]
        quarters[f'Q{q}'] = {
            'wages': round(sum(float(p.gross_earnings or 0) for p in q_payslips), 2),
            'tax_withheld': round(sum(float(p.tax_deduction or 0) for p in q_payslips), 2),
            'payslip_count': len(q_payslips)
        }
    
    report = {
        'year': year,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_wages': round(total_wages, 2),
        'total_tax_withheld': round(total_tax_withheld, 2),
        'total_deductions': round(total_deductions, 2),
        'total_net_pay': round(total_net_pay, 2),
        'total_employees': total_employees,
        'payslip_count': len(payslips),
        'quarterly_breakdown': quarters
    }
    
    return jsonify({'annual_report': report}), 200

