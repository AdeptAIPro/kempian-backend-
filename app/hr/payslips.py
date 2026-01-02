"""
Payslip Generation API
Generates payslips for employees based on their timesheets
"""
from flask import Blueprint, jsonify, request, send_file
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, EmployeeProfile, Timesheet, Payslip, db
from app.emails.ses import send_invite_email, send_payslip_email
from datetime import datetime, timedelta
from decimal import Decimal
import calendar
import csv
import io
import os

hr_payslips_bp = Blueprint('hr_payslips', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@hr_payslips_bp.route('/generate/<int:employee_id>', methods=['POST'])
def generate_payslip(employee_id):
    """Generate payslip for an employee for a specific pay period"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, employer, recruiter, and owner can generate payslips
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to generate payslips'}), 403
    
    data = request.get_json() or {}
    pay_period_start = data.get('pay_period_start')
    pay_period_end = data.get('pay_period_end')
    
    # If dates not provided, use current month
    if not pay_period_start or not pay_period_end:
        today = datetime.now().date()
        # First day of current month
        pay_period_start = today.replace(day=1)
        # Last day of current month
        last_day = calendar.monthrange(today.year, today.month)[1]
        pay_period_end = today.replace(day=last_day)
    else:
        try:
            pay_period_start = datetime.strptime(pay_period_start, '%Y-%m-%d').date()
            pay_period_end = datetime.strptime(pay_period_end, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    if employee.role != 'employee' and employee.user_type != 'employee':
        return jsonify({'error': 'User is not an employee'}), 400
    
    # Get employee profile
    employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
    if not employee_profile:
        return jsonify({'error': 'Employee profile not found'}), 404
    
    # Get approved timesheets for the pay period
    timesheets = Timesheet.query.filter(
        Timesheet.user_id == employee_id,
        Timesheet.status == 'approved',
        Timesheet.date >= pay_period_start,
        Timesheet.date <= pay_period_end
    ).all()
    
    # Calculate totals
    total_regular_hours = sum(float(ts.regular_hours or 0) for ts in timesheets)
    total_overtime_hours = sum(float(ts.overtime_hours or 0) for ts in timesheets)
    total_holiday_hours = sum(float(ts.holiday_hours or 0) for ts in timesheets)
    total_hours = total_regular_hours + total_overtime_hours + total_holiday_hours
    
    total_regular_earnings = sum(float(ts.regular_earnings or 0) for ts in timesheets)
    total_overtime_earnings = sum(float(ts.overtime_earnings or 0) for ts in timesheets)
    total_holiday_earnings = sum(float(ts.holiday_earnings or 0) for ts in timesheets)
    total_bonus = sum(float(ts.bonus_amount or 0) for ts in timesheets)
    gross_earnings = total_regular_earnings + total_overtime_earnings + total_holiday_earnings + total_bonus
    
    # Calculate taxes using new tax management system
    from app.models import TaxConfiguration, EmployeeTaxProfile
    
    def _calculate_progressive_tax(income, brackets):
        """Calculate tax using progressive brackets"""
        from decimal import Decimal
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
    
    tenant_id = get_current_tenant_id() or employee.tenant_id
    tax_profile = EmployeeTaxProfile.query.filter_by(employee_id=employee_id).first()
    
    # Get active tax configurations
    tax_configs = TaxConfiguration.query.filter(
        TaxConfiguration.tenant_id == tenant_id,
        TaxConfiguration.is_active == True
    ).all()
    
    tax_deduction = Decimal('0')
    tax_breakdown = {}
    
    for config in tax_configs:
        # Check exemptions
        if config.tax_type == 'federal' and tax_profile and tax_profile.exempt_from_federal:
            continue
        if config.tax_type == 'state' and tax_profile and tax_profile.exempt_from_state:
            continue
        if config.tax_type == 'local' and tax_profile and tax_profile.exempt_from_local:
            continue
        
        gross = Decimal(str(gross_earnings))
        tax_amount = Decimal('0')
        
        if config.tax_type in ['fica_social_security', 'fica_medicare']:
            # FICA calculations
            if config.tax_type == 'fica_social_security':
                rate = Decimal('0.062')
                wage_base = config.wage_base_limit if config.wage_base_limit else Decimal('160200')
                taxable_amount = min(gross, wage_base)
                tax_amount = taxable_amount * rate
            elif config.tax_type == 'fica_medicare':
                rate = Decimal('0.0145')
                tax_amount = gross * rate
                # Additional Medicare tax (0.9%) on earnings over $200k
                if gross > Decimal('200000'):
                    additional_tax = (gross - Decimal('200000')) * Decimal('0.009')
                    tax_amount += additional_tax
        elif config.tax_rate:
            tax_amount = gross * config.tax_rate
        elif config.tax_brackets:
            tax_amount = _calculate_progressive_tax(gross, config.tax_brackets)
        
        tax_breakdown[config.tax_type] = float(tax_amount)
        tax_deduction += tax_amount
    
    # Add additional withholding if specified
    if tax_profile and tax_profile.additional_withholding:
        tax_deduction += tax_profile.additional_withholding
        tax_breakdown['additional_withholding'] = float(tax_profile.additional_withholding)
    
    # Calculate deductions using new deductions system
    from app.models import EmployeeDeduction, DeductionType
    
    today = datetime.now().date()
    employee_deductions = EmployeeDeduction.query.filter(
        EmployeeDeduction.employee_id == employee_id,
        EmployeeDeduction.is_active == True,
        EmployeeDeduction.effective_date <= today,
        db.or_(
            EmployeeDeduction.end_date.is_(None),
            EmployeeDeduction.end_date >= today
        )
    ).join(DeductionType).all()
    
    pre_tax_deductions = Decimal('0')
    post_tax_deductions = Decimal('0')
    
    for emp_deduction in employee_deductions:
        deduction_type = emp_deduction.deduction_type
        amount = Decimal('0')
        
        if emp_deduction.amount:
            amount = emp_deduction.amount
        elif emp_deduction.percentage:
            amount = Decimal(str(gross_earnings)) * emp_deduction.percentage
        elif deduction_type.default_amount:
            amount = deduction_type.default_amount
        elif deduction_type.default_percentage:
            amount = Decimal(str(gross_earnings)) * deduction_type.default_percentage
        
        if deduction_type.is_pre_tax:
            pre_tax_deductions += amount
        else:
            post_tax_deductions += amount
    
    # Calculate tax on gross minus pre-tax deductions
    taxable_gross = Decimal(str(gross_earnings)) - pre_tax_deductions
    
    # Recalculate tax on taxable gross (if pre-tax deductions exist)
    if pre_tax_deductions > 0:
        tax_deduction = Decimal('0')
        tax_breakdown = {}
        
        for config in tax_configs:
            if config.tax_type == 'federal' and tax_profile and tax_profile.exempt_from_federal:
                continue
            if config.tax_type == 'state' and tax_profile and tax_profile.exempt_from_state:
                continue
            if config.tax_type == 'local' and tax_profile and tax_profile.exempt_from_local:
                continue
            
            tax_amount = Decimal('0')
            
            if config.tax_type in ['fica_social_security', 'fica_medicare']:
                if config.tax_type == 'fica_social_security':
                    rate = Decimal('0.062')
                    wage_base = config.wage_base_limit if config.wage_base_limit else Decimal('160200')
                    taxable_amount = min(taxable_gross, wage_base)
                    tax_amount = taxable_amount * rate
                elif config.tax_type == 'fica_medicare':
                    rate = Decimal('0.0145')
                    tax_amount = taxable_gross * rate
                    if taxable_gross > Decimal('200000'):
                        additional_tax = (taxable_gross - Decimal('200000')) * Decimal('0.009')
                        tax_amount += additional_tax
            elif config.tax_rate:
                tax_amount = taxable_gross * config.tax_rate
            elif config.tax_brackets:
                tax_amount = _calculate_progressive_tax(taxable_gross, config.tax_brackets)
            
            tax_breakdown[config.tax_type] = float(tax_amount)
            tax_deduction += tax_amount
        
        if tax_profile and tax_profile.additional_withholding:
            tax_deduction += tax_profile.additional_withholding
            tax_breakdown['additional_withholding'] = float(tax_profile.additional_withholding)
    
    other_deductions = float(pre_tax_deductions + post_tax_deductions)
    
    # India Compliance Calculations
    india_statutory_deductions = Decimal('0')
    pf_employee = Decimal('0')
    pf_employer = Decimal('0')
    esi_employee = Decimal('0')
    esi_employer = Decimal('0')
    professional_tax = Decimal('0')
    tds_amount = Decimal('0')
    hra_exemption = Decimal('0')
    lta_exemption = Decimal('0')
    
    # Check if employee is in India
    if employee_profile and employee_profile.country_code == 'IN':
        try:
            # Calculate basic salary (estimate 50% of gross if not available)
            basic_salary = Decimal(str(gross_earnings)) * Decimal('0.5')
            
            # PF Calculation
            from app.hr.pf_calculations import calculate_pf_contribution
            pf_calc = calculate_pf_contribution(basic_salary, employee_id)
            pf_employee = Decimal(str(pf_calc['employee_pf']))
            pf_employer = Decimal(str(pf_calc['employer_pf']))
            india_statutory_deductions += pf_employee
            
            # ESI Calculation
            from app.hr.esi_calculations import calculate_esi_contribution
            esi_calc = calculate_esi_contribution(gross_earnings, employee_id)
            esi_employee = Decimal(str(esi_calc['employee_esi']))
            esi_employer = Decimal(str(esi_calc['employer_esi']))
            india_statutory_deductions += esi_employee
            
            # Professional Tax
            if employee_profile.state_code:
                from app.hr.professional_tax import calculate_professional_tax
                pt_calc = calculate_professional_tax(gross_earnings, employee_profile.state_code)
                professional_tax = Decimal(str(pt_calc['professional_tax_amount']))
                india_statutory_deductions += professional_tax
            
            # TDS Calculation
            from app.hr.tds_calculations import calculate_tds
            regime = employee_profile.tax_regime or 'old'
            tds_calc = calculate_tds(gross_earnings, employee_id, regime)
            tds_amount = Decimal(str(tds_calc['monthly_tds']))
            india_statutory_deductions += tds_amount
            
            # HRA Exemption (if applicable)
            from app.hr.hra_exemption import get_hra_exemption
            hra_exemption_data = get_hra_exemption(employee_id)
            if hra_exemption_data:
                hra_exemption = Decimal(str(hra_exemption_data.get('hra_exemption_amount', 0)))
            
            # LTA Exemption (if applicable)
            from app.hr.lta_exemption import get_lta_exemption
            lta_exemption_data = get_lta_exemption(employee_id)
            if lta_exemption_data:
                lta_exemption = Decimal(str(lta_exemption_data.get('lta_exemption_amount', 0)))
            
        except Exception as e:
            logger.warning(f"Error calculating India compliance for employee {employee_id}: {str(e)}")
    
    # US Compliance Calculations (placeholder for future)
    state_tax = Decimal('0')
    local_tax = Decimal('0')
    sui_contribution = Decimal('0')
    workers_compensation = Decimal('0')
    garnishment_total = Decimal('0')
    
    if employee_profile and employee_profile.country_code == 'US':
        # US calculations will be added in Phase 3
        pass
    
    # Update total deductions with India statutory deductions
    total_deductions = float(tax_deduction) + other_deductions + float(india_statutory_deductions)
    net_pay = gross_earnings - total_deductions
    
    # Check if payslip already exists for this period
    existing_payslip = Payslip.query.filter(
        Payslip.employee_id == employee_id,
        Payslip.pay_period_start == pay_period_start,
        Payslip.pay_period_end == pay_period_end
    ).first()
    
    # Create or update payslip record
    if existing_payslip:
        payslip_record = existing_payslip
    else:
        payslip_record = Payslip(
            employee_id=employee_id,
            employee_profile_id=employee_profile.id if employee_profile else None,
            pay_period_start=pay_period_start,
            pay_period_end=pay_period_end,
            pay_date=datetime.now().date()
        )
        db.session.add(payslip_record)
    
    # Update payslip data
    payslip_record.total_regular_hours = Decimal(str(round(total_regular_hours, 2)))
    payslip_record.total_overtime_hours = Decimal(str(round(total_overtime_hours, 2)))
    payslip_record.total_holiday_hours = Decimal(str(round(total_holiday_hours, 2)))
    payslip_record.total_hours = Decimal(str(round(total_hours, 2)))
    payslip_record.regular_earnings = Decimal(str(round(total_regular_earnings, 2)))
    payslip_record.overtime_earnings = Decimal(str(round(total_overtime_earnings, 2)))
    payslip_record.holiday_earnings = Decimal(str(round(total_holiday_earnings, 2)))
    payslip_record.bonus_amount = Decimal(str(round(total_bonus, 2)))
    payslip_record.gross_earnings = Decimal(str(round(gross_earnings, 2)))
    payslip_record.tax_deduction = Decimal(str(round(tax_deduction, 2)))
    payslip_record.other_deductions = Decimal(str(round(other_deductions, 2)))
    payslip_record.total_deductions = Decimal(str(round(total_deductions, 2)))
    payslip_record.net_pay = Decimal(str(round(net_pay, 2)))
    payslip_record.currency = employee_profile.salary_currency or 'USD'
    payslip_record.status = 'generated'
    
    # India compliance fields
    payslip_record.pf_employee = Decimal(str(round(pf_employee, 2)))
    payslip_record.pf_employer = Decimal(str(round(pf_employer, 2)))
    payslip_record.esi_employee = Decimal(str(round(esi_employee, 2)))
    payslip_record.esi_employer = Decimal(str(round(esi_employer, 2)))
    payslip_record.professional_tax = Decimal(str(round(professional_tax, 2)))
    payslip_record.tds_amount = Decimal(str(round(tds_amount, 2)))
    payslip_record.hra_exemption = Decimal(str(round(hra_exemption, 2)))
    payslip_record.lta_exemption = Decimal(str(round(lta_exemption, 2)))
    
    # US compliance fields
    payslip_record.state_tax = Decimal(str(round(state_tax, 2)))
    payslip_record.local_tax = Decimal(str(round(local_tax, 2)))
    payslip_record.sui_contribution = Decimal(str(round(sui_contribution, 2)))
    payslip_record.workers_compensation = Decimal(str(round(workers_compensation, 2)))
    payslip_record.garnishment_total = Decimal(str(round(garnishment_total, 2)))
    
    db.session.commit()
    
    # Create India compliance contribution records if employee is in India
    if employee_profile and employee_profile.country_code == 'IN':
        try:
            from app.hr.pf_calculations import create_pf_contribution_record
            from app.hr.esi_calculations import create_esi_contribution_record
            from app.hr.professional_tax import create_professional_tax_record
            from app.hr.tds_calculations import create_tds_record
            
            # Create PF contribution record
            create_pf_contribution_record(
                employee_id=employee_id,
                basic_salary=float(basic_salary),
                payslip_id=payslip_record.id,
                contribution_month=pay_period_end.month,
                contribution_year=pay_period_end.year
            )
            
            # Create ESI contribution record
            create_esi_contribution_record(
                employee_id=employee_id,
                gross_salary=gross_earnings,
                payslip_id=payslip_record.id,
                contribution_month=pay_period_end.month,
                contribution_year=pay_period_end.year
            )
            
            # Create Professional Tax record
            if employee_profile.state_code and professional_tax > 0:
                create_professional_tax_record(
                    employee_id=employee_id,
                    gross_salary=gross_earnings,
                    state_code=employee_profile.state_code,
                    payslip_id=payslip_record.id,
                    deduction_month=pay_period_end.month,
                    deduction_year=pay_period_end.year
                )
            
            # Create TDS record
            if tds_amount > 0:
                create_tds_record(
                    employee_id=employee_id,
                    gross_salary=gross_earnings,
                    payslip_id=payslip_record.id,
                    tds_month=pay_period_end.month,
                    tds_year=pay_period_end.year
                )
        except Exception as e:
            logger.warning(f"Error creating India compliance records: {str(e)}")
    
    db.session.commit()
    
    # Build payslip response data
    payslip = {
        'id': payslip_record.id,
        'employee': {
            'id': employee.id,
            'email': employee.email,
            'first_name': employee_profile.first_name,
            'last_name': employee_profile.last_name,
            'department': employee_profile.department,
            'location': employee_profile.location,
            'employee_id': f"EMP{employee.id:05d}"
        },
        'pay_period': {
            'start_date': pay_period_start.isoformat(),
            'end_date': pay_period_end.isoformat(),
            'pay_date': datetime.now().date().isoformat()
        },
        'hours': {
            'regular': round(total_regular_hours, 2),
            'overtime': round(total_overtime_hours, 2),
            'holiday': round(total_holiday_hours, 2),
            'total': round(total_hours, 2)
        },
        'earnings': {
            'regular': round(total_regular_earnings, 2),
            'overtime': round(total_overtime_earnings, 2),
            'holiday': round(total_holiday_earnings, 2),
            'bonus': round(total_bonus, 2),
            'gross': round(gross_earnings, 2)
        },
        'deductions': {
            'tax': round(tax_deduction, 2),
            'other': round(other_deductions, 2),
            'total': round(total_deductions, 2)
        },
        'india_compliance': {
            'pf_employee': round(float(pf_employee), 2),
            'pf_employer': round(float(pf_employer), 2),
            'esi_employee': round(float(esi_employee), 2),
            'esi_employer': round(float(esi_employer), 2),
            'professional_tax': round(float(professional_tax), 2),
            'tds_amount': round(float(tds_amount), 2),
            'hra_exemption': round(float(hra_exemption), 2),
            'lta_exemption': round(float(lta_exemption), 2)
        } if employee_profile and employee_profile.country_code == 'IN' else None,
        'us_compliance': {
            'state_tax': round(float(state_tax), 2),
            'local_tax': round(float(local_tax), 2),
            'sui_contribution': round(float(sui_contribution), 2),
            'workers_compensation': round(float(workers_compensation), 2),
            'garnishment_total': round(float(garnishment_total), 2)
        } if employee_profile and employee_profile.country_code == 'US' else None,
        'net_pay': round(net_pay, 2),
        'currency': employee_profile.salary_currency or 'USD',
        'status': payslip_record.status,
        'email_sent': payslip_record.email_sent,
        'timesheets': [ts.to_dict() for ts in timesheets],
        'generated_at': datetime.utcnow().isoformat()
    }
    
    return jsonify({'payslip': payslip}), 200


@hr_payslips_bp.route('/employee/<int:employee_id>', methods=['GET'])
def get_employee_payslips(employee_id):
    """Get payslip history for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Check if user has permission (employee can view their own, managers can view any)
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Employees can only view their own payslips
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view this payslip'}), 403
    
    # Get pay period from query params
    pay_period_start = request.args.get('pay_period_start')
    pay_period_end = request.args.get('pay_period_end')
    
    if not pay_period_start or not pay_period_end:
        return jsonify({'error': 'pay_period_start and pay_period_end are required'}), 400
    
    try:
        pay_period_start = datetime.strptime(pay_period_start, '%Y-%m-%d').date()
        pay_period_end = datetime.strptime(pay_period_end, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    # Generate payslip
    # We'll reuse the generation logic
    employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
    if not employee_profile:
        return jsonify({'error': 'Employee profile not found'}), 404
    
    timesheets = Timesheet.query.filter(
        Timesheet.user_id == employee_id,
        Timesheet.status == 'approved',
        Timesheet.date >= pay_period_start,
        Timesheet.date <= pay_period_end
    ).all()
    
    if not timesheets:
        return jsonify({'error': 'No approved timesheets found for this period'}), 404
    
    # Calculate totals (same logic as generate_payslip)
    total_regular_hours = sum(float(ts.regular_hours or 0) for ts in timesheets)
    total_overtime_hours = sum(float(ts.overtime_hours or 0) for ts in timesheets)
    total_holiday_hours = sum(float(ts.holiday_hours or 0) for ts in timesheets)
    total_hours = total_regular_hours + total_overtime_hours + total_holiday_hours
    
    total_regular_earnings = sum(float(ts.regular_earnings or 0) for ts in timesheets)
    total_overtime_earnings = sum(float(ts.overtime_earnings or 0) for ts in timesheets)
    total_holiday_earnings = sum(float(ts.holiday_earnings or 0) for ts in timesheets)
    total_bonus = sum(float(ts.bonus_amount or 0) for ts in timesheets)
    gross_earnings = total_regular_earnings + total_overtime_earnings + total_holiday_earnings + total_bonus
    
    tax_rate = 0.10
    tax_deduction = gross_earnings * tax_rate
    other_deductions = 0
    total_deductions = tax_deduction + other_deductions
    net_pay = gross_earnings - total_deductions
    
    payslip = {
        'employee': {
            'id': employee.id,
            'email': employee.email,
            'first_name': employee_profile.first_name,
            'last_name': employee_profile.last_name,
            'department': employee_profile.department,
            'location': employee_profile.location,
            'employee_id': f"EMP{employee.id:05d}"
        },
        'pay_period': {
            'start_date': pay_period_start.isoformat(),
            'end_date': pay_period_end.isoformat(),
            'pay_date': datetime.now().date().isoformat()
        },
        'hours': {
            'regular': round(total_regular_hours, 2),
            'overtime': round(total_overtime_hours, 2),
            'holiday': round(total_holiday_hours, 2),
            'total': round(total_hours, 2)
        },
        'earnings': {
            'regular': round(total_regular_earnings, 2),
            'overtime': round(total_overtime_earnings, 2),
            'holiday': round(total_holiday_earnings, 2),
            'bonus': round(total_bonus, 2),
            'gross': round(gross_earnings, 2)
        },
        'deductions': {
            'tax': round(tax_deduction, 2),
            'other': round(other_deductions, 2),
            'total': round(total_deductions, 2)
        },
        'net_pay': round(net_pay, 2),
        'currency': employee_profile.salary_currency or 'USD',
        'timesheets': [ts.to_dict() for ts in timesheets],
        'generated_at': datetime.utcnow().isoformat()
    }
    
    return jsonify({'payslip': payslip}), 200


@hr_payslips_bp.route('/summary', methods=['GET'])
def get_payroll_summary():
    """Get payroll summary/KPIs for the current period"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404

    is_employee_user = db_user.role == 'employee' or db_user.user_type == 'employee'
    
    # Get date range (default to current month)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        # Default to current month
        today = datetime.now().date()
        start_date = today.replace(day=1)
        last_day = calendar.monthrange(today.year, today.month)[1]
        end_date = today.replace(day=last_day)
    else:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    # Get tenant_id for filtering
    tenant_id = None if is_employee_user else get_current_tenant_id()

    # Determine which employee IDs are in scope
    if is_employee_user:
        employee_ids_scope = [db_user.id]
    else:
        employee_ids_scope = None
        if tenant_id:
            employee_ids_scope = [u.id for u in User.query.filter_by(tenant_id=tenant_id).all()]

    # Get all payslips for the period
    query = Payslip.query.filter(
        Payslip.pay_period_start >= start_date,
        Payslip.pay_period_end <= end_date
    )

    if employee_ids_scope is not None:
        query = query.filter(Payslip.employee_id.in_(employee_ids_scope))
    
    payslips = query.all()
    
    # Calculate totals from payslips
    total_gross_earnings = sum(float(p.gross_earnings or 0) for p in payslips)
    total_net_pay = sum(float(p.net_pay or 0) for p in payslips)
    total_tax_deduction = sum(float(p.tax_deduction or 0) for p in payslips)
    total_other_deductions = sum(float(p.other_deductions or 0) for p in payslips)
    total_deductions = sum(float(p.total_deductions or 0) for p in payslips)
    
    # If no payslips, calculate from approved timesheets
    if not payslips:
        timesheet_query = Timesheet.query.filter(
            Timesheet.status == 'approved',
            Timesheet.date >= start_date,
            Timesheet.date <= end_date
        )

        if employee_ids_scope is not None:
            timesheet_query = timesheet_query.filter(Timesheet.user_id.in_(employee_ids_scope))
        elif tenant_id:
            scoped_ids = [u.id for u in User.query.filter_by(tenant_id=tenant_id).all()]
            if scoped_ids:
                timesheet_query = timesheet_query.filter(Timesheet.user_id.in_(scoped_ids))
        
        timesheets = timesheet_query.all()
        
        if timesheets:
            total_regular_earnings = sum(float(ts.regular_earnings or 0) for ts in timesheets)
            total_overtime_earnings = sum(float(ts.overtime_earnings or 0) for ts in timesheets)
            total_holiday_earnings = sum(float(ts.holiday_earnings or 0) for ts in timesheets)
            total_bonus = sum(float(ts.bonus_amount or 0) for ts in timesheets)
            total_gross_earnings = total_regular_earnings + total_overtime_earnings + total_holiday_earnings + total_bonus
            
            # Calculate deductions (10% tax rate)
            tax_rate = 0.10
            total_tax_deduction = total_gross_earnings * tax_rate
            total_other_deductions = 0
            total_deductions = total_tax_deduction + total_other_deductions
            total_net_pay = total_gross_earnings - total_deductions
    
    # Get currency (default to USD)
    currency = 'USD'
    if payslips:
        currency = payslips[0].currency
    
    summary = {
        'gross_pay': round(total_gross_earnings, 2),
        'net_pay': round(total_net_pay, 2),
        'tax_withholding': round(total_tax_deduction, 2),
        'other_deductions': round(total_other_deductions, 2),
        'total_deductions': round(total_deductions, 2),
        'currency': currency,
        'period_start': start_date.isoformat(),
        'period_end': end_date.isoformat(),
        'payslip_count': len(payslips)
    }
    
    return jsonify({'summary': summary}), 200


@hr_payslips_bp.route('/employee/<int:employee_id>/history', methods=['GET'])
def get_payslip_history(employee_id):
    """Get payslip history for an employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Check if user has permission (employee can view their own, managers can view any)
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view this payslip history'}), 403
    
    # Get payslips
    payslips = Payslip.query.filter_by(employee_id=employee_id).order_by(Payslip.pay_period_start.desc()).all()
    
    return jsonify({
        'payslips': [p.to_dict() for p in payslips],
        'total': len(payslips)
    }), 200


@hr_payslips_bp.route('/<int:payslip_id>', methods=['GET', 'OPTIONS'])
def get_payslip_by_id(payslip_id):
    """Get payslip by ID"""
    # Handle CORS preflight - Flask-CORS should handle this automatically, but explicit handling ensures it works
    if request.method == 'OPTIONS':
        response = jsonify({})
        return response, 200
    
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get payslip
    payslip = Payslip.query.get(payslip_id)
    if not payslip:
        return jsonify({'error': 'Payslip not found'}), 404
    
    # Get employee
    employee = User.query.get(payslip.employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Check permissions: employee can view their own, admins/managers can view any
    if payslip.employee_id != db_user.id and db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view this payslip'}), 403
    
    # Get employee profile
    employee_profile = EmployeeProfile.query.filter_by(user_id=payslip.employee_id).first()
    
    # Get timesheets for this pay period
    timesheets = Timesheet.query.filter(
        Timesheet.user_id == payslip.employee_id,
        Timesheet.status == 'approved',
        Timesheet.date >= payslip.pay_period_start,
        Timesheet.date <= payslip.pay_period_end
    ).all()
    
    # Build payslip response data (matching the structure from generate_payslip)
    payslip_data = {
        'id': payslip.id,
        'employee': {
            'id': employee.id,
            'email': employee.email,
            'first_name': employee_profile.first_name if employee_profile else None,
            'last_name': employee_profile.last_name if employee_profile else None,
            'department': employee_profile.department if employee_profile else None,
            'location': employee_profile.location if employee_profile else None,
            'employee_id': f"EMP{employee.id:05d}"
        },
        'pay_period': {
            'start_date': payslip.pay_period_start.isoformat() if payslip.pay_period_start else None,
            'end_date': payslip.pay_period_end.isoformat() if payslip.pay_period_end else None,
            'pay_date': payslip.pay_date.isoformat() if payslip.pay_date else None
        },
        'hours': {
            'regular': float(payslip.total_regular_hours) if payslip.total_regular_hours else 0,
            'overtime': float(payslip.total_overtime_hours) if payslip.total_overtime_hours else 0,
            'holiday': float(payslip.total_holiday_hours) if payslip.total_holiday_hours else 0,
            'total': float(payslip.total_hours) if payslip.total_hours else 0
        },
        'earnings': {
            'regular': float(payslip.regular_earnings) if payslip.regular_earnings else 0,
            'overtime': float(payslip.overtime_earnings) if payslip.overtime_earnings else 0,
            'holiday': float(payslip.holiday_earnings) if payslip.holiday_earnings else 0,
            'bonus': float(payslip.bonus_amount) if payslip.bonus_amount else 0,
            'gross': float(payslip.gross_earnings) if payslip.gross_earnings else 0
        },
        'deductions': {
            'tax': float(payslip.tax_deduction) if payslip.tax_deduction else 0,
            'other': float(payslip.other_deductions) if payslip.other_deductions else 0,
            'total': float(payslip.total_deductions) if payslip.total_deductions else 0
        },
        'net_pay': float(payslip.net_pay) if payslip.net_pay else 0,
        'currency': payslip.currency or 'USD',
        'status': payslip.status,
        'email_sent': payslip.email_sent,
        'timesheets': [ts.to_dict() for ts in timesheets],
        'generated_at': payslip.created_at.isoformat() if payslip.created_at else datetime.utcnow().isoformat()
    }
    
    return jsonify({'payslip': payslip_data}), 200


@hr_payslips_bp.route('/<int:payslip_id>/send-email', methods=['POST'])
def send_payslip_email_endpoint(payslip_id):
    """Send payslip via email to employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, employer, recruiter, and owner can send payslips
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to send payslips'}), 403
    
    # Get payslip
    payslip = Payslip.query.get(payslip_id)
    if not payslip:
        return jsonify({'error': 'Payslip not found'}), 404
    
    # Get employee
    employee = User.query.get(payslip.employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    employee_profile = EmployeeProfile.query.filter_by(user_id=payslip.employee_id).first()
    
    try:
        # Send payslip email
        frontend_url = os.getenv('FRONTEND_URL', 'https://kempian.ai')
        payslip_url = f"{frontend_url}/dashboard/payroll/payslip/{payslip_id}"
        
        employee_name = f"{employee_profile.first_name or ''} {employee_profile.last_name or ''}".strip() if employee_profile else employee.email
        
        email_sent = send_payslip_email(
            employee.email,
            employee_name,
            payslip.pay_period_start,
            payslip.pay_period_end,
            payslip.net_pay,
            payslip.currency,
            payslip_url
        )
        
        if email_sent:
            payslip.email_sent = True
            payslip.email_sent_at = datetime.utcnow()
            db.session.commit()
        
        return jsonify({
            'success': True,
            'email_sent': email_sent,
            'message': 'Payslip email sent successfully' if email_sent else 'Failed to send email'
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to send payslip email: {str(e)}'}), 500


@hr_payslips_bp.route('/bulk-generate', methods=['POST'])
def bulk_generate_payslips():
    """Generate payslips for multiple employees"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, employer, recruiter, and owner can bulk generate
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to bulk generate payslips'}), 403
    
    data = request.get_json() or {}
    pay_period_start = data.get('pay_period_start')
    pay_period_end = data.get('pay_period_end')
    employee_ids = data.get('employee_ids', [])  # Optional: specific employees
    organization_id = data.get('organization_id')  # Optional: filter by organization
    
    if not pay_period_start or not pay_period_end:
        return jsonify({'error': 'pay_period_start and pay_period_end are required'}), 400
    
    try:
        pay_period_start = datetime.strptime(pay_period_start, '%Y-%m-%d').date()
        pay_period_end = datetime.strptime(pay_period_end, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    # Get employees to process
    if employee_ids:
        employees = User.query.filter(
            User.id.in_(employee_ids),
            (User.user_type == 'employee') | (User.role == 'employee')
        ).all()
    elif organization_id:
        employees = User.query.filter(
            User.tenant_id == organization_id,
            (User.user_type == 'employee') | (User.role == 'employee')
        ).all()
    else:
        # Get all employees in current tenant
        tenant_id = get_current_tenant_id()
        employees = User.query.filter(
            User.tenant_id == tenant_id,
            (User.user_type == 'employee') | (User.role == 'employee')
        ).all()
    
    generated = []
    failed = []
    
    for employee in employees:
        try:
            # Reuse the generate logic
            employee_profile = EmployeeProfile.query.filter_by(user_id=employee.id).first()
            if not employee_profile:
                failed.append({'employee_id': employee.id, 'error': 'Employee profile not found'})
                continue
            
            # Get approved timesheets
            timesheets = Timesheet.query.filter(
                Timesheet.user_id == employee.id,
                Timesheet.status == 'approved',
                Timesheet.date >= pay_period_start,
                Timesheet.date <= pay_period_end
            ).all()
            
            if not timesheets:
                failed.append({'employee_id': employee.id, 'error': 'No approved timesheets found'})
                continue
            
            # Calculate totals (same as generate_payslip)
            total_regular_hours = sum(float(ts.regular_hours or 0) for ts in timesheets)
            total_overtime_hours = sum(float(ts.overtime_hours or 0) for ts in timesheets)
            total_holiday_hours = sum(float(ts.holiday_hours or 0) for ts in timesheets)
            total_hours = total_regular_hours + total_overtime_hours + total_holiday_hours
            
            total_regular_earnings = sum(float(ts.regular_earnings or 0) for ts in timesheets)
            total_overtime_earnings = sum(float(ts.overtime_earnings or 0) for ts in timesheets)
            total_holiday_earnings = sum(float(ts.holiday_earnings or 0) for ts in timesheets)
            total_bonus = sum(float(ts.bonus_amount or 0) for ts in timesheets)
            gross_earnings = total_regular_earnings + total_overtime_earnings + total_holiday_earnings + total_bonus
            
            tax_rate = 0.10
            tax_deduction = gross_earnings * tax_rate
            other_deductions = 0
            total_deductions = tax_deduction + other_deductions
            net_pay = gross_earnings - total_deductions
            
            # Create or update payslip
            existing_payslip = Payslip.query.filter(
                Payslip.employee_id == employee.id,
                Payslip.pay_period_start == pay_period_start,
                Payslip.pay_period_end == pay_period_end
            ).first()
            
            if existing_payslip:
                payslip_record = existing_payslip
            else:
                payslip_record = Payslip(
                    employee_id=employee.id,
                    employee_profile_id=employee_profile.id,
                    pay_period_start=pay_period_start,
                    pay_period_end=pay_period_end,
                    pay_date=datetime.now().date()
                )
                db.session.add(payslip_record)
            
            payslip_record.total_regular_hours = Decimal(str(round(total_regular_hours, 2)))
            payslip_record.total_overtime_hours = Decimal(str(round(total_overtime_hours, 2)))
            payslip_record.total_holiday_hours = Decimal(str(round(total_holiday_hours, 2)))
            payslip_record.total_hours = Decimal(str(round(total_hours, 2)))
            payslip_record.regular_earnings = Decimal(str(round(total_regular_earnings, 2)))
            payslip_record.overtime_earnings = Decimal(str(round(total_overtime_earnings, 2)))
            payslip_record.holiday_earnings = Decimal(str(round(total_holiday_earnings, 2)))
            payslip_record.bonus_amount = Decimal(str(round(total_bonus, 2)))
            payslip_record.gross_earnings = Decimal(str(round(gross_earnings, 2)))
            payslip_record.tax_deduction = Decimal(str(round(tax_deduction, 2)))
            payslip_record.other_deductions = Decimal(str(round(other_deductions, 2)))
            payslip_record.total_deductions = Decimal(str(round(total_deductions, 2)))
            payslip_record.net_pay = Decimal(str(round(net_pay, 2)))
            payslip_record.currency = employee_profile.salary_currency or 'USD'
            payslip_record.status = 'generated'
            
            generated.append({
                'employee_id': employee.id,
                'employee_email': employee.email,
                'payslip_id': payslip_record.id,
                'net_pay': float(net_pay)
            })
            
        except Exception as e:
            failed.append({'employee_id': employee.id, 'error': str(e)})
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'generated': generated,
        'failed': failed,
        'total_generated': len(generated),
        'total_failed': len(failed)
    }), 200
