"""
Provident Fund (PF) Calculation Service
Handles EPF contribution calculations for India payroll
"""

from app import db
from app.models import (
    ProvidentFundContribution, EmployeeProfile, Payslip, PayRun, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# PF Contribution Rates
PF_EMPLOYEE_RATE = Decimal('0.12')  # 12% of basic salary
PF_EMPLOYER_RATE = Decimal('0.12')  # 12% of basic salary
PF_MAX_SALARY = Decimal('15000')  # Maximum salary for PF (₹15,000 per month)


def calculate_pf_contribution(basic_salary, employee_id=None, employee_profile_id=None):
    """
    Calculate PF contributions for employee and employer
    
    Args:
        basic_salary: Basic salary amount (Decimal or float)
        employee_id: Employee user ID (optional)
        employee_profile_id: Employee profile ID (optional)
    
    Returns:
        dict: {
            'basic_salary': Decimal,
            'employee_pf': Decimal,
            'employer_pf': Decimal,
            'total_pf': Decimal,
            'is_above_max': bool
        }
    """
    try:
        basic_salary = Decimal(str(basic_salary))
        
        # PF is calculated on minimum of basic salary and ₹15,000
        pf_base_salary = min(basic_salary, PF_MAX_SALARY)
        
        # Calculate contributions
        employee_pf = (pf_base_salary * PF_EMPLOYEE_RATE).quantize(Decimal('0.01'))
        employer_pf = (pf_base_salary * PF_EMPLOYER_RATE).quantize(Decimal('0.01'))
        total_pf = employee_pf + employer_pf
        
        is_above_max = basic_salary > PF_MAX_SALARY
        
        return {
            'basic_salary': float(basic_salary),
            'pf_base_salary': float(pf_base_salary),
            'employee_pf': float(employee_pf),
            'employer_pf': float(employer_pf),
            'total_pf': float(total_pf),
            'is_above_max': is_above_max
        }
    except Exception as e:
        logger.error(f"Error calculating PF contribution: {str(e)}")
        raise


def create_pf_contribution_record(
    employee_id,
    basic_salary,
    payslip_id=None,
    pay_run_id=None,
    contribution_month=None,
    contribution_year=None,
    epf_account_number=None,
    uan_number=None
):
    """
    Create a PF contribution record in the database
    
    Args:
        employee_id: Employee user ID
        basic_salary: Basic salary amount
        payslip_id: Payslip ID (optional)
        pay_run_id: Pay run ID (optional)
        contribution_month: Month (1-12), defaults to current month
        contribution_year: Year, defaults to current year
        epf_account_number: EPF account number (optional)
        uan_number: UAN number (optional)
    
    Returns:
        ProvidentFundContribution: Created record
    """
    try:
        # Get employee profile
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        employee_profile_id = employee_profile.id if employee_profile else None
        
        # Get EPF/UAN from profile if not provided
        if not epf_account_number and employee_profile:
            epf_account_number = employee_profile.pf_account_number
        if not uan_number and employee_profile:
            uan_number = employee_profile.uan_number
        
        # Calculate PF
        pf_calc = calculate_pf_contribution(basic_salary, employee_id, employee_profile_id)
        
        # Set default month/year
        if not contribution_month:
            contribution_month = datetime.now().month
        if not contribution_year:
            contribution_year = datetime.now().year
        
        # Create record
        pf_record = ProvidentFundContribution(
            employee_id=employee_id,
            employee_profile_id=employee_profile_id,
            payslip_id=payslip_id,
            pay_run_id=pay_run_id,
            basic_salary=Decimal(str(pf_calc['pf_base_salary'])),
            employee_pf=Decimal(str(pf_calc['employee_pf'])),
            employer_pf=Decimal(str(pf_calc['employer_pf'])),
            total_pf=Decimal(str(pf_calc['total_pf'])),
            epf_account_number=epf_account_number,
            uan_number=uan_number,
            contribution_month=contribution_month,
            contribution_year=contribution_year
        )
        
        db.session.add(pf_record)
        db.session.commit()
        
        logger.info(f"Created PF contribution record for employee {employee_id}")
        return pf_record
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating PF contribution record: {str(e)}")
        raise


def get_pf_contributions_by_employee(employee_id, year=None, month=None):
    """
    Get PF contributions for an employee
    
    Args:
        employee_id: Employee user ID
        year: Year filter (optional)
        month: Month filter (optional)
    
    Returns:
        list: List of ProvidentFundContribution records
    """
    try:
        query = ProvidentFundContribution.query.filter_by(employee_id=employee_id)
        
        if year:
            query = query.filter_by(contribution_year=year)
        if month:
            query = query.filter_by(contribution_month=month)
        
        return query.order_by(
            ProvidentFundContribution.contribution_year.desc(),
            ProvidentFundContribution.contribution_month.desc()
        ).all()
    except Exception as e:
        logger.error(f"Error fetching PF contributions: {str(e)}")
        raise


def get_pf_contributions_by_payrun(pay_run_id):
    """
    Get all PF contributions for a pay run
    
    Args:
        pay_run_id: Pay run ID
    
    Returns:
        list: List of ProvidentFundContribution records
    """
    try:
        return ProvidentFundContribution.query.filter_by(
            pay_run_id=pay_run_id
        ).all()
    except Exception as e:
        logger.error(f"Error fetching PF contributions for pay run: {str(e)}")
        raise


def update_pf_challan_info(pf_contribution_id, challan_number, challan_date):
    """
    Update challan information for PF contribution
    
    Args:
        pf_contribution_id: PF contribution record ID
        challan_number: Challan number
        challan_date: Challan date
    """
    try:
        pf_record = ProvidentFundContribution.query.get(pf_contribution_id)
        if not pf_record:
            raise ValueError(f"PF contribution record {pf_contribution_id} not found")
        
        pf_record.challan_number = challan_number
        pf_record.challan_date = challan_date
        
        db.session.commit()
        logger.info(f"Updated challan info for PF contribution {pf_contribution_id}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating PF challan info: {str(e)}")
        raise

