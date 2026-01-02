"""
Employee State Insurance (ESI) Calculation Service
Handles ESI contribution calculations for India payroll
"""

from app import db
from app.models import (
    ESIContribution, EmployeeProfile, Payslip, PayRun, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# ESI Contribution Rates
ESI_EMPLOYEE_RATE = Decimal('0.0075')  # 0.75% of gross salary
ESI_EMPLOYER_RATE = Decimal('0.0325')  # 3.25% of gross salary
ESI_WAGE_CEILING = Decimal('21000')  # ₹21,000 per month wage ceiling


def calculate_esi_contribution(gross_salary, employee_id=None, employee_profile_id=None):
    """
    Calculate ESI contributions for employee and employer
    
    Args:
        gross_salary: Gross salary amount (Decimal or float)
        employee_id: Employee user ID (optional)
        employee_profile_id: Employee profile ID (optional)
    
    Returns:
        dict: {
            'gross_salary': Decimal,
            'employee_esi': Decimal,
            'employer_esi': Decimal,
            'total_esi': Decimal,
            'is_above_ceiling': bool
        }
    """
    try:
        gross_salary = Decimal(str(gross_salary))
        
        # Check if above wage ceiling
        is_above_ceiling = gross_salary > ESI_WAGE_CEILING
        
        # ESI is only applicable if gross salary is ₹21,000 or less
        if is_above_ceiling:
            return {
                'gross_salary': float(gross_salary),
                'employee_esi': 0.0,
                'employer_esi': 0.0,
                'total_esi': 0.0,
                'is_above_ceiling': True,
                'wage_ceiling': float(ESI_WAGE_CEILING)
            }
        
        # Calculate contributions
        employee_esi = (gross_salary * ESI_EMPLOYEE_RATE).quantize(Decimal('0.01'))
        employer_esi = (gross_salary * ESI_EMPLOYER_RATE).quantize(Decimal('0.01'))
        total_esi = employee_esi + employer_esi
        
        return {
            'gross_salary': float(gross_salary),
            'employee_esi': float(employee_esi),
            'employer_esi': float(employer_esi),
            'total_esi': float(total_esi),
            'is_above_ceiling': False,
            'wage_ceiling': float(ESI_WAGE_CEILING)
        }
    except Exception as e:
        logger.error(f"Error calculating ESI contribution: {str(e)}")
        raise


def create_esi_contribution_record(
    employee_id,
    gross_salary,
    payslip_id=None,
    pay_run_id=None,
    contribution_month=None,
    contribution_year=None,
    esi_card_number=None,
    esi_dispensary_code=None
):
    """
    Create an ESI contribution record in the database
    
    Args:
        employee_id: Employee user ID
        gross_salary: Gross salary amount
        payslip_id: Payslip ID (optional)
        pay_run_id: Pay run ID (optional)
        contribution_month: Month (1-12), defaults to current month
        contribution_year: Year, defaults to current year
        esi_card_number: ESI card number (optional)
        esi_dispensary_code: ESI dispensary code (optional)
    
    Returns:
        ESIContribution: Created record
    """
    try:
        # Get employee profile
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        employee_profile_id = employee_profile.id if employee_profile else None
        
        # Get ESI card number from profile if not provided
        if not esi_card_number and employee_profile:
            esi_card_number = employee_profile.esi_card_number
        
        # Calculate ESI
        esi_calc = calculate_esi_contribution(gross_salary, employee_id, employee_profile_id)
        
        # Set default month/year
        if not contribution_month:
            contribution_month = datetime.now().month
        if not contribution_year:
            contribution_year = datetime.now().year
        
        # Create record
        esi_record = ESIContribution(
            employee_id=employee_id,
            employee_profile_id=employee_profile_id,
            payslip_id=payslip_id,
            pay_run_id=pay_run_id,
            gross_salary=Decimal(str(esi_calc['gross_salary'])),
            employee_esi=Decimal(str(esi_calc['employee_esi'])),
            employer_esi=Decimal(str(esi_calc['employer_esi'])),
            total_esi=Decimal(str(esi_calc['total_esi'])),
            esi_card_number=esi_card_number,
            esi_dispensary_code=esi_dispensary_code,
            is_above_ceiling=esi_calc['is_above_ceiling'],
            wage_ceiling=ESI_WAGE_CEILING,
            contribution_month=contribution_month,
            contribution_year=contribution_year
        )
        
        db.session.add(esi_record)
        db.session.commit()
        
        logger.info(f"Created ESI contribution record for employee {employee_id}")
        return esi_record
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating ESI contribution record: {str(e)}")
        raise


def get_esi_contributions_by_employee(employee_id, year=None, month=None):
    """
    Get ESI contributions for an employee
    
    Args:
        employee_id: Employee user ID
        year: Year filter (optional)
        month: Month filter (optional)
    
    Returns:
        list: List of ESIContribution records
    """
    try:
        query = ESIContribution.query.filter_by(employee_id=employee_id)
        
        if year:
            query = query.filter_by(contribution_year=year)
        if month:
            query = query.filter_by(contribution_month=month)
        
        return query.order_by(
            ESIContribution.contribution_year.desc(),
            ESIContribution.contribution_month.desc()
        ).all()
    except Exception as e:
        logger.error(f"Error fetching ESI contributions: {str(e)}")
        raise


def get_esi_contributions_by_payrun(pay_run_id):
    """
    Get all ESI contributions for a pay run
    
    Args:
        pay_run_id: Pay run ID
    
    Returns:
        list: List of ESIContribution records
    """
    try:
        return ESIContribution.query.filter_by(
            pay_run_id=pay_run_id
        ).all()
    except Exception as e:
        logger.error(f"Error fetching ESI contributions for pay run: {str(e)}")
        raise


def update_esi_challan_info(esi_contribution_id, challan_number, challan_date):
    """
    Update challan information for ESI contribution
    
    Args:
        esi_contribution_id: ESI contribution record ID
        challan_number: Challan number
        challan_date: Challan date
    """
    try:
        esi_record = ESIContribution.query.get(esi_contribution_id)
        if not esi_record:
            raise ValueError(f"ESI contribution record {esi_contribution_id} not found")
        
        esi_record.challan_number = challan_number
        esi_record.challan_date = challan_date
        
        db.session.commit()
        logger.info(f"Updated challan info for ESI contribution {esi_contribution_id}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating ESI challan info: {str(e)}")
        raise


def validate_esi_card_number(esi_card_number):
    """
    Validate ESI card number format
    
    Args:
        esi_card_number: ESI card number to validate
    
    Returns:
        bool: True if valid format
    """
    if not esi_card_number:
        return False
    
    # ESI card number format: Usually 17 digits
    # Format: XXXXXXXXXXXXXXX (17 digits)
    esi_card_number = str(esi_card_number).strip().replace('-', '').replace(' ', '')
    
    if len(esi_card_number) != 17:
        return False
    
    if not esi_card_number.isdigit():
        return False
    
    return True

