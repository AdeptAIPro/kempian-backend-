"""
Workers' Compensation Calculation Service
Handles Workers' Compensation calculations for US payroll
"""

from app import db
from app.models import (
    WorkersCompensation, EmployeeProfile, Payslip, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Workers' Compensation Rates by State and Class Code (examples)
# Rates are per $100 of payroll
WC_RATES = {
    'CA': {  # California
        '8810': 0.50,  # Clerical
        '8742': 2.50,  # Salespersons
        '8816': 1.20,  # Computer Programmers
    },
    'NY': {  # New York
        '8810': 0.15,  # Clerical
        '8742': 1.50,  # Salespersons
        '8816': 0.80,  # Computer Programmers
    },
    'TX': {  # Texas
        '8810': 0.25,  # Clerical
        '8742': 1.20,  # Salespersons
        '8816': 0.60,  # Computer Programmers
    },
    'FL': {  # Florida
        '8810': 0.20,  # Clerical
        '8742': 1.00,  # Salespersons
        '8816': 0.50,  # Computer Programmers
    },
    'IL': {  # Illinois
        '8810': 0.18,  # Clerical
        '8742': 1.30,  # Salespersons
        '8816': 0.70,  # Computer Programmers
    },
    'PA': {  # Pennsylvania
        '8810': 0.22,  # Clerical
        '8742': 1.40,  # Salespersons
        '8816': 0.65,  # Computer Programmers
    },
    'OH': {  # Ohio
        '8810': 0.16,  # Clerical
        '8742': 1.10,  # Salespersons
        '8816': 0.55,  # Computer Programmers
    },
    'GA': {  # Georgia
        '8810': 0.19,  # Clerical
        '8742': 1.25,  # Salespersons
        '8816': 0.62,  # Computer Programmers
    },
    'NC': {  # North Carolina
        '8810': 0.17,  # Clerical
        '8742': 1.15,  # Salespersons
        '8816': 0.58,  # Computer Programmers
    },
    'MI': {  # Michigan
        '8810': 0.21,  # Clerical
        '8742': 1.35,  # Salespersons
        '8816': 0.68,  # Computer Programmers
    }
}

# Default WC rate if state/class code not found
DEFAULT_WC_RATE = Decimal('0.50')


def calculate_workers_compensation(gross_wages, state_code, wc_class_code=None):
    """
    Calculate Workers' Compensation
    
    Args:
        gross_wages: Gross wages
        state_code: State code
        wc_class_code: WC class code (optional)
    
    Returns:
        dict: WC calculation details
    """
    try:
        gross_wages = Decimal(str(gross_wages))
        state_code = state_code.upper() if state_code else None
        
        # Get WC rate
        if state_code and state_code in WC_RATES:
            if wc_class_code and wc_class_code in WC_RATES[state_code]:
                wc_rate = Decimal(str(WC_RATES[state_code][wc_class_code]))
            else:
                # Use default class code for state (clerical)
                default_class = '8810'
                wc_rate = Decimal(str(WC_RATES[state_code].get(default_class, DEFAULT_WC_RATE)))
        else:
            wc_rate = DEFAULT_WC_RATE
        
        # WC is calculated as rate per $100 of payroll
        wc_contribution = (gross_wages / Decimal('100')) * wc_rate
        
        return {
            'gross_wages': float(gross_wages),
            'state_code': state_code,
            'wc_class_code': wc_class_code or '8810',
            'wc_rate': float(wc_rate),
            'wc_contribution': float(wc_contribution)
        }
    except Exception as e:
        logger.error(f"Error calculating Workers' Compensation: {str(e)}")
        raise


def create_workers_compensation_record(
    employee_id,
    gross_wages,
    state_code,
    payslip_id=None,
    wc_month=None,
    wc_year=None,
    wc_class_code=None
):
    """
    Create Workers' Compensation record
    
    Args:
        employee_id: Employee user ID
        gross_wages: Gross wages
        state_code: State code
        payslip_id: Payslip ID (optional)
        wc_month: Month (1-12), defaults to current month
        wc_year: Year, defaults to current year
        wc_class_code: WC class code (optional)
    
    Returns:
        WorkersCompensation: Created record
    """
    try:
        # Get employee profile
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        employee_profile_id = employee_profile.id if employee_profile else None
        
        # Get state code from profile if not provided
        if not state_code and employee_profile:
            state_code = employee_profile.state_code
        
        if not state_code:
            raise ValueError("State code is required for Workers' Compensation calculation")
        
        # Calculate WC
        wc_calc = calculate_workers_compensation(gross_wages, state_code, wc_class_code)
        
        # Set default month/year
        if not wc_month:
            wc_month = datetime.now().month
        if not wc_year:
            wc_year = datetime.now().year
        
        # Create record
        wc_record = WorkersCompensation(
            employee_id=employee_id,
            employee_profile_id=employee_profile_id,
            payslip_id=payslip_id,
            state_code=state_code,
            wc_class_code=wc_calc['wc_class_code'],
            wc_rate=Decimal(str(wc_calc['wc_rate'])),
            gross_wages=Decimal(str(wc_calc['gross_wages'])),
            wc_contribution=Decimal(str(wc_calc['wc_contribution'])),
            wc_month=wc_month,
            wc_year=wc_year
        )
        
        db.session.add(wc_record)
        db.session.commit()
        
        logger.info(f"Created Workers' Compensation record for employee {employee_id}")
        return wc_record
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating Workers' Compensation record: {str(e)}")
        raise


def get_workers_compensation_by_employee(employee_id, year=None, month=None):
    """
    Get Workers' Compensation records for an employee
    
    Args:
        employee_id: Employee user ID
        year: Year filter (optional)
        month: Month filter (optional)
    
    Returns:
        list: List of WorkersCompensation records
    """
    try:
        query = WorkersCompensation.query.filter_by(employee_id=employee_id)
        
        if year:
            query = query.filter_by(wc_year=year)
        if month:
            query = query.filter_by(wc_month=month)
        
        return query.order_by(
            WorkersCompensation.wc_year.desc(),
            WorkersCompensation.wc_month.desc()
        ).all()
    except Exception as e:
        logger.error(f"Error fetching Workers' Compensation records: {str(e)}")
        raise


def get_wc_rates_by_state(state_code):
    """
    Get WC rates for a state
    
    Args:
        state_code: State code
    
    Returns:
        dict: WC rates by class code
    """
    state_code = state_code.upper() if state_code else None
    if not state_code or state_code not in WC_RATES:
        return {}
    
    return WC_RATES[state_code]

