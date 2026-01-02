"""
Professional Tax Calculation Service
Handles state-wise Professional Tax calculations for India payroll
"""

from app import db
from app.models import (
    ProfessionalTaxDeduction, EmployeeProfile, Payslip, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Professional Tax Slabs by State (Monthly)
# Format: {state_code: [(min_salary, max_salary, tax_amount), ...]}
PROFESSIONAL_TAX_SLABS = {
    'MH': [  # Maharashtra
        (0, 5000, 0),
        (5001, 10000, 150),
        (10001, 15000, 175),
        (15001, float('inf'), 200)
    ],
    'KA': [  # Karnataka
        (0, 10000, 0),
        (10001, 15000, 150),
        (15001, float('inf'), 200)
    ],
    'WB': [  # West Bengal
        (0, 10000, 110),
        (10001, 15000, 130),
        (15001, 20000, 150),
        (20001, float('inf'), 200)
    ],
    'TN': [  # Tamil Nadu
        (0, 21000, 0),
        (21001, 30000, 100),
        (30001, 45000, 235),
        (45001, 60000, 510),
        (60001, 75000, 760),
        (75001, float('inf'), 1095)
    ],
    'GJ': [  # Gujarat
        (0, 5000, 0),
        (5001, 10000, 100),
        (10001, float('inf'), 200)
    ],
    'AP': [  # Andhra Pradesh
        (0, 15000, 0),
        (15001, 20000, 150),
        (20001, float('inf'), 200)
    ],
    'TS': [  # Telangana
        (0, 15000, 0),
        (15001, 20000, 150),
        (20001, float('inf'), 200)
    ],
    'MP': [  # Madhya Pradesh
        (0, 10000, 0),
        (10001, float('inf'), 200)
    ],
    'RJ': [  # Rajasthan
        (0, 10000, 0),
        (10001, 20000, 150),
        (20001, float('inf'), 200)
    ],
    'UP': [  # Uttar Pradesh
        (0, 10000, 0),
        (10001, float('inf'), 200)
    ],
    'DL': [  # Delhi
        (0, 10000, 0),
        (10001, 20000, 150),
        (20001, float('inf'), 200)
    ],
    'HR': [  # Haryana
        (0, 10000, 0),
        (10001, 20000, 150),
        (20001, float('inf'), 200)
    ],
    'PB': [  # Punjab
        (0, 10000, 0),
        (10001, float('inf'), 200)
    ],
    'OR': [  # Odisha
        (0, 10000, 0),
        (10001, 20000, 150),
        (20001, float('inf'), 200)
    ],
    'BR': [  # Bihar
        (0, 10000, 0),
        (10001, 15000, 110),
        (15001, float('inf'), 200)
    ],
    'JH': [  # Jharkhand
        (0, 10000, 0),
        (10001, 15000, 110),
        (15001, float('inf'), 200)
    ],
    'CT': [  # Chhattisgarh
        (0, 10000, 0),
        (10001, float('inf'), 200)
    ],
    'AS': [  # Assam
        (0, 10000, 0),
        (10001, 15000, 150),
        (15001, float('inf'), 200)
    ],
    'KL': [  # Kerala
        (0, 10000, 0),
        (10001, 20000, 150),
        (20001, float('inf'), 200)
    ]
}

# Professional Tax Exemption Limits by State
PROFESSIONAL_TAX_EXEMPTION_LIMITS = {
    'MH': 5000,
    'KA': 10000,
    'WB': 10000,
    'TN': 21000,
    'GJ': 5000,
    'AP': 15000,
    'TS': 15000,
    'MP': 10000,
    'RJ': 10000,
    'UP': 10000,
    'DL': 10000,
    'HR': 10000,
    'PB': 10000,
    'OR': 10000,
    'BR': 10000,
    'JH': 10000,
    'CT': 10000,
    'AS': 10000,
    'KL': 10000
}

# State Names
STATE_NAMES = {
    'MH': 'Maharashtra',
    'KA': 'Karnataka',
    'WB': 'West Bengal',
    'TN': 'Tamil Nadu',
    'GJ': 'Gujarat',
    'AP': 'Andhra Pradesh',
    'TS': 'Telangana',
    'MP': 'Madhya Pradesh',
    'RJ': 'Rajasthan',
    'UP': 'Uttar Pradesh',
    'DL': 'Delhi',
    'HR': 'Haryana',
    'PB': 'Punjab',
    'OR': 'Odisha',
    'BR': 'Bihar',
    'JH': 'Jharkhand',
    'CT': 'Chhattisgarh',
    'AS': 'Assam',
    'KL': 'Kerala'
}


def calculate_professional_tax(gross_salary, state_code):
    """
    Calculate Professional Tax based on state and gross salary
    
    Args:
        gross_salary: Gross salary amount (Decimal or float)
        state_code: State code (e.g., 'MH', 'KA', 'WB')
    
    Returns:
        dict: {
            'gross_salary': Decimal,
            'state_code': str,
            'state_name': str,
            'professional_tax_amount': Decimal,
            'pt_slab_min': Decimal,
            'pt_slab_max': Decimal,
            'is_exempt': bool,
            'exemption_limit': Decimal
        }
    """
    try:
        gross_salary = Decimal(str(gross_salary))
        state_code = state_code.upper() if state_code else None
        
        if not state_code or state_code not in PROFESSIONAL_TAX_SLABS:
            logger.warning(f"State code {state_code} not found in PT slabs, returning 0")
            return {
                'gross_salary': float(gross_salary),
                'state_code': state_code,
                'state_name': None,
                'professional_tax_amount': 0.0,
                'pt_slab_min': None,
                'pt_slab_max': None,
                'is_exempt': True,
                'exemption_limit': None
            }
        
        # Get exemption limit
        exemption_limit = Decimal(str(PROFESSIONAL_TAX_EXEMPTION_LIMITS.get(state_code, 0)))
        is_exempt = gross_salary <= exemption_limit
        
        if is_exempt:
            return {
                'gross_salary': float(gross_salary),
                'state_code': state_code,
                'state_name': STATE_NAMES.get(state_code),
                'professional_tax_amount': 0.0,
                'pt_slab_min': None,
                'pt_slab_max': float(exemption_limit),
                'is_exempt': True,
                'exemption_limit': float(exemption_limit)
            }
        
        # Find applicable slab
        slabs = PROFESSIONAL_TAX_SLABS[state_code]
        professional_tax_amount = Decimal('0')
        pt_slab_min = None
        pt_slab_max = None
        
        for min_sal, max_sal, tax_amt in slabs:
            if min_sal == float('inf'):
                continue
            if gross_salary >= Decimal(str(min_sal)) and (max_sal == float('inf') or gross_salary <= Decimal(str(max_sal))):
                professional_tax_amount = Decimal(str(tax_amt))
                pt_slab_min = Decimal(str(min_sal))
                pt_slab_max = Decimal(str(max_sal)) if max_sal != float('inf') else None
                break
        
        return {
            'gross_salary': float(gross_salary),
            'state_code': state_code,
            'state_name': STATE_NAMES.get(state_code),
            'professional_tax_amount': float(professional_tax_amount),
            'pt_slab_min': float(pt_slab_min) if pt_slab_min else None,
            'pt_slab_max': float(pt_slab_max) if pt_slab_max else None,
            'is_exempt': False,
            'exemption_limit': float(exemption_limit)
        }
    except Exception as e:
        logger.error(f"Error calculating Professional Tax: {str(e)}")
        raise


def create_professional_tax_record(
    employee_id,
    gross_salary,
    state_code,
    payslip_id=None,
    deduction_month=None,
    deduction_year=None,
    pt_certificate_number=None
):
    """
    Create a Professional Tax deduction record in the database
    
    Args:
        employee_id: Employee user ID
        gross_salary: Gross salary amount
        state_code: State code
        payslip_id: Payslip ID (optional)
        deduction_month: Month (1-12), defaults to current month
        deduction_year: Year, defaults to current year
        pt_certificate_number: PT certificate number (optional)
    
    Returns:
        ProfessionalTaxDeduction: Created record
    """
    try:
        # Get employee profile
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        employee_profile_id = employee_profile.id if employee_profile else None
        
        # Get state code from profile if not provided
        if not state_code and employee_profile:
            state_code = employee_profile.state_code
        
        if not state_code:
            raise ValueError("State code is required for Professional Tax calculation")
        
        # Calculate Professional Tax
        pt_calc = calculate_professional_tax(gross_salary, state_code)
        
        # Set default month/year
        if not deduction_month:
            deduction_month = datetime.now().month
        if not deduction_year:
            deduction_year = datetime.now().year
        
        # Create record
        pt_record = ProfessionalTaxDeduction(
            employee_id=employee_id,
            employee_profile_id=employee_profile_id,
            payslip_id=payslip_id,
            state_code=pt_calc['state_code'],
            state_name=pt_calc['state_name'],
            gross_salary=Decimal(str(pt_calc['gross_salary'])),
            professional_tax_amount=Decimal(str(pt_calc['professional_tax_amount'])),
            pt_slab_min=Decimal(str(pt_calc['pt_slab_min'])) if pt_calc['pt_slab_min'] else None,
            pt_slab_max=Decimal(str(pt_calc['pt_slab_max'])) if pt_calc['pt_slab_max'] else None,
            is_exempt=pt_calc['is_exempt'],
            exemption_limit=Decimal(str(pt_calc['exemption_limit'])) if pt_calc['exemption_limit'] else None,
            pt_certificate_number=pt_certificate_number,
            deduction_month=deduction_month,
            deduction_year=deduction_year
        )
        
        db.session.add(pt_record)
        db.session.commit()
        
        logger.info(f"Created Professional Tax record for employee {employee_id}")
        return pt_record
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating Professional Tax record: {str(e)}")
        raise


def get_professional_tax_by_employee(employee_id, year=None, month=None):
    """
    Get Professional Tax deductions for an employee
    
    Args:
        employee_id: Employee user ID
        year: Year filter (optional)
        month: Month filter (optional)
    
    Returns:
        list: List of ProfessionalTaxDeduction records
    """
    try:
        query = ProfessionalTaxDeduction.query.filter_by(employee_id=employee_id)
        
        if year:
            query = query.filter_by(deduction_year=year)
        if month:
            query = query.filter_by(deduction_month=month)
        
        return query.order_by(
            ProfessionalTaxDeduction.deduction_year.desc(),
            ProfessionalTaxDeduction.deduction_month.desc()
        ).all()
    except Exception as e:
        logger.error(f"Error fetching Professional Tax deductions: {str(e)}")
        raise


def get_professional_tax_slabs(state_code):
    """
    Get Professional Tax slabs for a state
    
    Args:
        state_code: State code
    
    Returns:
        list: List of tax slabs
    """
    state_code = state_code.upper() if state_code else None
    if not state_code or state_code not in PROFESSIONAL_TAX_SLABS:
        return []
    
    return PROFESSIONAL_TAX_SLABS[state_code]


def get_professional_tax_exemption_limit(state_code):
    """
    Get Professional Tax exemption limit for a state
    
    Args:
        state_code: State code
    
    Returns:
        Decimal: Exemption limit
    """
    state_code = state_code.upper() if state_code else None
    if not state_code or state_code not in PROFESSIONAL_TAX_EXEMPTION_LIMITS:
        return Decimal('0')
    
    return Decimal(str(PROFESSIONAL_TAX_EXEMPTION_LIMITS[state_code]))

