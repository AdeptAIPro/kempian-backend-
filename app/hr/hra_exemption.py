"""
HRA (House Rent Allowance) Exemption Calculation Service
Handles HRA exemption calculations for India payroll
"""

from app import db
from app.models import IncomeTaxExemption, EmployeeProfile, User
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Metro cities for HRA calculation (50% of basic)
METRO_CITIES = ['Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 'Pune']


def calculate_hra_exemption(hra_received, basic_salary, rent_paid, is_metro_city=False):
    """
    Calculate HRA exemption
    
    HRA exemption is minimum of:
    1. Actual HRA received
    2. Rent paid - 10% of basic salary
    3. 50% of basic (metro) or 40% of basic (non-metro)
    
    Args:
        hra_received: HRA amount received
        basic_salary: Basic salary
        rent_paid: Rent paid per month
        is_metro_city: Whether employee is in metro city
    
    Returns:
        dict: HRA exemption details
    """
    try:
        hra_received = Decimal(str(hra_received))
        basic_salary = Decimal(str(basic_salary))
        rent_paid = Decimal(str(rent_paid))
        
        # Calculate three components
        # 1. Actual HRA received
        component1 = hra_received
        
        # 2. Rent paid - 10% of basic
        component2 = max(Decimal('0'), rent_paid - (basic_salary * Decimal('0.10')))
        
        # 3. 50% of basic (metro) or 40% of basic (non-metro)
        if is_metro_city:
            component3 = basic_salary * Decimal('0.50')
        else:
            component3 = basic_salary * Decimal('0.40')
        
        # HRA exemption is minimum of three
        hra_exemption = min(component1, component2, component3)
        
        return {
            'hra_received': float(hra_received),
            'basic_salary': float(basic_salary),
            'rent_paid': float(rent_paid),
            'is_metro_city': is_metro_city,
            'component1_actual_hra': float(component1),
            'component2_rent_minus_10pct': float(component2),
            'component3_percentage_of_basic': float(component3),
            'hra_exemption_amount': float(hra_exemption)
        }
    except Exception as e:
        logger.error(f"Error calculating HRA exemption: {str(e)}")
        raise


def update_hra_exemption(
    employee_id,
    hra_received,
    basic_salary,
    rent_paid,
    is_metro_city=False,
    rent_receipts_uploaded=False,
    financial_year=None
):
    """
    Update HRA exemption for an employee
    
    Args:
        employee_id: Employee user ID
        hra_received: HRA amount received
        basic_salary: Basic salary
        rent_paid: Rent paid per month
        is_metro_city: Whether employee is in metro city
        rent_receipts_uploaded: Whether rent receipts are uploaded
        financial_year: Financial year (optional)
    
    Returns:
        IncomeTaxExemption: Updated record
    """
    try:
        # Get or create exemption record
        exemption = IncomeTaxExemption.query.filter_by(employee_id=employee_id).first()
        
        if not exemption:
            employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
            employee_profile_id = employee_profile.id if employee_profile else None
            
            if not financial_year:
                financial_year = datetime.now().year
            
            exemption = IncomeTaxExemption(
                employee_id=employee_id,
                employee_profile_id=employee_profile_id,
                financial_year=financial_year,
                tax_regime='old'  # HRA only applicable in old regime
            )
            db.session.add(exemption)
        
        # Calculate HRA exemption
        hra_calc = calculate_hra_exemption(hra_received, basic_salary, rent_paid, is_metro_city)
        
        # Update exemption record
        exemption.hra_received = Decimal(str(hra_calc['hra_received']))
        exemption.hra_exemption_amount = Decimal(str(hra_calc['hra_exemption_amount']))
        exemption.rent_paid = Decimal(str(hra_calc['rent_paid']))
        exemption.rent_receipts_uploaded = rent_receipts_uploaded
        exemption.is_metro_city = is_metro_city
        
        db.session.commit()
        
        logger.info(f"Updated HRA exemption for employee {employee_id}")
        return exemption
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating HRA exemption: {str(e)}")
        raise


def get_hra_exemption(employee_id):
    """
    Get HRA exemption for an employee
    
    Args:
        employee_id: Employee user ID
    
    Returns:
        dict: HRA exemption details
    """
    try:
        exemption = IncomeTaxExemption.query.filter_by(employee_id=employee_id).first()
        
        if not exemption or not exemption.hra_received:
            return None
        
        return {
            'hra_received': float(exemption.hra_received) if exemption.hra_received else 0,
            'hra_exemption_amount': float(exemption.hra_exemption_amount) if exemption.hra_exemption_amount else 0,
            'rent_paid': float(exemption.rent_paid) if exemption.rent_paid else 0,
            'rent_receipts_uploaded': exemption.rent_receipts_uploaded,
            'is_metro_city': exemption.is_metro_city
        }
    except Exception as e:
        logger.error(f"Error fetching HRA exemption: {str(e)}")
        raise


def validate_rent_receipts(rent_receipts_data):
    """
    Validate rent receipts data
    
    Args:
        rent_receipts_data: List of rent receipt data
    
    Returns:
        bool: True if valid
    """
    if not rent_receipts_data:
        return False
    
    # Basic validation - should have required fields
    required_fields = ['month', 'year', 'amount', 'landlord_name']
    for receipt in rent_receipts_data:
        if not all(field in receipt for field in required_fields):
            return False
    
    return True

