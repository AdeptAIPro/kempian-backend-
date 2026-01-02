"""
LTA (Leave Travel Allowance) Exemption Calculation Service
Handles LTA exemption calculations for India payroll
"""

from app import db
from app.models import IncomeTaxExemption, EmployeeProfile, User
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# LTA Block Years
# Block 1: 2022-2023, 2024-2025
# Block 2: 2023-2024, 2025-2026
LTA_BLOCK_YEARS = {
    '2022-2025': [2022, 2024],
    '2023-2026': [2023, 2025]
}


def calculate_lta_exemption(lta_received, travel_cost, block_year, travel_dates=None, family_travel=False):
    """
    Calculate LTA exemption
    
    LTA exemption is available for:
    - Two journeys in a block of 4 years
    - Travel within India only
    - Can be claimed for self and family
    
    Args:
        lta_received: LTA amount received
        travel_cost: Actual travel cost
        block_year: Block year (e.g., '2022-2025')
        travel_dates: List of travel dates (optional)
        family_travel: Whether family traveled
    
    Returns:
        dict: LTA exemption details
    """
    try:
        lta_received = Decimal(str(lta_received))
        travel_cost = Decimal(str(travel_cost))
        
        # LTA exemption is minimum of:
        # 1. Actual LTA received
        # 2. Actual travel cost
        lta_exemption = min(lta_received, travel_cost)
        
        # Validate block year
        current_year = datetime.now().year
        is_valid_block = False
        if block_year in LTA_BLOCK_YEARS:
            if current_year in LTA_BLOCK_YEARS[block_year]:
                is_valid_block = True
        
        return {
            'lta_received': float(lta_received),
            'travel_cost': float(travel_cost),
            'lta_exemption_amount': float(lta_exemption),
            'block_year': block_year,
            'is_valid_block': is_valid_block,
            'travel_dates': travel_dates,
            'family_travel': family_travel
        }
    except Exception as e:
        logger.error(f"Error calculating LTA exemption: {str(e)}")
        raise


def update_lta_exemption(
    employee_id,
    lta_received,
    travel_cost,
    block_year,
    travel_dates=None,
    family_travel=False,
    financial_year=None
):
    """
    Update LTA exemption for an employee
    
    Args:
        employee_id: Employee user ID
        lta_received: LTA amount received
        travel_cost: Actual travel cost
        block_year: Block year (e.g., '2022-2025')
        travel_dates: List of travel dates (optional)
        family_travel: Whether family traveled
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
                tax_regime='old'  # LTA only applicable in old regime
            )
            db.session.add(exemption)
        
        # Calculate LTA exemption
        lta_calc = calculate_lta_exemption(lta_received, travel_cost, block_year, travel_dates, family_travel)
        
        # Update exemption record
        exemption.lta_received = Decimal(str(lta_calc['lta_received']))
        exemption.lta_exemption_amount = Decimal(str(lta_calc['lta_exemption_amount']))
        exemption.lta_block_year = block_year
        exemption.lta_travel_dates = travel_dates
        exemption.lta_family_travel = family_travel
        
        db.session.commit()
        
        logger.info(f"Updated LTA exemption for employee {employee_id}")
        return exemption
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating LTA exemption: {str(e)}")
        raise


def get_lta_exemption(employee_id):
    """
    Get LTA exemption for an employee
    
    Args:
        employee_id: Employee user ID
    
    Returns:
        dict: LTA exemption details
    """
    try:
        exemption = IncomeTaxExemption.query.filter_by(employee_id=employee_id).first()
        
        if not exemption or not exemption.lta_received:
            return None
        
        return {
            'lta_received': float(exemption.lta_received) if exemption.lta_received else 0,
            'lta_exemption_amount': float(exemption.lta_exemption_amount) if exemption.lta_exemption_amount else 0,
            'block_year': exemption.lta_block_year,
            'travel_dates': exemption.lta_travel_dates,
            'family_travel': exemption.lta_family_travel
        }
    except Exception as e:
        logger.error(f"Error fetching LTA exemption: {str(e)}")
        raise


def validate_lta_claim(block_year, travel_dates, financial_year):
    """
    Validate LTA claim
    
    Args:
        block_year: Block year
        travel_dates: List of travel dates
        financial_year: Financial year
    
    Returns:
        dict: Validation result
    """
    try:
        # Check if block year is valid
        if block_year not in LTA_BLOCK_YEARS:
            return {
                'is_valid': False,
                'error': 'Invalid block year'
            }
        
        # Check if financial year is in the block
        if financial_year not in LTA_BLOCK_YEARS[block_year]:
            return {
                'is_valid': False,
                'error': f'Financial year {financial_year} not in block {block_year}'
            }
        
        # Check travel dates
        if not travel_dates or len(travel_dates) == 0:
            return {
                'is_valid': False,
                'error': 'Travel dates are required'
            }
        
        # Maximum 2 journeys per block
        if len(travel_dates) > 2:
            return {
                'is_valid': False,
                'error': 'Maximum 2 journeys allowed per block'
            }
        
        return {
            'is_valid': True,
            'error': None
        }
    except Exception as e:
        logger.error(f"Error validating LTA claim: {str(e)}")
        return {
            'is_valid': False,
            'error': str(e)
        }

