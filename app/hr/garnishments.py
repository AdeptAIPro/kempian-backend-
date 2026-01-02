"""
Garnishment Management Service
Handles wage garnishments and attachments for US payroll
"""

from app import db
from app.models import (
    Garnishment, EmployeeProfile, Payslip, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Maximum garnishment limits by type (as percentage of disposable income)
GARNISHMENT_LIMITS = {
    'child_support': 0.60,  # 60% of disposable income (50% if not supporting another family)
    'tax_levy': 0.15,  # 15% of disposable income (varies)
    'creditor': 0.25,  # 25% of disposable income
    'student_loan': 0.15,  # 15% of disposable income
}

# Federal minimum wage (for garnishment calculations)
FEDERAL_MINIMUM_WAGE = Decimal('7.25')  # $7.25 per hour
PROTECTED_AMOUNT_MULTIPLIER = Decimal('30')  # 30 times federal minimum wage per week


def calculate_garnishment_amount(
    disposable_income,
    garnishment_type,
    amount_per_paycheck=None,
    percentage_of_wages=None,
    maximum_deduction_limit=None
):
    """
    Calculate garnishment amount
    
    Args:
        disposable_income: Disposable income (gross - required deductions)
        garnishment_type: Type of garnishment
        amount_per_paycheck: Fixed amount per paycheck (optional)
        percentage_of_wages: Percentage of wages (optional)
        maximum_deduction_limit: Maximum deduction limit (optional)
    
    Returns:
        dict: Garnishment calculation details
    """
    try:
        disposable_income = Decimal(str(disposable_income))
        
        # Get maximum limit for garnishment type
        max_percentage = GARNISHMENT_LIMITS.get(garnishment_type, 0.25)
        max_by_type = disposable_income * Decimal(str(max_percentage))
        
        # Calculate garnishment amount
        garnishment_amount = Decimal('0')
        
        if amount_per_paycheck:
            garnishment_amount = Decimal(str(amount_per_paycheck))
        elif percentage_of_wages:
            garnishment_amount = disposable_income * Decimal(str(percentage_of_wages))
        else:
            garnishment_amount = max_by_type
        
        # Apply maximum deduction limit
        if maximum_deduction_limit:
            garnishment_amount = min(garnishment_amount, Decimal(str(maximum_deduction_limit)))
        
        # Apply type-specific maximum
        garnishment_amount = min(garnishment_amount, max_by_type)
        
        # Ensure minimum protected amount (30x federal minimum wage per week)
        protected_amount = FEDERAL_MINIMUM_WAGE * PROTECTED_AMOUNT_MULTIPLIER
        if disposable_income <= protected_amount:
            garnishment_amount = Decimal('0')
        else:
            # Can't garnish below protected amount
            max_garnishable = disposable_income - protected_amount
            garnishment_amount = min(garnishment_amount, max_garnishable)
        
        return {
            'disposable_income': float(disposable_income),
            'garnishment_type': garnishment_type,
            'garnishment_amount': float(garnishment_amount),
            'max_percentage': float(max_percentage),
            'protected_amount': float(protected_amount)
        }
    except Exception as e:
        logger.error(f"Error calculating garnishment: {str(e)}")
        raise


def create_garnishment(
    employee_id,
    garnishment_type,
    priority_order,
    amount_per_paycheck=None,
    percentage_of_wages=None,
    maximum_deduction_limit=None,
    court_order_number=None,
    court_name=None,
    order_date=None,
    start_date=None,
    end_date=None
):
    """
    Create garnishment record
    
    Args:
        employee_id: Employee user ID
        garnishment_type: Type of garnishment
        priority_order: Priority order (1 = highest)
        amount_per_paycheck: Fixed amount per paycheck (optional)
        percentage_of_wages: Percentage of wages (optional)
        maximum_deduction_limit: Maximum deduction limit (optional)
        court_order_number: Court order number (optional)
        court_name: Court name (optional)
        order_date: Order date (optional)
        start_date: Start date (required)
        end_date: End date (optional)
    
    Returns:
        Garnishment: Created record
    """
    try:
        # Get employee profile
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        employee_profile_id = employee_profile.id if employee_profile else None
        
        if not start_date:
            start_date = datetime.now().date()
        
        # Create record
        garnishment = Garnishment(
            employee_id=employee_id,
            employee_profile_id=employee_profile_id,
            garnishment_type=garnishment_type,
            priority_order=priority_order,
            amount_per_paycheck=Decimal(str(amount_per_paycheck)) if amount_per_paycheck else None,
            percentage_of_wages=Decimal(str(percentage_of_wages)) if percentage_of_wages else None,
            maximum_deduction_limit=Decimal(str(maximum_deduction_limit)) if maximum_deduction_limit else None,
            court_order_number=court_order_number,
            court_name=court_name,
            order_date=order_date,
            start_date=start_date,
            end_date=end_date,
            is_active=True
        )
        
        db.session.add(garnishment)
        db.session.commit()
        
        logger.info(f"Created garnishment record for employee {employee_id}")
        return garnishment
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating garnishment record: {str(e)}")
        raise


def get_active_garnishments(employee_id):
    """
    Get active garnishments for an employee, sorted by priority
    
    Args:
        employee_id: Employee user ID
    
    Returns:
        list: List of Garnishment records sorted by priority
    """
    try:
        today = datetime.now().date()
        return Garnishment.query.filter(
            Garnishment.employee_id == employee_id,
            Garnishment.is_active == True,
            Garnishment.start_date <= today,
            db.or_(
                Garnishment.end_date.is_(None),
                Garnishment.end_date >= today
            )
        ).order_by(Garnishment.priority_order.asc()).all()
    except Exception as e:
        logger.error(f"Error fetching active garnishments: {str(e)}")
        raise


def calculate_total_garnishments(disposable_income, employee_id):
    """
    Calculate total garnishments for an employee
    
    Args:
        disposable_income: Disposable income
        employee_id: Employee user ID
    
    Returns:
        dict: Total garnishment calculation
    """
    try:
        active_garnishments = get_active_garnishments(employee_id)
        
        total_garnishment = Decimal('0')
        garnishment_breakdown = []
        
        for garnishment in active_garnishments:
            calc = calculate_garnishment_amount(
                disposable_income,
                garnishment.garnishment_type,
                float(garnishment.amount_per_paycheck) if garnishment.amount_per_paycheck else None,
                float(garnishment.percentage_of_wages) if garnishment.percentage_of_wages else None,
                float(garnishment.maximum_deduction_limit) if garnishment.maximum_deduction_limit else None
            )
            
            garnishment_amount = Decimal(str(calc['garnishment_amount']))
            total_garnishment += garnishment_amount
            
            garnishment_breakdown.append({
                'garnishment_id': garnishment.id,
                'garnishment_type': garnishment.garnishment_type,
                'priority_order': garnishment.priority_order,
                'amount': float(garnishment_amount)
            })
        
        return {
            'disposable_income': float(disposable_income),
            'total_garnishment': float(total_garnishment),
            'garnishment_breakdown': garnishment_breakdown,
            'remaining_disposable_income': float(disposable_income - total_garnishment)
        }
    except Exception as e:
        logger.error(f"Error calculating total garnishments: {str(e)}")
        raise


def update_garnishment_collected(garnishment_id, amount):
    """
    Update total collected amount for a garnishment
    
    Args:
        garnishment_id: Garnishment record ID
        amount: Amount to add to total collected
    """
    try:
        garnishment = Garnishment.query.get(garnishment_id)
        if not garnishment:
            raise ValueError(f"Garnishment {garnishment_id} not found")
        
        garnishment.total_collected += Decimal(str(amount))
        garnishment.updated_at = datetime.utcnow()
        
        db.session.commit()
        logger.info(f"Updated garnishment {garnishment_id} total collected by {amount}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating garnishment collected: {str(e)}")
        raise

