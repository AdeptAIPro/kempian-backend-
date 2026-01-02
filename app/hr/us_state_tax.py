"""
US State Tax Calculation Service
Handles state income tax calculations for US payroll
"""

from app import db
from app.models import (
    StateTaxConfiguration, EmployeeProfile, Payslip, TaxConfiguration, Tenant, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Top 10 US States Tax Information (2024)
# State codes: CA, TX, NY, FL, IL, PA, OH, GA, NC, MI

STATE_TAX_RATES = {
    'CA': {  # California
        'type': 'progressive',
        'brackets': [
            (0, 10099, 0.01),
            (10099, 23942, 0.02),
            (23942, 37788, 0.04),
            (37788, 52455, 0.06),
            (52455, 66295, 0.08),
            (66295, 338639, 0.093),
            (338639, 406364, 0.103),
            (406364, 677275, 0.113),
            (677275, float('inf'), 0.133)
        ]
    },
    'TX': {  # Texas - No state income tax
        'type': 'none',
        'rate': 0
    },
    'NY': {  # New York
        'type': 'progressive',
        'brackets': [
            (0, 8500, 0.04),
            (8500, 11700, 0.045),
            (11700, 13900, 0.0525),
            (13900, 21400, 0.059),
            (21400, 80650, 0.0621),
            (80650, 215400, 0.0649),
            (215400, 1077550, 0.0685),
            (1077550, 5000000, 0.0965),
            (5000000, 25000000, 0.103),
            (25000000, float('inf'), 0.109)
        ]
    },
    'FL': {  # Florida - No state income tax
        'type': 'none',
        'rate': 0
    },
    'IL': {  # Illinois - Flat rate
        'type': 'flat',
        'rate': 0.0495
    },
    'PA': {  # Pennsylvania - Flat rate
        'type': 'flat',
        'rate': 0.0307
    },
    'OH': {  # Ohio
        'type': 'progressive',
        'brackets': [
            (0, 26050, 0),
            (26050, 100000, 0.025),
            (100000, float('inf'), 0.035)
        ]
    },
    'GA': {  # Georgia
        'type': 'progressive',
        'brackets': [
            (0, 750, 0.01),
            (750, 2250, 0.02),
            (2250, 3750, 0.03),
            (3750, 5250, 0.04),
            (5250, 7000, 0.05),
            (7000, float('inf'), 0.0575)
        ]
    },
    'NC': {  # North Carolina - Flat rate
        'type': 'flat',
        'rate': 0.0475
    },
    'MI': {  # Michigan - Flat rate
        'type': 'flat',
        'rate': 0.0425
    }
}

# Reciprocal Agreements
# States that have reciprocal agreements with other states
RECIPROCAL_AGREEMENTS = {
    'IL': ['IA', 'KY', 'MI', 'WI'],
    'IN': ['KY', 'MI', 'OH', 'PA', 'WI'],
    'IA': ['IL'],
    'KY': ['IL', 'IN', 'MI', 'OH', 'VA', 'WV', 'WI'],
    'MD': ['DC', 'PA', 'VA', 'WV'],
    'MI': ['IL', 'IN', 'KY', 'MN', 'OH', 'WI'],
    'MN': ['MI', 'ND'],
    'MT': ['ND'],
    'ND': ['MN', 'MT'],
    'OH': ['IN', 'KY', 'MI', 'PA', 'WV'],
    'PA': ['IN', 'MD', 'NJ', 'OH', 'WV'],
    'VA': ['DC', 'KY', 'MD', 'WV'],
    'WI': ['IL', 'IN', 'KY', 'MI'],
    'WV': ['KY', 'MD', 'OH', 'PA', 'VA']
}


def calculate_state_tax(gross_income, state_code, filing_status='single', allowances=0):
    """
    Calculate state income tax
    
    Args:
        gross_income: Gross income
        state_code: State code (e.g., 'CA', 'NY')
        filing_status: Filing status (optional, for future use)
        allowances: Number of allowances (optional, for future use)
    
    Returns:
        dict: State tax calculation details
    """
    try:
        gross_income = Decimal(str(gross_income))
        state_code = state_code.upper() if state_code else None
        
        if not state_code or state_code not in STATE_TAX_RATES:
            logger.warning(f"State code {state_code} not found, returning 0")
            return {
                'gross_income': float(gross_income),
                'state_code': state_code,
                'state_tax': 0.0,
                'tax_type': 'unknown'
            }
        
        state_info = STATE_TAX_RATES[state_code]
        tax_type = state_info['type']
        state_tax = Decimal('0')
        
        if tax_type == 'none':
            state_tax = Decimal('0')
        elif tax_type == 'flat':
            rate = Decimal(str(state_info['rate']))
            state_tax = gross_income * rate
        elif tax_type == 'progressive':
            brackets = state_info['brackets']
            remaining_income = gross_income
            
            for min_income, max_income, rate in brackets:
                if remaining_income <= 0:
                    break
                
                if max_income == float('inf'):
                    slab_income = remaining_income
                else:
                    slab_income = min(remaining_income, Decimal(str(max_income)) - Decimal(str(min_income)) + Decimal('1'))
                
                if slab_income > 0:
                    tax_on_slab = (slab_income * Decimal(str(rate))).quantize(Decimal('0.01'))
                    state_tax += tax_on_slab
                    remaining_income -= slab_income
        
        return {
            'gross_income': float(gross_income),
            'state_code': state_code,
            'state_tax': float(state_tax),
            'tax_type': tax_type,
            'filing_status': filing_status
        }
    except Exception as e:
        logger.error(f"Error calculating state tax: {str(e)}")
        raise


def get_reciprocal_states(state_code):
    """
    Get states that have reciprocal agreements with the given state
    
    Args:
        state_code: State code
    
    Returns:
        list: List of state codes with reciprocal agreements
    """
    state_code = state_code.upper() if state_code else None
    if not state_code:
        return []
    
    return RECIPROCAL_AGREEMENTS.get(state_code, [])


def calculate_multi_state_tax(gross_income, primary_state, secondary_states=None):
    """
    Calculate tax for multi-state employees
    
    Args:
        gross_income: Gross income
        primary_state: Primary state code
        secondary_states: List of secondary state codes with allocation percentages
    
    Returns:
        dict: Multi-state tax calculation
    """
    try:
        gross_income = Decimal(str(gross_income))
        
        # Calculate primary state tax
        primary_tax = calculate_state_tax(gross_income, primary_state)
        total_tax = Decimal(str(primary_tax['state_tax']))
        
        secondary_taxes = []
        
        if secondary_states:
            for state_info in secondary_states:
                state_code = state_info.get('state_code')
                allocation_percent = Decimal(str(state_info.get('allocation_percent', 0)))
                
                if state_code and allocation_percent > 0:
                    allocated_income = gross_income * allocation_percent
                    state_tax_calc = calculate_state_tax(allocated_income, state_code)
                    secondary_taxes.append({
                        'state_code': state_code,
                        'allocation_percent': float(allocation_percent),
                        'allocated_income': float(allocated_income),
                        'state_tax': state_tax_calc['state_tax']
                    })
                    total_tax += Decimal(str(state_tax_calc['state_tax']))
        
        return {
            'gross_income': float(gross_income),
            'primary_state': primary_state,
            'primary_tax': primary_tax['state_tax'],
            'secondary_taxes': secondary_taxes,
            'total_state_tax': float(total_tax)
        }
    except Exception as e:
        logger.error(f"Error calculating multi-state tax: {str(e)}")
        raise


def create_state_tax_configuration(tenant_id, state_code, tax_type, tax_rate=None, tax_brackets=None):
    """
    Create or update state tax configuration
    
    Args:
        tenant_id: Tenant ID
        state_code: State code
        tax_type: Tax type ('income_tax', 'sui', 'local')
        tax_rate: Flat tax rate (optional)
        tax_brackets: Progressive tax brackets (optional)
    
    Returns:
        StateTaxConfiguration: Created/updated configuration
    """
    try:
        # Check if configuration exists
        existing = StateTaxConfiguration.query.filter_by(
            tenant_id=tenant_id,
            state_code=state_code,
            tax_type=tax_type
        ).first()
        
        if existing:
            if tax_rate:
                existing.tax_rate = Decimal(str(tax_rate))
            if tax_brackets:
                existing.tax_brackets = tax_brackets
            existing.updated_at = datetime.utcnow()
            config = existing
        else:
            config = StateTaxConfiguration(
                tenant_id=tenant_id,
                state_code=state_code,
                state_name=state_code,  # Can be enhanced with full state names
                tax_type=tax_type,
                tax_rate=Decimal(str(tax_rate)) if tax_rate else None,
                tax_brackets=tax_brackets,
                effective_date=datetime.utcnow().date()
            )
            db.session.add(config)
        
        db.session.commit()
        logger.info(f"Created/updated state tax configuration for {state_code}")
        return config
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating state tax configuration: {str(e)}")
        raise


def get_state_tax_configuration(tenant_id, state_code, tax_type='income_tax'):
    """
    Get state tax configuration
    
    Args:
        tenant_id: Tenant ID
        state_code: State code
        tax_type: Tax type
    
    Returns:
        StateTaxConfiguration: Configuration or None
    """
    try:
        return StateTaxConfiguration.query.filter_by(
            tenant_id=tenant_id,
            state_code=state_code,
            tax_type=tax_type,
            is_active=True
        ).first()
    except Exception as e:
        logger.error(f"Error fetching state tax configuration: {str(e)}")
        raise

