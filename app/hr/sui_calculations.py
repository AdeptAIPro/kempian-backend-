"""
SUI (State Unemployment Insurance) Calculation Service
Handles SUI contribution calculations for US payroll
"""

from app import db
from app.models import (
    SUIContribution, StateTaxConfiguration, PayRun, Tenant, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# SUI Rates by State (2024) - Examples
# These are employer contribution rates
SUI_RATES = {
    'CA': {'rate': 0.031, 'wage_base': 7000},  # California
    'NY': {'rate': 0.0275, 'wage_base': 12000},  # New York
    'TX': {'rate': 0.00625, 'wage_base': 9000},  # Texas
    'FL': {'rate': 0.0027, 'wage_base': 7000},  # Florida
    'IL': {'rate': 0.031, 'wage_base': 12820},  # Illinois
    'PA': {'rate': 0.037, 'wage_base': 10000},  # Pennsylvania
    'OH': {'rate': 0.027, 'wage_base': 9000},  # Ohio
    'GA': {'rate': 0.027, 'wage_base': 9500},  # Georgia
    'NC': {'rate': 0.01, 'wage_base': 28000},  # North Carolina
    'MI': {'rate': 0.027, 'wage_base': 9500},  # Michigan
}

# Default SUI rate if state not found
DEFAULT_SUI_RATE = 0.027
DEFAULT_WAGE_BASE = 7000


def calculate_sui_contribution(gross_wages, state_code, sui_rate=None, wage_base=None):
    """
    Calculate SUI contribution
    
    Args:
        gross_wages: Gross wages for the period
        state_code: State code
        sui_rate: SUI rate (optional, will use default if not provided)
        wage_base: Wage base limit (optional, will use default if not provided)
    
    Returns:
        dict: SUI calculation details
    """
    try:
        gross_wages = Decimal(str(gross_wages))
        state_code = state_code.upper() if state_code else None
        
        # Get SUI rate and wage base
        if not sui_rate:
            if state_code and state_code in SUI_RATES:
                sui_rate = Decimal(str(SUI_RATES[state_code]['rate']))
                wage_base = Decimal(str(SUI_RATES[state_code]['wage_base']))
            else:
                sui_rate = Decimal(str(DEFAULT_SUI_RATE))
                wage_base = Decimal(str(DEFAULT_WAGE_BASE))
        else:
            sui_rate = Decimal(str(sui_rate))
            wage_base = Decimal(str(wage_base)) if wage_base else Decimal(str(DEFAULT_WAGE_BASE))
        
        # SUI is calculated on wages up to the wage base
        taxable_wages = min(gross_wages, wage_base)
        sui_contribution = taxable_wages * sui_rate
        
        return {
            'gross_wages': float(gross_wages),
            'state_code': state_code,
            'sui_rate': float(sui_rate),
            'wage_base': float(wage_base),
            'taxable_wages': float(taxable_wages),
            'sui_contribution': float(sui_contribution),
            'is_above_wage_base': gross_wages > wage_base
        }
    except Exception as e:
        logger.error(f"Error calculating SUI contribution: {str(e)}")
        raise


def create_sui_contribution_record(
    tenant_id,
    state_code,
    total_wages,
    pay_run_id=None,
    contribution_quarter=None,
    contribution_year=None,
    sui_rate=None,
    wage_base=None
):
    """
    Create SUI contribution record
    
    Args:
        tenant_id: Tenant ID
        state_code: State code
        total_wages: Total wages for the period
        pay_run_id: Pay run ID (optional)
        contribution_quarter: Quarter (1-4), defaults to current quarter
        contribution_year: Year, defaults to current year
        sui_rate: SUI rate (optional)
        wage_base: Wage base limit (optional)
    
    Returns:
        SUIContribution: Created record
    """
    try:
        # Calculate SUI
        sui_calc = calculate_sui_contribution(total_wages, state_code, sui_rate, wage_base)
        
        # Get state name
        state_names = {
            'CA': 'California', 'NY': 'New York', 'TX': 'Texas', 'FL': 'Florida',
            'IL': 'Illinois', 'PA': 'Pennsylvania', 'OH': 'Ohio', 'GA': 'Georgia',
            'NC': 'North Carolina', 'MI': 'Michigan'
        }
        state_name = state_names.get(state_code, state_code)
        
        # Set default quarter/year
        if not contribution_quarter:
            month = datetime.now().month
            contribution_quarter = ((month - 1) // 3) + 1
        if not contribution_year:
            contribution_year = datetime.now().year
        
        # Check if record exists
        existing = SUIContribution.query.filter_by(
            tenant_id=tenant_id,
            state_code=state_code,
            contribution_quarter=contribution_quarter,
            contribution_year=contribution_year
        ).first()
        
        if existing:
            existing.total_wages = Decimal(str(sui_calc['gross_wages']))
            existing.sui_rate = Decimal(str(sui_calc['sui_rate']))
            existing.wage_base_limit = Decimal(str(sui_calc['wage_base']))
            existing.sui_contribution = Decimal(str(sui_calc['sui_contribution']))
            existing.updated_at = datetime.utcnow()
            sui_record = existing
        else:
            sui_record = SUIContribution(
                tenant_id=tenant_id,
                pay_run_id=pay_run_id,
                state_code=state_code,
                state_name=state_name,
                total_wages=Decimal(str(sui_calc['gross_wages'])),
                sui_rate=Decimal(str(sui_calc['sui_rate'])),
                wage_base_limit=Decimal(str(sui_calc['wage_base'])),
                sui_contribution=Decimal(str(sui_calc['sui_contribution'])),
                contribution_quarter=contribution_quarter,
                contribution_year=contribution_year
            )
            db.session.add(sui_record)
        
        db.session.commit()
        logger.info(f"Created SUI contribution record for tenant {tenant_id}, state {state_code}")
        return sui_record
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating SUI contribution record: {str(e)}")
        raise


def get_sui_contributions_by_tenant(tenant_id, year=None, quarter=None, state_code=None):
    """
    Get SUI contributions for a tenant
    
    Args:
        tenant_id: Tenant ID
        year: Year filter (optional)
        quarter: Quarter filter (optional)
        state_code: State code filter (optional)
    
    Returns:
        list: List of SUIContribution records
    """
    try:
        query = SUIContribution.query.filter_by(tenant_id=tenant_id)
        
        if year:
            query = query.filter_by(contribution_year=year)
        if quarter:
            query = query.filter_by(contribution_quarter=quarter)
        if state_code:
            query = query.filter_by(state_code=state_code)
        
        return query.order_by(
            SUIContribution.contribution_year.desc(),
            SUIContribution.contribution_quarter.desc()
        ).all()
    except Exception as e:
        logger.error(f"Error fetching SUI contributions: {str(e)}")
        raise


def mark_sui_quarterly_report_filed(sui_contribution_id, filed_by_user_id=None):
    """
    Mark SUI quarterly report as filed
    
    Args:
        sui_contribution_id: SUI contribution record ID
        filed_by_user_id: User ID who filed (optional)
    """
    try:
        sui_record = SUIContribution.query.get(sui_contribution_id)
        if not sui_record:
            raise ValueError(f"SUI contribution record {sui_contribution_id} not found")
        
        sui_record.quarterly_report_filed = True
        sui_record.quarterly_report_filed_at = datetime.utcnow()
        
        db.session.commit()
        logger.info(f"Marked SUI contribution {sui_contribution_id} as filed")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error marking SUI report as filed: {str(e)}")
        raise

