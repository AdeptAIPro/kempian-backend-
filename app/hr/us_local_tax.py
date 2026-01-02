"""
US Local Tax Calculation Service
Handles city, county, and school district tax calculations for US payroll
"""

from app import db
from app.models import (
    TaxConfiguration, EmployeeProfile, Payslip, Tenant
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Local Tax Rates by City/County
LOCAL_TAX_RATES = {
    'NYC': {  # New York City
        'city_tax': 0.03076,  # 3.076% for residents
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Philadelphia': {
        'city_tax': 0.0375,  # 3.75% wage tax
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Detroit': {
        'city_tax': 0.025,  # 2.5% income tax
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Cleveland': {
        'city_tax': 0.02,  # 2% income tax
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Columbus': {
        'city_tax': 0.025,  # 2.5% income tax
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Cincinnati': {
        'city_tax': 0.018,  # 1.8% income tax
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Pittsburgh': {
        'city_tax': 0.03,  # 3% local services tax
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Birmingham': {
        'city_tax': 0.01,  # 1% occupational tax
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Louisville': {
        'city_tax': 0.02,  # 2% occupational tax
        'county_tax': 0,
        'school_district_tax': 0
    },
    'Kansas City': {
        'city_tax': 0.01,  # 1% earnings tax
        'county_tax': 0,
        'school_district_tax': 0
    }
}

# County Tax Rates (examples)
COUNTY_TAX_RATES = {
    'Allegheny County, PA': 0.01,  # 1% local services tax
    'Cuyahoga County, OH': 0.008,  # 0.8% county tax
    'Wayne County, MI': 0.016,  # 1.6% income tax
}

# School District Tax Rates (examples)
SCHOOL_DISTRICT_TAX_RATES = {
    'Pittsburgh School District': 0.005,  # 0.5% school tax
    'Cleveland School District': 0.003,  # 0.3% school tax
}


def calculate_local_tax(gross_income, city=None, county=None, school_district=None):
    """
    Calculate local tax (city, county, school district)
    
    Args:
        gross_income: Gross income
        city: City name (optional)
        county: County name (optional)
        school_district: School district name (optional)
    
    Returns:
        dict: Local tax calculation details
    """
    try:
        gross_income = Decimal(str(gross_income))
        
        city_tax = Decimal('0')
        county_tax = Decimal('0')
        school_district_tax = Decimal('0')
        
        # Calculate city tax
        if city and city in LOCAL_TAX_RATES:
            city_rate = Decimal(str(LOCAL_TAX_RATES[city]['city_tax']))
            city_tax = gross_income * city_rate
        
        # Calculate county tax
        if county and county in COUNTY_TAX_RATES:
            county_rate = Decimal(str(COUNTY_TAX_RATES[county]))
            county_tax = gross_income * county_rate
        
        # Calculate school district tax
        if school_district and school_district in SCHOOL_DISTRICT_TAX_RATES:
            school_rate = Decimal(str(SCHOOL_DISTRICT_TAX_RATES[school_district]))
            school_district_tax = gross_income * school_rate
        
        total_local_tax = city_tax + county_tax + school_district_tax
        
        return {
            'gross_income': float(gross_income),
            'city': city,
            'county': county,
            'school_district': school_district,
            'city_tax': float(city_tax),
            'county_tax': float(county_tax),
            'school_district_tax': float(school_district_tax),
            'total_local_tax': float(total_local_tax)
        }
    except Exception as e:
        logger.error(f"Error calculating local tax: {str(e)}")
        raise


def create_local_tax_configuration(tenant_id, jurisdiction, tax_type, tax_rate, jurisdiction_type='city'):
    """
    Create or update local tax configuration
    
    Args:
        tenant_id: Tenant ID
        jurisdiction: City/County/School District name
        tax_type: Tax type ('city', 'county', 'school_district')
        tax_rate: Tax rate
        jurisdiction_type: Type of jurisdiction
    
    Returns:
        TaxConfiguration: Created/updated configuration
    """
    try:
        # Check if configuration exists
        existing = TaxConfiguration.query.filter_by(
            tenant_id=tenant_id,
            tax_type='local',
            jurisdiction=jurisdiction,
            country_code='US'
        ).first()
        
        if existing:
            existing.tax_rate = Decimal(str(tax_rate))
            existing.updated_at = datetime.utcnow()
            config = existing
        else:
            config = TaxConfiguration(
                tenant_id=tenant_id,
                tax_type='local',
                jurisdiction=jurisdiction,
                tax_rate=Decimal(str(tax_rate)),
                country_code='US',
                effective_date=datetime.utcnow().date()
            )
            db.session.add(config)
        
        db.session.commit()
        logger.info(f"Created/updated local tax configuration for {jurisdiction}")
        return config
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating local tax configuration: {str(e)}")
        raise


def get_local_tax_configurations(tenant_id, city=None, county=None):
    """
    Get local tax configurations
    
    Args:
        tenant_id: Tenant ID
        city: City filter (optional)
        county: County filter (optional)
    
    Returns:
        list: List of TaxConfiguration records
    """
    try:
        query = TaxConfiguration.query.filter_by(
            tenant_id=tenant_id,
            tax_type='local',
            country_code='US',
            is_active=True
        )
        
        if city:
            query = query.filter(TaxConfiguration.jurisdiction.like(f'%{city}%'))
        if county:
            query = query.filter(TaxConfiguration.jurisdiction.like(f'%{county}%'))
        
        return query.all()
    except Exception as e:
        logger.error(f"Error fetching local tax configurations: {str(e)}")
        raise

