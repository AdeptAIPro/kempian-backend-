"""
Multi-Country Tax Engine Framework
Handles tax calculations for UK, Canada, Australia, UAE, and other countries
"""

from app import db
from app.models import (
    CountryTaxConfiguration, EmployeeProfile, Payslip, Tenant
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# UK Tax Information (2024-25)
UK_PAYE_BRACKETS = [
    (0, 12570, 0),  # Personal Allowance
    (12570, 50270, 0.20),  # Basic rate
    (50270, 125140, 0.40),  # Higher rate
    (125140, float('inf'), 0.45)  # Additional rate
]

UK_NATIONAL_INSURANCE_RATE = Decimal('0.12')  # 12% on earnings between £12,570 and £50,270
UK_NI_UPPER_RATE = Decimal('0.02')  # 2% on earnings above £50,270
UK_NI_LOWER_THRESHOLD = Decimal('12570')
UK_NI_UPPER_THRESHOLD = Decimal('50270')

# Canada Tax Information (2024)
CANADA_CPP_RATE = Decimal('0.0595')  # 5.95% (employee and employer)
CANADA_CPP_MAX = Decimal('68600')  # Maximum pensionable earnings
CANADA_EI_RATE = Decimal('0.0166')  # 1.66% (employee)
CANADA_EI_MAX = Decimal('63100')  # Maximum insurable earnings

# Australia Tax Information (2024-25)
AUSTRALIA_PAYG_BRACKETS = [
    (0, 18200, 0),
    (18200, 45000, 0.19),
    (45000, 120000, 0.325),
    (120000, 180000, 0.37),
    (180000, float('inf'), 0.45)
]

AUSTRALIA_SUPERANNUATION_RATE = Decimal('0.11')  # 11% (employer contribution)

# UAE - No income tax, but gratuity calculations
UAE_GRATUITY_DAYS = 21  # Days per year for gratuity (if 5+ years, otherwise 7 days per year)


def calculate_uk_tax(gross_income):
    """
    Calculate UK PAYE tax and National Insurance
    
    Args:
        gross_income: Gross income (annual)
    
    Returns:
        dict: UK tax calculation
    """
    try:
        gross_income = Decimal(str(gross_income))
        
        # Calculate PAYE tax
        remaining_income = gross_income
        paye_tax = Decimal('0')
        
        for min_income, max_income, rate in UK_PAYE_BRACKETS:
            if remaining_income <= 0:
                break
            
            if max_income == float('inf'):
                slab_income = remaining_income
            else:
                slab_income = min(remaining_income, Decimal(str(max_income)) - Decimal(str(min_income)) + Decimal('1'))
            
            if slab_income > 0 and gross_income >= Decimal(str(min_income)):
                tax_on_slab = (slab_income * Decimal(str(rate))).quantize(Decimal('0.01'))
                paye_tax += tax_on_slab
                remaining_income -= slab_income
        
        # Calculate National Insurance
        ni_tax = Decimal('0')
        if gross_income > UK_NI_LOWER_THRESHOLD:
            if gross_income <= UK_NI_UPPER_THRESHOLD:
                ni_taxable = gross_income - UK_NI_LOWER_THRESHOLD
                ni_tax = ni_taxable * UK_NATIONAL_INSURANCE_RATE
            else:
                # Lower rate on income up to upper threshold
                ni_tax = (UK_NI_UPPER_THRESHOLD - UK_NI_LOWER_THRESHOLD) * UK_NATIONAL_INSURANCE_RATE
                # Higher rate on income above upper threshold
                ni_tax += (gross_income - UK_NI_UPPER_THRESHOLD) * UK_NI_UPPER_RATE
        
        total_tax = paye_tax + ni_tax
        
        return {
            'gross_income': float(gross_income),
            'paye_tax': float(paye_tax),
            'national_insurance': float(ni_tax),
            'total_tax': float(total_tax),
            'monthly_tax': float(total_tax / Decimal('12'))
        }
    except Exception as e:
        logger.error(f"Error calculating UK tax: {str(e)}")
        raise


def calculate_canada_tax(gross_income, province='ON'):
    """
    Calculate Canada tax (CPP, EI, and provincial tax)
    
    Args:
        gross_income: Gross income (annual)
        province: Province code (default: ON for Ontario)
    
    Returns:
        dict: Canada tax calculation
    """
    try:
        gross_income = Decimal(str(gross_income))
        
        # Calculate CPP (Canada Pension Plan)
        cpp_taxable = min(gross_income, CANADA_CPP_MAX)
        cpp_contribution = cpp_taxable * CANADA_CPP_RATE
        
        # Calculate EI (Employment Insurance)
        ei_taxable = min(gross_income, CANADA_EI_MAX)
        ei_contribution = ei_taxable * CANADA_EI_RATE
        
        # Provincial tax (simplified - Ontario example)
        # In production, would use province-specific rates
        provincial_tax_rate = Decimal('0.0505')  # Ontario: 5.05%
        provincial_tax = gross_income * provincial_tax_rate
        
        # Federal tax (simplified brackets)
        federal_tax = Decimal('0')
        if gross_income > 53359:
            federal_tax = gross_income * Decimal('0.15')  # Simplified
        
        total_tax = cpp_contribution + ei_contribution + provincial_tax + federal_tax
        
        return {
            'gross_income': float(gross_income),
            'cpp_contribution': float(cpp_contribution),
            'ei_contribution': float(ei_contribution),
            'provincial_tax': float(provincial_tax),
            'federal_tax': float(federal_tax),
            'total_tax': float(total_tax),
            'monthly_tax': float(total_tax / Decimal('12'))
        }
    except Exception as e:
        logger.error(f"Error calculating Canada tax: {str(e)}")
        raise


def calculate_australia_tax(gross_income):
    """
    Calculate Australia PAYG tax and Superannuation
    
    Args:
        gross_income: Gross income (annual)
    
    Returns:
        dict: Australia tax calculation
    """
    try:
        gross_income = Decimal(str(gross_income))
        
        # Calculate PAYG tax
        remaining_income = gross_income
        payg_tax = Decimal('0')
        
        for min_income, max_income, rate in AUSTRALIA_PAYG_BRACKETS:
            if remaining_income <= 0:
                break
            
            if max_income == float('inf'):
                slab_income = remaining_income
            else:
                slab_income = min(remaining_income, Decimal(str(max_income)) - Decimal(str(min_income)) + Decimal('1'))
            
            if slab_income > 0 and gross_income >= Decimal(str(min_income)):
                tax_on_slab = (slab_income * Decimal(str(rate))).quantize(Decimal('0.01'))
                payg_tax += tax_on_slab
                remaining_income -= slab_income
        
        # Superannuation (employer contribution, not deducted from employee)
        superannuation = gross_income * AUSTRALIA_SUPERANNUATION_RATE
        
        return {
            'gross_income': float(gross_income),
            'payg_tax': float(payg_tax),
            'superannuation': float(superannuation),
            'total_tax': float(payg_tax),
            'monthly_tax': float(payg_tax / Decimal('12'))
        }
    except Exception as e:
        logger.error(f"Error calculating Australia tax: {str(e)}")
        raise


def calculate_uae_gratuity(basic_salary, years_of_service):
    """
    Calculate UAE gratuity
    
    Args:
        basic_salary: Basic salary (monthly)
        years_of_service: Years of service
    
    Returns:
        dict: UAE gratuity calculation
    """
    try:
        basic_salary = Decimal(str(basic_salary))
        years_of_service = Decimal(str(years_of_service))
        
        # Gratuity calculation
        if years_of_service >= 5:
            # 21 days per year
            gratuity_days = years_of_service * Decimal(str(UAE_GRATUITY_DAYS))
        else:
            # 7 days per year for first 5 years
            gratuity_days = years_of_service * Decimal('7')
        
        daily_wage = basic_salary / Decimal('30')  # Assuming 30 days per month
        gratuity_amount = daily_wage * gratuity_days
        
        return {
            'basic_salary': float(basic_salary),
            'years_of_service': float(years_of_service),
            'gratuity_days': float(gratuity_days),
            'daily_wage': float(daily_wage),
            'gratuity_amount': float(gratuity_amount)
        }
    except Exception as e:
        logger.error(f"Error calculating UAE gratuity: {str(e)}")
        raise


def calculate_country_tax(gross_income, country_code, **kwargs):
    """
    Generic country tax calculation function
    
    Args:
        gross_income: Gross income
        country_code: Country code ('UK', 'CA', 'AU', 'AE', etc.)
        **kwargs: Country-specific parameters
    
    Returns:
        dict: Tax calculation
    """
    try:
        country_code = country_code.upper() if country_code else None
        
        if country_code == 'UK':
            return calculate_uk_tax(gross_income)
        elif country_code == 'CA':
            province = kwargs.get('province', 'ON')
            return calculate_canada_tax(gross_income, province)
        elif country_code == 'AU':
            return calculate_australia_tax(gross_income)
        elif country_code == 'AE':  # UAE
            # UAE has no income tax, return zero
            return {
                'gross_income': float(gross_income),
                'total_tax': 0.0,
                'monthly_tax': 0.0
            }
        else:
            logger.warning(f"Country code {country_code} not supported, returning zero tax")
            return {
                'gross_income': float(gross_income),
                'total_tax': 0.0,
                'monthly_tax': 0.0
            }
    except Exception as e:
        logger.error(f"Error calculating country tax: {str(e)}")
        raise


def create_country_tax_configuration(
    tenant_id,
    country_code,
    tax_type,
    tax_rate=None,
    tax_brackets=None,
    currency_code=None
):
    """
    Create or update country tax configuration
    
    Args:
        tenant_id: Tenant ID
        country_code: Country code
        tax_type: Tax type
        tax_rate: Tax rate (optional)
        tax_brackets: Tax brackets (optional)
        currency_code: Currency code (optional)
    
    Returns:
        CountryTaxConfiguration: Created/updated configuration
    """
    try:
        # Check if configuration exists
        existing = CountryTaxConfiguration.query.filter_by(
            tenant_id=tenant_id,
            country_code=country_code,
            tax_type=tax_type,
            is_active=True
        ).first()
        
        if existing:
            if tax_rate:
                existing.tax_rate = Decimal(str(tax_rate))
            if tax_brackets:
                existing.tax_brackets = tax_brackets
            if currency_code:
                existing.currency_code = currency_code
            existing.updated_at = datetime.utcnow()
            config = existing
        else:
            config = CountryTaxConfiguration(
                tenant_id=tenant_id,
                country_code=country_code,
                country_name=country_code,  # Can be enhanced with full country names
                tax_type=tax_type,
                tax_rate=Decimal(str(tax_rate)) if tax_rate else None,
                tax_brackets=tax_brackets,
                currency_code=currency_code,
                effective_date=datetime.utcnow().date(),
                tax_year=datetime.now().year
            )
            db.session.add(config)
        
        db.session.commit()
        logger.info(f"Created/updated country tax configuration for {country_code}")
        return config
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating country tax configuration: {str(e)}")
        raise

