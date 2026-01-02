"""
IRS Tax Tables Service
Handles IRS Publication 15-T (Circular E) tax tables for US payroll
"""

from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# IRS Publication 15-T Tax Tables (2024)
# Wage Bracket Method - Annual

# Federal Income Tax Withholding - Single
FEDERAL_TAX_SINGLE_2024 = [
    (0, 4850, 0, 0),
    (4850, 12200, 0.10, 0),
    (12200, 24750, 0.12, 485),
    (24750, 61200, 0.22, 1994),
    (61200, 102550, 0.24, 10016),
    (102550, 198050, 0.32, 19916),
    (198050, 250525, 0.35, 38016),
    (250525, float('inf'), 0.37, 57653)
]

# Federal Income Tax Withholding - Married Filing Jointly
FEDERAL_TAX_MARRIED_2024 = [
    (0, 13850, 0, 0),
    (13850, 24400, 0.10, 0),
    (24400, 49400, 0.12, 1055),
    (49400, 127100, 0.22, 4055),
    (127100, 204250, 0.24, 21115),
    (204250, 326600, 0.32, 38415),
    (326600, 414700, 0.35, 79415),
    (414700, float('inf'), 0.37, 102647)
]

# FICA Rates (2024)
FICA_SOCIAL_SECURITY_RATE = Decimal('0.062')  # 6.2%
FICA_SOCIAL_SECURITY_WAGE_BASE = Decimal('168600')  # $168,600
FICA_MEDICARE_RATE = Decimal('0.0145')  # 1.45%
FICA_ADDITIONAL_MEDICARE_RATE = Decimal('0.009')  # 0.9% on earnings over $200k
FICA_ADDITIONAL_MEDICARE_THRESHOLD = Decimal('200000')


def calculate_federal_income_tax(annual_income, filing_status='single', allowances=0):
    """
    Calculate federal income tax using wage bracket method
    
    Args:
        annual_income: Annual income
        filing_status: Filing status ('single', 'married_jointly', 'married_separately', 'head_of_household')
        allowances: Number of allowances (for future use)
    
    Returns:
        dict: Federal tax calculation
    """
    try:
        annual_income = Decimal(str(annual_income))
        
        # Select appropriate tax table
        if filing_status == 'married_jointly':
            brackets = FEDERAL_TAX_MARRIED_2024
        else:
            brackets = FEDERAL_TAX_SINGLE_2024
        
        # Calculate tax using brackets
        remaining_income = annual_income
        total_tax = Decimal('0')
        
        for min_income, max_income, rate, base_tax in brackets:
            if remaining_income <= 0:
                break
            
            if max_income == float('inf'):
                slab_income = remaining_income
            else:
                slab_income = min(remaining_income, Decimal(str(max_income)) - Decimal(str(min_income)) + Decimal('1'))
            
            if slab_income > 0 and annual_income >= Decimal(str(min_income)):
                if annual_income <= Decimal(str(max_income)):
                    # Use base tax for this bracket
                    total_tax = Decimal(str(base_tax)) + ((annual_income - Decimal(str(min_income))) * Decimal(str(rate)))
                    break
                else:
                    # Continue to next bracket
                    remaining_income -= slab_income
        
        return {
            'annual_income': float(annual_income),
            'filing_status': filing_status,
            'federal_tax': float(total_tax),
            'monthly_tax': float(total_tax / Decimal('12'))
        }
    except Exception as e:
        logger.error(f"Error calculating federal income tax: {str(e)}")
        raise


def calculate_fica_tax(gross_income):
    """
    Calculate FICA taxes (Social Security and Medicare)
    
    Args:
        gross_income: Gross income
    
    Returns:
        dict: FICA tax calculation
    """
    try:
        gross_income = Decimal(str(gross_income))
        
        # Social Security Tax (6.2% up to wage base)
        ss_taxable = min(gross_income, FICA_SOCIAL_SECURITY_WAGE_BASE)
        social_security_tax = ss_taxable * FICA_SOCIAL_SECURITY_RATE
        
        # Medicare Tax (1.45% on all income)
        medicare_tax = gross_income * FICA_MEDICARE_RATE
        
        # Additional Medicare Tax (0.9% on income over $200k)
        additional_medicare_tax = Decimal('0')
        if gross_income > FICA_ADDITIONAL_MEDICARE_THRESHOLD:
            additional_medicare_tax = (gross_income - FICA_ADDITIONAL_MEDICARE_THRESHOLD) * FICA_ADDITIONAL_MEDICARE_RATE
        
        total_fica = social_security_tax + medicare_tax + additional_medicare_tax
        
        return {
            'gross_income': float(gross_income),
            'social_security_tax': float(social_security_tax),
            'medicare_tax': float(medicare_tax),
            'additional_medicare_tax': float(additional_medicare_tax),
            'total_fica': float(total_fica),
            'is_above_ss_wage_base': gross_income > FICA_SOCIAL_SECURITY_WAGE_BASE
        }
    except Exception as e:
        logger.error(f"Error calculating FICA tax: {str(e)}")
        raise


def calculate_total_federal_tax(gross_income, filing_status='single', allowances=0):
    """
    Calculate total federal tax (income tax + FICA)
    
    Args:
        gross_income: Gross income (annual)
        filing_status: Filing status
        allowances: Number of allowances
    
    Returns:
        dict: Total federal tax calculation
    """
    try:
        # Calculate federal income tax
        income_tax = calculate_federal_income_tax(gross_income, filing_status, allowances)
        
        # Calculate FICA tax
        fica_tax = calculate_fica_tax(gross_income)
        
        total_tax = Decimal(str(income_tax['federal_tax'])) + Decimal(str(fica_tax['total_fica']))
        
        return {
            'gross_income': float(gross_income),
            'federal_income_tax': income_tax['federal_tax'],
            'fica_tax': fica_tax['total_fica'],
            'social_security_tax': fica_tax['social_security_tax'],
            'medicare_tax': fica_tax['medicare_tax'],
            'additional_medicare_tax': fica_tax['additional_medicare_tax'],
            'total_federal_tax': float(total_tax),
            'monthly_withholding': float(total_tax / Decimal('12'))
        }
    except Exception as e:
        logger.error(f"Error calculating total federal tax: {str(e)}")
        raise


def get_tax_table_by_year(year):
    """
    Get tax table for a specific year
    
    Args:
        year: Tax year
    
    Returns:
        dict: Tax table information
    """
    # For now, return 2024 tables
    # In production, this would load year-specific tables
    if year == 2024:
        return {
            'year': 2024,
            'federal_single': FEDERAL_TAX_SINGLE_2024,
            'federal_married': FEDERAL_TAX_MARRIED_2024,
            'fica_ss_rate': float(FICA_SOCIAL_SECURITY_RATE),
            'fica_ss_wage_base': float(FICA_SOCIAL_SECURITY_WAGE_BASE),
            'fica_medicare_rate': float(FICA_MEDICARE_RATE)
        }
    else:
        logger.warning(f"Tax table for year {year} not available, using 2024")
        return get_tax_table_by_year(2024)

