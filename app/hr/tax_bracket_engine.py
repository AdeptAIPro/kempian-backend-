"""
Enhanced Tax Bracket Calculation Engine
Handles progressive tax brackets, country-specific rules, year-specific tables, and tax optimization
"""

from app import db
from app.models import (
    TaxConfiguration, CountryTaxConfiguration, EmployeeProfile, Payslip
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)


def calculate_progressive_tax(income, brackets, country_code=None):
    """
    Calculate tax using progressive brackets
    
    Args:
        income: Taxable income
        brackets: List of tax brackets [(min, max, rate, base_tax), ...]
        country_code: Country code (optional, for country-specific rules)
    
    Returns:
        dict: Tax calculation details
    """
    try:
        income = Decimal(str(income))
        remaining_income = income
        total_tax = Decimal('0')
        bracket_breakdown = []
        
        # Sort brackets by min_income
        sorted_brackets = sorted(brackets, key=lambda x: x[0] if isinstance(x, (list, tuple)) else x.get('min_income', 0))
        
        for bracket in sorted_brackets:
            if remaining_income <= 0:
                break
            
            # Handle both tuple and dict formats
            if isinstance(bracket, (list, tuple)):
                min_income = Decimal(str(bracket[0]))
                max_income = Decimal(str(bracket[1])) if bracket[1] != float('inf') else None
                rate = Decimal(str(bracket[2]))
                base_tax = Decimal(str(bracket[3])) if len(bracket) > 3 else Decimal('0')
            else:
                min_income = Decimal(str(bracket.get('min_income', 0)))
                max_income = Decimal(str(bracket.get('max_income'))) if bracket.get('max_income') != float('inf') else None
                rate = Decimal(str(bracket.get('rate', 0)))
                base_tax = Decimal(str(bracket.get('base_tax', 0)))
            
            if income < min_income:
                continue
            
            if max_income is None:
                # Last bracket (infinity)
                slab_income = remaining_income
                tax_on_slab = base_tax + (remaining_income - min_income) * rate
                total_tax = tax_on_slab
                bracket_breakdown.append({
                    'min_income': float(min_income),
                    'max_income': None,
                    'rate': float(rate),
                    'income_in_bracket': float(remaining_income),
                    'tax_in_bracket': float(tax_on_slab)
                })
                break
            else:
                max_income_dec = Decimal(str(max_income))
                if income <= max_income_dec:
                    # Income falls in this bracket
                    slab_income = income - min_income
                    tax_on_slab = base_tax + (slab_income * rate)
                    total_tax = tax_on_slab
                    bracket_breakdown.append({
                        'min_income': float(min_income),
                        'max_income': float(max_income),
                        'rate': float(rate),
                        'income_in_bracket': float(slab_income),
                        'tax_in_bracket': float(tax_on_slab)
                    })
                    break
                else:
                    # Income exceeds this bracket
                    slab_income = max_income_dec - min_income
                    tax_on_slab = base_tax + (slab_income * rate)
                    total_tax += tax_on_slab
                    remaining_income -= slab_income
                    bracket_breakdown.append({
                        'min_income': float(min_income),
                        'max_income': float(max_income),
                        'rate': float(rate),
                        'income_in_bracket': float(slab_income),
                        'tax_in_bracket': float(tax_on_slab)
                    })
        
        return {
            'income': float(income),
            'total_tax': float(total_tax),
            'effective_rate': float(total_tax / income) if income > 0 else 0,
            'bracket_breakdown': bracket_breakdown
        }
    except Exception as e:
        logger.error(f"Error calculating progressive tax: {str(e)}")
        raise


def get_tax_configuration_by_country_year(tenant_id, country_code, tax_type, tax_year=None):
    """
    Get tax configuration for a specific country and year
    
    Args:
        tenant_id: Tenant ID
        country_code: Country code
        tax_type: Tax type
        tax_year: Tax year (optional, defaults to current year)
    
    Returns:
        TaxConfiguration or CountryTaxConfiguration: Tax configuration
    """
    try:
        if not tax_year:
            tax_year = datetime.now().year
        
        # Try country-specific configuration first
        country_config = CountryTaxConfiguration.query.filter_by(
            tenant_id=tenant_id,
            country_code=country_code,
            tax_type=tax_type,
            tax_year=tax_year,
            is_active=True
        ).first()
        
        if country_config:
            return country_config
        
        # Fall back to generic tax configuration
        generic_config = TaxConfiguration.query.filter_by(
            tenant_id=tenant_id,
            country_code=country_code,
            tax_type=tax_type,
            is_active=True
        ).first()
        
        return generic_config
    except Exception as e:
        logger.error(f"Error getting tax configuration: {str(e)}")
        raise


def optimize_tax_with_deductions(gross_income, deductions, tax_brackets, country_code=None):
    """
    Optimize tax by applying deductions strategically
    
    Args:
        gross_income: Gross income
        deductions: List of available deductions
        tax_brackets: Tax brackets
        country_code: Country code (optional)
    
    Returns:
        dict: Optimized tax calculation
    """
    try:
        gross_income = Decimal(str(gross_income))
        
        # Calculate tax without deductions
        tax_without = calculate_progressive_tax(gross_income, tax_brackets, country_code)
        
        # Apply deductions (simplified - apply all pre-tax deductions)
        total_deductions = sum(Decimal(str(d.get('amount', 0))) for d in deductions if d.get('is_pre_tax', True))
        taxable_income = max(Decimal('0'), gross_income - total_deductions)
        
        # Calculate tax with deductions
        tax_with = calculate_progressive_tax(taxable_income, tax_brackets, country_code)
        
        tax_savings = Decimal(str(tax_without['total_tax'])) - Decimal(str(tax_with['total_tax']))
        
        return {
            'gross_income': float(gross_income),
            'total_deductions': float(total_deductions),
            'taxable_income': float(taxable_income),
            'tax_without_deductions': tax_without['total_tax'],
            'tax_with_deductions': tax_with['total_tax'],
            'tax_savings': float(tax_savings),
            'effective_rate_before': tax_without['effective_rate'],
            'effective_rate_after': tax_with['effective_rate']
        }
    except Exception as e:
        logger.error(f"Error optimizing tax: {str(e)}")
        raise


def validate_tax_brackets(brackets):
    """
    Validate tax brackets
    
    Args:
        brackets: List of tax brackets
    
    Returns:
        dict: Validation result
    """
    try:
        errors = []
        
        if not brackets or len(brackets) == 0:
            errors.append("Tax brackets cannot be empty")
            return {'is_valid': False, 'errors': errors}
        
        # Check bracket format
        for i, bracket in enumerate(brackets):
            if isinstance(bracket, (list, tuple)):
                if len(bracket) < 3:
                    errors.append(f"Bracket {i+1}: Invalid format, need at least (min, max, rate)")
            elif isinstance(bracket, dict):
                if 'min_income' not in bracket or 'rate' not in bracket:
                    errors.append(f"Bracket {i+1}: Missing required fields")
            else:
                errors.append(f"Bracket {i+1}: Invalid format")
        
        # Check for overlapping brackets
        sorted_brackets = sorted(brackets, key=lambda x: x[0] if isinstance(x, (list, tuple)) else x.get('min_income', 0))
        for i in range(len(sorted_brackets) - 1):
            current = sorted_brackets[i]
            next_bracket = sorted_brackets[i + 1]
            
            current_max = current[1] if isinstance(current, (list, tuple)) else current.get('max_income', float('inf'))
            next_min = next_bracket[0] if isinstance(next_bracket, (list, tuple)) else next_bracket.get('min_income', 0)
            
            if current_max != float('inf') and current_max >= next_min:
                errors.append(f"Brackets {i+1} and {i+2} overlap")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    except Exception as e:
        logger.error(f"Error validating tax brackets: {str(e)}")
        return {
            'is_valid': False,
            'errors': [str(e)]
        }

