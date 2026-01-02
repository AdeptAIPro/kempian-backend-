"""
Currency Service
Handles multi-currency payroll, exchange rates, and currency conversion
"""

from app import db
from app.models import (
    CurrencyExchangeRate, EmployeeProfile, Payslip
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)


def get_exchange_rate(base_currency, target_currency, rate_date=None):
    """
    Get exchange rate between currencies
    
    Args:
        base_currency: Base currency code (e.g., 'USD')
        target_currency: Target currency code (e.g., 'INR')
        rate_date: Date for rate (optional, defaults to today)
    
    Returns:
        Decimal: Exchange rate
    """
    try:
        if not rate_date:
            rate_date = datetime.now().date()
        
        # Get most recent exchange rate
        rate = CurrencyExchangeRate.query.filter(
            CurrencyExchangeRate.base_currency == base_currency,
            CurrencyExchangeRate.target_currency == target_currency,
            CurrencyExchangeRate.rate_date <= rate_date,
            CurrencyExchangeRate.is_active == True
        ).order_by(CurrencyExchangeRate.rate_date.desc()).first()
        
        if rate:
            return rate.exchange_rate
        else:
            # Default to 1.0 if same currency
            if base_currency == target_currency:
                return Decimal('1.0')
            logger.warning(f"Exchange rate not found for {base_currency} to {target_currency}, using 1.0")
            return Decimal('1.0')
    except Exception as e:
        logger.error(f"Error getting exchange rate: {str(e)}")
        return Decimal('1.0')


def convert_currency(amount, from_currency, to_currency, rate_date=None):
    """
    Convert currency amount
    
    Args:
        amount: Amount to convert
        from_currency: Source currency
        to_currency: Target currency
        rate_date: Date for rate (optional)
    
    Returns:
        Decimal: Converted amount
    """
    try:
        amount = Decimal(str(amount))
        
        if from_currency == to_currency:
            return amount
        
        exchange_rate = get_exchange_rate(from_currency, to_currency, rate_date)
        converted_amount = amount * exchange_rate
        
        return converted_amount.quantize(Decimal('0.01'))
    except Exception as e:
        logger.error(f"Error converting currency: {str(e)}")
        raise


def create_exchange_rate(
    base_currency,
    target_currency,
    exchange_rate,
    rate_source='manual',
    rate_date=None
):
    """
    Create or update exchange rate
    
    Args:
        base_currency: Base currency code
        target_currency: Target currency code
        exchange_rate: Exchange rate
        rate_source: Rate source ('api', 'manual', 'bank')
        rate_date: Rate date (optional, defaults to today)
    
    Returns:
        CurrencyExchangeRate: Created/updated rate
    """
    try:
        if not rate_date:
            rate_date = datetime.now().date()
        
        # Check if rate exists for this date
        existing = CurrencyExchangeRate.query.filter_by(
            base_currency=base_currency,
            target_currency=target_currency,
            rate_date=rate_date
        ).first()
        
        if existing:
            existing.exchange_rate = Decimal(str(exchange_rate))
            existing.rate_source = rate_source
            existing.updated_at = datetime.utcnow()
            rate = existing
        else:
            # Deactivate old rates
            CurrencyExchangeRate.query.filter_by(
                base_currency=base_currency,
                target_currency=target_currency,
                is_active=True
            ).update({'is_active': False})
            
            rate = CurrencyExchangeRate(
                base_currency=base_currency,
                target_currency=target_currency,
                exchange_rate=Decimal(str(exchange_rate)),
                rate_source=rate_source,
                rate_date=rate_date
            )
            db.session.add(rate)
        
        db.session.commit()
        logger.info(f"Created/updated exchange rate: {base_currency} to {target_currency} = {exchange_rate}")
        return rate
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating exchange rate: {str(e)}")
        raise


def format_currency(amount, currency_code):
    """
    Format currency amount based on currency code
    
    Args:
        amount: Amount to format
        currency_code: Currency code
    
    Returns:
        str: Formatted currency string
    """
    try:
        amount = Decimal(str(amount))
        
        currency_symbols = {
            'USD': '$',
            'INR': '₹',
            'GBP': '£',
            'EUR': '€',
            'CAD': 'C$',
            'AUD': 'A$',
            'AED': 'AED',
            'SGD': 'S$',
            'JPY': '¥',
            'CNY': '¥'
        }
        
        symbol = currency_symbols.get(currency_code, currency_code)
        
        # Format based on currency
        if currency_code in ['JPY', 'KRW']:
            # No decimal places for some currencies
            return f"{symbol}{amount:,.0f}"
        else:
            return f"{symbol}{amount:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting currency: {str(e)}")
        return str(amount)

