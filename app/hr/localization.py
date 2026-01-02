"""
Localization Service
Handles country-specific date formats, number formats, and address formats
"""

from datetime import datetime
from decimal import Decimal
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Date formats by country
DATE_FORMATS = {
    'US': '%m/%d/%Y',  # MM/DD/YYYY
    'IN': '%d/%m/%Y',  # DD/MM/YYYY
    'UK': '%d/%m/%Y',  # DD/MM/YYYY
    'CA': '%Y-%m-%d',  # YYYY-MM-DD
    'AU': '%d/%m/%Y',  # DD/MM/YYYY
    'AE': '%d/%m/%Y',  # DD/MM/YYYY
    'SG': '%d/%m/%Y',  # DD/MM/YYYY
    'DE': '%d.%m.%Y',  # DD.MM.YYYY
    'FR': '%d/%m/%Y',  # DD/MM/YYYY
}

# Number formats by country
NUMBER_FORMATS = {
    'US': {'decimal_separator': '.', 'thousands_separator': ','},
    'IN': {'decimal_separator': '.', 'thousands_separator': ','},
    'UK': {'decimal_separator': '.', 'thousands_separator': ','},
    'CA': {'decimal_separator': '.', 'thousands_separator': ','},
    'AU': {'decimal_separator': '.', 'thousands_separator': ','},
    'AE': {'decimal_separator': '.', 'thousands_separator': ','},
    'SG': {'decimal_separator': '.', 'thousands_separator': ','},
    'DE': {'decimal_separator': ',', 'thousands_separator': '.'},
    'FR': {'decimal_separator': ',', 'thousands_separator': ' '},
}

# Address formats by country
ADDRESS_FORMATS = {
    'US': ['street', 'city', 'state', 'zip', 'country'],
    'IN': ['street', 'city', 'state', 'pincode', 'country'],
    'UK': ['street', 'city', 'postcode', 'country'],
    'CA': ['street', 'city', 'province', 'postal_code', 'country'],
    'AU': ['street', 'city', 'state', 'postcode', 'country'],
    'AE': ['street', 'city', 'emirate', 'country'],
    'SG': ['street', 'city', 'postal_code', 'country'],
    'DE': ['street', 'city', 'postal_code', 'country'],
    'FR': ['street', 'city', 'postal_code', 'country'],
}


def format_date(date_obj, country_code='US'):
    """
    Format date based on country
    
    Args:
        date_obj: Date object
        country_code: Country code
    
    Returns:
        str: Formatted date string
    """
    try:
        if not date_obj:
            return ''
        
        date_format = DATE_FORMATS.get(country_code, DATE_FORMATS['US'])
        
        if isinstance(date_obj, str):
            # Try to parse if string
            try:
                date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
            except:
                return date_obj
        
        return date_obj.strftime(date_format)
    except Exception as e:
        logger.error(f"Error formatting date: {str(e)}")
        return str(date_obj)


def format_number(number, country_code='US', decimals=2):
    """
    Format number based on country
    
    Args:
        number: Number to format
        country_code: Country code
        decimals: Number of decimal places
    
    Returns:
        str: Formatted number string
    """
    try:
        number = Decimal(str(number))
        format_info = NUMBER_FORMATS.get(country_code, NUMBER_FORMATS['US'])
        
        decimal_sep = format_info['decimal_separator']
        thousands_sep = format_info['thousands_separator']
        
        # Format number
        if decimals == 0:
            formatted = f"{number:,.0f}"
        else:
            formatted = f"{number:,.{decimals}f}"
        
        # Replace separators
        if decimal_sep != '.':
            formatted = formatted.replace('.', 'TEMP').replace(',', thousands_sep).replace('TEMP', decimal_sep)
        elif thousands_sep != ',':
            formatted = formatted.replace(',', thousands_sep)
        
        return formatted
    except Exception as e:
        logger.error(f"Error formatting number: {str(e)}")
        return str(number)


def format_address(address_data, country_code='US'):
    """
    Format address based on country
    
    Args:
        address_data: Dictionary with address fields
        country_code: Country code
    
    Returns:
        str: Formatted address string
    """
    try:
        format_fields = ADDRESS_FORMATS.get(country_code, ADDRESS_FORMATS['US'])
        
        address_parts = []
        for field in format_fields:
            value = address_data.get(field, '')
            if value:
                address_parts.append(str(value))
        
        return ', '.join(address_parts)
    except Exception as e:
        logger.error(f"Error formatting address: {str(e)}")
        return str(address_data)


def get_localization_settings(country_code):
    """
    Get all localization settings for a country
    
    Args:
        country_code: Country code
    
    Returns:
        dict: Localization settings
    """
    try:
        return {
            'country_code': country_code,
            'date_format': DATE_FORMATS.get(country_code, DATE_FORMATS['US']),
            'number_format': NUMBER_FORMATS.get(country_code, NUMBER_FORMATS['US']),
            'address_format': ADDRESS_FORMATS.get(country_code, ADDRESS_FORMATS['US'])
        }
    except Exception as e:
        logger.error(f"Error getting localization settings: {str(e)}")
        return {}

