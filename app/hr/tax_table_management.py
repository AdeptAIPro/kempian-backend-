"""
Tax Table Management Service
Handles tax table versioning, historical storage, import/export, and automatic updates
"""

from app import db
from app.models import (
    TaxConfiguration, CountryTaxConfiguration
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger
import json

logger = get_logger(__name__)


def create_tax_table_version(
    tenant_id,
    country_code,
    tax_type,
    tax_year,
    tax_rate=None,
    tax_brackets=None,
    wage_base_limit=None,
    source='manual'
):
    """
    Create a new version of tax table
    
    Args:
        tenant_id: Tenant ID
        country_code: Country code
        tax_type: Tax type
        tax_year: Tax year
        tax_rate: Tax rate (optional)
        tax_brackets: Tax brackets (optional)
        wage_base_limit: Wage base limit (optional)
        source: Source of tax table ('manual', 'irs', 'government_api')
    
    Returns:
        TaxConfiguration or CountryTaxConfiguration: Created tax table
    """
    try:
        # Deactivate old versions
        if country_code:
            CountryTaxConfiguration.query.filter_by(
                tenant_id=tenant_id,
                country_code=country_code,
                tax_type=tax_type,
                tax_year=tax_year,
                is_active=True
            ).update({'is_active': False})
        else:
            TaxConfiguration.query.filter_by(
                tenant_id=tenant_id,
                country_code=country_code,
                tax_type=tax_type,
                is_active=True
            ).update({'is_active': False})
        
        # Create new version
        if country_code:
            tax_table = CountryTaxConfiguration(
                tenant_id=tenant_id,
                country_code=country_code,
                tax_type=tax_type,
                tax_rate=Decimal(str(tax_rate)) if tax_rate else None,
                tax_brackets=tax_brackets,
                wage_base_limit=Decimal(str(wage_base_limit)) if wage_base_limit else None,
                tax_year=tax_year,
                effective_date=datetime.now().date(),
                is_active=True
            )
        else:
            tax_table = TaxConfiguration(
                tenant_id=tenant_id,
                tax_type=tax_type,
                jurisdiction=country_code,
                tax_rate=Decimal(str(tax_rate)) if tax_rate else None,
                tax_brackets=tax_brackets,
                wage_base_limit=Decimal(str(wage_base_limit)) if wage_base_limit else None,
                country_code=country_code,
                tax_year=tax_year,
                effective_date=datetime.now().date(),
                is_active=True
            )
        
        db.session.add(tax_table)
        db.session.commit()
        
        logger.info(f"Created tax table version for {country_code}, {tax_type}, {tax_year}")
        return tax_table
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating tax table version: {str(e)}")
        raise


def get_tax_table_history(tenant_id, country_code, tax_type, tax_year=None):
    """
    Get historical tax tables
    
    Args:
        tenant_id: Tenant ID
        country_code: Country code
        tax_type: Tax type
        tax_year: Tax year (optional)
    
    Returns:
        list: List of tax table versions
    """
    try:
        query = CountryTaxConfiguration.query.filter_by(
            tenant_id=tenant_id,
            country_code=country_code,
            tax_type=tax_type
        )
        
        if tax_year:
            query = query.filter_by(tax_year=tax_year)
        
        return query.order_by(
            CountryTaxConfiguration.tax_year.desc(),
            CountryTaxConfiguration.effective_date.desc()
        ).all()
    except Exception as e:
        logger.error(f"Error fetching tax table history: {str(e)}")
        raise


def export_tax_table(tax_table_id, format='json'):
    """
    Export tax table
    
    Args:
        tax_table_id: Tax table ID
        format: Export format ('json', 'csv')
    
    Returns:
        str or dict: Exported tax table data
    """
    try:
        tax_table = CountryTaxConfiguration.query.get(tax_table_id)
        if not tax_table:
            tax_table = TaxConfiguration.query.get(tax_table_id)
        
        if not tax_table:
            raise ValueError(f"Tax table {tax_table_id} not found")
        
        if format == 'json':
            return {
                'id': tax_table.id,
                'country_code': tax_table.country_code if hasattr(tax_table, 'country_code') else None,
                'tax_type': tax_table.tax_type,
                'tax_year': tax_table.tax_year if hasattr(tax_table, 'tax_year') else None,
                'tax_rate': float(tax_table.tax_rate) if tax_table.tax_rate else None,
                'tax_brackets': tax_table.tax_brackets,
                'wage_base_limit': float(tax_table.wage_base_limit) if tax_table.wage_base_limit else None,
                'effective_date': tax_table.effective_date.isoformat() if tax_table.effective_date else None
            }
        elif format == 'csv':
            # CSV format would be implemented here
            return "CSV export not yet implemented"
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        logger.error(f"Error exporting tax table: {str(e)}")
        raise


def import_tax_table(tenant_id, tax_table_data, source='manual'):
    """
    Import tax table
    
    Args:
        tenant_id: Tenant ID
        tax_table_data: Tax table data (dict)
        source: Source of import
    
    Returns:
        TaxConfiguration or CountryTaxConfiguration: Imported tax table
    """
    try:
        country_code = tax_table_data.get('country_code')
        tax_type = tax_table_data.get('tax_type')
        tax_year = tax_table_data.get('tax_year', datetime.now().year)
        
        return create_tax_table_version(
            tenant_id=tenant_id,
            country_code=country_code,
            tax_type=tax_type,
            tax_year=tax_year,
            tax_rate=tax_table_data.get('tax_rate'),
            tax_brackets=tax_table_data.get('tax_brackets'),
            wage_base_limit=tax_table_data.get('wage_base_limit'),
            source=source
        )
    except Exception as e:
        logger.error(f"Error importing tax table: {str(e)}")
        raise


def update_tax_tables_annually():
    """
    Update tax tables annually (should be run as a scheduled job)
    
    Returns:
        dict: Update summary
    """
    try:
        current_year = datetime.now().year
        updated_count = 0
        
        # Get all active tax tables from previous year
        old_tables = CountryTaxConfiguration.query.filter(
            CountryTaxConfiguration.tax_year == current_year - 1,
            CountryTaxConfiguration.is_active == True
        ).all()
        
        for old_table in old_tables:
            # Create new version for current year
            create_tax_table_version(
                tenant_id=old_table.tenant_id,
                country_code=old_table.country_code,
                tax_type=old_table.tax_type,
                tax_year=current_year,
                tax_rate=float(old_table.tax_rate) if old_table.tax_rate else None,
                tax_brackets=old_table.tax_brackets,
                wage_base_limit=float(old_table.wage_base_limit) if old_table.wage_base_limit else None,
                source='annual_update'
            )
            updated_count += 1
        
        logger.info(f"Updated {updated_count} tax tables for year {current_year}")
        return {
            'updated_count': updated_count,
            'year': current_year
        }
    except Exception as e:
        logger.error(f"Error updating tax tables annually: {str(e)}")
        raise

