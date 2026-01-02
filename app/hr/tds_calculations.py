"""
TDS (Tax Deducted at Source) Calculation Service
Handles income tax calculations for India payroll (old and new regime)
"""

from app import db
from app.models import (
    TDSRecord, IncomeTaxExemption, EmployeeProfile, Payslip, PayRun, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger

logger = get_logger(__name__)

# Income Tax Slabs - Old Regime (FY 2024-25)
OLD_REGIME_SLABS = [
    (0, 250000, 0, 0),           # 0% - Up to ₹2.5L
    (250001, 500000, 0.05, 0),   # 5% - ₹2.5L to ₹5L
    (500001, 1000000, 0.20, 12500),  # 20% - ₹5L to ₹10L (with rebate)
    (1000001, float('inf'), 0.30, 112500)  # 30% - Above ₹10L (with rebate)
]

# Income Tax Slabs - New Regime (FY 2024-25)
NEW_REGIME_SLABS = [
    (0, 300000, 0, 0),           # 0% - Up to ₹3L
    (300001, 700000, 0.05, 0),   # 5% - ₹3L to ₹7L
    (700001, 1000000, 0.10, 20000),  # 10% - ₹7L to ₹10L (with rebate)
    (1000001, 1200000, 0.15, 50000),  # 15% - ₹10L to ₹12L (with rebate)
    (1200001, 1500000, 0.20, 90000),  # 20% - ₹12L to ₹15L (with rebate)
    (1500001, float('inf'), 0.30, 150000)  # 30% - Above ₹15L (with rebate)
]

# Standard Deduction
STANDARD_DEDUCTION = Decimal('50000')  # ₹50,000

# Section 80C Maximum
SECTION_80C_MAX = Decimal('150000')  # ₹1.5L


def calculate_income_tax_old_regime(taxable_income, exemptions=None):
    """
    Calculate income tax using old regime
    
    Args:
        taxable_income: Taxable income after deductions
        exemptions: IncomeTaxExemption object (optional)
    
    Returns:
        dict: Tax calculation details
    """
    try:
        taxable_income = Decimal(str(taxable_income))
        total_tax = Decimal('0')
        
        # Apply standard deduction
        if exemptions and exemptions.standard_deduction:
            taxable_income = max(Decimal('0'), taxable_income - exemptions.standard_deduction)
        else:
            taxable_income = max(Decimal('0'), taxable_income - STANDARD_DEDUCTION)
        
        # Calculate tax using slabs
        remaining_income = taxable_income
        for min_income, max_income, rate, rebate in OLD_REGIME_SLABS:
            if remaining_income <= 0:
                break
            
            if max_income == float('inf'):
                slab_income = remaining_income
            else:
                slab_income = min(remaining_income, Decimal(str(max_income)) - Decimal(str(min_income)) + Decimal('1'))
            
            if slab_income > 0:
                tax_on_slab = (slab_income * Decimal(str(rate))).quantize(Decimal('0.01'))
                total_tax += tax_on_slab
                remaining_income -= slab_income
        
        # Apply rebate (Section 87A - up to ₹12,500 for income up to ₹5L)
        if taxable_income <= 500000:
            rebate = min(total_tax, Decimal('12500'))
            total_tax = max(Decimal('0'), total_tax - rebate)
        
        # Add cess (4% of tax)
        cess = (total_tax * Decimal('0.04')).quantize(Decimal('0.01'))
        total_tax_with_cess = total_tax + cess
        
        return {
            'taxable_income': float(taxable_income),
            'tax_amount': float(total_tax),
            'cess': float(cess),
            'total_tax': float(total_tax_with_cess),
            'regime': 'old'
        }
    except Exception as e:
        logger.error(f"Error calculating old regime tax: {str(e)}")
        raise


def calculate_income_tax_new_regime(taxable_income):
    """
    Calculate income tax using new regime
    
    Args:
        taxable_income: Taxable income
    
    Returns:
        dict: Tax calculation details
    """
    try:
        taxable_income = Decimal(str(taxable_income))
        total_tax = Decimal('0')
        
        # New regime has built-in standard deduction
        # No separate standard deduction needed
        
        # Calculate tax using slabs
        remaining_income = taxable_income
        for min_income, max_income, rate, rebate in NEW_REGIME_SLABS:
            if remaining_income <= 0:
                break
            
            if max_income == float('inf'):
                slab_income = remaining_income
            else:
                slab_income = min(remaining_income, Decimal(str(max_income)) - Decimal(str(min_income)) + Decimal('1'))
            
            if slab_income > 0:
                tax_on_slab = (slab_income * Decimal(str(rate))).quantize(Decimal('0.01'))
                total_tax += tax_on_slab
                remaining_income -= slab_income
        
        # Apply rebate (Section 87A - up to ₹25,000 for income up to ₹7L)
        if taxable_income <= 700000:
            rebate = min(total_tax, Decimal('25000'))
            total_tax = max(Decimal('0'), total_tax - rebate)
        
        # Add cess (4% of tax)
        cess = (total_tax * Decimal('0.04')).quantize(Decimal('0.01'))
        total_tax_with_cess = total_tax + cess
        
        return {
            'taxable_income': float(taxable_income),
            'tax_amount': float(total_tax),
            'cess': float(cess),
            'total_tax': float(total_tax_with_cess),
            'regime': 'new'
        }
    except Exception as e:
        logger.error(f"Error calculating new regime tax: {str(e)}")
        raise


def calculate_tds(
    gross_salary,
    employee_id,
    regime='old',
    exemptions=None
):
    """
    Calculate TDS based on gross salary and exemptions
    
    Args:
        gross_salary: Gross salary
        employee_id: Employee user ID
        regime: Tax regime ('old' or 'new')
        exemptions: IncomeTaxExemption object (optional)
    
    Returns:
        dict: TDS calculation details
    """
    try:
        gross_salary = Decimal(str(gross_salary))
        
        # Get exemptions if not provided
        if not exemptions:
            exemptions = IncomeTaxExemption.query.filter_by(employee_id=employee_id).first()
        
        # Calculate taxable income
        taxable_income = gross_salary
        
        # Apply exemptions (only for old regime)
        if regime == 'old' and exemptions:
            # HRA exemption
            if exemptions.hra_exemption_amount:
                taxable_income -= exemptions.hra_exemption_amount
            
            # LTA exemption
            if exemptions.lta_exemption_amount:
                taxable_income -= exemptions.lta_exemption_amount
            
            # Section 80C
            if exemptions.section_80c_amount:
                taxable_income -= min(exemptions.section_80c_amount, SECTION_80C_MAX)
            
            # Section 80D
            if exemptions.section_80d_amount:
                taxable_income -= exemptions.section_80d_amount
            
            # Section 80G
            if exemptions.section_80g_amount:
                taxable_income -= exemptions.section_80g_amount
            
            # Section 24 (Home loan interest)
            if exemptions.section_24_amount:
                taxable_income -= exemptions.section_24_amount
        
        # Calculate tax based on regime
        if regime == 'old':
            tax_calc = calculate_income_tax_old_regime(taxable_income, exemptions)
        else:
            tax_calc = calculate_income_tax_new_regime(taxable_income)
        
        # TDS is tax divided by 12 (monthly)
        monthly_tds = (Decimal(str(tax_calc['total_tax'])) / Decimal('12')).quantize(Decimal('0.01'))
        
        return {
            'gross_salary': float(gross_salary),
            'taxable_income': tax_calc['taxable_income'],
            'annual_tax': tax_calc['total_tax'],
            'monthly_tds': float(monthly_tds),
            'regime': regime,
            'tax_breakdown': tax_calc
        }
    except Exception as e:
        logger.error(f"Error calculating TDS: {str(e)}")
        raise


def create_tds_record(
    employee_id,
    gross_salary,
    payslip_id=None,
    pay_run_id=None,
    tds_month=None,
    tds_year=None,
    tan_number=None
):
    """
    Create a TDS record in the database
    
    Args:
        employee_id: Employee user ID
        gross_salary: Gross salary
        payslip_id: Payslip ID (optional)
        pay_run_id: Pay run ID (optional)
        tds_month: Month (1-12), defaults to current month
        tds_year: Year, defaults to current year
        tan_number: TAN number (optional)
    
    Returns:
        TDSRecord: Created record
    """
    try:
        # Get employee profile
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        employee_profile_id = employee_profile.id if employee_profile else None
        
        # Get tax regime
        regime = employee_profile.tax_regime if employee_profile and employee_profile.tax_regime else 'old'
        
        # Get exemptions
        exemptions = IncomeTaxExemption.query.filter_by(employee_id=employee_id).first()
        
        # Calculate TDS
        tds_calc = calculate_tds(gross_salary, employee_id, regime, exemptions)
        
        # Set default month/year
        if not tds_month:
            tds_month = datetime.now().month
        if not tds_year:
            tds_year = datetime.now().year
        
        # Create record
        tds_record = TDSRecord(
            employee_id=employee_id,
            employee_profile_id=employee_profile_id,
            payslip_id=payslip_id,
            pay_run_id=pay_run_id,
            gross_salary=Decimal(str(tds_calc['gross_salary'])),
            taxable_income=Decimal(str(tds_calc['taxable_income'])),
            tds_amount=Decimal(str(tds_calc['monthly_tds'])),
            tan_number=tan_number,
            tds_month=tds_month,
            tds_year=tds_year
        )
        
        db.session.add(tds_record)
        db.session.commit()
        
        logger.info(f"Created TDS record for employee {employee_id}")
        return tds_record
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating TDS record: {str(e)}")
        raise


def get_tds_records_by_employee(employee_id, year=None, month=None):
    """
    Get TDS records for an employee
    
    Args:
        employee_id: Employee user ID
        year: Year filter (optional)
        month: Month filter (optional)
    
    Returns:
        list: List of TDSRecord records
    """
    try:
        query = TDSRecord.query.filter_by(employee_id=employee_id)
        
        if year:
            query = query.filter_by(tds_year=year)
        if month:
            query = query.filter_by(tds_month=month)
        
        return query.order_by(
            TDSRecord.tds_year.desc(),
            TDSRecord.tds_month.desc()
        ).all()
    except Exception as e:
        logger.error(f"Error fetching TDS records: {str(e)}")
        raise

