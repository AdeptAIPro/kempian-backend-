"""
Challan Generation Service
Generates challans for various payments (TDS, PF, ESI, PT, LWF) for India payroll
"""

from app import db
from app.models import (
    ChallanRecord, TDSRecord, ProvidentFundContribution, ESIContribution,
    ProfessionalTaxDeduction, PayRun, Tenant
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger
import json

logger = get_logger(__name__)


def generate_challan_281(
    tenant_id,
    payment_month,
    payment_year,
    tan_number=None
):
    """
    Generate Challan 281 for TDS payment
    
    Args:
        tenant_id: Tenant ID
        payment_month: Payment month (1-12)
        payment_year: Payment year
        tan_number: TAN number (optional)
    
    Returns:
        ChallanRecord: Generated challan record
    """
    try:
        # Get all TDS records for the month (join through User to get tenant_id)
        tds_records = TDSRecord.query.join(
            User, TDSRecord.employee_id == User.id
        ).filter(
            User.tenant_id == tenant_id,
            TDSRecord.tds_month == payment_month,
            TDSRecord.tds_year == payment_year
        ).all()
        
        if not tds_records:
            raise ValueError(f"No TDS records found for month {payment_month}, year {payment_year}")
        
        # Calculate total TDS
        total_tds = sum(Decimal(str(record.tds_amount)) for record in tds_records)
        
        # Get TAN from first record if not provided
        if not tan_number and tds_records:
            tan_number = tds_records[0].tan_number
        
        # Generate challan number
        challan_number = f"CH281-{tenant_id}-{payment_year}{payment_month:02d}-{datetime.now().strftime('%H%M%S')}"
        challan_date = datetime(payment_year, payment_month, 1).date()
        
        # Prepare challan data
        challan_data = {
            'challan_type': '281',
            'tan': tan_number,
            'tds_records': [
                {
                    'employee_id': record.employee_id,
                    'tds_amount': float(record.tds_amount),
                    'payslip_id': record.payslip_id
                }
                for record in tds_records
            ],
            'total_records': len(tds_records)
        }
        
        # Create challan record
        challan = ChallanRecord(
            tenant_id=tenant_id,
            challan_type='281',
            challan_number=challan_number,
            challan_date=challan_date,
            amount=total_tds,
            payment_mode='online',
            payment_status='pending',
            payment_month=payment_month,
            payment_year=payment_year,
            challan_data=challan_data
        )
        
        db.session.add(challan)
        db.session.commit()
        
        # Update TDS records with challan number
        for record in tds_records:
            record.challan_281_number = challan_number
            record.challan_281_date = challan_date
        
        db.session.commit()
        
        logger.info(f"Generated Challan 281: {challan_number} for tenant {tenant_id}")
        return challan
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating Challan 281: {str(e)}")
        raise


def generate_epf_challan(
    tenant_id,
    payment_month,
    payment_year,
    pay_run_id=None
):
    """
    Generate EPF challan for PF contributions
    
    Args:
        tenant_id: Tenant ID
        payment_month: Payment month (1-12)
        payment_year: Payment year
        pay_run_id: Pay run ID (optional)
    
    Returns:
        ChallanRecord: Generated challan record
    """
    try:
        # Get all PF contributions for the month
        if pay_run_id:
            pf_contributions = ProvidentFundContribution.query.filter_by(
                pay_run_id=pay_run_id
            ).all()
        else:
            pf_contributions = ProvidentFundContribution.query.join(
                PayRun
            ).filter(
                PayRun.tenant_id == tenant_id,
                ProvidentFundContribution.contribution_month == payment_month,
                ProvidentFundContribution.contribution_year == payment_year
            ).all()
        
        if not pf_contributions:
            raise ValueError(f"No PF contributions found for month {payment_month}, year {payment_year}")
        
        # Calculate totals
        total_employee_pf = sum(Decimal(str(contrib.employee_pf)) for contrib in pf_contributions)
        total_employer_pf = sum(Decimal(str(contrib.employer_pf)) for contrib in pf_contributions)
        total_pf = total_employee_pf + total_employer_pf
        
        # Generate challan number
        challan_number = f"EPF-{tenant_id}-{payment_year}{payment_month:02d}-{datetime.now().strftime('%H%M%S')}"
        challan_date = datetime(payment_year, payment_month, 1).date()
        
        # Prepare challan data
        challan_data = {
            'challan_type': 'EPF',
            'employee_pf': float(total_employee_pf),
            'employer_pf': float(total_employer_pf),
            'total_pf': float(total_pf),
            'contributions': [
                {
                    'employee_id': contrib.employee_id,
                    'employee_pf': float(contrib.employee_pf),
                    'employer_pf': float(contrib.employer_pf),
                    'uan_number': contrib.uan_number
                }
                for contrib in pf_contributions
            ],
            'total_contributions': len(pf_contributions)
        }
        
        # Create challan record
        challan = ChallanRecord(
            tenant_id=tenant_id,
            pay_run_id=pay_run_id,
            challan_type='EPF',
            challan_number=challan_number,
            challan_date=challan_date,
            amount=total_pf,
            payment_mode='online',
            payment_status='pending',
            payment_month=payment_month,
            payment_year=payment_year,
            challan_data=challan_data
        )
        
        db.session.add(challan)
        db.session.commit()
        
        # Update PF contributions with challan number
        for contrib in pf_contributions:
            contrib.challan_number = challan_number
            contrib.challan_date = challan_date
        
        db.session.commit()
        
        logger.info(f"Generated EPF Challan: {challan_number} for tenant {tenant_id}")
        return challan
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating EPF challan: {str(e)}")
        raise


def generate_esi_challan(
    tenant_id,
    payment_month,
    payment_year,
    pay_run_id=None
):
    """
    Generate ESI challan for ESI contributions
    
    Args:
        tenant_id: Tenant ID
        payment_month: Payment month (1-12)
        payment_year: Payment year
        pay_run_id: Pay run ID (optional)
    
    Returns:
        ChallanRecord: Generated challan record
    """
    try:
        # Get all ESI contributions for the month
        if pay_run_id:
            esi_contributions = ESIContribution.query.filter_by(
                pay_run_id=pay_run_id
            ).all()
        else:
            esi_contributions = ESIContribution.query.join(
                PayRun
            ).filter(
                PayRun.tenant_id == tenant_id,
                ESIContribution.contribution_month == payment_month,
                ESIContribution.contribution_year == payment_year
            ).all()
        
        if not esi_contributions:
            raise ValueError(f"No ESI contributions found for month {payment_month}, year {payment_year}")
        
        # Calculate totals
        total_employee_esi = sum(Decimal(str(contrib.employee_esi)) for contrib in esi_contributions)
        total_employer_esi = sum(Decimal(str(contrib.employer_esi)) for contrib in esi_contributions)
        total_esi = total_employee_esi + total_employer_esi
        
        # Generate challan number
        challan_number = f"ESI-{tenant_id}-{payment_year}{payment_month:02d}-{datetime.now().strftime('%H%M%S')}"
        challan_date = datetime(payment_year, payment_month, 1).date()
        
        # Prepare challan data
        challan_data = {
            'challan_type': 'ESI',
            'employee_esi': float(total_employee_esi),
            'employer_esi': float(total_employer_esi),
            'total_esi': float(total_esi),
            'contributions': [
                {
                    'employee_id': contrib.employee_id,
                    'employee_esi': float(contrib.employee_esi),
                    'employer_esi': float(contrib.employer_esi),
                    'esi_card_number': contrib.esi_card_number
                }
                for contrib in esi_contributions
            ],
            'total_contributions': len(esi_contributions)
        }
        
        # Create challan record
        challan = ChallanRecord(
            tenant_id=tenant_id,
            pay_run_id=pay_run_id,
            challan_type='ESI',
            challan_number=challan_number,
            challan_date=challan_date,
            amount=total_esi,
            payment_mode='online',
            payment_status='pending',
            payment_month=payment_month,
            payment_year=payment_year,
            challan_data=challan_data
        )
        
        db.session.add(challan)
        db.session.commit()
        
        # Update ESI contributions with challan number
        for contrib in esi_contributions:
            contrib.challan_number = challan_number
            contrib.challan_date = challan_date
        
        db.session.commit()
        
        logger.info(f"Generated ESI Challan: {challan_number} for tenant {tenant_id}")
        return challan
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating ESI challan: {str(e)}")
        raise


def generate_pt_challan(
    tenant_id,
    payment_month,
    payment_year,
    state_code
):
    """
    Generate Professional Tax challan
    
    Args:
        tenant_id: Tenant ID
        payment_month: Payment month (1-12)
        payment_year: Payment year
        state_code: State code
    
    Returns:
        ChallanRecord: Generated challan record
    """
    try:
        # Get all PT deductions for the month and state
        pt_deductions = ProfessionalTaxDeduction.query.filter_by(
            state_code=state_code,
            deduction_month=payment_month,
            deduction_year=payment_year
        ).join(
            EmployeeProfile
        ).join(
            User
        ).filter(
            User.tenant_id == tenant_id
        ).all()
        
        if not pt_deductions:
            raise ValueError(f"No PT deductions found for state {state_code}, month {payment_month}, year {payment_year}")
        
        # Calculate total PT
        total_pt = sum(Decimal(str(ded.professional_tax_amount)) for ded in pt_deductions)
        
        # Generate challan number
        challan_number = f"PT-{state_code}-{tenant_id}-{payment_year}{payment_month:02d}-{datetime.now().strftime('%H%M%S')}"
        challan_date = datetime(payment_year, payment_month, 1).date()
        
        # Prepare challan data
        challan_data = {
            'challan_type': 'PT',
            'state_code': state_code,
            'deductions': [
                {
                    'employee_id': ded.employee_id,
                    'pt_amount': float(ded.professional_tax_amount),
                    'gross_salary': float(ded.gross_salary)
                }
                for ded in pt_deductions
            ],
            'total_deductions': len(pt_deductions)
        }
        
        # Create challan record
        challan = ChallanRecord(
            tenant_id=tenant_id,
            challan_type='PT',
            challan_number=challan_number,
            challan_date=challan_date,
            amount=total_pt,
            payment_mode='online',
            payment_status='pending',
            payment_month=payment_month,
            payment_year=payment_year,
            challan_data=challan_data
        )
        
        db.session.add(challan)
        db.session.commit()
        
        logger.info(f"Generated PT Challan: {challan_number} for tenant {tenant_id}, state {state_code}")
        return challan
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating PT challan: {str(e)}")
        raise


def get_challans_by_tenant(tenant_id, challan_type=None, year=None, month=None):
    """
    Get challans for a tenant
    
    Args:
        tenant_id: Tenant ID
        challan_type: Challan type filter (optional)
        year: Year filter (optional)
        month: Month filter (optional)
    
    Returns:
        list: List of ChallanRecord records
    """
    try:
        query = ChallanRecord.query.filter_by(tenant_id=tenant_id)
        
        if challan_type:
            query = query.filter_by(challan_type=challan_type)
        if year:
            query = query.filter_by(payment_year=year)
        if month:
            query = query.filter_by(payment_month=month)
        
        return query.order_by(
            ChallanRecord.payment_year.desc(),
            ChallanRecord.payment_month.desc()
        ).all()
    except Exception as e:
        logger.error(f"Error fetching challans: {str(e)}")
        raise


def update_challan_payment_status(challan_id, payment_status, payment_reference=None):
    """
    Update challan payment status
    
    Args:
        challan_id: Challan record ID
        payment_status: Payment status ('pending', 'paid', 'failed')
        payment_reference: Payment reference number (optional)
    """
    try:
        challan = ChallanRecord.query.get(challan_id)
        if not challan:
            raise ValueError(f"Challan {challan_id} not found")
        
        challan.payment_status = payment_status
        if payment_reference:
            challan.payment_reference = payment_reference
        if payment_status == 'paid':
            challan.paid_at = datetime.utcnow()
        
        db.session.commit()
        logger.info(f"Updated challan {challan_id} payment status to {payment_status}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating challan payment status: {str(e)}")
        raise

