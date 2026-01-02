"""
Form 941 Generation Service
Generates Form 941 Quarterly Federal Tax Return for US payroll
"""

from app import db
from app.models import (
    Form941Return, PayRun, Payslip, Tenant, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger
import json

logger = get_logger(__name__)


def generate_form941(
    tenant_id,
    quarter,
    tax_year,
    ein=None
):
    """
    Generate Form 941 quarterly return
    
    Args:
        tenant_id: Tenant ID
        quarter: Quarter (1, 2, 3, 4)
        tax_year: Tax year (e.g., 2024)
        ein: Employer Identification Number (optional)
    
    Returns:
        Form941Return: Generated Form 941 record
    """
    try:
        # Validate quarter
        if quarter not in [1, 2, 3, 4]:
            raise ValueError("Quarter must be 1, 2, 3, or 4")
        
        # Get tenant
        tenant = Tenant.query.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Calculate quarter date range
        quarter_months = {
            1: [1, 2, 3],   # Q1: Jan, Feb, Mar
            2: [4, 5, 6],   # Q2: Apr, May, Jun
            3: [7, 8, 9],   # Q3: Jul, Aug, Sep
            4: [10, 11, 12]  # Q4: Oct, Nov, Dec
        }
        
        months = quarter_months[quarter]
        
        # Get all payslips for the quarter
        # Join through PayRun to get tenant_id
        payslips = Payslip.query.join(
            User, Payslip.employee_id == User.id
        ).filter(
            User.tenant_id == tenant_id,
            db.extract('year', Payslip.pay_date) == tax_year,
            db.extract('month', Payslip.pay_date).in_(months)
        ).all()
        
        if not payslips:
            raise ValueError(f"No payslips found for quarter {quarter}, tax year {tax_year}")
        
        # Calculate totals
        total_wages = sum(Decimal(str(ps.gross_earnings)) for ps in payslips)
        
        # Calculate federal income tax withheld
        federal_income_tax_withheld = sum(Decimal(str(ps.tax_deduction)) for ps in payslips)
        
        # Calculate Social Security tax (6.2% up to wage base)
        from app.hr.irs_tax_tables import FICA_SOCIAL_SECURITY_RATE, FICA_SOCIAL_SECURITY_WAGE_BASE
        social_security_tax = Decimal('0')
        for ps in payslips:
            wages = Decimal(str(ps.gross_earnings))
            taxable_wages = min(wages, FICA_SOCIAL_SECURITY_WAGE_BASE)
            social_security_tax += taxable_wages * FICA_SOCIAL_SECURITY_RATE
        
        # Calculate Medicare tax (1.45% on all wages)
        from app.hr.irs_tax_tables import FICA_MEDICARE_RATE
        medicare_tax = total_wages * FICA_MEDICARE_RATE
        
        # Total tax liability
        total_tax_liability = federal_income_tax_withheld + social_security_tax + medicare_tax
        
        # Prepare form data
        return_type = f"Q{quarter}"
        form_data = {
            'quarter': quarter,
            'tax_year': tax_year,
            'ein': ein,
            'total_wages': float(total_wages),
            'federal_income_tax_withheld': float(federal_income_tax_withheld),
            'social_security_tax': float(social_security_tax),
            'medicare_tax': float(medicare_tax),
            'total_tax_liability': float(total_tax_liability),
            'total_employees': len(set(ps.employee_id for ps in payslips)),
            'payslip_ids': [ps.id for ps in payslips]
        }
        
        # Check if Form 941 already exists
        existing = Form941Return.query.filter_by(
            tenant_id=tenant_id,
            quarter=quarter,
            tax_year=tax_year
        ).first()
        
        if existing:
            # Update existing
            existing.total_wages = total_wages
            existing.federal_income_tax_withheld = federal_income_tax_withheld
            existing.social_security_tax = social_security_tax
            existing.medicare_tax = medicare_tax
            existing.total_tax_liability = total_tax_liability
            existing.form_data = form_data
            existing.ein = ein
            existing.updated_at = datetime.utcnow()
            form941 = existing
        else:
            # Create new
            form941 = Form941Return(
                tenant_id=tenant_id,
                return_type=return_type,
                tax_year=tax_year,
                quarter=quarter,
                ein=ein,
                total_wages=total_wages,
                federal_income_tax_withheld=federal_income_tax_withheld,
                social_security_tax=social_security_tax,
                medicare_tax=medicare_tax,
                total_tax_liability=total_tax_liability,
                form_data=form_data
            )
            db.session.add(form941)
        
        db.session.commit()
        
        logger.info(f"Generated Form 941 for tenant {tenant_id}, Q{quarter}, tax year {tax_year}")
        return form941
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating Form 941: {str(e)}")
        raise


def generate_form941_pdf(form941_id, output_path=None):
    """
    Generate PDF for Form 941
    
    Args:
        form941_id: Form 941 return ID
        output_path: Output file path (optional)
    
    Returns:
        str: Path to generated PDF
    """
    try:
        form941 = Form941Return.query.get(form941_id)
        if not form941:
            raise ValueError(f"Form 941 {form941_id} not found")
        
        # Import PDF generation library
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            logger.warning("reportlab not installed, PDF generation skipped")
            return None
        
        # Create output path if not provided
        if not output_path:
            import os
            output_dir = os.path.join(os.getcwd(), 'generated_forms')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"Form941_{form941.tenant_id}_Q{form941.quarter}_{form941.tax_year}.pdf"
            )
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#1a1a1a'),
            alignment=TA_CENTER,
            spaceAfter=20
        )
        story.append(Paragraph("FORM 941", title_style))
        story.append(Paragraph("Employer's Quarterly Federal Tax Return", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Form Data
        form_data = form941.form_data or {}
        form_data_table = [
            ['Quarter', f"Q{form941.quarter}"],
            ['Tax Year', str(form941.tax_year)],
            ['EIN', form941.ein or ''],
            ['Total Wages', f"${form_data.get('total_wages', 0):,.2f}"],
            ['Federal Income Tax Withheld', f"${form_data.get('federal_income_tax_withheld', 0):,.2f}"],
            ['Social Security Tax', f"${form_data.get('social_security_tax', 0):,.2f}"],
            ['Medicare Tax', f"${form_data.get('medicare_tax', 0):,.2f}"],
            ['Total Tax Liability', f"${form_data.get('total_tax_liability', 0):,.2f}"],
            ['Total Employees', str(form_data.get('total_employees', 0))],
        ]
        
        form_table = Table(form_data_table, colWidths=[3*inch, 3*inch])
        form_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(form_table)
        
        # Build PDF
        doc.build(story)
        
        # Update form941 record
        form941.pdf_path = output_path
        form941.pdf_generated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Generated Form 941 PDF: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating Form 941 PDF: {str(e)}")
        raise


def get_form941(tenant_id, quarter, tax_year):
    """
    Get Form 941 for a tenant
    
    Args:
        tenant_id: Tenant ID
        quarter: Quarter (1, 2, 3, 4)
        tax_year: Tax year
    
    Returns:
        Form941Return: Form 941 record or None
    """
    try:
        return Form941Return.query.filter_by(
            tenant_id=tenant_id,
            quarter=quarter,
            tax_year=tax_year
        ).first()
    except Exception as e:
        logger.error(f"Error fetching Form 941: {str(e)}")
        raise


def mark_form941_efiled(form941_id, acknowledgment_number, filed_by_user_id):
    """
    Mark Form 941 as e-filed
    
    Args:
        form941_id: Form 941 return ID
        acknowledgment_number: E-filing acknowledgment number
        filed_by_user_id: User ID who filed
    """
    try:
        form941 = Form941Return.query.get(form941_id)
        if not form941:
            raise ValueError(f"Form 941 {form941_id} not found")
        
        form941.e_filing_status = 'submitted'
        form941.e_filing_acknowledgment = acknowledgment_number
        form941.e_filed_at = datetime.utcnow()
        form941.e_filed_by = filed_by_user_id
        
        db.session.commit()
        logger.info(f"Form 941 {form941_id} marked as e-filed with acknowledgment {acknowledgment_number}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error marking Form 941 as e-filed: {str(e)}")
        raise

