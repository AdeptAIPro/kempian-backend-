"""
Form 940 Generation Service
Generates Form 940 Annual Federal Unemployment Tax Return for US payroll
"""

from app import db
from app.models import (
    Form940Return, PayRun, Payslip, Tenant, User
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger
import json

logger = get_logger(__name__)

# FUTA Tax Rate (2024)
FUTA_TAX_RATE = Decimal('0.006')  # 0.6%
FUTA_WAGE_BASE = Decimal('7000')  # $7,000 per employee


def generate_form940(
    tenant_id,
    tax_year,
    ein=None
):
    """
    Generate Form 940 annual return
    
    Args:
        tenant_id: Tenant ID
        tax_year: Tax year (e.g., 2024)
        ein: Employer Identification Number (optional)
    
    Returns:
        Form940Return: Generated Form 940 record
    """
    try:
        # Get tenant
        tenant = Tenant.query.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Get all payslips for the tax year
        payslips = Payslip.query.join(
            User, Payslip.employee_id == User.id
        ).filter(
            User.tenant_id == tenant_id,
            db.extract('year', Payslip.pay_date) == tax_year
        ).all()
        
        if not payslips:
            raise ValueError(f"No payslips found for tax year {tax_year}")
        
        # Calculate total wages
        total_wages = sum(Decimal(str(ps.gross_earnings)) for ps in payslips)
        
        # Calculate FUTA tax
        # FUTA is 0.6% on first $7,000 of wages per employee
        unique_employees = set(ps.employee_id for ps in payslips)
        futa_taxable_wages = min(total_wages, FUTA_WAGE_BASE * Decimal(str(len(unique_employees))))
        futa_tax_liability = futa_taxable_wages * FUTA_TAX_RATE
        
        # Prepare form data
        form_data = {
            'tax_year': tax_year,
            'ein': ein,
            'total_wages': float(total_wages),
            'total_employees': len(unique_employees),
            'futa_taxable_wages': float(futa_taxable_wages),
            'futa_tax_rate': float(FUTA_TAX_RATE),
            'futa_tax_liability': float(futa_tax_liability),
            'payslip_ids': [ps.id for ps in payslips]
        }
        
        # Check if Form 940 already exists
        existing = Form940Return.query.filter_by(
            tenant_id=tenant_id,
            tax_year=tax_year
        ).first()
        
        if existing:
            # Update existing
            existing.total_wages = total_wages
            existing.futa_tax_liability = futa_tax_liability
            existing.form_data = form_data
            existing.ein = ein
            existing.updated_at = datetime.utcnow()
            form940 = existing
        else:
            # Create new
            form940 = Form940Return(
                tenant_id=tenant_id,
                tax_year=tax_year,
                ein=ein,
                total_wages=total_wages,
                futa_tax_liability=futa_tax_liability,
                form_data=form_data
            )
            db.session.add(form940)
        
        db.session.commit()
        
        logger.info(f"Generated Form 940 for tenant {tenant_id}, tax year {tax_year}")
        return form940
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating Form 940: {str(e)}")
        raise


def generate_form940_pdf(form940_id, output_path=None):
    """
    Generate PDF for Form 940
    
    Args:
        form940_id: Form 940 return ID
        output_path: Output file path (optional)
    
    Returns:
        str: Path to generated PDF
    """
    try:
        form940 = Form940Return.query.get(form940_id)
        if not form940:
            raise ValueError(f"Form 940 {form940_id} not found")
        
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
                f"Form940_{form940.tenant_id}_{form940.tax_year}.pdf"
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
        story.append(Paragraph("FORM 940", title_style))
        story.append(Paragraph("Employer's Annual Federal Unemployment (FUTA) Tax Return", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Form Data
        form_data = form940.form_data or {}
        form_data_table = [
            ['Tax Year', str(form940.tax_year)],
            ['EIN', form940.ein or ''],
            ['Total Wages', f"${form_data.get('total_wages', 0):,.2f}"],
            ['Total Employees', str(form_data.get('total_employees', 0))],
            ['FUTA Taxable Wages', f"${form_data.get('futa_taxable_wages', 0):,.2f}"],
            ['FUTA Tax Rate', f"{form_data.get('futa_tax_rate', 0.006)*100:.2f}%"],
            ['FUTA Tax Liability', f"${form_data.get('futa_tax_liability', 0):,.2f}"],
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
        
        # Update form940 record
        form940.pdf_path = output_path
        form940.pdf_generated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Generated Form 940 PDF: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating Form 940 PDF: {str(e)}")
        raise


def get_form940(tenant_id, tax_year):
    """
    Get Form 940 for a tenant
    
    Args:
        tenant_id: Tenant ID
        tax_year: Tax year
    
    Returns:
        Form940Return: Form 940 record or None
    """
    try:
        return Form940Return.query.filter_by(
            tenant_id=tenant_id,
            tax_year=tax_year
        ).first()
    except Exception as e:
        logger.error(f"Error fetching Form 940: {str(e)}")
        raise


def mark_form940_efiled(form940_id, acknowledgment_number, filed_by_user_id):
    """
    Mark Form 940 as e-filed
    
    Args:
        form940_id: Form 940 return ID
        acknowledgment_number: E-filing acknowledgment number
        filed_by_user_id: User ID who filed
    """
    try:
        form940 = Form940Return.query.get(form940_id)
        if not form940:
            raise ValueError(f"Form 940 {form940_id} not found")
        
        form940.e_filing_status = 'submitted'
        form940.e_filing_acknowledgment = acknowledgment_number
        form940.e_filed_at = datetime.utcnow()
        form940.e_filed_by = filed_by_user_id
        
        db.session.commit()
        logger.info(f"Form 940 {form940_id} marked as e-filed with acknowledgment {acknowledgment_number}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error marking Form 940 as e-filed: {str(e)}")
        raise

