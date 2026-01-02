"""
Form 16 Generation Service
Generates Form 16 TDS Certificate (Part A & Part B) for India payroll
"""

from app import db
from app.models import (
    Form16Certificate, TDSRecord, EmployeeProfile, User, Tenant, IncomeTaxExemption, OrganizationMetadata
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger
import json

logger = get_logger(__name__)


def generate_form16(
    employee_id,
    financial_year,
    tenant_id=None,
    tan_number=None
):
    """
    Generate Form 16 certificate for an employee
    
    Args:
        employee_id: Employee user ID
        financial_year: Financial year (e.g., 2024)
        tenant_id: Tenant ID (optional, will be fetched from employee)
        tan_number: TAN number (optional)
    
    Returns:
        Form16Certificate: Generated Form 16 record
    """
    try:
        # Get employee
        employee = User.query.get(employee_id)
        if not employee:
            raise ValueError(f"Employee {employee_id} not found")
        
        # Get employee profile
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        if not employee_profile:
            raise ValueError(f"Employee profile not found for employee {employee_id}")
        
        # Get tenant ID
        if not tenant_id:
            tenant_id = employee.tenant_id
        
        # Get tenant
        tenant = Tenant.query.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Get all TDS records for the financial year
        tds_records = TDSRecord.query.filter_by(
            employee_id=employee_id,
            tds_year=financial_year
        ).all()
        
        if not tds_records:
            raise ValueError(f"No TDS records found for employee {employee_id} for financial year {financial_year}")
        
        # Calculate totals
        total_tds_deducted = sum(Decimal(str(record.tds_amount)) for record in tds_records)
        total_gross_salary = sum(Decimal(str(record.gross_salary)) for record in tds_records)
        
        # Get exemptions
        exemptions = IncomeTaxExemption.query.filter_by(
            employee_id=employee_id,
            financial_year=financial_year
        ).first()
        
        # Calculate taxable income
        taxable_income = total_gross_salary
        
        if exemptions:
            # Apply exemptions (old regime only)
            if employee_profile.tax_regime == 'old':
                # HRA exemption
                if exemptions.hra_exemption_amount:
                    taxable_income -= exemptions.hra_exemption_amount
                
                # LTA exemption
                if exemptions.lta_exemption_amount:
                    taxable_income -= exemptions.lta_exemption_amount
                
                # Section 80C
                if exemptions.section_80c_amount:
                    taxable_income -= min(exemptions.section_80c_amount, Decimal('150000'))
                
                # Section 80D
                if exemptions.section_80d_amount:
                    taxable_income -= exemptions.section_80d_amount
                
                # Section 80G
                if exemptions.section_80g_amount:
                    taxable_income -= exemptions.section_80g_amount
                
                # Section 24
                if exemptions.section_24_amount:
                    taxable_income -= exemptions.section_24_amount
                
                # Standard deduction
                if exemptions.standard_deduction:
                    taxable_income -= exemptions.standard_deduction
                else:
                    taxable_income -= Decimal('50000')
        
        # Calculate total tax payable (annual)
        from app.hr.tds_calculations import calculate_tds
        tax_calc = calculate_tds(
            float(total_gross_salary),
            employee_id,
            employee_profile.tax_regime or 'old',
            exemptions
        )
        
        total_tax_payable = Decimal(str(tax_calc['annual_tax']))
        tax_paid = total_tds_deducted * Decimal('12')  # Monthly TDS * 12
        
        # Calculate refund
        refund_amount = max(Decimal('0'), tax_paid - total_tax_payable)
        
        # Get employer details from tenant/organization
        org_metadata = OrganizationMetadata.query.filter_by(tenant_id=tenant_id).first()
        
        employer_name = org_metadata.name if org_metadata else "Company Name"
        employer_address = org_metadata.description if org_metadata else "Company Address"
        
        # Get TAN from first TDS record if not provided
        if not tan_number and tds_records:
            tan_number = tds_records[0].tan_number
        
        # Generate certificate number
        certificate_number = f"FORM16-{employee_id}-{financial_year}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Prepare Part A data (TDS Details)
        part_a_data = {
            'tan': tan_number,
            'employer_name': employer_name,
            'employer_address': employer_address,
            'employee_pan': employee_profile.pan_number,
            'employee_name': f"{employee_profile.first_name or ''} {employee_profile.last_name or ''}".strip(),
            'assessment_year': financial_year + 1,
            'total_tds_deducted': float(total_tds_deducted),
            'tds_details': [
                {
                    'month': record.tds_month,
                    'tds_amount': float(record.tds_amount),
                    'challan_number': record.challan_281_number,
                    'challan_date': record.challan_281_date.isoformat() if record.challan_281_date else None
                }
                for record in tds_records
            ]
        }
        
        # Prepare Part B data (Salary Details)
        part_b_data = {
            'gross_salary': float(total_gross_salary),
            'allowances': {
                'hra_received': float(exemptions.hra_received) if exemptions and exemptions.hra_received else 0,
                'lta_received': float(exemptions.lta_received) if exemptions and exemptions.lta_received else 0,
            },
            'deductions': {
                'standard_deduction': float(exemptions.standard_deduction) if exemptions and exemptions.standard_deduction else 50000,
                'section_80c': float(exemptions.section_80c_amount) if exemptions and exemptions.section_80c_amount else 0,
                'section_80d': float(exemptions.section_80d_amount) if exemptions and exemptions.section_80d_amount else 0,
                'section_80g': float(exemptions.section_80g_amount) if exemptions and exemptions.section_80g_amount else 0,
                'section_24': float(exemptions.section_24_amount) if exemptions and exemptions.section_24_amount else 0,
                'hra_exemption': float(exemptions.hra_exemption_amount) if exemptions and exemptions.hra_exemption_amount else 0,
                'lta_exemption': float(exemptions.lta_exemption_amount) if exemptions and exemptions.lta_exemption_amount else 0,
            },
            'total_deductions': float(total_gross_salary - taxable_income),
            'taxable_income': float(taxable_income),
            'total_tax_payable': float(total_tax_payable),
            'tax_paid': float(tax_paid),
            'refund_amount': float(refund_amount),
            'tax_regime': employee_profile.tax_regime or 'old'
        }
        
        # Prepare Annexure (Detailed breakdown)
        annexure_data = {
            'salary_breakdown': {
                'basic_salary': float(total_gross_salary * Decimal('0.5')),  # Estimate 50% as basic
                'hra': float(exemptions.hra_received) if exemptions and exemptions.hra_received else 0,
                'lta': float(exemptions.lta_received) if exemptions and exemptions.lta_received else 0,
                'other_allowances': float(total_gross_salary * Decimal('0.3')),  # Estimate
            },
            'exemptions_breakdown': part_b_data['deductions'],
            'tax_calculation': {
                'annual_tax': float(total_tax_payable),
                'cess': float(total_tax_payable * Decimal('0.04')),
                'total_tax': float(total_tax_payable * Decimal('1.04'))
            }
        }
        
        # Create Form 16 record
        form16 = Form16Certificate(
            employee_id=employee_id,
            employee_profile_id=employee_profile.id,
            tenant_id=tenant_id,
            certificate_number=certificate_number,
            financial_year=financial_year,
            tan_number=tan_number,
            employer_name=employer_name,
            employer_address=employer_address,
            employee_pan=employee_profile.pan_number,
            total_tds_deducted=total_tds_deducted,
            gross_salary=total_gross_salary,
            total_deductions=total_gross_salary - taxable_income,
            taxable_income=taxable_income,
            total_tax_payable=total_tax_payable,
            tax_paid=tax_paid,
            refund_amount=refund_amount,
            part_a_data=part_a_data,
            part_b_data=part_b_data,
            annexure_data=annexure_data
        )
        
        db.session.add(form16)
        db.session.commit()
        
        logger.info(f"Generated Form 16 for employee {employee_id}, financial year {financial_year}")
        return form16
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating Form 16: {str(e)}")
        raise


def get_form16(employee_id, financial_year):
    """
    Get Form 16 for an employee
    
    Args:
        employee_id: Employee user ID
        financial_year: Financial year
    
    Returns:
        Form16Certificate: Form 16 record or None
    """
    try:
        return Form16Certificate.query.filter_by(
            employee_id=employee_id,
            financial_year=financial_year
        ).first()
    except Exception as e:
        logger.error(f"Error fetching Form 16: {str(e)}")
        raise


def generate_form16_pdf(form16_id, output_path=None):
    """
    Generate PDF for Form 16
    
    Args:
        form16_id: Form 16 certificate ID
        output_path: Output file path (optional)
    
    Returns:
        str: Path to generated PDF
    """
    try:
        form16 = Form16Certificate.query.get(form16_id)
        if not form16:
            raise ValueError(f"Form 16 {form16_id} not found")
        
        # Import PDF generation library
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
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
                f"Form16_{form16.employee_id}_{form16.financial_year}.pdf"
            )
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a1a1a'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        story.append(Paragraph("FORM 16", title_style))
        story.append(Paragraph("Certificate under Section 203 of the Income-tax Act, 1961", styles['Normal']))
        story.append(Paragraph(f"for tax deducted at source on salary", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Part A - TDS Details
        story.append(Paragraph("<b>PART A</b>", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        part_a = form16.part_a_data or {}
        part_a_data_table = [
            ['TAN', part_a.get('tan', '')],
            ['Employer Name', part_a.get('employer_name', '')],
            ['Employee PAN', part_a.get('employee_pan', '')],
            ['Assessment Year', str(part_a.get('assessment_year', ''))],
            ['Total TDS Deducted', f"₹{part_a.get('total_tds_deducted', 0):,.2f}"],
        ]
        
        part_a_table = Table(part_a_data_table, colWidths=[2*inch, 4*inch])
        part_a_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(part_a_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Part B - Salary Details
        story.append(Paragraph("<b>PART B</b>", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        part_b = form16.part_b_data or {}
        part_b_data_table = [
            ['Gross Salary', f"₹{part_b.get('gross_salary', 0):,.2f}"],
            ['Total Deductions', f"₹{part_b.get('total_deductions', 0):,.2f}"],
            ['Taxable Income', f"₹{part_b.get('taxable_income', 0):,.2f}"],
            ['Total Tax Payable', f"₹{part_b.get('total_tax_payable', 0):,.2f}"],
            ['Tax Paid', f"₹{part_b.get('tax_paid', 0):,.2f}"],
            ['Refund Amount', f"₹{part_b.get('refund_amount', 0):,.2f}"],
        ]
        
        part_b_table = Table(part_b_data_table, colWidths=[2*inch, 4*inch])
        part_b_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(part_b_table)
        
        # Build PDF
        doc.build(story)
        
        # Update form16 record
        form16.pdf_path = output_path
        form16.pdf_generated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Generated Form 16 PDF: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating Form 16 PDF: {str(e)}")
        raise


def sign_form16(form16_id, signed_by_user_id, digital_signature=None):
    """
    Sign Form 16 with digital signature
    
    Args:
        form16_id: Form 16 certificate ID
        signed_by_user_id: User ID who is signing
        digital_signature: Digital signature data (optional)
    """
    try:
        form16 = Form16Certificate.query.get(form16_id)
        if not form16:
            raise ValueError(f"Form 16 {form16_id} not found")
        
        form16.signed_by = signed_by_user_id
        form16.signed_at = datetime.utcnow()
        if digital_signature:
            form16.digital_signature = digital_signature
        
        db.session.commit()
        logger.info(f"Form 16 {form16_id} signed by user {signed_by_user_id}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error signing Form 16: {str(e)}")
        raise

