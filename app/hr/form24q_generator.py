"""
Form 24Q Generation Service
Generates Form 24Q Quarterly TDS Return for India payroll
"""

from app import db
from app.models import (
    Form24QReturn, TDSRecord, PayRun, Tenant, User, EmployeeProfile
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger
import json

logger = get_logger(__name__)


def generate_form24q(
    tenant_id,
    quarter,
    financial_year,
    tan_number=None
):
    """
    Generate Form 24Q quarterly return
    
    Args:
        tenant_id: Tenant ID
        quarter: Quarter (1, 2, 3, 4)
        financial_year: Financial year (e.g., 2024)
        tan_number: TAN number (optional)
    
    Returns:
        Form24QReturn: Generated Form 24Q record
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
            1: [4, 5, 6],   # Q1: Apr, May, Jun
            2: [7, 8, 9],   # Q2: Jul, Aug, Sep
            3: [10, 11, 12], # Q3: Oct, Nov, Dec
            4: [1, 2, 3]     # Q4: Jan, Feb, Mar (next year)
        }
        
        months = quarter_months[quarter]
        if quarter == 4:
            # Q4 spans into next calendar year
            year_start = financial_year
            year_end = financial_year + 1
        else:
            year_start = financial_year
            year_end = financial_year
        
        # Get all TDS records for the quarter (join through User to get tenant_id)
        if quarter == 4:
            # Q4: months 1-3 of next year
            tds_records = TDSRecord.query.join(
                User, TDSRecord.employee_id == User.id
            ).filter(
                User.tenant_id == tenant_id,
                TDSRecord.tds_year == financial_year + 1,
                TDSRecord.tds_month.in_([1, 2, 3])
            ).all()
        else:
            # Q1-Q3: months in same year
            tds_records = TDSRecord.query.join(
                User, TDSRecord.employee_id == User.id
            ).filter(
                User.tenant_id == tenant_id,
                TDSRecord.tds_year == financial_year,
                TDSRecord.tds_month.in_(months)
            ).all()
        
        if not tds_records:
            raise ValueError(f"No TDS records found for quarter {quarter}, financial year {financial_year}")
        
        # Get unique employees
        employee_ids = list(set(record.employee_id for record in tds_records))
        total_employees = len(employee_ids)
        
        # Calculate totals
        total_tds_deducted = sum(Decimal(str(record.tds_amount)) for record in tds_records)
        
        # Get TAN from first record if not provided
        if not tan_number and tds_records:
            tan_number = tds_records[0].tan_number
        
        # Get challan details
        challan_records = {}
        for record in tds_records:
            if record.challan_281_number:
                challan_key = record.challan_281_number
                if challan_key not in challan_records:
                    challan_records[challan_key] = {
                        'challan_number': record.challan_281_number,
                        'challan_date': record.challan_281_date.isoformat() if record.challan_281_date else None,
                        'amount': Decimal('0')
                    }
                challan_records[challan_key]['amount'] += record.tds_amount
        
        total_tds_paid = sum(Decimal(str(ch['amount'])) for ch in challan_records.values())
        
        # Prepare return data
        return_type = f"Q{quarter}"
        
        # Employee-wise breakdown
        employee_breakdown = []
        for emp_id in employee_ids:
            emp_records = [r for r in tds_records if r.employee_id == emp_id]
            emp_total_tds = sum(Decimal(str(r.tds_amount)) for r in emp_records)
            emp_total_salary = sum(Decimal(str(r.gross_salary)) for r in emp_records)
            
            employee_profile = EmployeeProfile.query.filter_by(user_id=emp_id).first()
            employee = User.query.get(emp_id)
            
            employee_breakdown.append({
                'employee_id': emp_id,
                'employee_name': f"{employee_profile.first_name or ''} {employee_profile.last_name or ''}".strip() if employee_profile else employee.email,
                'pan': employee_profile.pan_number if employee_profile else None,
                'total_salary': float(emp_total_salary),
                'total_tds': float(emp_total_tds),
                'months': [r.tds_month for r in emp_records]
            })
        
        return_data = {
            'quarter': quarter,
            'financial_year': financial_year,
            'tan': tan_number,
            'total_employees': total_employees,
            'total_tds_deducted': float(total_tds_deducted),
            'total_tds_paid': float(total_tds_paid),
            'challan_details': list(challan_records.values()),
            'employee_breakdown': employee_breakdown,
            'months_covered': months
        }
        
        # Check if Form 24Q already exists
        existing = Form24QReturn.query.filter_by(
            tenant_id=tenant_id,
            quarter=quarter,
            financial_year=financial_year
        ).first()
        
        if existing:
            # Update existing
            existing.total_tds_deducted = total_tds_deducted
            existing.total_tds_paid = total_tds_paid
            existing.total_employees = total_employees
            existing.return_data = return_data
            existing.tan_number = tan_number
            existing.updated_at = datetime.utcnow()
            form24q = existing
        else:
            # Create new
            form24q = Form24QReturn(
                tenant_id=tenant_id,
                return_type=return_type,
                financial_year=financial_year,
                quarter=quarter,
                tan_number=tan_number,
                total_tds_deducted=total_tds_deducted,
                total_tds_paid=total_tds_paid,
                total_employees=total_employees,
                return_data=return_data
            )
            db.session.add(form24q)
        
        db.session.commit()
        
        logger.info(f"Generated Form 24Q for tenant {tenant_id}, Q{quarter}, FY {financial_year}")
        return form24q
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating Form 24Q: {str(e)}")
        raise


def generate_form24q_xml(form24q_id, output_path=None):
    """
    Generate XML file for Form 24Q e-filing
    
    Args:
        form24q_id: Form 24Q return ID
        output_path: Output file path (optional)
    
    Returns:
        str: Path to generated XML file
    """
    try:
        form24q = Form24QReturn.query.get(form24q_id)
        if not form24q:
            raise ValueError(f"Form 24Q {form24q_id} not found")
        
        # Create output path if not provided
        if not output_path:
            import os
            output_dir = os.path.join(os.getcwd(), 'generated_forms')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"Form24Q_{form24q.tenant_id}_Q{form24q.quarter}_{form24q.financial_year}.xml"
            )
        
        # Generate XML
        import xml.etree.ElementTree as ET
        
        root = ET.Element("Form24Q")
        root.set("xmlns", "http://incometaxindia.gov.in/Form24Q")
        
        # Header
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "TAN").text = form24q.tan_number or ""
        ET.SubElement(header, "FinancialYear").text = str(form24q.financial_year)
        ET.SubElement(header, "Quarter").text = str(form24q.quarter)
        ET.SubElement(header, "TotalTDSDeducted").text = str(form24q.total_tds_deducted)
        ET.SubElement(header, "TotalTDSPaid").text = str(form24q.total_tds_paid)
        ET.SubElement(header, "TotalEmployees").text = str(form24q.total_employees)
        
        # Employee Details
        return_data = form24q.return_data or {}
        employee_breakdown = return_data.get('employee_breakdown', [])
        
        employees = ET.SubElement(root, "Employees")
        for emp in employee_breakdown:
            emp_elem = ET.SubElement(employees, "Employee")
            ET.SubElement(emp_elem, "EmployeeID").text = str(emp.get('employee_id', ''))
            ET.SubElement(emp_elem, "EmployeeName").text = emp.get('employee_name', '')
            ET.SubElement(emp_elem, "PAN").text = emp.get('pan', '')
            ET.SubElement(emp_elem, "TotalSalary").text = str(emp.get('total_salary', 0))
            ET.SubElement(emp_elem, "TotalTDS").text = str(emp.get('total_tds', 0))
        
        # Challan Details
        challans = ET.SubElement(root, "Challans")
        challan_details = return_data.get('challan_details', [])
        for challan in challan_details:
            challan_elem = ET.SubElement(challans, "Challan")
            ET.SubElement(challan_elem, "ChallanNumber").text = challan.get('challan_number', '')
            ET.SubElement(challan_elem, "ChallanDate").text = challan.get('challan_date', '')
            ET.SubElement(challan_elem, "Amount").text = str(challan.get('amount', 0))
        
        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        # Update form24q record
        form24q.xml_path = output_path
        form24q.xml_generated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Generated Form 24Q XML: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating Form 24Q XML: {str(e)}")
        raise


def get_form24q(tenant_id, quarter, financial_year):
    """
    Get Form 24Q for a tenant
    
    Args:
        tenant_id: Tenant ID
        quarter: Quarter (1, 2, 3, 4)
        financial_year: Financial year
    
    Returns:
        Form24QReturn: Form 24Q record or None
    """
    try:
        return Form24QReturn.query.filter_by(
            tenant_id=tenant_id,
            quarter=quarter,
            financial_year=financial_year
        ).first()
    except Exception as e:
        logger.error(f"Error fetching Form 24Q: {str(e)}")
        raise


def mark_form24q_efiled(form24q_id, acknowledgment_number, filed_by_user_id):
    """
    Mark Form 24Q as e-filed
    
    Args:
        form24q_id: Form 24Q return ID
        acknowledgment_number: E-filing acknowledgment number
        filed_by_user_id: User ID who filed
    """
    try:
        form24q = Form24QReturn.query.get(form24q_id)
        if not form24q:
            raise ValueError(f"Form 24Q {form24q_id} not found")
        
        form24q.e_filing_status = 'submitted'
        form24q.e_filing_acknowledgment = acknowledgment_number
        form24q.e_filed_at = datetime.utcnow()
        form24q.e_filed_by = filed_by_user_id
        
        db.session.commit()
        logger.info(f"Form 24Q {form24q_id} marked as e-filed with acknowledgment {acknowledgment_number}")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error marking Form 24Q as e-filed: {str(e)}")
        raise

