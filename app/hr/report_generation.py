"""
Enhanced Report Generation Service
Handles PDF/XML generation, templates, batch processing, and compliance dashboard
"""

from app import db
from app.models import (
    Payslip, PayRun, Tenant, Form16Certificate, Form24QReturn,
    Form941Return, Form940Return, ComplianceForm
)
from decimal import Decimal
from datetime import datetime, timedelta
from app.simple_logger import get_logger
import json
import os

logger = get_logger(__name__)


def generate_compliance_report(
    tenant_id,
    report_type,
    start_date,
    end_date,
    country_code=None,
    output_format='pdf'
):
    """
    Generate compliance report
    
    Args:
        tenant_id: Tenant ID
        report_type: Report type ('payroll_summary', 'tax_summary', 'compliance_summary')
        start_date: Start date
        end_date: End date
        country_code: Country code filter (optional)
        output_format: Output format ('pdf', 'xml', 'json', 'csv')
    
    Returns:
        dict: Report generation result
    """
    try:
        # Get data based on report type
        if report_type == 'payroll_summary':
            data = _generate_payroll_summary(tenant_id, start_date, end_date, country_code)
        elif report_type == 'tax_summary':
            data = _generate_tax_summary(tenant_id, start_date, end_date, country_code)
        elif report_type == 'compliance_summary':
            data = _generate_compliance_summary(tenant_id, start_date, end_date, country_code)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        # Generate output based on format
        if output_format == 'pdf':
            output_path = _generate_pdf_report(data, report_type, tenant_id)
        elif output_format == 'xml':
            output_path = _generate_xml_report(data, report_type, tenant_id)
        elif output_format == 'json':
            output_path = _generate_json_report(data, report_type, tenant_id)
        elif output_format == 'csv':
            output_path = _generate_csv_report(data, report_type, tenant_id)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return {
            'report_type': report_type,
            'output_format': output_format,
            'output_path': output_path,
            'generated_at': datetime.utcnow().isoformat(),
            'data_summary': {
                'total_records': len(data.get('records', [])),
                'total_amount': data.get('total_amount', 0)
            }
        }
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise


def _generate_payroll_summary(tenant_id, start_date, end_date, country_code):
    """Generate payroll summary data"""
    from app.models import User, Payslip
    
    payslips = Payslip.query.join(
        User, Payslip.employee_id == User.id
    ).filter(
        User.tenant_id == tenant_id,
        Payslip.pay_date >= start_date,
        Payslip.pay_date <= end_date
    )
    
    if country_code:
        from app.models import EmployeeProfile
        payslips = payslips.join(
            EmployeeProfile, Payslip.employee_id == EmployeeProfile.user_id
        ).filter(EmployeeProfile.country_code == country_code)
    
    payslips = payslips.all()
    
    total_gross = sum(Decimal(str(ps.gross_earnings)) for ps in payslips)
    total_deductions = sum(Decimal(str(ps.total_deductions)) for ps in payslips)
    total_net = sum(Decimal(str(ps.net_pay)) for ps in payslips)
    
    return {
        'report_type': 'payroll_summary',
        'period': {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        },
        'total_gross': float(total_gross),
        'total_deductions': float(total_deductions),
        'total_net': float(total_net),
        'total_employees': len(set(ps.employee_id for ps in payslips)),
        'records': [{
            'payslip_id': ps.id,
            'employee_id': ps.employee_id,
            'pay_date': ps.pay_date.isoformat() if ps.pay_date else None,
            'gross_earnings': float(ps.gross_earnings),
            'net_pay': float(ps.net_pay)
        } for ps in payslips]
    }


def _generate_tax_summary(tenant_id, start_date, end_date, country_code):
    """Generate tax summary data"""
    from app.models import User, Payslip
    
    payslips = Payslip.query.join(
        User, Payslip.employee_id == User.id
    ).filter(
        User.tenant_id == tenant_id,
        Payslip.pay_date >= start_date,
        Payslip.pay_date <= end_date
    )
    
    if country_code:
        from app.models import EmployeeProfile
        payslips = payslips.join(
            EmployeeProfile, Payslip.employee_id == EmployeeProfile.user_id
        ).filter(EmployeeProfile.country_code == country_code)
    
    payslips = payslips.all()
    
    total_tax = sum(Decimal(str(ps.tax_deduction)) for ps in payslips)
    
    # Country-specific tax breakdown
    tax_breakdown = {}
    if country_code == 'IN':
        total_pf = sum(Decimal(str(ps.pf_employee or 0)) for ps in payslips)
        total_esi = sum(Decimal(str(ps.esi_employee or 0)) for ps in payslips)
        total_pt = sum(Decimal(str(ps.professional_tax or 0)) for ps in payslips)
        total_tds = sum(Decimal(str(ps.tds_amount or 0)) for ps in payslips)
        tax_breakdown = {
            'pf': float(total_pf),
            'esi': float(total_esi),
            'professional_tax': float(total_pt),
            'tds': float(total_tds)
        }
    elif country_code == 'US':
        total_state = sum(Decimal(str(ps.state_tax or 0)) for ps in payslips)
        total_local = sum(Decimal(str(ps.local_tax or 0)) for ps in payslips)
        total_sui = sum(Decimal(str(ps.sui_contribution or 0)) for ps in payslips)
        tax_breakdown = {
            'state_tax': float(total_state),
            'local_tax': float(total_local),
            'sui': float(total_sui)
        }
    
    return {
        'report_type': 'tax_summary',
        'period': {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        },
        'total_tax': float(total_tax),
        'tax_breakdown': tax_breakdown,
        'records': [{
            'payslip_id': ps.id,
            'employee_id': ps.employee_id,
            'tax_deduction': float(ps.tax_deduction)
        } for ps in payslips]
    }


def _generate_compliance_summary(tenant_id, start_date, end_date, country_code):
    """Generate compliance summary data"""
    compliance_data = {
        'report_type': 'compliance_summary',
        'period': {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        },
        'forms_generated': [],
        'forms_filed': [],
        'challans_generated': [],
        'challans_paid': []
    }
    
    if country_code == 'IN':
        # Form 16
        form16s = Form16Certificate.query.filter(
            Form16Certificate.tenant_id == tenant_id,
            Form16Certificate.created_at >= start_date,
            Form16Certificate.created_at <= end_date
        ).all()
        compliance_data['forms_generated'].extend([{'type': 'Form16', 'id': f.id} for f in form16s])
        compliance_data['forms_filed'].extend([{'type': 'Form16', 'id': f.id} for f in form16s if f.e_filing_status == 'submitted'])
        
        # Form 24Q
        form24qs = Form24QReturn.query.filter(
            Form24QReturn.tenant_id == tenant_id,
            Form24QReturn.created_at >= start_date,
            Form24QReturn.created_at <= end_date
        ).all()
        compliance_data['forms_generated'].extend([{'type': 'Form24Q', 'id': f.id} for f in form24qs])
        compliance_data['forms_filed'].extend([{'type': 'Form24Q', 'id': f.id} for f in form24qs if f.e_filing_status == 'submitted'])
    
    elif country_code == 'US':
        # Form 941
        form941s = Form941Return.query.filter(
            Form941Return.tenant_id == tenant_id,
            Form941Return.created_at >= start_date,
            Form941Return.created_at <= end_date
        ).all()
        compliance_data['forms_generated'].extend([{'type': 'Form941', 'id': f.id} for f in form941s])
        compliance_data['forms_filed'].extend([{'type': 'Form941', 'id': f.id} for f in form941s if f.e_filing_status == 'submitted'])
        
        # Form 940
        form940s = Form940Return.query.filter(
            Form940Return.tenant_id == tenant_id,
            Form940Return.created_at >= start_date,
            Form940Return.created_at <= end_date
        ).all()
        compliance_data['forms_generated'].extend([{'type': 'Form940', 'id': f.id} for f in form940s])
        compliance_data['forms_filed'].extend([{'type': 'Form940', 'id': f.id} for f in form940s if f.e_filing_status == 'submitted'])
    
    return compliance_data


def _generate_pdf_report(data, report_type, tenant_id):
    """Generate PDF report"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        logger.warning("reportlab not installed, PDF generation skipped")
        return None
    
    try:
        output_dir = os.path.join(os.getcwd(), 'generated_reports')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{report_type}_{tenant_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        )
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        story.append(Paragraph(f"{report_type.replace('_', ' ').title()} Report", styles['Heading1']))
        story.append(Spacer(1, 0.3*inch))
        
        # Summary table
        summary_data = [
            ['Total Amount', f"${data.get('total_gross', data.get('total_tax', 0)):,.2f}"],
            ['Total Records', str(len(data.get('records', [])))],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        
        doc.build(story)
        return output_path
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None


def _generate_xml_report(data, report_type, tenant_id):
    """Generate XML report"""
    import xml.etree.ElementTree as ET
    
    output_dir = os.path.join(os.getcwd(), 'generated_reports')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{report_type}_{tenant_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.xml"
    )
    
    root = ET.Element("Report")
    root.set("type", report_type)
    root.set("generated_at", datetime.utcnow().isoformat())
    
    # Add data
    for key, value in data.items():
        if key != 'records':
            elem = ET.SubElement(root, key)
            elem.text = str(value)
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    return output_path


def _generate_json_report(data, report_type, tenant_id):
    """Generate JSON report"""
    output_dir = os.path.join(os.getcwd(), 'generated_reports')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{report_type}_{tenant_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    )
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return output_path


def _generate_csv_report(data, report_type, tenant_id):
    """Generate CSV report"""
    import csv
    
    output_dir = os.path.join(os.getcwd(), 'generated_reports')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{report_type}_{tenant_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    
    records = data.get('records', [])
    if records:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    
    return output_path


def batch_generate_reports(tenant_id, report_configs):
    """
    Batch generate multiple reports
    
    Args:
        tenant_id: Tenant ID
        report_configs: List of report configurations
    
    Returns:
        list: List of generated report paths
    """
    try:
        results = []
        for config in report_configs:
            result = generate_compliance_report(
                tenant_id=tenant_id,
                report_type=config.get('report_type'),
                start_date=datetime.strptime(config['start_date'], '%Y-%m-%d').date(),
                end_date=datetime.strptime(config['end_date'], '%Y-%m-%d').date(),
                country_code=config.get('country_code'),
                output_format=config.get('output_format', 'pdf')
            )
            results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Error in batch report generation: {str(e)}")
        raise

