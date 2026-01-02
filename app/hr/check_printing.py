"""
Check Printing Service
Handles check printing with MICR encoding for US payroll
"""

from app import db
from app.models import (
    Payslip, PayRun, User, EmployeeProfile
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger
import os

logger = get_logger(__name__)


def generate_micr_line(
    routing_number,
    account_number,
    check_number,
    amount=None
):
    """
    Generate MICR (Magnetic Ink Character Recognition) line for check
    
    Args:
        routing_number: Bank routing number
        account_number: Bank account number
        check_number: Check number
        amount: Check amount (optional, for amount box)
    
    Returns:
        str: MICR line
    """
    try:
        # MICR format: ⑆ routing ⑈ account ⑉ check ⑊
        # Using special MICR characters (represented as text)
        routing = str(routing_number).zfill(9)
        account = str(account_number).ljust(17)
        check = str(check_number).zfill(10)
        
        # In actual implementation, use proper MICR fonts
        # For now, using text representation
        micr_line = f"⑆{routing}⑈{account}⑉{check}⑊"
        
        return micr_line
    except Exception as e:
        logger.error(f"Error generating MICR line: {str(e)}")
        raise


def generate_check_pdf(
    payslip_id,
    output_path=None,
    company_name=None,
    company_address=None,
    bank_name=None,
    bank_routing=None,
    bank_account=None
):
    """
    Generate check PDF for a payslip
    
    Args:
        payslip_id: Payslip ID
        output_path: Output file path (optional)
        company_name: Company name (optional)
        company_address: Company address (optional)
        bank_name: Bank name (optional)
        bank_routing: Bank routing number (optional)
        bank_account: Bank account number (optional)
    
    Returns:
        str: Path to generated check PDF
    """
    try:
        # Get payslip
        payslip = Payslip.query.get(payslip_id)
        if not payslip:
            raise ValueError(f"Payslip {payslip_id} not found")
        
        # Get employee
        employee = User.query.get(payslip.employee_id)
        employee_profile = EmployeeProfile.query.filter_by(user_id=payslip.employee_id).first()
        
        if not employee:
            raise ValueError(f"Employee not found for payslip {payslip_id}")
        
        # Import PDF generation library
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_LEFT, TA_RIGHT
        except ImportError:
            logger.warning("reportlab not installed, PDF generation skipped")
            return None
        
        # Create output path if not provided
        if not output_path:
            output_dir = os.path.join(os.getcwd(), 'generated_checks')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"Check_{payslip_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            )
        
        # Get company information
        if not company_name:
            from app.models import OrganizationMetadata
            org = OrganizationMetadata.query.filter_by(tenant_id=employee.tenant_id).first()
            company_name = org.name if org else "Company Name"
            company_address = org.description if org else "Company Address"
        
        # Generate check number (use payslip ID)
        check_number = payslip.id
        
        # Create PDF document (check size: 6" x 2.75")
        doc = SimpleDocTemplate(output_path, pagesize=(6*inch, 2.75*inch), 
                               leftMargin=0.25*inch, rightMargin=0.25*inch,
                               topMargin=0.25*inch, bottomMargin=0.25*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Check layout
        # Top section: Company info and date
        company_info = f"{company_name}\n{company_address or ''}"
        date_str = payslip.pay_date.strftime('%B %d, %Y') if payslip.pay_date else datetime.now().strftime('%B %d, %Y')
        
        # Pay to the order of
        payee_name = f"{employee_profile.first_name or ''} {employee_profile.last_name or ''}".strip() if employee_profile else employee.email
        
        # Amount in words
        amount = Decimal(str(payslip.net_pay))
        amount_words = _number_to_words(float(amount))
        
        # Amount in numbers
        amount_numbers = f"${amount:,.2f}"
        
        # MICR line
        if bank_routing:
            micr_line = generate_micr_line(bank_routing, bank_account or '', check_number)
        else:
            micr_line = f"Check #{check_number}"
        
        # Create check table
        check_data = [
            [company_info, '', date_str],
            ['', '', ''],
            ['Pay to the', '', ''],
            ['order of:', '', ''],
            [payee_name, '', amount_numbers],
            ['', '', ''],
            [f"{amount_words} dollars", '', ''],
            ['', '', ''],
            ['', '', micr_line]
        ]
        
        check_table = Table(check_data, colWidths=[3*inch, 0.5*inch, 2*inch])
        check_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(check_table)
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Generated check PDF: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating check PDF: {str(e)}")
        raise


def _number_to_words(number):
    """Convert number to words (simplified version)"""
    ones = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    teens = ['Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 
             'Seventeen', 'Eighteen', 'Nineteen']
    tens = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
    
    if number == 0:
        return 'Zero'
    
    dollars = int(number)
    cents = int((number - dollars) * 100)
    
    def convert_hundreds(n):
        if n == 0:
            return ''
        result = ''
        if n >= 100:
            result += ones[n // 100] + ' Hundred '
            n %= 100
        if n >= 20:
            result += tens[n // 10] + ' '
            n %= 10
        elif n >= 10:
            result += teens[n - 10] + ' '
            return result
        if n > 0:
            result += ones[n] + ' '
        return result
    
    result = convert_hundreds(dollars).strip()
    if cents > 0:
        result += f' and {cents}/100'
    
    return result


def void_check(check_number, reason=None):
    """
    Void a check
    
    Args:
        check_number: Check number
        reason: Void reason (optional)
    
    Returns:
        dict: Void confirmation
    """
    try:
        # In a real system, this would mark the check as voided in the database
        # For now, just log it
        logger.info(f"Check {check_number} voided. Reason: {reason or 'Not specified'}")
        
        return {
            'check_number': check_number,
            'voided_at': datetime.utcnow().isoformat(),
            'reason': reason
        }
    except Exception as e:
        logger.error(f"Error voiding check: {str(e)}")
        raise

