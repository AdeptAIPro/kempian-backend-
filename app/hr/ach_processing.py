"""
ACH Direct Deposit Processing Service
Handles ACH file generation (NACHA format) for US payroll
"""

from app import db
from app.models import (
    Payslip, PayRun, User, UserBankAccount, EmployeeProfile
)
from decimal import Decimal
from datetime import datetime
from app.simple_logger import get_logger
import os

logger = get_logger(__name__)


def validate_routing_number(routing_number):
    """
    Validate US bank routing number using MOD 10 check
    
    Args:
        routing_number: 9-digit routing number
    
    Returns:
        bool: True if valid
    """
    try:
        routing_number = str(routing_number).strip()
        
        if len(routing_number) != 9:
            return False
        
        if not routing_number.isdigit():
            return False
        
        # MOD 10 check algorithm
        digits = [int(d) for d in routing_number]
        checksum = (
            (digits[0] + digits[3] + digits[6]) * 3 +
            (digits[1] + digits[4] + digits[7]) * 7 +
            (digits[2] + digits[5] + digits[8]) * 1
        ) % 10
        
        return checksum == 0
    except Exception:
        return False


def generate_nacha_file(
    pay_run_id,
    output_path=None,
    company_name=None,
    company_id=None,
    immediate_destination=None,
    immediate_origin=None
):
    """
    Generate NACHA format ACH file
    
    Args:
        pay_run_id: Pay run ID
        output_path: Output file path (optional)
        company_name: Company name (optional)
        company_id: Company ID (optional)
        immediate_destination: Bank routing number (optional)
        immediate_origin: Company routing number (optional)
    
    Returns:
        str: Path to generated NACHA file
    """
    try:
        # Get pay run
        pay_run = PayRun.query.get(pay_run_id)
        if not pay_run:
            raise ValueError(f"Pay run {pay_run_id} not found")
        
        # Get all payslips for the pay run
        from app.models import PayRunPayslip
        pay_run_payslips = PayRunPayslip.query.filter_by(pay_run_id=pay_run_id).all()
        payslips = [prp.payslip for prp in pay_run_payslips if prp.payslip]
        
        if not payslips:
            raise ValueError(f"No payslips found for pay run {pay_run_id}")
        
        # Create output path if not provided
        if not output_path:
            output_dir = os.path.join(os.getcwd(), 'generated_ach_files')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"ACH_{pay_run_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.ach"
            )
        
        # Get company information
        if not company_name:
            from app.models import OrganizationMetadata
            org = OrganizationMetadata.query.filter_by(tenant_id=pay_run.tenant_id).first()
            company_name = org.name if org else "Company Name"
        
        # Generate NACHA file
        nacha_lines = []
        
        # File Header Record (Record Type 1)
        file_id_modifier = 'A'  # Single file
        immediate_dest = immediate_destination or '000000000'
        immediate_origin = immediate_origin or '000000000'
        file_creation_date = datetime.now().strftime('%y%m%d')
        file_creation_time = datetime.now().strftime('%H%M')
        
        file_header = (
            '1' +  # Record Type Code
            '01' +  # Priority Code
            immediate_dest.ljust(10) +  # Immediate Destination
            immediate_origin.ljust(10) +  # Immediate Origin
            file_creation_date +  # File Creation Date
            file_creation_time +  # File Creation Time
            file_id_modifier +  # File ID Modifier
            '094' +  # Record Size
            '10' +  # Blocking Factor
            '1' +  # Format Code
            immediate_dest.ljust(10) +  # Immediate Destination Name
            company_name[:23].ljust(23) +  # Immediate Origin Name
            ' ' * 8  # Reference Code
        )
        nacha_lines.append(file_header)
        
        # Batch Header Record (Record Type 5)
        batch_number = '1'
        service_class_code = '200'  # Mixed debits and credits
        company_name_short = company_name[:16].ljust(16)
        company_discretionary = ' ' * 20
        company_id_field = company_id or '0000000000'
        standard_entry_class = 'PPD'  # Prearranged Payment and Deposit
        company_entry_description = 'PAYROLL'.ljust(10)
        company_descriptive_date = pay_run.pay_date.strftime('%y%m%d') if pay_run.pay_date else datetime.now().strftime('%y%m%d')
        effective_entry_date = pay_run.pay_date.strftime('%y%m%d') if pay_run.pay_date else datetime.now().strftime('%y%m%d')
        settlement_date = '   '  # Bank fills this
        originator_status_code = '1'  # ACH Operator
        originating_dfi_id = immediate_origin[:8]
        batch_number_field = batch_number.zfill(7)
        
        batch_header = (
            '5' +  # Record Type Code
            service_class_code +  # Service Class Code
            company_name_short +  # Company Name
            company_discretionary +  # Company Discretionary Data
            company_id_field +  # Company Identification
            standard_entry_class +  # Standard Entry Class Code
            company_entry_description +  # Company Entry Description
            company_descriptive_date +  # Company Descriptive Date
            effective_entry_date +  # Effective Entry Date
            settlement_date +  # Settlement Date
            originator_status_code +  # Originator Status Code
            originating_dfi_id +  # Originating DFI Identification
            batch_number_field +  # Batch Number
            ' ' * 39  # Reserved
        )
        nacha_lines.append(batch_header)
        
        # Entry Detail Records (Record Type 6)
        total_debits = Decimal('0')
        total_credits = Decimal('0')
        entry_count = 0
        
        for payslip in payslips:
            if payslip.status != 'generated' and payslip.status != 'paid':
                continue
            
            # Get employee bank account
            bank_account = UserBankAccount.query.filter_by(
                user_id=payslip.employee_id,
                is_active=True
            ).first()
            
            if not bank_account or not bank_account.routing_number:
                logger.warning(f"No bank account found for employee {payslip.employee_id}")
                continue
            
            # Validate routing number
            if not validate_routing_number(bank_account.routing_number):
                logger.warning(f"Invalid routing number for employee {payslip.employee_id}")
                continue
            
            transaction_code = '21'  # Checking account credit
            if bank_account.account_type == 'savings':
                transaction_code = '32'  # Savings account credit
            
            routing_number = str(bank_account.routing_number).zfill(9)
            account_number = str(bank_account.account_number).ljust(17)
            amount = int(Decimal(str(payslip.net_pay)) * Decimal('100'))  # Amount in cents
            individual_id = f"EMP{payslip.employee_id:06d}".ljust(15)
            individual_name = f"{payslip.employee.email[:22]}".ljust(22)
            discretionary_data = ' ' * 2
            addenda_record_indicator = '0'  # No addenda
            trace_number = f"{originating_dfi_id}{entry_count + 1:07d}"
            
            entry_detail = (
                '6' +  # Record Type Code
                transaction_code +  # Transaction Code
                routing_number[:8] +  # Receiving DFI Identification
                routing_number[8] +  # Check Digit
                account_number +  # DFI Account Number
                str(amount).zfill(10) +  # Amount
                individual_id +  # Individual Identification Number
                individual_name +  # Individual Name
                discretionary_data +  # Discretionary Data
                addenda_record_indicator +  # Addenda Record Indicator
                trace_number +  # Trace Number
                ' ' * 7  # Reserved
            )
            nacha_lines.append(entry_detail)
            
            total_credits += Decimal(str(payslip.net_pay))
            entry_count += 1
        
        # Batch Control Record (Record Type 8)
        entry_hash = str(entry_count)[:10].zfill(10)  # Simplified
        total_debit_entry_dollar = str(int(total_debits * Decimal('100'))).zfill(12)
        total_credit_entry_dollar = str(int(total_credits * Decimal('100'))).zfill(12)
        company_id_control = company_id_field
        message_authentication_code = ' ' * 19
        reserved = ' ' * 6
        originating_dfi_id_control = originating_dfi_id
        batch_number_control = batch_number_field
        
        batch_control = (
            '8' +  # Record Type Code
            service_class_code +  # Service Class Code
            str(entry_count).zfill(6) +  # Entry/Addenda Count
            entry_hash +  # Entry Hash
            total_debit_entry_dollar +  # Total Debit Entry Dollar Amount
            total_credit_entry_dollar +  # Total Credit Entry Dollar Amount
            company_id_control +  # Company Identification
            message_authentication_code +  # Message Authentication Code
            reserved +  # Reserved
            originating_dfi_id_control +  # Originating DFI Identification
            batch_number_control +  # Batch Number
            ' ' * 6  # Reserved
        )
        nacha_lines.append(batch_control)
        
        # File Control Record (Record Type 9)
        file_control = (
            '9' +  # Record Type Code
            service_class_code +  # Batch Count
            str(1).zfill(6) +  # Block Count
            str(entry_count).zfill(6) +  # Entry/Addenda Count
            entry_hash +  # Entry Hash
            total_debit_entry_dollar +  # Total Debit Entry Dollar Amount
            total_credit_entry_dollar +  # Total Credit Entry Dollar Amount
            ' ' * 39  # Reserved
        )
        nacha_lines.append(file_control)
        
        # Write file
        with open(output_path, 'w') as f:
            for line in nacha_lines:
                f.write(line + '\n')
        
        logger.info(f"Generated NACHA file: {output_path} with {entry_count} entries")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating NACHA file: {str(e)}")
        raise


def process_ach_returns(ach_file_path):
    """
    Process ACH return file
    
    Args:
        ach_file_path: Path to ACH return file
    
    Returns:
        list: List of return records
    """
    try:
        returns = []
        
        with open(ach_file_path, 'r') as f:
            for line in f:
                if line.startswith('6'):  # Entry Detail Record
                    return_code = line[79:81]  # Return code position
                    trace_number = line[82:94]  # Trace number
                    
                    if return_code and return_code != '  ':
                        returns.append({
                            'trace_number': trace_number.strip(),
                            'return_code': return_code.strip(),
                            'reason': _get_return_code_reason(return_code)
                        })
        
        return returns
    except Exception as e:
        logger.error(f"Error processing ACH returns: {str(e)}")
        raise


def _get_return_code_reason(return_code):
    """Get reason for ACH return code"""
    return_codes = {
        'R01': 'Insufficient Funds',
        'R02': 'Account Closed',
        'R03': 'No Account/Unable to Locate Account',
        'R04': 'Invalid Account Number',
        'R05': 'Unauthorized Debit to Consumer Account',
        'R06': 'Returned per ODFI Request',
        'R07': 'Authorization Revoked by Customer',
        'R08': 'Payment Stopped',
        'R09': 'Uncollected Funds',
        'R10': 'Customer Advises Not Authorized',
        'R11': 'Check Truncation Entry Return',
        'R12': 'Branch Sold to Another DFI',
        'R13': 'RDFI Not Qualified to Participate',
        'R14': 'Representative Payee Deceased',
        'R15': 'Beneficiary or Account Holder Deceased',
        'R16': 'Account Frozen',
        'R17': 'File Record Edit Error',
        'R18': 'Improper Effective Entry Date',
        'R19': 'Amount Field Error',
        'R20': 'Non-Transaction Account',
        'R21': 'Invalid Company Identification',
        'R22': 'Invalid Individual ID Number',
        'R23': 'Credit Entry Refused by Receiver',
        'R24': 'Duplicate Entry',
        'R25': 'Addenda Error',
        'R26': 'Mandatory Field Error',
        'R27': 'Trace Number Error',
        'R28': 'Routing Number Check Digit Error',
        'R29': 'Corporate Customer Advises Not Authorized',
        'R30': 'RDFI Not Participant in Check Truncation Program',
        'R31': 'Permissible Return Entry Not Accepted',
        'R32': 'RDFI Not Participant in Automated Notification',
        'R33': 'Return of Improper Debit Entry',
        'R34': 'Limited Participation DFI',
        'R35': 'Return of Improper Credit Entry',
        'R36': 'Return of XCK Entry',
        'R37': 'Source Document Presented for Payment',
        'R38': 'Stop Payment on Source Document',
        'R39': 'Improper Source Document',
        'R40': 'Return of ENR Entry',
        'R41': 'Invalid Transaction Code',
        'R42': 'Routing Number/Check Digit Error',
        'R43': 'Invalid DFI Account Number',
        'R44': 'Invalid Individual ID Number',
        'R45': 'Invalid Individual Name',
        'R46': 'Invalid Representative Payee Indicator',
        'R47': 'Duplicate Enrollment',
        'R50': 'State Law Affecting RCK Acceptance',
        'R51': 'Item is Ineligible',
        'R52': 'Stop Payment on Item',
        'R53': 'Item and ACH Entry Presented for Payment',
        'R61': 'Misrouted Return',
        'R62': 'Incorrect Trace Number',
        'R63': 'Incorrect Dollar Amount',
        'R64': 'Incorrect Individual Identification',
        'R65': 'Incorrect Transaction Code',
        'R66': 'Incorrect Company Identification',
        'R67': 'Duplicate Return',
        'R68': 'Untimely Return',
        'R69': 'Field Error(s)',
        'R70': 'Permissible Return Entry Not Accepted',
        'R71': 'Misrouted Dishonored Return',
        'R72': 'Untimely Dishonored Return',
        'R73': 'Timely Original Return',
        'R74': 'Corrected Return',
        'R75': 'Return Not a Duplicate',
        'R76': 'No Errors Found',
        'R77': 'Non-Accepted Return',
        'R80': 'Cross-Border Payment Coding Error',
        'R81': 'Non-Participant in Cross-Border Program',
        'R82': 'Invalid Foreign Receiving DFI Identification',
        'R83': 'Foreign Receiving DFI Unable to Settle',
        'R84': 'Entry Not Processed by Gateway',
        'R85': 'Invalid Individual ID Number',
        'R86': 'Invalid Transaction Code',
        'R87': 'Invalid Account Number',
        'R88': 'Invalid Identification Number',
        'R89': 'Invalid Receiving DFI Identification',
        'R90': 'Cross-Border Entry',
        'R91': 'Invalid Individual ID Number',
        'R92': 'Entry Not Processed by Gateway',
        'R93': 'Invalid Receiving DFI Identification',
        'R94': 'Duplicate Entry',
        'R95': 'Addenda Error',
        'R96': 'Mandatory Field Error',
        'R97': 'Invalid Receiving DFI Identification',
        'R98': 'Cross-Border Payment Coding Error',
        'R99': 'Non-Participant in Cross-Border Program'
    }
    
    return return_codes.get(return_code, 'Unknown Return Code')

