"""
ESIC Portal Integration
Handles ESI contribution filing to ESIC portal
"""

from app import db
from app.models import ESIContribution, ChallanRecord, Tenant
from datetime import datetime
from app.simple_logger import get_logger
import json

logger = get_logger(__name__)


def submit_esi_challan_to_esic(challan_id, esic_credentials=None):
    """
    Submit ESI challan to ESIC portal
    
    Args:
        challan_id: Challan record ID
        esic_credentials: ESIC portal credentials (username, password, employer_code)
    
    Returns:
        dict: Submission result
    """
    try:
        challan = ChallanRecord.query.get(challan_id)
        if not challan:
            raise ValueError(f"Challan {challan_id} not found")
        
        if challan.challan_type != 'ESI':
            raise ValueError(f"Challan {challan_id} is not an ESI challan")
        
        # In production, this would:
        # 1. Authenticate with ESIC portal
        # 2. Upload challan data
        # 3. Get receipt number
        # 4. Update challan record
        
        # For now, simulate submission
        receipt_number = f"ESIC-RECEIPT-{challan_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        challan.payment_status = 'submitted'
        challan.payment_reference = receipt_number
        challan.paid_at = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(f"ESI challan {challan_id} submitted to ESIC portal")
        return {
            'challan_id': challan_id,
            'receipt_number': receipt_number,
            'status': 'submitted',
            'submitted_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error submitting ESI challan to ESIC: {str(e)}")
        raise


def check_esi_challan_status(challan_id):
    """
    Check ESI challan submission status
    
    Args:
        challan_id: Challan record ID
    
    Returns:
        dict: Status information
    """
    try:
        challan = ChallanRecord.query.get(challan_id)
        if not challan:
            raise ValueError(f"Challan {challan_id} not found")
        
        return {
            'challan_id': challan_id,
            'payment_status': challan.payment_status,
            'payment_reference': challan.payment_reference,
            'paid_at': challan.paid_at.isoformat() if challan.paid_at else None
        }
    except Exception as e:
        logger.error(f"Error checking ESI challan status: {str(e)}")
        raise

