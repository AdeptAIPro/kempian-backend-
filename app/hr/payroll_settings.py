"""
Payroll Settings API
Handles payroll configuration and holiday calendars
"""
from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, PayrollSettings, HolidayCalendar, db
from datetime import datetime
from decimal import Decimal

hr_payroll_settings_bp = Blueprint('hr_payroll_settings', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


@hr_payroll_settings_bp.route('/', methods=['GET'], strict_slashes=False)
def get_payroll_settings():
    """Get payroll settings for current tenant"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    settings = PayrollSettings.query.filter_by(tenant_id=tenant_id).first()
    
    if not settings:
        # Create default settings
        settings = PayrollSettings(
            tenant_id=tenant_id,
            pay_frequency='monthly',
            overtime_threshold_hours=Decimal('40'),
            overtime_multiplier=Decimal('1.5'),
            holiday_multiplier=Decimal('2.0'),
            default_currency='USD'
        )
        db.session.add(settings)
        db.session.commit()
    
    return jsonify({'settings': settings.to_dict()}), 200


@hr_payroll_settings_bp.route('/', methods=['PUT'], strict_slashes=False)
def update_payroll_settings():
    """Update payroll settings"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to update payroll settings'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    settings = PayrollSettings.query.filter_by(tenant_id=tenant_id).first()
    
    if not settings:
        settings = PayrollSettings(tenant_id=tenant_id)
        db.session.add(settings)
    
    data = request.get_json()
    
    if 'pay_frequency' in data:
        valid_frequencies = ['weekly', 'bi_weekly', 'semi_monthly', 'monthly']
        if data['pay_frequency'] not in valid_frequencies:
            return jsonify({'error': f'pay_frequency must be one of: {", ".join(valid_frequencies)}'}), 400
        settings.pay_frequency = data['pay_frequency']
    
    if 'pay_day_of_week' in data:
        settings.pay_day_of_week = data['pay_day_of_week']
    if 'pay_day_of_month' in data:
        settings.pay_day_of_month = data['pay_day_of_month']
    if 'overtime_threshold_hours' in data:
        settings.overtime_threshold_hours = Decimal(str(data['overtime_threshold_hours']))
    if 'overtime_multiplier' in data:
        settings.overtime_multiplier = Decimal(str(data['overtime_multiplier']))
    if 'holiday_multiplier' in data:
        settings.holiday_multiplier = Decimal(str(data['holiday_multiplier']))
    if 'default_currency' in data:
        settings.default_currency = data['default_currency']
    if 'holiday_calendar_id' in data:
        settings.holiday_calendar_id = data['holiday_calendar_id']
    
    # Payment Gateway Settings with encryption
    if 'payment_gateway' in data:
        settings.payment_gateway = data['payment_gateway']
    if 'razorpay_key_id' in data:
        # Store key ID (can be stored as-is, but encrypt in production)
        settings.razorpay_key_id = data['razorpay_key_id']
    if 'razorpay_key_secret' in data:
        # Encrypt the secret key before storing
        from app.utils.payment_security import PaymentEncryption
        encryption = PaymentEncryption()
        encrypted_secret = encryption.encrypt(data['razorpay_key_secret'])
        settings.razorpay_key_secret = f"enc:{encrypted_secret}"
    if 'razorpay_webhook_secret' in data:
        # Encrypt the webhook secret before storing
        from app.utils.payment_security import PaymentEncryption
        encryption = PaymentEncryption()
        encrypted_webhook_secret = encryption.encrypt(data['razorpay_webhook_secret'])
        settings.razorpay_webhook_secret = f"enc:{encrypted_webhook_secret}"
    if 'razorpay_fund_account_id' in data:
        old_fund_account_id = settings.razorpay_fund_account_id
        settings.razorpay_fund_account_id = data['razorpay_fund_account_id']
        # Reset validation when fund account ID changes
        if old_fund_account_id != data['razorpay_fund_account_id']:
            settings.razorpay_fund_account_validated = False
            settings.razorpay_fund_account_validated_at = None
    if 'payment_mode' in data:
        valid_modes = ['NEFT', 'RTGS', 'IMPS', 'UPI']
        if data['payment_mode'] not in valid_modes:
            return jsonify({'error': f'payment_mode must be one of: {", ".join(valid_modes)}'}), 400
        settings.payment_mode = data['payment_mode']
    
    db.session.commit()
    
    return jsonify({'settings': settings.to_dict()}), 200


@hr_payroll_settings_bp.route('/holidays', methods=['GET'], strict_slashes=False)
def list_holiday_calendars():
    """List holiday calendars for current tenant"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    calendars = HolidayCalendar.query.filter_by(tenant_id=tenant_id).order_by(HolidayCalendar.name).all()
    
    return jsonify({'holiday_calendars': [c.to_dict() for c in calendars]}), 200


@hr_payroll_settings_bp.route('/holidays', methods=['POST'], strict_slashes=False)
def create_holiday_calendar():
    """Create a holiday calendar"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to create holiday calendars'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json()
    
    name = data.get('name')
    if not name:
        return jsonify({'error': 'name is required'}), 400
    
    # If this is set as default, unset other defaults
    is_default = data.get('is_default', False)
    if is_default:
        HolidayCalendar.query.filter_by(tenant_id=tenant_id, is_default=True).update({'is_default': False})
    
    calendar = HolidayCalendar(
        tenant_id=tenant_id,
        name=name,
        holidays=data.get('holidays', []),
        is_default=is_default
    )
    
    db.session.add(calendar)
    db.session.commit()
    
    return jsonify({'holiday_calendar': calendar.to_dict()}), 201


@hr_payroll_settings_bp.route('/holidays/<int:calendar_id>', methods=['PUT'], strict_slashes=False)
def update_holiday_calendar(calendar_id):
    """Update a holiday calendar"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to update holiday calendars'}), 403
    
    calendar = HolidayCalendar.query.get(calendar_id)
    if not calendar:
        return jsonify({'error': 'Holiday calendar not found'}), 404
    
    data = request.get_json()
    
    if 'name' in data:
        calendar.name = data['name']
    if 'holidays' in data:
        calendar.holidays = data['holidays']
    if 'is_default' in data:
        is_default = data['is_default']
        if is_default:
            # Unset other defaults
            HolidayCalendar.query.filter_by(
                tenant_id=calendar.tenant_id,
                is_default=True
            ).filter(HolidayCalendar.id != calendar_id).update({'is_default': False})
        calendar.is_default = is_default
    
    db.session.commit()
    
    return jsonify({'holiday_calendar': calendar.to_dict()}), 200


@hr_payroll_settings_bp.route('/test-razorpay-connection', methods=['POST'], strict_slashes=False)
def test_razorpay_connection():
    """Test Razorpay API connection and validate credentials"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to test Razorpay connection'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    settings = PayrollSettings.query.filter_by(tenant_id=tenant_id).first()
    if not settings or settings.payment_gateway != 'razorpay':
        return jsonify({'error': 'Razorpay not configured'}), 400
    
    # Get credentials
    from app.utils.payment_security import PaymentEncryption
    encryption = PaymentEncryption()
    
    key_id = settings.razorpay_key_id
    key_secret = settings.razorpay_key_secret
    if key_secret and key_secret.startswith('enc:'):
        key_secret = encryption.decrypt(key_secret[4:])
    
    if not key_id or not key_secret:
        return jsonify({'error': 'Razorpay credentials not configured'}), 400
    
    # Test connection by fetching account details
    import requests
    import base64
    credentials = f"{key_id}:{key_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json"
    }
    
    try:
        # Test with a simple API call (fetch account details)
        response = requests.get(
            "https://api.razorpay.com/v1/account",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            account_data = response.json()
            return jsonify({
                'success': True,
                'message': 'Connection successful',
                'account': {
                    'id': account_data.get('id'),
                    'name': account_data.get('name'),
                    'email': account_data.get('email'),
                    'type': account_data.get('type')
                }
            }), 200
        else:
            error_data = response.json() if response.text else {}
            return jsonify({
                'success': False,
                'error': f'Connection failed: {error_data.get("error", {}).get("description", "Unknown error")}',
                'status_code': response.status_code
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Connection test failed: {str(e)}'
        }), 500


@hr_payroll_settings_bp.route('/validate-fund-account', methods=['POST'], strict_slashes=False)
def validate_fund_account():
    """Validate Razorpay fund account ID"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to validate fund account'}), 403
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    settings = PayrollSettings.query.filter_by(tenant_id=tenant_id).first()
    if not settings or settings.payment_gateway != 'razorpay':
        return jsonify({'error': 'Razorpay not configured'}), 400
    
    if not settings.razorpay_fund_account_id:
        return jsonify({'error': 'Fund account ID not configured'}), 400
    
    # Get credentials
    from app.utils.payment_security import PaymentEncryption
    encryption = PaymentEncryption()
    
    key_id = settings.razorpay_key_id
    key_secret = settings.razorpay_key_secret
    if key_secret and key_secret.startswith('enc:'):
        key_secret = encryption.decrypt(key_secret[4:])
    
    if not key_id or not key_secret:
        return jsonify({'error': 'Razorpay credentials not configured'}), 400
    
    # Validate fund account by fetching it
    import requests
    import base64
    credentials = f"{key_id}:{key_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json"
    }
    
    try:
        # Fetch fund account details
        response = requests.get(
            f"https://api.razorpay.com/v1/fund_accounts/{settings.razorpay_fund_account_id}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            fund_account_data = response.json()
            # Update validation status
            settings.razorpay_fund_account_validated = True
            settings.razorpay_fund_account_validated_at = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'success': True,
                'validated': True,
                'message': 'Fund account validated successfully',
                'fund_account': {
                    'id': fund_account_data.get('id'),
                    'account_type': fund_account_data.get('account_type'),
                    'active': fund_account_data.get('active', False)
                }
            }), 200
        elif response.status_code == 404:
            settings.razorpay_fund_account_validated = False
            settings.razorpay_fund_account_validated_at = None
            db.session.commit()
            return jsonify({
                'success': False,
                'validated': False,
                'error': 'Fund account not found. Please check the fund account ID.'
            }), 404
        else:
            error_data = response.json() if response.text else {}
            return jsonify({
                'success': False,
                'validated': False,
                'error': f'Validation failed: {error_data.get("error", {}).get("description", "Unknown error")}',
                'status_code': response.status_code
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'validated': False,
            'error': f'Validation failed: {str(e)}'
        }), 500

