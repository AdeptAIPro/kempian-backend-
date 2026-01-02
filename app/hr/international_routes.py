"""
International Compliance API Routes
Handles multi-country tax, currency, and localization endpoints
"""

from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, db
from app.hr.country_tax_engines import (
    calculate_country_tax, create_country_tax_configuration, get_tax_configuration_by_country_year
)
from app.hr.currency_service import (
    get_exchange_rate, convert_currency, create_exchange_rate, format_currency
)
from app.hr.localization import (
    format_date, format_number, format_address, get_localization_settings
)
from datetime import datetime
from decimal import Decimal

international_bp = Blueprint('international', __name__)


def _auth_or_401():
    """Helper to get current user or return 401"""
    user = get_current_user_flexible()
    if not user:
        return None, (jsonify({'error': 'Unauthorized'}), 401)
    return user, None


# ============================================================================
# Country Tax Routes
# ============================================================================

@international_bp.route('/country-tax', methods=['GET'])
def get_country_tax_configs():
    """Get country tax configurations"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    country_code = request.args.get('country_code')
    tax_year = request.args.get('tax_year', type=int)
    
    try:
        from app.models import CountryTaxConfiguration
        query = CountryTaxConfiguration.query.filter_by(tenant_id=tenant_id)
        if country_code:
            query = query.filter_by(country_code=country_code)
        if tax_year:
            query = query.filter_by(tax_year=tax_year)
        
        configs = query.all()
        return jsonify({'configs': [c.to_dict() for c in configs]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@international_bp.route('/country-tax', methods=['POST'])
def create_country_tax_config():
    """Create country tax configuration"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    
    try:
        config = create_country_tax_configuration(
            tenant_id=tenant_id,
            country_code=data.get('country_code'),
            tax_type=data.get('tax_type'),
            tax_rate=data.get('tax_rate'),
            tax_brackets=data.get('tax_brackets'),
            currency_code=data.get('currency_code')
        )
        return jsonify(config.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Currency Routes
# ============================================================================

@international_bp.route('/currency/rates', methods=['GET'])
def get_exchange_rates():
    """Get exchange rates"""
    user, error = _auth_or_401()
    if error:
        return error
    
    base_currency = request.args.get('base_currency')
    target_currency = request.args.get('target_currency')
    
    try:
        from app.models import CurrencyExchangeRate
        query = CurrencyExchangeRate.query.filter_by(is_active=True)
        if base_currency:
            query = query.filter_by(base_currency=base_currency)
        if target_currency:
            query = query.filter_by(target_currency=target_currency)
        
        rates = query.order_by(CurrencyExchangeRate.rate_date.desc()).all()
        return jsonify({'rates': [r.to_dict() for r in rates]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@international_bp.route('/currency/rates', methods=['POST'])
def create_exchange_rate_endpoint():
    """Create exchange rate"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    
    try:
        rate = create_exchange_rate(
            base_currency=data.get('base_currency'),
            target_currency=data.get('target_currency'),
            exchange_rate=data.get('exchange_rate'),
            rate_source=data.get('rate_source', 'manual')
        )
        return jsonify(rate.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@international_bp.route('/currency/convert', methods=['POST'])
def convert_currency_endpoint():
    """Convert currency"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    amount = data.get('amount')
    from_currency = data.get('from_currency')
    to_currency = data.get('to_currency')
    
    if not all([amount, from_currency, to_currency]):
        return jsonify({'error': 'Amount, from_currency, and to_currency are required'}), 400
    
    try:
        converted_amount = convert_currency(amount, from_currency, to_currency)
        return jsonify({
            'amount': amount,
            'from_currency': from_currency,
            'to_currency': to_currency,
            'converted_amount': float(converted_amount)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Localization Routes
# ============================================================================

@international_bp.route('/localization/settings', methods=['GET'])
def get_localization_settings_endpoint():
    """Get localization settings"""
    user, error = _auth_or_401()
    if error:
        return error
    
    country_code = request.args.get('country_code', 'US')
    
    try:
        settings = get_localization_settings(country_code)
        return jsonify(settings), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Compliance Forms Routes
# ============================================================================

@international_bp.route('/compliance-forms', methods=['GET'])
def get_compliance_forms():
    """Get compliance forms"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    country_code = request.args.get('country_code')
    
    try:
        from app.models import ComplianceForm
        query = ComplianceForm.query.filter_by(tenant_id=tenant_id)
        if country_code:
            query = query.filter_by(country_code=country_code)
        
        forms = query.all()
        return jsonify({'forms': [f.to_dict() for f in forms]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@international_bp.route('/compliance-forms/generate', methods=['POST'])
def generate_compliance_form():
    """Generate compliance form"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    
    try:
        from app.models import ComplianceForm
        form = ComplianceForm(
            tenant_id=tenant_id,
            country_code=data.get('country_code'),
            form_type=data.get('form_type'),
            tax_year=data.get('tax_year'),
            status='generated'
        )
        db.session.add(form)
        db.session.commit()
        return jsonify(form.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@international_bp.route('/compliance-forms/<int:form_id>/download', methods=['GET'])
def download_compliance_form(form_id):
    """Download compliance form"""
    user, error = _auth_or_401()
    if error:
        return error
    
    try:
        from app.models import ComplianceForm
        form = ComplianceForm.query.get(form_id)
        if not form:
            return jsonify({'error': 'Form not found'}), 404
        
        if form.pdf_path:
            import os
            if os.path.exists(form.pdf_path):
                from flask import send_file
                return send_file(form.pdf_path, as_attachment=True)
        
        return jsonify({'error': 'PDF not available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

