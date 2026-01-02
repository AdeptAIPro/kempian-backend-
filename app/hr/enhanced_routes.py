"""
Enhanced Features API Routes
Handles tax bracket engine, tax table management, e-filing dashboard, and enhanced reporting
"""

from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, db
from app.hr.tax_bracket_engine import (
    calculate_progressive_tax, get_tax_configuration_by_country_year,
    optimize_tax_with_deductions, validate_tax_brackets
)
from app.hr.tax_table_management import (
    create_tax_table_version, get_tax_table_history, export_tax_table,
    import_tax_table, update_tax_tables_annually
)
from app.hr.report_generation import (
    generate_compliance_report, batch_generate_reports
)
from datetime import datetime
from decimal import Decimal

enhanced_bp = Blueprint('enhanced', __name__)


def _auth_or_401():
    """Helper to get current user or return 401"""
    user = get_current_user_flexible()
    if not user:
        return None, (jsonify({'error': 'Unauthorized'}), 401)
    return user, None


# ============================================================================
# Tax Bracket Engine Routes
# ============================================================================

@enhanced_bp.route('/tax-bracket/calculate', methods=['POST'])
def calculate_tax_bracket():
    """Calculate tax using progressive brackets"""
    user, error = _auth_or_401()
    if error:
        return error
    
    data = request.get_json() or {}
    income = data.get('income')
    country_code = data.get('country_code', 'US')
    tax_type = data.get('tax_type', 'income_tax')
    deductions = data.get('deductions', 0)
    
    if not income:
        return jsonify({'error': 'Income is required'}), 400
    
    try:
        tenant_id = get_current_tenant_id()
        config = get_tax_configuration_by_country_year(tenant_id, country_code, tax_type)
        
        if not config or not config.tax_brackets:
            return jsonify({'error': 'Tax configuration not found'}), 404
        
        result = calculate_progressive_tax(income, config.tax_brackets, country_code)
        
        # Apply deductions if provided
        if deductions > 0:
            optimized = optimize_tax_with_deductions(
                income,
                [{'amount': deductions, 'is_pre_tax': True}],
                config.tax_brackets,
                country_code
            )
            result['tax_savings'] = optimized.get('tax_savings', 0)
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Tax Table Management Routes
# ============================================================================

@enhanced_bp.route('/tax-tables', methods=['GET'])
def get_tax_tables():
    """Get tax tables"""
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
        
        tables = query.order_by(CountryTaxConfiguration.tax_year.desc()).all()
        return jsonify({'tables': [t.to_dict() for t in tables]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@enhanced_bp.route('/tax-tables/<int:table_id>/export', methods=['GET'])
def export_tax_table_endpoint(table_id):
    """Export tax table"""
    user, error = _auth_or_401()
    if error:
        return error
    
    format_type = request.args.get('format', 'json')
    
    try:
        result = export_tax_table(table_id, format_type)
        
        if format_type == 'json':
            return jsonify(result), 200
        else:
            from flask import send_file
            import os
            if os.path.exists(result):
                return send_file(result, as_attachment=True)
            return jsonify({'error': 'Export file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# E-filing Dashboard Routes
# ============================================================================

@enhanced_bp.route('/efiling/dashboard', methods=['GET'])
def get_efiling_dashboard():
    """Get e-filing dashboard data"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    country_code = request.args.get('country_code')
    
    try:
        records = []
        
        # Get India forms
        if not country_code or country_code == 'IN':
            from app.models import Form16Certificate, Form24QReturn
            form16s = Form16Certificate.query.filter_by(tenant_id=tenant_id).all()
            form24qs = Form24QReturn.query.filter_by(tenant_id=tenant_id).all()
            
            for f16 in form16s:
                records.append({
                    'id': f16.id,
                    'form_type': 'Form16',
                    'country_code': 'IN',
                    'form_id': f16.id,
                    'e_filing_status': f16.e_filing_status,
                    'e_filing_acknowledgment': f16.e_filing_acknowledgment,
                    'e_filed_at': f16.e_filed_at.isoformat() if f16.e_filed_at else None,
                })
            
            for f24q in form24qs:
                records.append({
                    'id': f24q.id,
                    'form_type': 'Form24Q',
                    'country_code': 'IN',
                    'form_id': f24q.id,
                    'e_filing_status': f24q.e_filing_status,
                    'e_filing_acknowledgment': f24q.e_filing_acknowledgment,
                    'e_filed_at': f24q.e_filed_at.isoformat() if f24q.e_filed_at else None,
                })
        
        # Get US forms
        if not country_code or country_code == 'US':
            from app.models import Form941Return, Form940Return
            form941s = Form941Return.query.filter_by(tenant_id=tenant_id).all()
            form940s = Form940Return.query.filter_by(tenant_id=tenant_id).all()
            
            for f941 in form941s:
                records.append({
                    'id': f941.id,
                    'form_type': 'Form941',
                    'country_code': 'US',
                    'form_id': f941.id,
                    'e_filing_status': f941.e_filing_status,
                    'e_filing_acknowledgment': f941.e_filing_acknowledgment,
                    'e_filed_at': f941.e_filed_at.isoformat() if f941.e_filed_at else None,
                })
            
            for f940 in form940s:
                records.append({
                    'id': f940.id,
                    'form_type': 'Form940',
                    'country_code': 'US',
                    'form_id': f940.id,
                    'e_filing_status': f940.e_filing_status,
                    'e_filing_acknowledgment': f940.e_filing_acknowledgment,
                    'e_filed_at': f940.e_filed_at.isoformat() if f940.e_filed_at else None,
                })
        
        return jsonify({'records': records}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Enhanced Reporting Routes
# ============================================================================

@enhanced_bp.route('/reports', methods=['GET'])
def get_reports():
    """Get generated reports"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    
    try:
        # In production, this would query a Reports table
        # For now, return empty list
        return jsonify({'reports': []}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@enhanced_bp.route('/reports/generate', methods=['POST'])
def generate_report():
    """Generate compliance report"""
    user, error = _auth_or_401()
    if error:
        return error
    
    tenant_id = get_current_tenant_id()
    if not tenant_id:
        return jsonify({'error': 'No tenant/organization found'}), 400
    
    data = request.get_json() or {}
    report_type = data.get('report_type')
    start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
    end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()
    country_code = data.get('country_code')
    output_format = data.get('output_format', 'pdf')
    
    if not report_type:
        return jsonify({'error': 'Report type is required'}), 400
    
    try:
        result = generate_compliance_report(
            tenant_id=tenant_id,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            country_code=country_code,
            output_format=output_format
        )
        return jsonify(result), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@enhanced_bp.route('/reports/<int:report_id>/download', methods=['GET'])
def download_report(report_id):
    """Download report"""
    user, error = _auth_or_401()
    if error:
        return error
    
    try:
        # In production, this would query a Reports table
        # For now, return error
        return jsonify({'error': 'Report download not yet implemented'}), 501
    except Exception as e:
        return jsonify({'error': str(e)}), 500

