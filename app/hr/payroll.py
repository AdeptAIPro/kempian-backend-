from flask import Blueprint, jsonify
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, Tenant


hr_payroll_bp = Blueprint('hr_payroll', __name__)


@hr_payroll_bp.route('/', methods=['GET'])
def list_payroll():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    db_user = User.query.filter_by(email=user.get('email')).first()
    tenant_id = get_current_tenant_id()
    tenant = Tenant.query.get(tenant_id) if tenant_id else None
    return jsonify({
        'payroll': [],
        'tenant': tenant.id if tenant else None,
        'user': db_user.id if db_user else None
    }), 200


