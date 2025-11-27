from flask import Blueprint, jsonify
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import Tenant


hr_financials_bp = Blueprint('hr_financials', __name__)


@hr_financials_bp.route('/', methods=['GET'])
def list_financials():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    tenant_id = get_current_tenant_id()
    t = Tenant.query.get(tenant_id) if tenant_id else None
    return jsonify({'financials': [], 'tenant': t.id if t else None}), 200


