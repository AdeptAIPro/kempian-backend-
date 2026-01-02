from flask import Blueprint, jsonify
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User


hr_client_companies_bp = Blueprint('hr_client_companies', __name__)


@hr_client_companies_bp.route('/', methods=['GET'])
def list_companies():
    user = get_current_user_flexible() or get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    tenant_id = get_current_tenant_id()
    query = User.query
    if tenant_id:
        query = query.filter_by(tenant_id=tenant_id)
    companies = query.filter(User.company_name.isnot(None), User.company_name != '').distinct(User.company_name).all()
    names = sorted({u.company_name for u in companies if u.company_name})
    return jsonify({'companies': names}), 200


