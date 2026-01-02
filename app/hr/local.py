from flask import Blueprint, jsonify


hr_local_bp = Blueprint('hr_local', __name__)


@hr_local_bp.route('/ping', methods=['GET'])
def ping():
    return jsonify({'ok': True, 'service': 'hr-local'}), 200


