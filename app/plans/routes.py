import logging
from flask import Blueprint, jsonify, current_app
from app.simple_logger import get_logger
from app.models import Plan

logger = get_logger("plans")

plans_bp = Blueprint('plans', __name__)

@plans_bp.route('/', methods=['GET'])
def get_plans():
    print("PLANS ROUTE DB URI:", current_app.config['SQLALCHEMY_DATABASE_URI'])
    plans = Plan.query.all()
    return jsonify([plan.to_dict() for plan in plans]) 