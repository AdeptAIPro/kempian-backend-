import logging
import os
import stripe

stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
# print("[DEBUG] STRIPE_SECRET_KEY:", repr(stripe.api_key))
# print("[DEBUG] stripe.api_key:", stripe.api_key)
print("[DEBUG] stripe module:", stripe)
print("[DEBUG] stripe module file:", getattr(stripe, '__file__', 'no __file__'))
try:
    print("[DEBUG] stripe.Account.retrieve():", stripe.Account.retrieve())
except Exception as e:
    print("[DEBUG] Stripe API test call failed:", e)
from flask import Blueprint, request, jsonify
from app.models import Plan
from app.db import db
from app.utils import get_current_user

logger = logging.getLogger(__name__)

stripe_bp = Blueprint('stripe', __name__)

@stripe_bp.route('/create-session', methods=['POST'])
def create_checkout_session():
    data = request.get_json()
    plan_id = data.get('plan_id')
    success_url = data.get('success_url')
    cancel_url = data.get('cancel_url')
    auth_header = request.headers.get('Authorization')
    user_jwt = get_current_user()
    print(f"[DEBUG] /checkout/create-session: Authorization header: {auth_header}")
    print(f"[DEBUG] /checkout/create-session: Decoded user_jwt: {user_jwt}")
    if not plan_id or not success_url or not cancel_url:
        return jsonify({'error': 'Missing required fields'}), 400
    if not user_jwt or not user_jwt.get('email'):
        print(f"[ERROR] /checkout/create-session: Unauthorized. user_jwt: {user_jwt}, auth_header: {auth_header}")
        return jsonify({'error': 'Unauthorized: Invalid or missing token. See backend logs for details.'}), 401
    plan = Plan.query.get(plan_id)
    if not plan:
        return jsonify({'error': 'Plan not found'}), 404
    if plan.price_cents == 0:
        return jsonify({'error': 'Free Trial plan is assigned automatically. No checkout required.'}), 400
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': plan.stripe_price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=True
        )
        return jsonify({'url': session.url}), 200
    except Exception as e:
        logger.error(f"Error in /stripe/create-session: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 