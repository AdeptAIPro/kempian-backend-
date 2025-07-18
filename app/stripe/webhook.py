import logging
import os
import stripe
from flask import Blueprint, request, jsonify
from app.models import Plan, Tenant, db
from app.auth.cognito import cognito_admin_update_user_attributes

logger = logging.getLogger(__name__)

stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

webhook_bp = Blueprint('stripe_webhook', __name__)

@webhook_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        logger.error(f"Error verifying Stripe webhook: {str(e)}", exc_info=True)
        return jsonify({'error': 'Invalid payload or signature', 'details': str(e)}), 400
    try:
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            price_id = session['display_items'][0]['price']['id'] if 'display_items' in session else session['line_items']['data'][0]['price']['id']
            plan = Plan.query.filter_by(stripe_price_id=price_id).first()
            if not plan:
                return jsonify({'error': 'Plan not found'}), 400
            customer_id = session['customer']
            subscription_id = session['subscription']
            email = session['customer_email']
            # Create tenant
            tenant = Tenant(
                plan_id=plan.id,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                status='active'
            )
            db.session.add(tenant)
            db.session.commit()
            # Update Cognito user with tenant_id
            cognito_admin_update_user_attributes(email, {'custom:tenant_id': str(tenant.id)})
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logger.error(f"Error in Stripe webhook handler: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 