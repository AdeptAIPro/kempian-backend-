import logging
import os
import stripe
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import Plan, Tenant, User, SubscriptionTransaction, SubscriptionHistory, db
from app.auth.cognito import cognito_admin_update_user_attributes
from app.emails.subscription import send_subscription_purchase_receipt
from datetime import datetime, timedelta

logger = get_logger("stripe")

# Initialize Stripe with error handling
stripe_secret_key = os.getenv('STRIPE_SECRET_KEY')
if stripe_secret_key:
    stripe.api_key = stripe_secret_key
else:
    print("[DEBUG] STRIPE_SECRET_KEY not found, Stripe webhook functionality will be limited")

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
            
            # Extract price_id more robustly
            price_id = None
            if 'line_items' in session and 'data' in session['line_items'] and len(session['line_items']['data']) > 0:
                price_id = session['line_items']['data'][0]['price']['id']
            elif 'display_items' in session and len(session['display_items']) > 0:
                price_id = session['display_items'][0]['price']['id']
            else:
                logger.error(f"Could not extract price_id from session: {session}")
                return jsonify({'error': 'Could not extract price_id'}), 400
            
            # Find plan
            plan = Plan.query.filter_by(stripe_price_id=price_id).first()
            if not plan:
                logger.error(f"Plan not found for price_id: {price_id}")
                return jsonify({'error': 'Plan not found'}), 400
            
            customer_id = session['customer']
            subscription_id = session['subscription']
            email = session['customer_email']
            
            # Find existing user
            from app.models import User
            user = User.query.filter_by(email=email).first()
            if not user:
                logger.error(f"User not found for email: {email}")
                return jsonify({'error': 'User not found'}), 400
            
            # Update existing tenant or create new one
            old_plan_id = None
            if user.tenant_id:
                # Update existing tenant
                tenant = Tenant.query.get(user.tenant_id)
                if tenant:
                    old_plan_id = tenant.plan_id
                    tenant.plan_id = plan.id
                    tenant.stripe_customer_id = customer_id
                    tenant.stripe_subscription_id = subscription_id
                    tenant.status = 'active'
                    logger.info(f"Updated tenant {tenant.id} for user {email} to plan {plan.name}")
                else:
                    logger.error(f"Tenant {user.tenant_id} not found for user {email}")
                    return jsonify({'error': 'Tenant not found'}), 400
            else:
                # Create new tenant
                tenant = Tenant(
                    plan_id=plan.id,
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                    status='active'
                )
                db.session.add(tenant)
                db.session.flush()  # Get tenant ID
                user.tenant_id = tenant.id
                logger.info(f"Created new tenant {tenant.id} for user {email} with plan {plan.name}")
            
            # Create transaction record
            transaction_type = 'purchase' if not old_plan_id else 'upgrade'
            transaction = SubscriptionTransaction(
                tenant_id=tenant.id,
                user_id=user.id,
                transaction_type=transaction_type,
                stripe_payment_intent_id=session.get('payment_intent'),
                amount_cents=plan.price_cents,
                plan_id=plan.id,
                previous_plan_id=old_plan_id,
                status='succeeded',
                payment_method='card',
                notes=f'Stripe checkout session completed'
            )
            db.session.add(transaction)
            
            # Create subscription history record
            history_action = 'created' if not old_plan_id else 'upgraded'
            history = SubscriptionHistory(
                tenant_id=tenant.id,
                user_id=user.id,
                action=history_action,
                from_plan_id=old_plan_id,
                to_plan_id=plan.id,
                reason='Stripe checkout session completed'
            )
            db.session.add(history)
            
            db.session.commit()
            
            # Update Cognito user with tenant_id
            try:
                cognito_admin_update_user_attributes(email, {'custom:tenant_id': str(tenant.id)})
                logger.info(f"Updated Cognito user {email} with tenant_id {tenant.id}")
            except Exception as e:
                logger.error(f"Failed to update Cognito user {email}: {e}")
            
            # Send purchase receipt email
            try:
                next_billing_date = None
                if plan.billing_cycle == 'monthly':
                    next_billing_date = (datetime.utcnow() + timedelta(days=30)).strftime('%B %d, %Y')
                elif plan.billing_cycle == 'yearly':
                    next_billing_date = (datetime.utcnow() + timedelta(days=365)).strftime('%B %d, %Y')
                
                send_subscription_purchase_receipt(
                    to_email=email,
                    user_name=email.split('@')[0],  # Use email prefix as name
                    plan_name=plan.name,
                    amount_display=f"${plan.price_cents / 100:.2f}",
                    transaction_id=session.get('id', 'N/A'),
                    payment_method='Card',
                    purchase_date=datetime.utcnow().strftime('%B %d, %Y at %I:%M %p'),
                    billing_cycle=plan.billing_cycle.title(),
                    next_billing_date=next_billing_date,
                    jd_quota=plan.jd_quota_per_month,
                    max_subaccounts=plan.max_subaccounts,
                    dashboard_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:8081')}/dashboard"
                )
                logger.info(f"Purchase receipt email sent to {email}")
            except Exception as email_error:
                logger.error(f"Failed to send purchase receipt email to {email}: {str(email_error)}")
            
            return jsonify({'status': 'success'}), 200
            
        elif event['type'] == 'invoice.payment_succeeded':
            # Handle recurring payments
            invoice = event['data']['object']
            subscription_id = invoice['subscription']
            customer_id = invoice['customer']
            
            # Find tenant by subscription_id
            tenant = Tenant.query.filter_by(stripe_subscription_id=subscription_id).first()
            if tenant:
                tenant.status = 'active'
                db.session.commit()
                logger.info(f"Updated tenant {tenant.id} status to active for subscription {subscription_id}")
            
            return jsonify({'status': 'success'}), 200
            
        elif event['type'] == 'customer.subscription.deleted':
            # Handle subscription cancellation
            subscription = event['data']['object']
            subscription_id = subscription['id']
            
            # Find tenant by subscription_id
            tenant = Tenant.query.filter_by(stripe_subscription_id=subscription_id).first()
            if tenant:
                tenant.status = 'cancelled'
                db.session.commit()
                logger.info(f"Updated tenant {tenant.id} status to cancelled for subscription {subscription_id}")
            
            return jsonify({'status': 'success'}), 200
            
        elif event['type'] == 'customer.subscription.updated':
            # Handle subscription updates
            subscription = event['data']['object']
            subscription_id = subscription['id']
            status = subscription['status']
            
            # Find tenant by subscription_id
            tenant = Tenant.query.filter_by(stripe_subscription_id=subscription_id).first()
            if tenant:
                tenant.status = status
                db.session.commit()
                logger.info(f"Updated tenant {tenant.id} status to {status} for subscription {subscription_id}")
            
            return jsonify({'status': 'success'}), 200
            
        else:
            logger.info(f"Unhandled event type: {event['type']}")
            return jsonify({'status': 'ignored'}), 200
            
    except Exception as e:
        logger.error(f"Error in Stripe webhook handler: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 