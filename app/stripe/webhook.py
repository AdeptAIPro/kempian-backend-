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

def get_payment_method_from_stripe(payment_intent_id=None, invoice=None, session=None):
    """
    Extract payment method type from Stripe objects.
    Returns a human-readable payment method string (e.g., 'card', 'us_bank_account', 'link').
    Falls back to 'card' if unable to determine.
    """
    try:
        # Try to get from payment intent
        if payment_intent_id:
            try:
                payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id, expand=['payment_method'])
                if hasattr(payment_intent, 'payment_method') and payment_intent.payment_method:
                    pm = payment_intent.payment_method
                    if isinstance(pm, str):
                        pm_obj = stripe.PaymentMethod.retrieve(pm)
                    else:
                        pm_obj = pm
                    
                    pm_type = pm_obj.type if hasattr(pm_obj, 'type') else pm_obj.get('type')
                    if pm_type:
                        # Map Stripe payment method types to readable names
                        pm_type_map = {
                            'card': 'card',
                            'us_bank_account': 'ach',
                            'link': 'link',
                            'cashapp': 'cashapp',
                            'paypal': 'paypal'
                        }
                        return pm_type_map.get(pm_type, pm_type)
                
                # Fallback: check payment_method_types array
                if hasattr(payment_intent, 'payment_method_types') and payment_intent.payment_method_types:
                    pm_types = payment_intent.payment_method_types
                    if isinstance(pm_types, list) and len(pm_types) > 0:
                        pm_type = pm_types[0]
                        pm_type_map = {
                            'card': 'card',
                            'us_bank_account': 'ach',
                            'link': 'link'
                        }
                        return pm_type_map.get(pm_type, pm_type)
            except Exception as e:
                logger.warning(f"Could not retrieve payment method from payment intent {payment_intent_id}: {str(e)}")
        
        # Try to get from invoice
        if invoice:
            try:
                payment_intent_id_from_invoice = invoice.get('payment_intent')
                if payment_intent_id_from_invoice:
                    if isinstance(payment_intent_id_from_invoice, dict):
                        payment_intent_id_from_invoice = payment_intent_id_from_invoice.get('id')
                    return get_payment_method_from_stripe(payment_intent_id=payment_intent_id_from_invoice)
            except Exception as e:
                logger.warning(f"Could not extract payment method from invoice: {str(e)}")
        
        # Try to get from checkout session
        if session:
            try:
                pm_types = session.get('payment_method_types', [])
                if pm_types and len(pm_types) > 0:
                    pm_type = pm_types[0]
                    pm_type_map = {
                        'card': 'card',
                        'us_bank_account': 'ach',
                        'link': 'link'
                    }
                    return pm_type_map.get(pm_type, pm_type)
            except Exception as e:
                logger.warning(f"Could not extract payment method from session: {str(e)}")
        
    except Exception as e:
        logger.warning(f"Error extracting payment method: {str(e)}")
    
    # Default fallback
    return 'card'

@webhook_bp.route('/stripe', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    
    # Log webhook receipt for debugging
    logger.info(f"[WEBHOOK] Received webhook request. Headers: {dict(request.headers)}")
    logger.info(f"[WEBHOOK] Payload size: {len(payload)} bytes")
    logger.info(f"[WEBHOOK] Signature header present: {bool(sig_header)}")
    logger.info(f"[WEBHOOK] Webhook secret configured: {bool(STRIPE_WEBHOOK_SECRET)}")
    
    try:
        if not STRIPE_WEBHOOK_SECRET:
            logger.error("[WEBHOOK] STRIPE_WEBHOOK_SECRET not configured!")
            return jsonify({'error': 'Webhook secret not configured'}), 500
            
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
        logger.info(f"[WEBHOOK] Event verified successfully. Event ID: {event.get('id')}, Type: {event.get('type')}")
    except ValueError as e:
        logger.error(f"[WEBHOOK] Invalid payload: {str(e)}")
        return jsonify({'error': 'Invalid payload', 'details': str(e)}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"[WEBHOOK] Invalid signature: {str(e)}")
        logger.error(f"[WEBHOOK] Expected secret: {STRIPE_WEBHOOK_SECRET[:10]}... (first 10 chars)")
        return jsonify({'error': 'Invalid signature', 'details': str(e)}), 400
    except Exception as e:
        logger.error(f"[WEBHOOK] Error verifying Stripe webhook: {str(e)}", exc_info=True)
        return jsonify({'error': 'Invalid payload or signature', 'details': str(e)}), 400
    
    try:
        event_type = event.get('type', 'unknown')
        logger.info(f"Processing Stripe webhook event: {event_type} (event_id: {event.get('id', 'N/A')})")
        
        if event_type == 'checkout.session.completed':
            session = event['data']['object']
            
            # Retrieve full session with expanded line items to get price_id
            try:
                expanded_session = stripe.checkout.Session.retrieve(
                    session['id'],
                    expand=['line_items']
                )
            except Exception as e:
                logger.error(f"Error retrieving expanded session: {str(e)}")
                expanded_session = session
            
            # Extract price_id more robustly
            price_id = None
            if hasattr(expanded_session, 'line_items') and expanded_session.line_items:
                line_items_data = expanded_session.line_items.data if hasattr(expanded_session.line_items, 'data') else expanded_session.line_items
                if line_items_data and len(line_items_data) > 0:
                    price_id = line_items_data[0].price.id if hasattr(line_items_data[0].price, 'id') else line_items_data[0].get('price', {}).get('id')
            elif 'line_items' in session and 'data' in session['line_items'] and len(session['line_items']['data']) > 0:
                price_id = session['line_items']['data'][0]['price']['id']
            elif 'display_items' in session and len(session['display_items']) > 0:
                price_id = session['display_items'][0]['price']['id']
            
            if not price_id:
                logger.error(f"Could not extract price_id from session: {session}")
                return jsonify({'error': 'Could not extract price_id'}), 400
            
            # Find plan
            plan = Plan.query.filter_by(stripe_price_id=price_id).first()
            if not plan:
                logger.error(f"Plan not found for price_id: {price_id}")
                return jsonify({'error': 'Plan not found'}), 400
            
            customer_id = session.get('customer') or session.get('customer_id')
            subscription_id = session.get('subscription') or session.get('subscription_id')
            
            # Extract email more robustly
            email = None
            if 'customer_email' in session:
                email = session['customer_email']
            elif 'customer_details' in session and session['customer_details']:
                email = session['customer_details'].get('email')
            elif customer_id:
                # Fallback: retrieve customer to get email
                try:
                    customer = stripe.Customer.retrieve(customer_id)
                    email = customer.email
                except Exception as e:
                    logger.error(f"Error retrieving customer {customer_id}: {str(e)}")
            
            if not email:
                logger.error(f"Could not extract email from session: {session}")
                return jsonify({'error': 'Could not extract customer email'}), 400
            
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
            
            # Retrieve invoice details from Stripe for proper billing information
            invoice_id = None
            invoice_number = None
            receipt_url = None
            invoice_url = None
            payment_intent_id = session.get('payment_intent')
            invoice_obj = None
            
            try:
                # Get the invoice from the subscription
                if subscription_id:
                    try:
                        subscription_obj = stripe.Subscription.retrieve(subscription_id)
                        latest_invoice_id = subscription_obj.get('latest_invoice')
                        
                        if latest_invoice_id:
                            if isinstance(latest_invoice_id, str):
                                invoice_id = latest_invoice_id
                            else:
                                invoice_id = latest_invoice_id.get('id') if hasattr(latest_invoice_id, 'get') else str(latest_invoice_id)
                            
                            # Retrieve full invoice details with error handling
                            try:
                                invoice_obj = stripe.Invoice.retrieve(invoice_id, expand=['payment_intent'])
                                invoice_number = invoice_obj.get('number') or invoice_obj.get('id')
                                receipt_url = invoice_obj.get('receipt_url')
                                invoice_url = invoice_obj.get('hosted_invoice_url')
                                
                                # If payment_intent not set, get it from invoice
                                if not payment_intent_id:
                                    payment_intent_id = invoice_obj.get('payment_intent')
                                    if isinstance(payment_intent_id, dict):
                                        payment_intent_id = payment_intent_id.get('id')
                                
                                logger.info(f"Retrieved invoice details: invoice_id={invoice_id}, invoice_number={invoice_number}, receipt_url={bool(receipt_url)}, invoice_url={bool(invoice_url)}")
                            except stripe.error.InvalidRequestError as invoice_retrieve_error:
                                logger.warning(f"Could not retrieve invoice {invoice_id}: {str(invoice_retrieve_error)}")
                            except Exception as invoice_retrieve_error:
                                logger.warning(f"Unexpected error retrieving invoice {invoice_id}: {str(invoice_retrieve_error)}")
                    except stripe.error.InvalidRequestError as sub_error:
                        logger.warning(f"Could not retrieve subscription {subscription_id}: {str(sub_error)}")
                    except Exception as sub_error:
                        logger.warning(f"Unexpected error retrieving subscription {subscription_id}: {str(sub_error)}")
            except Exception as invoice_error:
                logger.error(f"Error in invoice retrieval process: {str(invoice_error)}", exc_info=True)
            
            # Generate bill/invoice number if not available from Stripe
            if not invoice_number:
                # Format: INV-YYYYMMDD-XXXXX (date + transaction ID)
                date_prefix = datetime.utcnow().strftime('%Y%m%d')
                txn_suffix = str(tenant.id).zfill(4) + str(user.id).zfill(4)
                invoice_number = f"INV-{date_prefix}-{txn_suffix}"
                logger.info(f"Generated invoice number: {invoice_number}")
            
            # Extract payment method from Stripe
            payment_method = get_payment_method_from_stripe(
                payment_intent_id=payment_intent_id,
                invoice=invoice_obj,
                session=session
            )
            logger.info(f"Detected payment method: {payment_method} for checkout session {session.get('id')}")
            
            # Check if transaction already exists (idempotency check)
            existing_transaction = SubscriptionTransaction.query.filter_by(
                stripe_invoice_id=invoice_id,
                status='succeeded'
            ).first() if invoice_id else None
            
            if not existing_transaction:
                # Create transaction record with all billing details
                transaction_type = 'purchase' if not old_plan_id else 'upgrade'
                transaction = SubscriptionTransaction(
                    tenant_id=tenant.id,
                    user_id=user.id,
                    transaction_type=transaction_type,
                    stripe_payment_intent_id=payment_intent_id,
                    stripe_invoice_id=invoice_id,
                    amount_cents=plan.price_cents,
                    plan_id=plan.id,
                    previous_plan_id=old_plan_id,
                    status='succeeded',
                    payment_method=payment_method,
                    receipt_url=receipt_url,
                    invoice_url=invoice_url,
                    notes=f'Stripe checkout session completed. Invoice: {invoice_number}'
                )
                db.session.add(transaction)
                db.session.flush()  # Get transaction ID for invoice number update if needed
                logger.info(f"Created new transaction record for tenant {tenant.id}")
            else:
                logger.info(f"Transaction already exists for invoice {invoice_id}, skipping duplicate")
                transaction = existing_transaction
            
            # Create subscription history record (check for duplicates)
            history_action = 'created' if not old_plan_id else 'upgraded'
            existing_history = SubscriptionHistory.query.filter_by(
                tenant_id=tenant.id,
                action=history_action,
                to_plan_id=plan.id
            ).first()
            
            if not existing_history:
                history = SubscriptionHistory(
                    tenant_id=tenant.id,
                    user_id=user.id,
                    action=history_action,
                    from_plan_id=old_plan_id,
                    to_plan_id=plan.id,
                    reason='Stripe checkout session completed'
                )
                db.session.add(history)
                logger.info(f"Created subscription history record for tenant {tenant.id}")
            else:
                logger.info(f"Subscription history already exists for tenant {tenant.id}, skipping duplicate")
            
            db.session.commit()
            logger.info(f"[WEBHOOK] Successfully committed subscription update for user {email}, tenant {tenant.id}, plan {plan.name}")
            
            # Update Cognito user with tenant_id
            try:
                cognito_admin_update_user_attributes(email, {'custom:tenant_id': str(tenant.id)})
                logger.info(f"[WEBHOOK] Updated Cognito user {email} with tenant_id {tenant.id}")
            except Exception as e:
                logger.error(f"[WEBHOOK] Failed to update Cognito user {email}: {e}", exc_info=True)
                # Don't fail the webhook if Cognito update fails
            
            # Send purchase receipt email with proper invoice/bill details
            logger.info(f"[WEBHOOK] Attempting to send purchase receipt email to {email}")
            try:
                next_billing_date = None
                if plan.billing_cycle == 'monthly':
                    next_billing_date = (datetime.utcnow() + timedelta(days=30)).strftime('%B %d, %Y')
                elif plan.billing_cycle == 'yearly':
                    next_billing_date = (datetime.utcnow() + timedelta(days=365)).strftime('%B %d, %Y')
                
                # Use invoice number as transaction ID for proper billing reference
                transaction_id_display = invoice_number or f"TXN-{transaction.id:06d}" or session.get('id', 'N/A')
                
                # Format payment method for display (capitalize first letter, uppercase ACH)
                payment_method_display = payment_method.upper() if payment_method == 'ach' else payment_method.capitalize()
                
                email_sent = send_subscription_purchase_receipt(
                    to_email=email,
                    user_name=email.split('@')[0],  # Use email prefix as name
                    plan_name=plan.name,
                    amount_display=f"${plan.price_cents / 100:.2f}",
                    transaction_id=transaction_id_display,
                    payment_method=payment_method_display,
                    purchase_date=datetime.utcnow().strftime('%B %d, %Y at %I:%M %p'),
                    billing_cycle=plan.billing_cycle.title(),
                    next_billing_date=next_billing_date,
                    jd_quota=plan.jd_quota_per_month,
                    max_subaccounts=plan.max_subaccounts,
                    dashboard_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:8081')}/dashboard",
                    invoice_number=invoice_number,
                    invoice_url=invoice_url,
                    receipt_url=receipt_url
                )
                
                if email_sent:
                    logger.info(f"[WEBHOOK] Purchase receipt email sent successfully to {email} with invoice number {invoice_number}")
                else:
                    logger.error(f"[WEBHOOK] Purchase receipt email FAILED to send to {email} - both SMTP and SES failed. Check email logs for details.")
                    # Don't fail the webhook if email fails - subscription is already updated
                    logger.warning(f"[WEBHOOK] Continuing despite email failure - subscription update was successful")
            except Exception as email_error:
                logger.error(f"[WEBHOOK] Exception while sending purchase receipt email to {email}: {str(email_error)}", exc_info=True)
                # Don't fail the webhook if email fails - subscription is already updated
                logger.warning(f"[WEBHOOK] Continuing despite email failure - subscription update was successful")
            
            logger.info(f"[WEBHOOK] checkout.session.completed processed successfully for user {email}")
            return jsonify({'status': 'success', 'message': f'Subscription activated for {email}'}), 200
            
        elif event_type == 'invoice.payment_succeeded':
            # Handle recurring payments - store invoice details properly
            invoice = event['data']['object']
            subscription_id = invoice.get('subscription')
            customer_id = invoice.get('customer')
            invoice_id = invoice.get('id')
            invoice_number = invoice.get('number') or invoice_id
            amount_paid = invoice.get('amount_paid', 0)
            receipt_url = invoice.get('receipt_url')
            invoice_url = invoice.get('hosted_invoice_url')
            payment_intent_id = invoice.get('payment_intent')
            
            # Extract payment method from invoice
            # Retrieve full invoice with expanded payment_intent if needed
            invoice_obj = None
            try:
                if payment_intent_id:
                    # If payment_intent is just an ID string, we'll retrieve it in the helper
                    # Otherwise, if it's already expanded, we can use it
                    if isinstance(payment_intent_id, str):
                        invoice_obj = invoice
                    else:
                        invoice_obj = invoice
            except Exception as e:
                logger.warning(f"Could not prepare invoice object for payment method extraction: {str(e)}")
            
            payment_method = get_payment_method_from_stripe(
                payment_intent_id=payment_intent_id if isinstance(payment_intent_id, str) else None,
                invoice=invoice_obj or invoice
            )
            logger.info(f"Detected payment method: {payment_method} for recurring payment invoice {invoice_id}")
            
            # Validate subscription_id before querying
            if not subscription_id:
                logger.warning(f"invoice.payment_succeeded event has no subscription_id. Invoice ID: {invoice_id}, Customer: {customer_id}")
                # Try to find tenant by customer_id as fallback
                if customer_id:
                    tenant = Tenant.query.filter_by(stripe_customer_id=customer_id).first()
                    if tenant:
                        logger.info(f"Found tenant {tenant.id} by customer_id {customer_id} for invoice {invoice_id}")
                    else:
                        logger.warning(f"Tenant not found for customer_id {customer_id} in invoice.payment_succeeded event")
                        return jsonify({'status': 'ignored', 'reason': 'Tenant not found'}), 200
                else:
                    logger.warning(f"Both subscription_id and customer_id are missing in invoice.payment_succeeded event")
                    return jsonify({'status': 'ignored', 'reason': 'No subscription or customer ID'}), 200
            else:
                # Find tenant by subscription_id
                tenant = Tenant.query.filter_by(stripe_subscription_id=subscription_id).first()
            
            if tenant:
                tenant.status = 'active'
                
                # Find the plan for this tenant
                plan = Plan.query.get(tenant.plan_id)
                
                # Create transaction record for recurring payment
                if plan:
                    # Find the owner user for this tenant
                    owner_user = User.query.filter_by(tenant_id=tenant.id, role='owner').first()
                    if not owner_user:
                        # Fallback to any user in the tenant
                        owner_user = User.query.filter_by(tenant_id=tenant.id).first()
                    
                    if owner_user:
                        transaction = SubscriptionTransaction(
                            tenant_id=tenant.id,
                            user_id=owner_user.id,
                            transaction_type='renewal',
                            stripe_payment_intent_id=payment_intent_id if isinstance(payment_intent_id, str) else None,
                            stripe_invoice_id=invoice_id,
                            amount_cents=amount_paid,
                            plan_id=plan.id,
                            status='succeeded',
                            payment_method=payment_method,
                            receipt_url=receipt_url,
                            invoice_url=invoice_url,
                            notes=f'Recurring payment - Invoice: {invoice_number}'
                        )
                        db.session.add(transaction)
                        logger.info(f"Created renewal transaction for tenant {tenant.id}, invoice {invoice_number} with payment method {payment_method}")
                
                db.session.commit()
                logger.info(f"Updated tenant {tenant.id} status to active for subscription {subscription_id}, invoice {invoice_number}")
            else:
                logger.warning(f"Tenant not found for subscription {subscription_id} in invoice.payment_succeeded event")
            
            return jsonify({'status': 'success'}), 200
            
        elif event_type == 'customer.subscription.deleted':
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
            
        elif event_type == 'customer.subscription.updated':
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
            logger.info(f"Unhandled event type: {event_type} - ignoring")
            return jsonify({'status': 'ignored', 'event_type': event_type}), 200
            
    except KeyError as key_error:
        logger.error(f"Missing required key in Stripe webhook event: {str(key_error)}", exc_info=True)
        return jsonify({'error': f'Missing required field: {str(key_error)}'}), 400
    except stripe.error.StripeError as stripe_error:
        logger.error(f"Stripe API error in webhook handler: {str(stripe_error)}", exc_info=True)
        return jsonify({'error': f'Stripe API error: {str(stripe_error)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in Stripe webhook handler: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error processing webhook'}), 500 