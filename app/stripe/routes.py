import logging
import os
import stripe

# Initialize Stripe with error handling
stripe_secret_key = os.getenv('STRIPE_SECRET_KEY')
if stripe_secret_key:
    stripe.api_key = stripe_secret_key
    print("[DEBUG] stripe module:", stripe)
    print("[DEBUG] stripe module file:", getattr(stripe, '__file__', 'no __file__'))
    try:
        print("[DEBUG] stripe.Account.retrieve():", stripe.Account.retrieve())
    except Exception as e:
        print("[DEBUG] Stripe API test call failed:", e)
else:
    print("[DEBUG] STRIPE_SECRET_KEY not found, Stripe functionality will be limited")
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import Plan
from app.db import db
from app.utils import get_current_user
from datetime import datetime

logger = get_logger("stripe")

stripe_bp = Blueprint('stripe', __name__)

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
        user_email = user_jwt.get('email')
        
        session = stripe.checkout.Session.create(
            payment_method_types=['card', 'us_bank_account'],
            line_items=[{
                'price': plan.stripe_price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=True,
            customer_email=user_email,  # Pre-fill email for better UX
            metadata={
                'user_email': user_email,
                'plan_id': str(plan_id),
                'plan_name': plan.name
            }
        )
        return jsonify({'url': session.url}), 200
    except Exception as e:
        logger.error(f"Error in /stripe/create-session: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@stripe_bp.route('/sync-subscription', methods=['POST'])
def sync_subscription():
    """Manually sync subscription status from Stripe (useful if webhook didn't fire)"""
    debug_log = {
        'timestamp': datetime.utcnow().isoformat(),
        'steps': [],
        'errors': [],
        'success': False
    }
    
    try:
        debug_log['steps'].append({'step': '1', 'action': 'Get current user', 'status': 'in_progress'})
        user_jwt = get_current_user()
        if not user_jwt or not user_jwt.get('email'):
            debug_log['errors'].append('Unauthorized: No user JWT or email')
            debug_log['steps'].append({'step': '1', 'action': 'Get current user', 'status': 'error', 'details': 'Unauthorized'})
            return jsonify({'error': 'Unauthorized', 'debug': debug_log}), 401
        
        debug_log['steps'].append({'step': '1', 'action': 'Get current user', 'status': 'success', 'details': f"User email: {user_jwt.get('email')}"})
        
        from app.models import User, Tenant, Plan, SubscriptionTransaction, SubscriptionHistory
        from app.auth.cognito import cognito_admin_update_user_attributes
        from app.emails.subscription import send_subscription_purchase_receipt
        from datetime import timedelta
        import os
        
        user_email = user_jwt.get('email')
        debug_log['steps'].append({'step': '2', 'action': 'Find user in database', 'status': 'in_progress'})
        user = User.query.filter_by(email=user_email).first()
        if not user:
            debug_log['errors'].append(f'User not found for email: {user_email}')
            debug_log['steps'].append({'step': '2', 'action': 'Find user in database', 'status': 'error', 'details': f'User not found for {user_email}'})
            return jsonify({'error': 'User not found', 'debug': debug_log}), 404
        
        debug_log['steps'].append({'step': '2', 'action': 'Find user in database', 'status': 'success', 'details': f'User ID: {user.id}, Tenant ID: {user.tenant_id}'})
        
        # Get customer ID from user's tenant or Stripe
        customer_id = None
        subscription_id = None
        
        debug_log['steps'].append({'step': '3', 'action': 'Get Stripe customer ID', 'status': 'in_progress'})
        if user.tenant_id:
            tenant = Tenant.query.get(user.tenant_id)
            if tenant:
                customer_id = tenant.stripe_customer_id
                subscription_id = tenant.stripe_subscription_id
                debug_log['steps'].append({'step': '3a', 'action': 'Get from tenant', 'status': 'success', 'details': f'Customer ID: {customer_id}, Subscription ID: {subscription_id}'})
        
        # If no customer ID, try to find by email in Stripe
        if not customer_id:
            debug_log['steps'].append({'step': '3b', 'action': 'Search Stripe by email', 'status': 'in_progress'})
            try:
                customers = stripe.Customer.list(email=user_email, limit=1)
                if customers.data:
                    customer_id = customers.data[0].id
                    debug_log['steps'].append({'step': '3b', 'action': 'Search Stripe by email', 'status': 'success', 'details': f'Found customer: {customer_id}'})
                    logger.info(f"Found Stripe customer {customer_id} for email {user_email}")
                else:
                    debug_log['steps'].append({'step': '3b', 'action': 'Search Stripe by email', 'status': 'error', 'details': 'No customers found in Stripe'})
            except Exception as e:
                debug_log['errors'].append(f'Error searching Stripe: {str(e)}')
                debug_log['steps'].append({'step': '3b', 'action': 'Search Stripe by email', 'status': 'error', 'details': str(e)})
                logger.error(f"Error searching for Stripe customer: {str(e)}")
        
        if not customer_id:
            debug_log['steps'].append({'step': '3', 'action': 'Get Stripe customer ID', 'status': 'error', 'details': 'No customer ID found'})
            return jsonify({'error': 'No Stripe customer found for this user', 'debug': debug_log}), 404
        
        debug_log['steps'].append({'step': '3', 'action': 'Get Stripe customer ID', 'status': 'success', 'details': f'Customer ID: {customer_id}'})
        
        # Get customer's subscriptions
        debug_log['steps'].append({'step': '4', 'action': 'Retrieve Stripe customer and subscriptions', 'status': 'in_progress'})
        try:
            customer = stripe.Customer.retrieve(customer_id)
            debug_log['steps'].append({'step': '4a', 'action': 'Retrieve customer', 'status': 'success', 'details': f'Customer: {customer.email if hasattr(customer, "email") else customer_id}'})
            
            subscriptions = stripe.Subscription.list(customer=customer_id, status='active', limit=1, expand=['data.items.data.price'])
            debug_log['steps'].append({'step': '4b', 'action': 'List active subscriptions', 'status': 'success', 'details': f'Found {len(subscriptions.data)} active subscription(s)'})
            
            if not subscriptions.data:
                debug_log['errors'].append('No active subscription found in Stripe')
                debug_log['steps'].append({'step': '4', 'action': 'Retrieve Stripe customer and subscriptions', 'status': 'error', 'details': 'No active subscriptions'})
                return jsonify({'error': 'No active subscription found in Stripe', 'debug': debug_log}), 404
            
            subscription = subscriptions.data[0]
            subscription_id = subscription.id
            debug_log['steps'].append({'step': '4c', 'action': 'Get subscription details', 'status': 'success', 'details': f'Subscription ID: {subscription_id}'})
            
            # Retrieve full subscription with expanded items to get price_id
            try:
                expanded_subscription = stripe.Subscription.retrieve(
                    subscription_id,
                    expand=['items.data.price']
                )
                subscription = expanded_subscription
                debug_log['steps'].append({'step': '4d', 'action': 'Retrieve expanded subscription', 'status': 'success', 'details': 'Subscription retrieved with expanded items'})
            except Exception as e:
                debug_log['steps'].append({'step': '4d', 'action': 'Retrieve expanded subscription', 'status': 'warning', 'details': f'Could not expand: {str(e)}, using basic subscription'})
                logger.warning(f"Could not retrieve expanded subscription: {str(e)}")
            
            # Get the price ID from subscription - handle both dict and object access
            debug_log['steps'].append({'step': '5', 'action': 'Extract price ID', 'status': 'in_progress'})
            price_id = None
            
            # Try accessing as object first
            if hasattr(subscription, 'items'):
                items = subscription.items
                # Check if items is a method or property
                if callable(items):
                    items = items()
                # Access items.data
                if hasattr(items, 'data') and items.data:
                    price_id = items.data[0].price.id if hasattr(items.data[0].price, 'id') else None
                elif hasattr(items, '__iter__'):
                    # If items is iterable directly
                    items_list = list(items)
                    if items_list and len(items_list) > 0:
                        first_item = items_list[0]
                        if hasattr(first_item, 'price'):
                            price_id = first_item.price.id if hasattr(first_item.price, 'id') else None
            
            # Try accessing as dictionary if object access failed
            if not price_id:
                subscription_dict = subscription if isinstance(subscription, dict) else subscription.to_dict() if hasattr(subscription, 'to_dict') else {}
                if 'items' in subscription_dict:
                    items_data = subscription_dict['items']
                    if isinstance(items_data, dict) and 'data' in items_data and len(items_data['data']) > 0:
                        price_id = items_data['data'][0].get('price', {}).get('id')
                    elif isinstance(items_data, list) and len(items_data) > 0:
                        price_id = items_data[0].get('price', {}).get('id')
            
            if not price_id:
                debug_log['errors'].append('Could not extract price ID from subscription')
                debug_log['steps'].append({'step': '5', 'action': 'Extract price ID', 'status': 'error', 'details': f'No price ID found. Subscription items: {str(subscription.items) if hasattr(subscription, "items") else "N/A"}'})
                return jsonify({'error': 'Could not extract price ID from subscription', 'debug': debug_log}), 400
            
            debug_log['steps'].append({'step': '5', 'action': 'Extract price ID', 'status': 'success', 'details': f'Price ID: {price_id}'})
            
            # Find plan by price ID
            debug_log['steps'].append({'step': '6', 'action': 'Find plan in database', 'status': 'in_progress'})
            plan = Plan.query.filter_by(stripe_price_id=price_id).first()
            if not plan:
                debug_log['errors'].append(f'Plan not found for price_id: {price_id}')
                debug_log['steps'].append({'step': '6', 'action': 'Find plan in database', 'status': 'error', 'details': f'Price ID {price_id} not found in database'})
                logger.error(f"Plan not found for price_id: {price_id}")
                return jsonify({'error': f'Plan not found for price_id: {price_id}. Please check your pricing plans migration.', 'debug': debug_log}), 404
            
            debug_log['steps'].append({'step': '6', 'action': 'Find plan in database', 'status': 'success', 'details': f'Plan: {plan.name} (ID: {plan.id})'})
            
            # Update or create tenant
            debug_log['steps'].append({'step': '7', 'action': 'Update or create tenant', 'status': 'in_progress'})
            old_plan_id = None
            if user.tenant_id:
                tenant = Tenant.query.get(user.tenant_id)
                if tenant:
                    old_plan_id = tenant.plan_id
                    tenant.plan_id = plan.id
                    tenant.stripe_customer_id = customer_id
                    tenant.stripe_subscription_id = subscription_id
                    tenant.status = 'active'
                    debug_log['steps'].append({'step': '7', 'action': 'Update or create tenant', 'status': 'success', 'details': f'Updated tenant {tenant.id} from plan {old_plan_id} to {plan.id}'})
                    logger.info(f"Updated tenant {tenant.id} for user {user_email} to plan {plan.name}")
            else:
                tenant = Tenant(
                    plan_id=plan.id,
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                    status='active'
                )
                db.session.add(tenant)
                db.session.flush()
                user.tenant_id = tenant.id
                debug_log['steps'].append({'step': '7', 'action': 'Update or create tenant', 'status': 'success', 'details': f'Created new tenant {tenant.id}'})
                logger.info(f"Created new tenant {tenant.id} for user {user_email} with plan {plan.name}")
            
            # Get invoice details
            debug_log['steps'].append({'step': '8', 'action': 'Retrieve invoice details', 'status': 'in_progress'})
            invoice_id = None
            invoice_number = None
            receipt_url = None
            invoice_url = None
            payment_intent_id = None
            invoice_obj = None
            
            try:
                latest_invoice_id = subscription.latest_invoice
                if latest_invoice_id:
                    if isinstance(latest_invoice_id, str):
                        invoice_id = latest_invoice_id
                    else:
                        invoice_id = latest_invoice_id.get('id') if hasattr(latest_invoice_id, 'get') else str(latest_invoice_id)
                    
                    debug_log['steps'].append({'step': '8a', 'action': 'Get invoice ID', 'status': 'success', 'details': f'Invoice ID: {invoice_id}'})
                    
                    invoice_obj = stripe.Invoice.retrieve(invoice_id, expand=['payment_intent'])
                    invoice_number = invoice_obj.get('number') or invoice_obj.get('id')
                    receipt_url = invoice_obj.get('receipt_url')
                    invoice_url = invoice_obj.get('hosted_invoice_url')
                    payment_intent_id = invoice_obj.get('payment_intent')
                    if isinstance(payment_intent_id, dict):
                        payment_intent_id = payment_intent_id.get('id')
                    
                    debug_log['steps'].append({'step': '8b', 'action': 'Retrieve invoice', 'status': 'success', 'details': f'Invoice: {invoice_number}, Amount: ${invoice_obj.amount_paid/100 if hasattr(invoice_obj, "amount_paid") else "N/A"}'})
            except stripe.error.InvalidRequestError as e:
                debug_log['steps'].append({'step': '8', 'action': 'Retrieve invoice details', 'status': 'warning', 'details': f'Invoice not found: {str(e)}'})
                logger.warning(f"Could not retrieve invoice details: {str(e)}")
            except Exception as e:
                debug_log['steps'].append({'step': '8', 'action': 'Retrieve invoice details', 'status': 'warning', 'details': f'Unexpected error: {str(e)}'})
                logger.warning(f"Unexpected error retrieving invoice details: {str(e)}")
            
            # Generate invoice number if not available
            if not invoice_number:
                date_prefix = datetime.utcnow().strftime('%Y%m%d')
                txn_suffix = str(tenant.id).zfill(4) + str(user.id).zfill(4)
                invoice_number = f"INV-{date_prefix}-{txn_suffix}"
                debug_log['steps'].append({'step': '8c', 'action': 'Generate invoice number', 'status': 'success', 'details': f'Generated: {invoice_number}'})
            
            # Extract payment method from Stripe
            payment_method = get_payment_method_from_stripe(
                payment_intent_id=payment_intent_id,
                invoice=invoice_obj
            )
            debug_log['steps'].append({'step': '8d', 'action': 'Extract payment method', 'status': 'success', 'details': f'Payment method: {payment_method}'})
            logger.info(f"Detected payment method: {payment_method} for synced subscription")
            
            # Check if transaction already exists
            # Note: SubscriptionTransaction doesn't have stripe_subscription_id field
            # We'll check by tenant_id, plan_id, and stripe_invoice_id if available
            debug_log['steps'].append({'step': '9', 'action': 'Create transaction record', 'status': 'in_progress'})
            existing_transaction = None
            
            # Try to find existing transaction by invoice_id first (most reliable)
            if invoice_id:
                existing_transaction = SubscriptionTransaction.query.filter_by(
                    stripe_invoice_id=invoice_id,
                    status='succeeded'
                ).first()
                if existing_transaction:
                    debug_log['steps'].append({'step': '9a', 'action': 'Check existing transaction by invoice', 'status': 'success', 'details': f'Found transaction by invoice_id: {invoice_id}'})
            
            # If not found by invoice, check by tenant and plan (for same subscription)
            if not existing_transaction:
                existing_transaction = SubscriptionTransaction.query.filter_by(
                    tenant_id=tenant.id,
                    plan_id=plan.id,
                    status='succeeded'
                ).order_by(SubscriptionTransaction.created_at.desc()).first()
                if existing_transaction:
                    debug_log['steps'].append({'step': '9b', 'action': 'Check existing transaction by tenant/plan', 'status': 'success', 'details': f'Found transaction by tenant_id and plan_id'})
            
            if not existing_transaction:
                # Create transaction record
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
                    notes=f'Manually synced from Stripe. Invoice: {invoice_number}'
                )
                db.session.add(transaction)
                debug_log['steps'].append({'step': '9', 'action': 'Create transaction record', 'status': 'success', 'details': f'Created {transaction_type} transaction'})
            else:
                debug_log['steps'].append({'step': '9', 'action': 'Create transaction record', 'status': 'success', 'details': 'Transaction already exists, skipped'})
            
            # Create subscription history if needed (check for duplicates more precisely)
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
                    reason='Manually synced from Stripe'
                )
                db.session.add(history)
                debug_log['steps'].append({'step': '9a', 'action': 'Create history record', 'status': 'success', 'details': f'Created {history_action} history'})
            else:
                debug_log['steps'].append({'step': '9a', 'action': 'Create history record', 'status': 'success', 'details': 'History already exists, skipped'})
            
            debug_log['steps'].append({'step': '10', 'action': 'Commit database changes', 'status': 'in_progress'})
            db.session.commit()
            debug_log['steps'].append({'step': '10', 'action': 'Commit database changes', 'status': 'success'})
            
            # Update Cognito
            debug_log['steps'].append({'step': '11', 'action': 'Update Cognito user', 'status': 'in_progress'})
            try:
                cognito_admin_update_user_attributes(user_email, {'custom:tenant_id': str(tenant.id)})
                debug_log['steps'].append({'step': '11', 'action': 'Update Cognito user', 'status': 'success'})
                logger.info(f"Updated Cognito user {user_email} with tenant_id {tenant.id}")
            except Exception as e:
                debug_log['steps'].append({'step': '11', 'action': 'Update Cognito user', 'status': 'error', 'details': str(e)})
                logger.error(f"Failed to update Cognito user {user_email}: {e}")
            
            # Note: Email sending is skipped for manual sync to avoid duplicate emails
            # The webhook handler sends the purchase receipt email automatically
            debug_log['steps'].append({'step': '12', 'action': 'Skip email (manual sync)', 'status': 'success', 'details': 'Email sending skipped for manual sync to avoid duplicates'})
            
            debug_log['success'] = True
            debug_log['steps'].append({'step': '13', 'action': 'Complete', 'status': 'success', 'details': f'Synced: Tenant {tenant.id}, Plan {plan.name}, Status {tenant.status}'})
            logger.info(f"Successfully synced subscription for user {user_email}, tenant {tenant.id}, plan {plan.name}")
            
            return jsonify({
                'success': True,
                'message': 'Subscription synced successfully',
                'subscription': {
                    'tenant_id': tenant.id,
                    'plan': plan.name,
                    'status': tenant.status,
                    'stripe_customer_id': customer_id,
                    'stripe_subscription_id': subscription_id
                },
                'debug': debug_log
            }), 200
            
        except stripe.error.StripeError as e:
            debug_log['errors'].append(f'Stripe API error: {str(e)}')
            debug_log['steps'].append({'step': 'X', 'action': 'Stripe API Error', 'status': 'error', 'details': str(e)})
            logger.error(f"Stripe API error in sync-subscription: {str(e)}", exc_info=True)
            return jsonify({'error': f'Stripe API error: {str(e)}', 'debug': debug_log}), 500
        except Exception as e:
            debug_log['errors'].append(str(e))
            debug_log['steps'].append({'step': 'X', 'action': 'Exception', 'status': 'error', 'details': str(e)})
            logger.error(f"Error in sync-subscription: {str(e)}", exc_info=True)
            db.session.rollback()
            return jsonify({'error': str(e), 'debug': debug_log}), 500
            
    except Exception as e:
        debug_log['errors'].append(str(e))
        debug_log['steps'].append({'step': 'X', 'action': 'Top-level Exception', 'status': 'error', 'details': str(e)})
        logger.error(f"Error in /stripe/sync-subscription: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'debug': debug_log}), 500 