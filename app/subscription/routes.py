"""
Subscription management API routes
"""
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import db, User, Tenant, Plan, SubscriptionTransaction, SubscriptionHistory
from app.utils import get_current_user
from app.emails.subscription import (
    send_subscription_purchase_receipt,
    send_subscription_plan_changed,
    send_subscription_cancelled
)
import stripe
import os
from datetime import datetime, timedelta

logger = get_logger("subscription")

subscription_bp = Blueprint('subscription', __name__)

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

def get_user_email_from_jwt(user_jwt):
    """Extract email from JWT payload, handling different JWT formats"""
    if not user_jwt:
        return None
    
    # Try different possible email fields
    email = user_jwt.get('email') or user_jwt.get('Email') or user_jwt.get('EMAIL')
    
    # Try Cognito format
    if not email:
        email = user_jwt.get('cognito:username') or user_jwt.get('sub')
        # If sub is an email, use it; otherwise we need to look up the user
        if email and '@' in email:
            return email
    
    return email

@subscription_bp.route('/plans', methods=['GET'])
def get_available_plans():
    """Get all available subscription plans"""
    try:
        plans = Plan.query.filter_by(is_trial=False).all()
        return jsonify({
            'success': True,
            'plans': [plan.to_dict() for plan in plans]
        }), 200
    except Exception as e:
        logger.error(f"Error fetching plans: {str(e)}")
        return jsonify({'error': 'Failed to fetch plans'}), 500

@subscription_bp.route('/current', methods=['GET'])
def get_current_subscription():
    """Get current user's subscription details"""
    try:
        user_jwt = get_current_user()
        if not user_jwt:
            return jsonify({'error': 'Unauthorized'}), 401
        
        email = get_user_email_from_jwt(user_jwt)
        if not email:
            return jsonify({'error': 'Unauthorized - Invalid token'}), 401
        
        user = User.query.filter_by(email=email).first()
        if not user or not user.tenant_id:
            return jsonify({'error': 'No subscription found'}), 404
        
        tenant = Tenant.query.get(user.tenant_id)
        if not tenant:
            return jsonify({'error': 'Tenant not found'}), 404
        
        plan = Plan.query.get(tenant.plan_id)
        
        return jsonify({
            'success': True,
            'subscription': {
                'tenant_id': tenant.id,
                'plan': plan.to_dict() if plan else None,
                'status': tenant.status,
                'stripe_customer_id': tenant.stripe_customer_id,
                'stripe_subscription_id': tenant.stripe_subscription_id,
                'created_at': tenant.created_at.isoformat() if tenant.created_at else None,
                'updated_at': tenant.updated_at.isoformat() if tenant.updated_at else None
            }
        }), 200
    except Exception as e:
        logger.error(f"Error fetching current subscription: {str(e)}")
        return jsonify({'error': 'Failed to fetch subscription'}), 500

@subscription_bp.route('/transactions', methods=['GET'])
def get_transaction_history():
    """Get user's transaction history"""
    try:
        user_jwt = get_current_user()
        if not user_jwt:
            return jsonify({'error': 'Unauthorized'}), 401
        
        email = get_user_email_from_jwt(user_jwt)
        if not email:
            return jsonify({'error': 'Unauthorized - Invalid token'}), 401
        
        user = User.query.filter_by(email=email).first()
        if not user or not user.tenant_id:
            return jsonify({'error': 'No subscription found'}), 404
        
        transactions = SubscriptionTransaction.query.filter_by(
            tenant_id=user.tenant_id
        ).order_by(SubscriptionTransaction.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'transactions': [transaction.to_dict() for transaction in transactions]
        }), 200
    except Exception as e:
        logger.error(f"Error fetching transaction history: {str(e)}")
        return jsonify({'error': 'Failed to fetch transaction history'}), 500

@subscription_bp.route('/history', methods=['GET'])
def get_subscription_history():
    """Get user's subscription change history"""
    try:
        user_jwt = get_current_user()
        if not user_jwt:
            return jsonify({'error': 'Unauthorized'}), 401
        
        email = get_user_email_from_jwt(user_jwt)
        if not email:
            return jsonify({'error': 'Unauthorized - Invalid token'}), 401
        
        user = User.query.filter_by(email=email).first()
        if not user or not user.tenant_id:
            return jsonify({'error': 'No subscription found'}), 404
        
        history = SubscriptionHistory.query.filter_by(
            tenant_id=user.tenant_id
        ).order_by(SubscriptionHistory.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'history': [entry.to_dict() for entry in history]
        }), 200
    except Exception as e:
        logger.error(f"Error fetching subscription history: {str(e)}")
        return jsonify({'error': 'Failed to fetch subscription history'}), 500

@subscription_bp.route('/upgrade', methods=['POST'])
def upgrade_subscription():
    """Upgrade user's subscription plan"""
    try:
        user_jwt = get_current_user()
        if not user_jwt:
            logger.warning("Upgrade subscription: No user JWT found")
            return jsonify({'error': 'Unauthorized - Please log in'}), 401
        
        email = get_user_email_from_jwt(user_jwt)
        if not email:
            logger.warning(f"Upgrade subscription: JWT found but no email. JWT keys: {list(user_jwt.keys()) if user_jwt else 'None'}")
            return jsonify({'error': 'Unauthorized - Invalid token'}), 401
        
        data = request.get_json()
        new_plan_id = data.get('plan_id')
        reason = data.get('reason', 'User requested upgrade')
        
        if not new_plan_id:
            return jsonify({'error': 'Plan ID is required'}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user or not user.tenant_id:
            return jsonify({'error': 'No subscription found'}), 404
        
        tenant = Tenant.query.get(user.tenant_id)
        if not tenant:
            return jsonify({'error': 'Tenant not found'}), 404
        
        new_plan = Plan.query.get(new_plan_id)
        if not new_plan:
            return jsonify({'error': 'Plan not found'}), 404
        
        current_plan = Plan.query.get(tenant.plan_id)
        if not current_plan:
            return jsonify({'error': 'Current plan not found'}), 404
        
        # Check if it's actually an upgrade
        if new_plan.price_cents <= current_plan.price_cents:
            return jsonify({'error': 'New plan must be higher priced for upgrade'}), 400
        
        # Update tenant plan
        old_plan_id = tenant.plan_id
        tenant.plan_id = new_plan.id
        tenant.updated_at = datetime.utcnow()
        
        # Create transaction record
        transaction = SubscriptionTransaction(
            tenant_id=tenant.id,
            user_id=user.id,
            transaction_type='upgrade',
            amount_cents=new_plan.price_cents - current_plan.price_cents,
            plan_id=new_plan.id,
            previous_plan_id=old_plan_id,
            status='succeeded',
            notes=f'Upgraded from {current_plan.name} to {new_plan.name}'
        )
        db.session.add(transaction)
        
        # Create history record
        history = SubscriptionHistory(
            tenant_id=tenant.id,
            user_id=user.id,
            action='upgraded',
            from_plan_id=old_plan_id,
            to_plan_id=new_plan.id,
            reason=reason
        )
        db.session.add(history)
        
        db.session.commit()
        
        # Send email notification
        try:
            send_subscription_plan_changed(
                to_email=user.email,
                user_name=user.email.split('@')[0],  # Use email prefix as name
                change_type='upgrade',
                from_plan_name=current_plan.name,
                to_plan_name=new_plan.name,
                effective_date=datetime.utcnow().strftime('%B %d, %Y'),
                proration_amount=f"${(new_plan.price_cents - current_plan.price_cents) / 100:.2f}",
                reason=reason,
                new_jd_quota=new_plan.jd_quota_per_month,
                new_max_subaccounts=new_plan.max_subaccounts,
                dashboard_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:8081')}/dashboard"
            )
        except Exception as email_error:
            logger.error(f"Failed to send upgrade email: {str(email_error)}")
        
        return jsonify({
            'success': True,
            'message': 'Subscription upgraded successfully',
            'new_plan': new_plan.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Error upgrading subscription: {str(e)}")
        return jsonify({'error': 'Failed to upgrade subscription'}), 500

@subscription_bp.route('/downgrade', methods=['POST'])
def downgrade_subscription():
    """Downgrade user's subscription plan"""
    try:
        user_jwt = get_current_user()
        if not user_jwt:
            logger.warning("Downgrade subscription: No user JWT found")
            return jsonify({'error': 'Unauthorized - Please log in'}), 401
        
        email = get_user_email_from_jwt(user_jwt)
        if not email:
            logger.warning(f"Downgrade subscription: JWT found but no email. JWT keys: {list(user_jwt.keys()) if user_jwt else 'None'}")
            return jsonify({'error': 'Unauthorized - Invalid token'}), 401
        
        data = request.get_json()
        new_plan_id = data.get('plan_id')
        reason = data.get('reason', 'User requested downgrade')
        
        if not new_plan_id:
            return jsonify({'error': 'Plan ID is required'}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user or not user.tenant_id:
            return jsonify({'error': 'No subscription found'}), 404
        
        tenant = Tenant.query.get(user.tenant_id)
        if not tenant:
            return jsonify({'error': 'Tenant not found'}), 404
        
        new_plan = Plan.query.get(new_plan_id)
        if not new_plan:
            return jsonify({'error': 'Plan not found'}), 404
        
        current_plan = Plan.query.get(tenant.plan_id)
        if not current_plan:
            return jsonify({'error': 'Current plan not found'}), 404
        
        # Check if it's actually a downgrade
        if new_plan.price_cents >= current_plan.price_cents:
            return jsonify({'error': 'New plan must be lower priced for downgrade'}), 400
        
        # Update tenant plan
        old_plan_id = tenant.plan_id
        tenant.plan_id = new_plan.id
        tenant.updated_at = datetime.utcnow()
        
        # Create transaction record
        transaction = SubscriptionTransaction(
            tenant_id=tenant.id,
            user_id=user.id,
            transaction_type='downgrade',
            amount_cents=0,  # No charge for downgrade
            plan_id=new_plan.id,
            previous_plan_id=old_plan_id,
            status='succeeded',
            notes=f'Downgraded from {current_plan.name} to {new_plan.name}'
        )
        db.session.add(transaction)
        
        # Create history record
        history = SubscriptionHistory(
            tenant_id=tenant.id,
            user_id=user.id,
            action='downgraded',
            from_plan_id=old_plan_id,
            to_plan_id=new_plan.id,
            reason=reason
        )
        db.session.add(history)
        
        db.session.commit()
        
        # Send email notification
        try:
            send_subscription_plan_changed(
                to_email=user.email,
                user_name=user.email.split('@')[0],  # Use email prefix as name
                change_type='downgrade',
                from_plan_name=current_plan.name,
                to_plan_name=new_plan.name,
                effective_date=datetime.utcnow().strftime('%B %d, %Y'),
                proration_amount=None,
                reason=reason,
                new_jd_quota=new_plan.jd_quota_per_month,
                new_max_subaccounts=new_plan.max_subaccounts,
                dashboard_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:8081')}/dashboard"
            )
        except Exception as email_error:
            logger.error(f"Failed to send downgrade email: {str(email_error)}")
        
        return jsonify({
            'success': True,
            'message': 'Subscription downgraded successfully',
            'new_plan': new_plan.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Error downgrading subscription: {str(e)}")
        return jsonify({'error': 'Failed to downgrade subscription'}), 500

@subscription_bp.route('/cancel', methods=['POST'])
def cancel_subscription():
    """Cancel user's subscription"""
    try:
        user_jwt = get_current_user()
        if not user_jwt:
            logger.warning("Cancel subscription: No user JWT found")
            return jsonify({'error': 'Unauthorized - Please log in'}), 401
        
        email = get_user_email_from_jwt(user_jwt)
        if not email:
            logger.warning(f"Cancel subscription: JWT found but no email. JWT keys: {list(user_jwt.keys()) if user_jwt else 'None'}")
            return jsonify({'error': 'Unauthorized - Invalid token'}), 401
        
        data = request.get_json()
        reason = data.get('reason', 'User requested cancellation')
        immediate = data.get('immediate', False)
        
        user = User.query.filter_by(email=email).first()
        if not user or not user.tenant_id:
            return jsonify({'error': 'No subscription found'}), 404
        
        tenant = Tenant.query.get(user.tenant_id)
        if not tenant:
            return jsonify({'error': 'Tenant not found'}), 404
        
        current_plan = Plan.query.get(tenant.plan_id)
        if not current_plan:
            return jsonify({'error': 'Current plan not found'}), 404
        
        # Update tenant status
        old_status = tenant.status
        tenant.status = 'cancelled'
        tenant.updated_at = datetime.utcnow()
        
        # Create transaction record
        transaction = SubscriptionTransaction(
            tenant_id=tenant.id,
            user_id=user.id,
            transaction_type='cancellation',
            amount_cents=0,
            plan_id=current_plan.id,
            status='succeeded',
            notes=f'Cancelled {current_plan.name} subscription'
        )
        db.session.add(transaction)
        
        # Create history record
        history = SubscriptionHistory(
            tenant_id=tenant.id,
            user_id=user.id,
            action='cancelled',
            from_plan_id=current_plan.id,
            to_plan_id=current_plan.id,
            reason=reason
        )
        db.session.add(history)
        
        db.session.commit()
        
        # Send email notification
        try:
            effective_date = None if immediate else (datetime.utcnow() + timedelta(days=30)).strftime('%B %d, %Y')
            send_subscription_cancelled(
                to_email=user.email,
                user_name=user.email.split('@')[0],  # Use email prefix as name
                plan_name=current_plan.name,
                cancellation_date=datetime.utcnow().strftime('%B %d, %Y'),
                effective_date=effective_date,
                reason=reason,
                refund_amount=None,
                refund_status=None,
                feedback_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:8081')}/feedback",
                reactivate_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:8081')}/subscription/reactivate",
                dashboard_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:8081')}/dashboard"
            )
        except Exception as email_error:
            logger.error(f"Failed to send cancellation email: {str(email_error)}")
        
        return jsonify({
            'success': True,
            'message': 'Subscription cancelled successfully',
            'effective_date': effective_date
        }), 200
    except Exception as e:
        logger.error(f"Error cancelling subscription: {str(e)}")
        return jsonify({'error': 'Failed to cancel subscription'}), 500

@subscription_bp.route('/reactivate', methods=['POST'])
def reactivate_subscription():
    """Reactivate user's cancelled subscription"""
    try:
        user_jwt = get_current_user()
        if not user_jwt:
            return jsonify({'error': 'Unauthorized'}), 401
        
        email = get_user_email_from_jwt(user_jwt)
        if not email:
            return jsonify({'error': 'Unauthorized - Invalid token'}), 401
        
        user = User.query.filter_by(email=email).first()
        if not user or not user.tenant_id:
            return jsonify({'error': 'No subscription found'}), 404
        
        tenant = Tenant.query.get(user.tenant_id)
        if not tenant:
            return jsonify({'error': 'Tenant not found'}), 404
        
        if tenant.status != 'cancelled':
            return jsonify({'error': 'Subscription is not cancelled'}), 400
        
        current_plan = Plan.query.get(tenant.plan_id)
        if not current_plan:
            return jsonify({'error': 'Current plan not found'}), 404
        
        # Reactivate tenant
        tenant.status = 'active'
        tenant.updated_at = datetime.utcnow()
        
        # Create transaction record
        transaction = SubscriptionTransaction(
            tenant_id=tenant.id,
            user_id=user.id,
            transaction_type='renewal',
            amount_cents=current_plan.price_cents,
            plan_id=current_plan.id,
            status='succeeded',
            notes=f'Reactivated {current_plan.name} subscription'
        )
        db.session.add(transaction)
        
        # Create history record
        history = SubscriptionHistory(
            tenant_id=tenant.id,
            user_id=user.id,
            action='reactivated',
            from_plan_id=current_plan.id,
            to_plan_id=current_plan.id,
            reason='User requested reactivation'
        )
        db.session.add(history)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Subscription reactivated successfully',
            'plan': current_plan.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Error reactivating subscription: {str(e)}")
        return jsonify({'error': 'Failed to reactivate subscription'}), 500
