"""
Script to manually resend subscription purchase receipt emails.
This is useful when emails weren't sent due to webhook failures or other issues.

Usage:
    python -m scripts.resend_subscription_email --email nishant@adeptaipro.com
    OR
    python -m scripts.resend_subscription_email --tenant-id 624
    OR
    python -m scripts.resend_subscription_email --transaction-id <transaction_id>
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variable to skip heavy initialization
os.environ['SKIP_HEAVY_INIT'] = '1'
os.environ['SKIP_SEARCH_INIT'] = '1'
os.environ['MIGRATION_MODE'] = '1'

from flask import Flask
from app.db import db
from app.models import Plan, Tenant, User, SubscriptionTransaction
from app.config import Config
from app.emails.subscription import send_subscription_purchase_receipt
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def create_minimal_app():
    """Create a minimal Flask app without heavy dependencies"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def resend_email_for_transaction(transaction_id=None, tenant_id=None, email=None):
    """Resend purchase receipt email for a specific transaction"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            # Find the transaction
            transaction = None
            if transaction_id:
                transaction = SubscriptionTransaction.query.get(transaction_id)
            elif tenant_id:
                # Get the latest successful transaction for this tenant
                transaction = SubscriptionTransaction.query.filter_by(
                    tenant_id=tenant_id,
                    status='succeeded'
                ).order_by(SubscriptionTransaction.created_at.desc()).first()
            elif email:
                # Find user by email, then get their tenant's latest transaction
                user = User.query.filter_by(email=email).first()
                if user and user.tenant_id:
                    transaction = SubscriptionTransaction.query.filter_by(
                        tenant_id=user.tenant_id,
                        status='succeeded'
                    ).order_by(SubscriptionTransaction.created_at.desc()).first()
            
            if not transaction:
                print(f"[ERROR] No successful transaction found")
                if transaction_id:
                    print(f"  Transaction ID: {transaction_id}")
                elif tenant_id:
                    print(f"  Tenant ID: {tenant_id}")
                elif email:
                    print(f"  Email: {email}")
                return False
            
            print(f"[INFO] Found transaction ID: {transaction.id}")
            print(f"  Type: {transaction.transaction_type}")
            print(f"  Status: {transaction.status}")
            print(f"  Amount: ${transaction.amount_cents / 100:.2f}")
            print(f"  Created: {transaction.created_at}")
            
            # Get tenant and plan
            tenant = Tenant.query.get(transaction.tenant_id)
            if not tenant:
                print(f"[ERROR] Tenant {transaction.tenant_id} not found")
                return False
            
            plan = Plan.query.get(transaction.plan_id)
            if not plan:
                print(f"[ERROR] Plan {transaction.plan_id} not found")
                return False
            
            # Get user
            user = User.query.get(transaction.user_id)
            if not user:
                print(f"[ERROR] User {transaction.user_id} not found")
                return False
            
            print(f"\n[INFO] Sending email to: {user.email}")
            print(f"  Tenant ID: {tenant.id}")
            print(f"  Plan: {plan.name} ({plan.billing_cycle})")
            
            # Prepare email parameters
            user_name = user.company_name if user.company_name else user.email.split('@')[0]
            amount_display = f"${transaction.amount_cents / 100:.2f}"
            
            # Get invoice number from transaction
            invoice_number = None
            if transaction.notes and 'Invoice:' in transaction.notes:
                try:
                    invoice_number = transaction.notes.split('Invoice:')[1].strip().split()[0]
                except:
                    pass
            if not invoice_number and transaction.stripe_invoice_id:
                invoice_number = transaction.stripe_invoice_id
            
            transaction_id_display = invoice_number or f"TXN-{transaction.id:06d}"
            
            # Calculate next billing date
            next_billing_date = None
            if plan.billing_cycle == 'monthly':
                next_billing_date = (transaction.created_at + timedelta(days=30)).strftime('%B %d, %Y')
            elif plan.billing_cycle == 'yearly':
                next_billing_date = (transaction.created_at + timedelta(days=365)).strftime('%B %d, %Y')
            
            # Format purchase date
            purchase_date = transaction.created_at.strftime('%B %d, %Y at %I:%M %p') if transaction.created_at else datetime.utcnow().strftime('%B %d, %Y at %I:%M %p')
            
            # Get URLs from transaction
            invoice_url = transaction.invoice_url
            receipt_url = transaction.receipt_url
            
            # Send email
            print(f"\n[INFO] Sending purchase receipt email...")
            try:
                result = send_subscription_purchase_receipt(
                    to_email=user.email,
                    user_name=user_name,
                    plan_name=plan.name,
                    amount_display=amount_display,
                    transaction_id=transaction_id_display,
                    payment_method=transaction.payment_method or 'Card',
                    purchase_date=purchase_date,
                    billing_cycle=plan.billing_cycle.title(),
                    next_billing_date=next_billing_date,
                    jd_quota=plan.jd_quota_per_month,
                    max_subaccounts=plan.max_subaccounts,
                    dashboard_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:8081')}/dashboard",
                    invoice_number=invoice_number,
                    invoice_url=invoice_url,
                    receipt_url=receipt_url
                )
                
                if result:
                    print(f"[SUCCESS] Purchase receipt email sent successfully to {user.email}")
                    print(f"  Transaction ID: {transaction_id_display}")
                    print(f"  Invoice Number: {invoice_number or 'N/A'}")
                    return True
                else:
                    print(f"[ERROR] Failed to send email (function returned False)")
                    return False
                    
            except Exception as email_error:
                print(f"[ERROR] Exception while sending email: {str(email_error)}")
                import traceback
                traceback.print_exc()
                return False
            
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resend subscription purchase receipt email')
    parser.add_argument('--email', type=str, help='User email address')
    parser.add_argument('--tenant-id', type=int, help='Tenant ID')
    parser.add_argument('--transaction-id', type=int, help='Transaction ID')
    
    args = parser.parse_args()
    
    if not args.email and not args.tenant_id and not args.transaction_id:
        print("[ERROR] Please provide one of: --email, --tenant-id, or --transaction-id")
        print("\nUsage examples:")
        print("  python scripts/resend_subscription_email.py --email nishant@adeptaipro.com")
        print("  python scripts/resend_subscription_email.py --tenant-id 624")
        print("  python scripts/resend_subscription_email.py --transaction-id 123")
        sys.exit(1)
    
    print("="*80)
    print("RESEND SUBSCRIPTION PURCHASE RECEIPT EMAIL")
    print("="*80)
    print()
    
    success = resend_email_for_transaction(
        transaction_id=args.transaction_id,
        tenant_id=args.tenant_id,
        email=args.email
    )
    
    if success:
        print("\n" + "="*80)
        print("[SUCCESS] Email sent successfully!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("[ERROR] Failed to send email. Check logs above for details.")
        print("="*80)
        sys.exit(1)

