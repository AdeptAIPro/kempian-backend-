"""
Script to check who has purchased plans and verify subscription status.
This script provides a comprehensive report of all subscriptions, transactions, and their status.

Usage:
    python -m scripts.check_subscription_purchases
    OR
    cd backend && python scripts/check_subscription_purchases.py

Options:
    --verify-stripe    : Verify subscription status with Stripe API (requires STRIPE_SECRET_KEY)
    --detailed         : Show detailed transaction history for each tenant
    --export-csv       : Export results to CSV file
"""

import sys
import os
import argparse
from datetime import datetime
from decimal import Decimal

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variable to skip heavy initialization
os.environ['SKIP_HEAVY_INIT'] = '1'
os.environ['SKIP_SEARCH_INIT'] = '1'
os.environ['MIGRATION_MODE'] = '1'

from flask import Flask
from app.db import db
from app.models import Plan, Tenant, User, SubscriptionTransaction, SubscriptionHistory
from app.config import Config
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Initialize Stripe (optional)
stripe = None
stripe_available = False
try:
    import stripe as stripe_module
    stripe_secret_key = os.getenv('STRIPE_SECRET_KEY')
    if stripe_secret_key:
        stripe_module.api_key = stripe_secret_key
        stripe = stripe_module
        stripe_available = True
except ImportError:
    pass

def create_minimal_app():
    """Create a minimal Flask app without heavy dependencies"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def format_currency(cents, currency='USD'):
    """Format cents to currency string"""
    if cents is None:
        return 'N/A'
    amount = Decimal(cents) / 100
    return f"${amount:,.2f} {currency}"

def verify_stripe_subscription(tenant, stripe):
    """Verify subscription status with Stripe API"""
    if not stripe or not tenant.stripe_customer_id:
        return None
    
    try:
        # Get customer from Stripe
        customer = stripe.Customer.retrieve(tenant.stripe_customer_id)
        
        # Get active subscriptions
        subscriptions = stripe.Subscription.list(
            customer=tenant.stripe_customer_id,
            status='all',
            limit=10
        )
        
        # Get latest invoice
        invoices = stripe.Invoice.list(
            customer=tenant.stripe_customer_id,
            limit=5
        )
        
        return {
            'customer_exists': True,
            'customer_email': customer.get('email', 'N/A'),
            'subscriptions': [
                {
                    'id': sub.id,
                    'status': sub.status,
                    'current_period_start': datetime.fromtimestamp(sub.current_period_start).isoformat() if sub.current_period_start else None,
                    'current_period_end': datetime.fromtimestamp(sub.current_period_end).isoformat() if sub.current_period_end else None,
                    'cancel_at_period_end': sub.cancel_at_period_end,
                    'price_id': sub.items.data[0].price.id if sub.items.data else None
                }
                for sub in subscriptions.data
            ],
            'latest_invoices': [
                {
                    'id': inv.id,
                    'status': inv.status,
                    'amount_paid': inv.amount_paid,
                    'created': datetime.fromtimestamp(inv.created).isoformat() if inv.created else None
                }
                for inv in invoices.data[:3]
            ]
        }
    except Exception as e:
        return {
            'customer_exists': False,
            'error': str(e)
        }

def check_subscription_purchases(verify_stripe_flag=False, detailed=False, export_csv=False):
    """Check all subscription purchases and their status"""
    app = create_minimal_app()
    
    with app.app_context():
        try:
            print("="*80)
            print("SUBSCRIPTION PURCHASE VERIFICATION REPORT")
            print("="*80)
            print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Get all tenants with subscriptions (excluding free tier)
            all_tenants = Tenant.query.all()
            tenants_with_plans = []
            
            for tenant in all_tenants:
                plan = db.session.get(Plan, tenant.plan_id)
                if plan and plan.name.lower() != 'free trial':
                    tenants_with_plans.append((tenant, plan))
            
            print(f"Total tenants with paid plans: {len(tenants_with_plans)}")
            print()
            
            if len(tenants_with_plans) == 0:
                print("[INFO] No tenants with paid plans found.")
                return
            
            # Summary statistics
            plan_counts = {}
            status_counts = {'active': 0, 'inactive': 0, 'cancelled': 0}
            total_revenue = 0
            
            for tenant, plan in tenants_with_plans:
                plan_counts[plan.name] = plan_counts.get(plan.name, 0) + 1
                status_counts[tenant.status] = status_counts.get(tenant.status, 0) + 1
                
                # Get successful transactions for revenue calculation
                successful_transactions = SubscriptionTransaction.query.filter_by(
                    tenant_id=tenant.id,
                    status='succeeded'
                ).all()
                for trans in successful_transactions:
                    total_revenue += trans.amount_cents
            
            print("="*80)
            print("SUMMARY STATISTICS")
            print("="*80)
            print(f"Total tenants with paid plans: {len(tenants_with_plans)}")
            print(f"Total revenue (from successful transactions): {format_currency(total_revenue)}")
            print()
            print("Plans distribution:")
            for plan_name, count in sorted(plan_counts.items()):
                print(f"  - {plan_name}: {count} tenant(s)")
            print()
            print("Status distribution:")
            for status, count in status_counts.items():
                print(f"  - {status}: {count} tenant(s)")
            print()
            
            # Detailed tenant information
            print("="*80)
            print("DETAILED TENANT INFORMATION")
            print("="*80)
            print()
            
            csv_data = []
            
            for idx, (tenant, plan) in enumerate(tenants_with_plans, 1):
                print(f"\n[{idx}] Tenant ID: {tenant.id}")
                print("-" * 80)
                
                # Get tenant owner/primary user
                owner = User.query.filter_by(tenant_id=tenant.id, role='owner').first()
                if not owner:
                    owner = User.query.filter_by(tenant_id=tenant.id).first()
                
                owner_email = owner.email if owner else 'N/A'
                # User model doesn't have first_name/last_name, use company_name or email
                if owner:
                    owner_name = owner.company_name if owner.company_name else owner.email
                else:
                    owner_name = 'N/A'
                
                print(f"  Owner: {owner_name} ({owner_email})")
                print(f"  Plan: {plan.name} ({plan.billing_cycle})")
                print(f"  Plan Price: {format_currency(plan.price_cents)}")
                print(f"  Plan Features:")
                print(f"    - JD Quota: {'Unlimited' if plan.jd_quota_per_month == -1 else plan.jd_quota_per_month} per month")
                print(f"    - Max Subaccounts: {'Unlimited' if plan.max_subaccounts == -1 else plan.max_subaccounts}")
                print(f"  Tenant Status: {tenant.status}")
                print(f"  Stripe Customer ID: {tenant.stripe_customer_id or 'N/A'}")
                print(f"  Stripe Subscription ID: {tenant.stripe_subscription_id or 'N/A'}")
                print(f"  Created: {tenant.created_at.strftime('%Y-%m-%d %H:%M:%S') if tenant.created_at else 'N/A'}")
                print(f"  Last Updated: {tenant.updated_at.strftime('%Y-%m-%d %H:%M:%S') if tenant.updated_at else 'N/A'}")
                
                # Get transactions
                transactions = SubscriptionTransaction.query.filter_by(
                    tenant_id=tenant.id
                ).order_by(SubscriptionTransaction.created_at.desc()).all()
                
                print(f"  Total Transactions: {len(transactions)}")
                
                if transactions:
                    successful_count = sum(1 for t in transactions if t.status == 'succeeded')
                    print(f"  Successful Transactions: {successful_count}")
                    
                    if detailed:
                        print(f"  Transaction History:")
                        for trans in transactions[:10]:  # Show last 10
                            print(f"    - {trans.transaction_type.upper()} | {trans.status.upper()} | {format_currency(trans.amount_cents)} | {trans.created_at.strftime('%Y-%m-%d') if trans.created_at else 'N/A'}")
                            if trans.stripe_invoice_id:
                                print(f"      Invoice ID: {trans.stripe_invoice_id}")
                    else:
                        # Show latest transaction
                        latest = transactions[0]
                        print(f"  Latest Transaction: {latest.transaction_type.upper()} | {latest.status.upper()} | {format_currency(latest.amount_cents)} | {latest.created_at.strftime('%Y-%m-%d') if latest.created_at else 'N/A'}")
                
                # Verify with Stripe if requested
                stripe_info = None
                if verify_stripe_flag and stripe_available:
                    print(f"  Verifying with Stripe...")
                    stripe_info = verify_stripe_subscription(tenant, stripe)
                    
                    if stripe_info:
                        if stripe_info.get('customer_exists'):
                            print(f"  ✓ Stripe Customer Found")
                            print(f"    Email: {stripe_info.get('customer_email', 'N/A')}")
                            
                            if stripe_info.get('subscriptions'):
                                print(f"    Active Subscriptions in Stripe: {len([s for s in stripe_info['subscriptions'] if s['status'] == 'active'])}")
                                for sub in stripe_info['subscriptions']:
                                    print(f"      - {sub['id']}: {sub['status']} (Price: {sub['price_id']})")
                                    if sub['current_period_end']:
                                        print(f"        Period End: {sub['current_period_end']}")
                                    if sub['cancel_at_period_end']:
                                        print(f"        ⚠ Will cancel at period end")
                            
                            # Check for mismatches
                            if tenant.stripe_subscription_id:
                                stripe_sub_ids = [s['id'] for s in stripe_info.get('subscriptions', [])]
                                if tenant.stripe_subscription_id not in stripe_sub_ids:
                                    print(f"    ⚠ WARNING: Tenant subscription ID {tenant.stripe_subscription_id} not found in Stripe!")
                        else:
                            print(f"  ✗ Stripe Customer NOT Found: {stripe_info.get('error', 'Unknown error')}")
                    else:
                        print(f"  ⚠ Could not verify with Stripe (check STRIPE_SECRET_KEY)")
                
                # Collect data for CSV export
                if export_csv:
                    csv_data.append({
                        'tenant_id': tenant.id,
                        'owner_name': owner_name if owner_name != 'N/A' else '',
                        'owner_email': owner_email,
                        'plan_name': plan.name,
                        'billing_cycle': plan.billing_cycle,
                        'plan_price': plan.price_cents,
                        'tenant_status': tenant.status,
                        'stripe_customer_id': tenant.stripe_customer_id or '',
                        'stripe_subscription_id': tenant.stripe_subscription_id or '',
                        'total_transactions': len(transactions),
                        'successful_transactions': sum(1 for t in transactions if t.status == 'succeeded'),
                        'latest_transaction_date': transactions[0].created_at.strftime('%Y-%m-%d') if transactions else '',
                        'latest_transaction_status': transactions[0].status if transactions else '',
                        'stripe_verified': 'Yes' if (stripe_info and stripe_info.get('customer_exists')) else 'No',
                        'created_at': tenant.created_at.strftime('%Y-%m-%d %H:%M:%S') if tenant.created_at else ''
                    })
            
            # Export to CSV if requested
            if export_csv and csv_data:
                try:
                    import csv
                    csv_filename = f"subscription_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    csv_path = os.path.join(os.path.dirname(__file__), '..', csv_filename)
                    
                    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                        if csv_data:
                            fieldnames = csv_data[0].keys()
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(csv_data)
                    
                    print("\n" + "="*80)
                    print(f"CSV Report exported to: {csv_path}")
                    print("="*80)
                except Exception as e:
                    print(f"\n[ERROR] Failed to export CSV: {e}")
            
            # Check for potential issues
            print("\n" + "="*80)
            print("POTENTIAL ISSUES CHECK")
            print("="*80)
            
            issues_found = []
            
            for tenant, plan in tenants_with_plans:
                # Check for tenants with active status but no successful transactions
                successful_transactions = SubscriptionTransaction.query.filter_by(
                    tenant_id=tenant.id,
                    status='succeeded'
                ).count()
                
                if tenant.status == 'active' and successful_transactions == 0:
                    issues_found.append(f"Tenant {tenant.id} ({User.query.filter_by(tenant_id=tenant.id, role='owner').first().email if User.query.filter_by(tenant_id=tenant.id, role='owner').first() else 'N/A'}) is active but has no successful transactions")
                
                # Check for missing Stripe IDs
                if not tenant.stripe_customer_id:
                    issues_found.append(f"Tenant {tenant.id} has no Stripe customer ID")
                
                if not tenant.stripe_subscription_id:
                    issues_found.append(f"Tenant {tenant.id} has no Stripe subscription ID")
            
            if issues_found:
                print(f"\n⚠ Found {len(issues_found)} potential issue(s):")
                for issue in issues_found:
                    print(f"  - {issue}")
            else:
                print("\n✓ No obvious issues found!")
            
            print("\n" + "="*80)
            print("REPORT COMPLETE")
            print("="*80)
            
        except Exception as e:
            print(f"\n[ERROR] Error generating report: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check subscription purchases and verify status')
    parser.add_argument('--verify-stripe', action='store_true', 
                       help='Verify subscription status with Stripe API')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed transaction history')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV file')
    
    args = parser.parse_args()
    
    if args.verify_stripe and not stripe_available:
        print("[WARNING] --verify-stripe requested but Stripe is not available (check STRIPE_SECRET_KEY)")
        print("Continuing without Stripe verification...\n")
    
    check_subscription_purchases(
        verify_stripe_flag=args.verify_stripe,
        detailed=args.detailed,
        export_csv=args.export_csv
    )

