#!/usr/bin/env python3
"""
Database Table Creation Script for Payroll Payment System

This script creates all necessary tables for the payment system.
Run this after setting up your database connection.

Usage:
    python scripts/create_payment_tables.py

Requirements:
    - Database connection configured in .env
    - SQLAlchemy models imported
    - Database user has CREATE TABLE permissions
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
from app import create_app, db
from app.models import (
    EmployerWalletBalance,
    FraudAlert,
    PaymentTransaction,
    PayRun,
    UserBankAccount,
    PayrollSettings
)
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

# Load environment variables
load_dotenv(os.path.join(backend_dir, '.env'))

def create_tables():
    """Create all payment-related tables"""
    app = create_app()
    
    with app.app_context():
        print("=" * 80)
        print("PAYROLL PAYMENT SYSTEM - DATABASE TABLE CREATION")
        print("=" * 80)
        print()
        
        try:
            # Test database connection
            print("1. Testing database connection...")
            db.session.execute(text("SELECT 1"))
            print("   ✅ Database connection successful")
            print()
            
            # Check if tables exist
            print("2. Checking existing tables...")
            inspector = db.inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            tables_to_create = {
                'employer_wallet_balances': EmployerWalletBalance,
                'fraud_alerts': FraudAlert,
                'payment_transactions': PaymentTransaction,
            }
            
            tables_to_alter = {
                'pay_runs': PayRun,
                'user_bank_accounts': UserBankAccount,
                'payroll_settings': PayrollSettings,
            }
            
            print(f"   Found {len(existing_tables)} existing tables")
            print()
            
            # Create new tables
            print("3. Creating new tables...")
            for table_name, model in tables_to_create.items():
                if table_name in existing_tables:
                    print(f"   ⚠️  Table '{table_name}' already exists (skipping)")
                else:
                    try:
                        model.__table__.create(db.engine, checkfirst=True)
                        print(f"   ✅ Created table '{table_name}'")
                    except Exception as e:
                        print(f"   ❌ Failed to create '{table_name}': {e}")
            print()
            
            # Add columns to existing tables
            print("4. Adding columns to existing tables...")
            
            # Add columns to user_bank_accounts
            if 'user_bank_accounts' in existing_tables:
                print("   Checking user_bank_accounts...")
                columns = [col['name'] for col in inspector.get_columns('user_bank_accounts')]
                
                new_columns = {
                    'ifsc_code': "ALTER TABLE user_bank_accounts ADD COLUMN ifsc_code VARCHAR(11) NULL",
                    'account_type': "ALTER TABLE user_bank_accounts ADD COLUMN account_type VARCHAR(20) NULL",
                    'branch_name': "ALTER TABLE user_bank_accounts ADD COLUMN branch_name VARCHAR(255) NULL",
                    'branch_address': "ALTER TABLE user_bank_accounts ADD COLUMN branch_address TEXT NULL",
                    'consent_given_at': "ALTER TABLE user_bank_accounts ADD COLUMN consent_given_at DATETIME NULL",
                    'consent_ip': "ALTER TABLE user_bank_accounts ADD COLUMN consent_ip VARCHAR(45) NULL",
                    'verified_by_penny_drop': "ALTER TABLE user_bank_accounts ADD COLUMN verified_by_penny_drop BOOLEAN NOT NULL DEFAULT FALSE",
                    'verification_reference_id': "ALTER TABLE user_bank_accounts ADD COLUMN verification_reference_id VARCHAR(255) NULL",
                    'verification_date': "ALTER TABLE user_bank_accounts ADD COLUMN verification_date DATETIME NULL",
                    'last_updated_by': "ALTER TABLE user_bank_accounts ADD COLUMN last_updated_by INT NULL",
                    'bank_change_cooldown_until': "ALTER TABLE user_bank_accounts ADD COLUMN bank_change_cooldown_until DATETIME NULL",
                }
                
                for col_name, sql in new_columns.items():
                    if col_name not in columns:
                        try:
                            db.session.execute(text(sql))
                            print(f"      ✅ Added column '{col_name}'")
                        except Exception as e:
                            if 'Duplicate column' in str(e) or 'already exists' in str(e).lower():
                                print(f"      ⚠️  Column '{col_name}' already exists")
                            else:
                                print(f"      ❌ Failed to add '{col_name}': {e}")
                    else:
                        print(f"      ⚠️  Column '{col_name}' already exists")
            
            # Add columns to payroll_settings
            if 'payroll_settings' in existing_tables:
                print("   Checking payroll_settings...")
                columns = [col['name'] for col in inspector.get_columns('payroll_settings')]
                
                new_columns = {
                    'payment_gateway': "ALTER TABLE payroll_settings ADD COLUMN payment_gateway VARCHAR(50) NULL",
                    'razorpay_key_id': "ALTER TABLE payroll_settings ADD COLUMN razorpay_key_id VARCHAR(255) NULL",
                    'razorpay_key_secret': "ALTER TABLE payroll_settings ADD COLUMN razorpay_key_secret TEXT NULL",
                    'razorpay_webhook_secret': "ALTER TABLE payroll_settings ADD COLUMN razorpay_webhook_secret VARCHAR(255) NULL",
                    'razorpay_fund_account_id': "ALTER TABLE payroll_settings ADD COLUMN razorpay_fund_account_id VARCHAR(255) NULL",
                    'payment_mode': "ALTER TABLE payroll_settings ADD COLUMN payment_mode VARCHAR(20) NOT NULL DEFAULT 'NEFT'",
                }
                
                for col_name, sql in new_columns.items():
                    if col_name not in columns:
                        try:
                            db.session.execute(text(sql))
                            print(f"      ✅ Added column '{col_name}'")
                        except Exception as e:
                            if 'Duplicate column' in str(e) or 'already exists' in str(e).lower():
                                print(f"      ⚠️  Column '{col_name}' already exists")
                            else:
                                print(f"      ❌ Failed to add '{col_name}': {e}")
                    else:
                        print(f"      ⚠️  Column '{col_name}' already exists")
            
            # Add columns to pay_runs
            if 'pay_runs' in existing_tables:
                print("   Checking pay_runs...")
                columns = [col['name'] for col in inspector.get_columns('pay_runs')]
                
                new_columns = {
                    'funds_locked': "ALTER TABLE pay_runs ADD COLUMN funds_locked BOOLEAN NOT NULL DEFAULT FALSE",
                    'funds_locked_at': "ALTER TABLE pay_runs ADD COLUMN funds_locked_at DATETIME NULL",
                    'funds_locked_amount': "ALTER TABLE pay_runs ADD COLUMN funds_locked_amount DECIMAL(12, 2) NULL",
                    'payments_initiated': "ALTER TABLE pay_runs ADD COLUMN payments_initiated INT NOT NULL DEFAULT 0",
                    'payments_successful': "ALTER TABLE pay_runs ADD COLUMN payments_successful INT NOT NULL DEFAULT 0",
                    'payments_failed': "ALTER TABLE pay_runs ADD COLUMN payments_failed INT NOT NULL DEFAULT 0",
                    'payments_pending': "ALTER TABLE pay_runs ADD COLUMN payments_pending INT NOT NULL DEFAULT 0",
                    'funds_validated_by': "ALTER TABLE pay_runs ADD COLUMN funds_validated_by INT NULL",
                    'funds_validated_at': "ALTER TABLE pay_runs ADD COLUMN funds_validated_at DATETIME NULL",
                    'correlation_id': "ALTER TABLE pay_runs ADD COLUMN correlation_id VARCHAR(255) NULL",
                }
                
                for col_name, sql in new_columns.items():
                    if col_name not in columns:
                        try:
                            db.session.execute(text(sql))
                            print(f"      ✅ Added column '{col_name}'")
                        except Exception as e:
                            if 'Duplicate column' in str(e) or 'already exists' in str(e).lower():
                                print(f"      ⚠️  Column '{col_name}' already exists")
                            else:
                                print(f"      ❌ Failed to add '{col_name}': {e}")
                    else:
                        print(f"      ⚠️  Column '{col_name}' already exists")
            
            # Add columns to payment_transactions (if table exists)
            if 'payment_transactions' in existing_tables:
                print("   Checking payment_transactions...")
                columns = [col['name'] for col in inspector.get_columns('payment_transactions')]
                
                new_columns = {
                    'idempotency_key': "ALTER TABLE payment_transactions ADD COLUMN idempotency_key VARCHAR(255) NULL",
                    'retry_count': "ALTER TABLE payment_transactions ADD COLUMN retry_count INT NOT NULL DEFAULT 0",
                    'max_retries': "ALTER TABLE payment_transactions ADD COLUMN max_retries INT NOT NULL DEFAULT 3",
                    'last_retry_at': "ALTER TABLE payment_transactions ADD COLUMN last_retry_at DATETIME NULL",
                    'purpose_code': "ALTER TABLE payment_transactions ADD COLUMN purpose_code VARCHAR(20) NOT NULL DEFAULT 'SALARY'",
                    'payout_category': "ALTER TABLE payment_transactions ADD COLUMN payout_category VARCHAR(20) NOT NULL DEFAULT 'salary'",
                    'fraud_risk_score': "ALTER TABLE payment_transactions ADD COLUMN fraud_risk_score DECIMAL(5, 2) NULL",
                    'fraud_flags': "ALTER TABLE payment_transactions ADD COLUMN fraud_flags JSON NULL",
                    'requires_manual_review': "ALTER TABLE payment_transactions ADD COLUMN requires_manual_review BOOLEAN NOT NULL DEFAULT FALSE",
                    'reviewed_by': "ALTER TABLE payment_transactions ADD COLUMN reviewed_by INT NULL",
                    'reviewed_at': "ALTER TABLE payment_transactions ADD COLUMN reviewed_at DATETIME NULL",
                    'review_notes': "ALTER TABLE payment_transactions ADD COLUMN review_notes TEXT NULL",
                }
                
                for col_name, sql in new_columns.items():
                    if col_name not in columns:
                        try:
                            db.session.execute(text(sql))
                            print(f"      ✅ Added column '{col_name}'")
                        except Exception as e:
                            if 'Duplicate column' in str(e) or 'already exists' in str(e).lower():
                                print(f"      ⚠️  Column '{col_name}' already exists")
                            else:
                                print(f"      ❌ Failed to add '{col_name}': {e}")
                    else:
                        print(f"      ⚠️  Column '{col_name}' already exists")
            
            print()
            
            # Create indexes
            print("5. Creating indexes...")
            indexes = [
                ("CREATE INDEX IF NOT EXISTS idx_cooldown ON user_bank_accounts(bank_change_cooldown_until)"),
                ("CREATE INDEX IF NOT EXISTS idx_idempotency_key ON payment_transactions(idempotency_key)"),
                ("CREATE INDEX IF NOT EXISTS idx_fraud_review ON payment_transactions(requires_manual_review)"),
                ("CREATE INDEX IF NOT EXISTS idx_fraud_tenant ON fraud_alerts(tenant_id)"),
                ("CREATE INDEX IF NOT EXISTS idx_fraud_status ON fraud_alerts(status)"),
                ("CREATE INDEX IF NOT EXISTS idx_wallet_tenant ON employer_wallet_balances(tenant_id)"),
            ]
            
            for index_sql in indexes:
                try:
                    db.session.execute(text(index_sql))
                    print(f"   ✅ Created index")
                except Exception as e:
                    if 'already exists' in str(e).lower() or 'Duplicate key' in str(e):
                        print(f"   ⚠️  Index already exists")
                    else:
                        print(f"   ❌ Failed to create index: {e}")
            
            print()
            
            # Commit all changes
            db.session.commit()
            print("6. ✅ All changes committed successfully!")
            print()
            print("=" * 80)
            print("DATABASE SETUP COMPLETE")
            print("=" * 80)
            print()
            print("Next steps:")
            print("1. Configure Razorpay keys in PayrollSettings")
            print("2. Set up webhook endpoint: /api/hr/payments/webhooks/razorpay")
            print("3. Run cron jobs for reconciliation and balance sync")
            print("4. Test with a small payment before production use")
            print()
            
        except OperationalError as e:
            print(f"❌ Database connection failed: {e}")
            print()
            print("Please check:")
            print("1. Database server is running")
            print("2. DATABASE_URL in .env is correct")
            print("3. Database user has CREATE TABLE permissions")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            sys.exit(1)

if __name__ == '__main__':
    create_tables()

