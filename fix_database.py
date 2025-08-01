import pymysql
import os
from app.config import Config

def fix_database():
    # Parse database URL to get connection details
    db_url = Config.SQLALCHEMY_DATABASE_URI
    # mysql+pymysql://kempianai:AdeptAi2025@127.0.0.1:3307/kempianDB
    parts = db_url.replace('mysql+pymysql://', '').split('@')
    user_pass = parts[0].split(':')
    host_port_db = parts[1].split('/')
    host_port = host_port_db[0].split(':')
    
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else 3306
    user = user_pass[0]
    password = user_pass[1]
    database = host_port_db[1]
    
    print(f"Connecting to database: {host}:{port}/{database}")
    
    try:
        # Connect to database
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        cursor = connection.cursor()
        
        print("✅ Connected to database successfully")
        
        # Add missing columns to plans table
        try:
            cursor.execute("""
                ALTER TABLE plans 
                ADD COLUMN is_trial BOOLEAN DEFAULT FALSE,
                ADD COLUMN trial_days INTEGER DEFAULT 0,
                ADD COLUMN billing_cycle VARCHAR(20) DEFAULT 'monthly'
            """)
            connection.commit()
            print("✅ Added missing columns to plans table")
        except Exception as e:
            print(f"⚠️  Plans table columns may already exist: {e}")
        
        # Create user_trials table
        try:
            cursor.execute("""
                CREATE TABLE user_trials (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL UNIQUE,
                    trial_start_date DATETIME NOT NULL,
                    trial_end_date DATETIME NOT NULL,
                    searches_used_today INT DEFAULT 0 NOT NULL,
                    last_search_date DATE,
                    is_active BOOLEAN DEFAULT TRUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            connection.commit()
            print("✅ Created user_trials table")
        except Exception as e:
            print(f"⚠️  user_trials table may already exist: {e}")
        
        # Insert/Update plans
        plans_data = [
            ("Free Trial", 0, "price_free_trial", 5, 1, True, 7, "monthly"),
            ("Recruiters plan", 2900, "price_1RjMKNKVj190YQJbhBuhcjfQ", 100, 2, False, 0, "monthly"),
            ("Recruiters plan (Yearly)", 29000, "price_1RjMKNKVj190YQJbhBuhcjfQ_yearly", 100, 2, False, 0, "yearly"),
            ("Pro", 290000, "price_1RjMKNKVj190YQJbhBuhcjfQ", 500, 3, False, 0, "monthly"),
            ("Pro (Yearly)", 2900000, "price_1RjMKNKVj190YQJbhBuhcjfQ_pro_yearly", 500, 3, False, 0, "yearly"),
            ("Business", 59900, "price_1RjMMuKVj190YQJbBX2LhoTs", 2500, 10, False, 0, "monthly"),
            ("Business (Yearly)", 599000, "price_1RjMMuKVj190YQJbBX2LhoTs_yearly", 2500, 10, False, 0, "yearly")
        ]
        
        for plan_data in plans_data:
            name, price_cents, stripe_price_id, jd_quota, max_subaccounts, is_trial, trial_days, billing_cycle = plan_data
            
            # Check if plan exists
            cursor.execute("SELECT id FROM plans WHERE name = %s", (name,))
            existing_plan = cursor.fetchone()
            
            if existing_plan:
                # Update existing plan
                cursor.execute("""
                    UPDATE plans SET 
                    price_cents = %s, stripe_price_id = %s, jd_quota_per_month = %s, 
                    max_subaccounts = %s, is_trial = %s, trial_days = %s, billing_cycle = %s
                    WHERE name = %s
                """, (price_cents, stripe_price_id, jd_quota, max_subaccounts, is_trial, trial_days, billing_cycle, name))
                print(f"✅ Updated plan: {name}")
            else:
                # Insert new plan
                cursor.execute("""
                    INSERT INTO plans (name, price_cents, stripe_price_id, jd_quota_per_month, 
                    max_subaccounts, is_trial, trial_days, billing_cycle, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """, (name, price_cents, stripe_price_id, jd_quota, max_subaccounts, is_trial, trial_days, billing_cycle))
                print(f"✅ Added plan: {name}")
        
        connection.commit()
        print("✅ Database fix completed successfully!")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        raise

if __name__ == "__main__":
    fix_database() 