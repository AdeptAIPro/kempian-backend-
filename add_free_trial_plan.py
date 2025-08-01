import pymysql
from app.config import Config

def add_free_trial_plan():
    # Parse database URL to get connection details
    db_url = Config.SQLALCHEMY_DATABASE_URI
    parts = db_url.replace('mysql+pymysql://', '').split('@')
    user_pass = parts[0].split(':')
    host_port_db = parts[1].split('/')
    host_port = host_port_db[0].split(':')
    
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else 3306
    user = user_pass[0]
    password = user_pass[1]
    database = host_port_db[1]
    
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        cursor = connection.cursor()
        
        # Check if Free Trial plan exists
        cursor.execute("SELECT id FROM plans WHERE name = 'Free Trial'")
        existing_plan = cursor.fetchone()
        
        if not existing_plan:
            # Insert Free Trial plan
            cursor.execute("""
                INSERT INTO plans (name, price_cents, stripe_price_id, jd_quota_per_month, 
                max_subaccounts, is_trial, trial_days, billing_cycle, created_at, updated_at)
                VALUES ('Free Trial', 0, 'price_free_trial', 5, 1, TRUE, 7, 'monthly', NOW(), NOW())
            """)
            connection.commit()
            print("✅ Added Free Trial plan")
        else:
            print("✅ Free Trial plan already exists")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ Error adding Free Trial plan: {e}")
        raise

if __name__ == "__main__":
    add_free_trial_plan() 