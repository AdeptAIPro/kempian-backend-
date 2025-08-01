import pymysql
from app.config import Config

def fix_user_plans():
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
        
        print("🔧 Fixing user plans to Free Trial...")
        
        # Get Free Trial plan ID
        cursor.execute("SELECT id FROM plans WHERE name = 'Free Trial'")
        free_trial_plan = cursor.fetchone()
        
        if not free_trial_plan:
            print("❌ Free Trial plan not found!")
            return
        
        free_trial_plan_id = free_trial_plan[0]
        print(f"✅ Found Free Trial plan ID: {free_trial_plan_id}")
        
        # Get all users
        cursor.execute("SELECT id, email, tenant_id FROM users ORDER BY id")
        users = cursor.fetchall()
        
        for user_data in users:
            user_id, email, tenant_id = user_data
            print(f"👤 Processing user {user_id} ({email})...")
            
            # Update tenant to use Free Trial plan
            cursor.execute("UPDATE tenants SET plan_id = %s WHERE id = %s", (free_trial_plan_id, tenant_id))
            print(f"   ✅ Updated tenant {tenant_id} to Free Trial plan")
            
            # Check if user already has trial record
            cursor.execute("SELECT id FROM user_trials WHERE user_id = %s", (user_id,))
            existing_trial = cursor.fetchone()
            
            if existing_trial:
                print(f"   ✅ User already has trial record")
            else:
                # Create trial record
                cursor.execute("""
                    INSERT INTO user_trials (user_id, trial_start_date, trial_end_date, 
                    searches_used_today, is_active, created_at, updated_at)
                    VALUES (%s, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 0, TRUE, NOW(), NOW())
                """, (user_id,))
                print(f"   ✅ Created trial record")
            
            print()
        
        connection.commit()
        print("✅ All users moved to Free Trial plan successfully!")
        
        # Verify the changes
        print("\n🔍 Verifying changes...")
        cursor.execute("""
            SELECT u.id, u.email, p.name as plan_name, p.is_trial
            FROM users u
            LEFT JOIN tenants t ON u.tenant_id = t.id
            LEFT JOIN plans p ON t.plan_id = p.id
            ORDER BY u.id
        """)
        
        users = cursor.fetchall()
        for user_data in users:
            user_id, email, plan_name, is_trial = user_data
            print(f"👤 User {user_id} ({email}): {plan_name} (Trial: {is_trial})")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ Error fixing user plans: {e}")
        raise

if __name__ == "__main__":
    fix_user_plans() 