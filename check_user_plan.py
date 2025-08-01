import pymysql
from app.config import Config

def check_user_plan():
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
        
        print("🔍 Checking user plans and trials...")
        
        # Check all users and their plans
        cursor.execute("""
            SELECT u.id, u.email, u.tenant_id, t.plan_id, p.name as plan_name, p.is_trial
            FROM users u
            LEFT JOIN tenants t ON u.tenant_id = t.id
            LEFT JOIN plans p ON t.plan_id = p.id
            ORDER BY u.id
        """)
        
        users = cursor.fetchall()
        
        for user_data in users:
            user_id, email, tenant_id, plan_id, plan_name, is_trial = user_data
            print(f"👤 User {user_id} ({email}):")
            print(f"   Tenant ID: {tenant_id}")
            print(f"   Plan ID: {plan_id}")
            print(f"   Plan Name: {plan_name}")
            print(f"   Is Trial: {is_trial}")
            
            # Check if user has trial record
            cursor.execute("SELECT * FROM user_trials WHERE user_id = %s", (user_id,))
            trial_record = cursor.fetchone()
            
            if trial_record:
                print(f"   ✅ Has trial record: {trial_record}")
            else:
                print(f"   ❌ No trial record")
                
                # If user is on Free Trial plan but no trial record, create one
                if plan_name == "Free Trial":
                    print(f"   🔧 Creating trial record for Free Trial user...")
                    cursor.execute("""
                        INSERT INTO user_trials (user_id, trial_start_date, trial_end_date, 
                        searches_used_today, is_active, created_at, updated_at)
                        VALUES (%s, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 0, TRUE, NOW(), NOW())
                    """, (user_id,))
                    connection.commit()
                    print(f"   ✅ Created trial record")
            
            print()
        
        # Check Free Trial plan
        cursor.execute("SELECT * FROM plans WHERE name = 'Free Trial'")
        free_trial_plan = cursor.fetchone()
        
        if free_trial_plan:
            print(f"✅ Free Trial plan exists: {free_trial_plan}")
        else:
            print("❌ Free Trial plan not found!")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ Error checking user plans: {e}")
        raise

if __name__ == "__main__":
    check_user_plan() 