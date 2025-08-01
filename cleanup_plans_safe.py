import pymysql
from app.config import Config

def cleanup_plans_safe():
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
    
    # Plans that should be kept (from create_tables_and_seed_plans.py)
    KEEP_PLANS = [
        "Free Trial",
        "Recruiters plan",
        "Recruiters plan (Yearly)",
        "Basic plan",
        "Basic plan (Yearly)",
        "Growth",
        "Growth (Yearly)",
        "Professional",
        "Professional (Yearly)"
    ]
    
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        cursor = connection.cursor()
        
        print("🧹 Safely cleaning up plans database...")
        
        # First, let's see what plans currently exist
        cursor.execute("SELECT id, name FROM plans ORDER BY id")
        existing_plans = cursor.fetchall()
        
        print("📋 Current plans in database:")
        for plan_id, plan_name in existing_plans:
            print(f"   ID {plan_id}: {plan_name}")
        
        print(f"\n✅ Plans to keep: {KEEP_PLANS}")
        
        # Find the Free Trial plan ID to use as fallback
        cursor.execute("SELECT id FROM plans WHERE name = 'Free Trial'")
        free_trial_result = cursor.fetchone()
        if free_trial_result:
            free_trial_id = free_trial_result[0]
            print(f"✅ Found Free Trial plan ID: {free_trial_id}")
        else:
            print("❌ Free Trial plan not found!")
            return
        
        # Update tenants that reference plans we want to delete
        for plan_id, plan_name in existing_plans:
            if plan_name not in KEEP_PLANS:
                print(f"🔄 Updating tenants using plan: {plan_name} (ID: {plan_id})")
                cursor.execute("UPDATE tenants SET plan_id = %s WHERE plan_id = %s", (free_trial_id, plan_id))
                updated_count = cursor.rowcount
                if updated_count > 0:
                    print(f"   ✅ Updated {updated_count} tenants to Free Trial plan")
        
        connection.commit()
        
        # Now delete the plans that are not in the keep list
        for plan_id, plan_name in existing_plans:
            if plan_name not in KEEP_PLANS:
                print(f"🗑️  Deleting plan: {plan_name} (ID: {plan_id})")
                cursor.execute("DELETE FROM plans WHERE id = %s", (plan_id,))
            else:
                print(f"✅ Keeping plan: {plan_name} (ID: {plan_id})")
        
        connection.commit()
        print("\n✅ Plan cleanup completed!")
        
        # Verify the final state
        print("\n📋 Final plans in database:")
        cursor.execute("SELECT id, name, price_cents, jd_quota_per_month FROM plans ORDER BY id")
        final_plans = cursor.fetchall()
        
        for plan_id, plan_name, price_cents, quota in final_plans:
            price_dollars = price_cents / 100 if price_cents > 0 else 0
            print(f"   ID {plan_id}: {plan_name} - ${price_dollars} - {quota} searches/month")
        
        # Also verify tenant assignments
        print("\n👥 Tenant plan assignments:")
        cursor.execute("""
            SELECT t.id, p.name as plan_name 
            FROM tenants t 
            LEFT JOIN plans p ON t.plan_id = p.id 
            ORDER BY t.id
        """)
        tenant_assignments = cursor.fetchall()
        
        for tenant_id, plan_name in tenant_assignments:
            print(f"   Tenant {tenant_id}: {plan_name}")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ Error cleaning up plans: {e}")
        raise

if __name__ == "__main__":
    cleanup_plans_safe() 