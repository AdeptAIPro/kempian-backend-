import pymysql
from app.config import Config

def reset_search_logs():
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
        
        print("🧹 Resetting search logs for trial users...")
        
        # Clear all search logs
        cursor.execute("DELETE FROM jd_search_logs")
        deleted_count = cursor.rowcount
        print(f"✅ Deleted {deleted_count} search log entries")
        
        # Reset trial search counts to 0
        cursor.execute("UPDATE user_trials SET searches_used_today = 0, last_search_date = NULL")
        updated_count = cursor.rowcount
        print(f"✅ Reset {updated_count} trial records")
        
        connection.commit()
        print("✅ Search logs reset successfully!")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ Error resetting search logs: {e}")
        raise

if __name__ == "__main__":
    reset_search_logs() 