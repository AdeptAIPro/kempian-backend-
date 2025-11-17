import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from app import create_app, db
from app.simple_logger import get_logger
except ImportError:
    from .. import create_app, db
from sqlalchemy import text

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        print('Flask app DB URI:', app.config['SQLALCHEMY_DATABASE_URI'])
        try:
            db.session.execute(text('SELECT 1'))
            print('DB connection successful')
        except Exception as e:
            print('DB connection failed:', e) 