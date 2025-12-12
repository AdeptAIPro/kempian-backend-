# Production entrypoint for Kempian backend
from app import create_app
import os

# Set production environment
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_DEBUG'] = '0'

app = create_app()

if __name__ == "__main__":
    # Production settings
    app.run(
        host="0.0.0.0", 
        port=int(os.environ.get('PORT', 8000)), 
        debug=False,
        threaded=True
    )
