import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask
from flask_cors import CORS
from .config import Config
from .db import db
from .models import CeipalIntegration

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    # Only allow your frontend origin and support credentials
    # Do NOT use '*' for origins when supports_credentials=True, per CORS spec.
    CORS(app, supports_credentials=True, origins=[
        "http://localhost:8081", "http://127.0.0.1:8081",
        "https://kempian.ai", "https://new.kempian.ai",
        "http://localhost:8080", "http://127.0.0.1:8080",
        "http://localhost:5173", "http://127.0.0.1:5173"
    ])
    # NOTE: If you change CORS config, restart the backend server to apply changes.

    db.init_app(app)

    # Register blueprints
    from .auth.routes import auth_bp
    from .tenants.routes import tenants_bp
    from .plans.routes import plans_bp
    from .search.routes import search_bp
    from .stripe.routes import stripe_bp
    from .stripe.webhook import webhook_bp
    from .ceipal.routes import ceipal_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(tenants_bp, url_prefix="/tenant")
    app.register_blueprint(plans_bp, url_prefix="/plans")
    app.register_blueprint(search_bp, url_prefix="/search")
    app.register_blueprint(stripe_bp, url_prefix="/checkout")
    app.register_blueprint(webhook_bp, url_prefix="/webhook")
    app.register_blueprint(ceipal_bp, url_prefix="")

    return app 