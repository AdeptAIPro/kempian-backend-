from .db import db
from datetime import datetime

class Plan(db.Model):
    __tablename__ = "plans"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    price_cents = db.Column(db.Integer, nullable=False)
    stripe_price_id = db.Column(db.String(128), nullable=False)
    jd_quota_per_month = db.Column(db.Integer, nullable=False)
    max_subaccounts = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    tenants = db.relationship('Tenant', backref='plan', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'price_cents': self.price_cents,
            'stripe_price_id': self.stripe_price_id,
            'jd_quota_per_month': self.jd_quota_per_month,
            'max_subaccounts': self.max_subaccounts
        }

class Tenant(db.Model):
    __tablename__ = "tenants"
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey("plans.id"), nullable=False)
    stripe_customer_id = db.Column(db.String(128), nullable=False)
    stripe_subscription_id = db.Column(db.String(128), nullable=False)
    status = db.Column(db.Enum("active", "inactive", "cancelled"), default="active", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    users = db.relationship('User', backref='tenant', lazy=True)
    search_logs = db.relationship('JDSearchLog', backref='tenant', lazy=True)

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))  # nullable if using Cognito
    role = db.Column(db.Enum("owner", "subuser"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    search_logs = db.relationship('JDSearchLog', backref='user', lazy=True)

class JDSearchLog(db.Model):
    __tablename__ = "jd_search_logs"
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    searched_at = db.Column(db.DateTime, default=datetime.utcnow)

class TenantAlert(db.Model):
    __tablename__ = "tenant_alerts"
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    alert_type = db.Column(db.String(64), nullable=False)
    alert_month = db.Column(db.String(7), nullable=False)  # e.g. '2024-06'
    sent_at = db.Column(db.DateTime, default=datetime.utcnow) 

class CeipalIntegration(db.Model):
    __tablename__ = "ceipal_integrations"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    ceipal_email = db.Column(db.String(255), nullable=False)
    ceipal_api_key = db.Column(db.String(255), nullable=False)
    ceipal_password = db.Column(db.String(255), nullable=False)  # Store encrypted in production
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('ceipal_integration', uselist=False)) 