from .db import db
from app.simple_logger import get_logger
from datetime import datetime, timedelta
import json

class Plan(db.Model):
    __tablename__ = "plans"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    price_cents = db.Column(db.Integer, nullable=False)
    stripe_price_id = db.Column(db.String(128), nullable=False)
    jd_quota_per_month = db.Column(db.Integer, nullable=False)
    max_subaccounts = db.Column(db.Integer, nullable=False)
    is_trial = db.Column(db.Boolean, default=False, nullable=False)
    trial_days = db.Column(db.Integer, default=0, nullable=False)
    billing_cycle = db.Column(db.String(20), default='monthly', nullable=False)  # 'monthly' or 'yearly'
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
            'max_subaccounts': self.max_subaccounts,
            'is_trial': self.is_trial,
            'trial_days': self.trial_days,
            'billing_cycle': self.billing_cycle
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
    transactions = db.relationship('SubscriptionTransaction', backref='tenant', lazy=True)
    subscription_history = db.relationship('SubscriptionHistory', backref='tenant', lazy=True)

class SubscriptionTransaction(db.Model):
    __tablename__ = "subscription_transactions"
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    transaction_type = db.Column(db.Enum("purchase", "upgrade", "downgrade", "renewal", "cancellation", "refund"), nullable=False)
    stripe_payment_intent_id = db.Column(db.String(128), nullable=True)
    stripe_invoice_id = db.Column(db.String(128), nullable=True)
    amount_cents = db.Column(db.Integer, nullable=False)
    currency = db.Column(db.String(3), default='USD', nullable=False)
    plan_id = db.Column(db.Integer, db.ForeignKey("plans.id"), nullable=False)
    previous_plan_id = db.Column(db.Integer, db.ForeignKey("plans.id"), nullable=True)
    status = db.Column(db.Enum("pending", "succeeded", "failed", "cancelled"), default="pending", nullable=False)
    payment_method = db.Column(db.String(50), nullable=True)  # card, bank_transfer, etc.
    receipt_url = db.Column(db.String(500), nullable=True)
    invoice_url = db.Column(db.String(500), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    plan = db.relationship('Plan', foreign_keys=[plan_id], backref='transactions')
    previous_plan = db.relationship('Plan', foreign_keys=[previous_plan_id])
    user = db.relationship('User', backref='transactions')

    def to_dict(self):
        return {
            'id': self.id,
            'transaction_type': self.transaction_type,
            'amount_cents': self.amount_cents,
            'currency': self.currency,
            'plan_name': self.plan.name if self.plan else None,
            'previous_plan_name': self.previous_plan.name if self.previous_plan else None,
            'status': self.status,
            'payment_method': self.payment_method,
            'receipt_url': self.receipt_url,
            'invoice_url': self.invoice_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'notes': self.notes
        }

class SubscriptionHistory(db.Model):
    __tablename__ = "subscription_history"
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    action = db.Column(db.Enum("created", "upgraded", "downgraded", "cancelled", "renewed", "reactivated"), nullable=False)
    from_plan_id = db.Column(db.Integer, db.ForeignKey("plans.id"), nullable=True)
    to_plan_id = db.Column(db.Integer, db.ForeignKey("plans.id"), nullable=False)
    reason = db.Column(db.String(255), nullable=True)
    effective_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    from_plan = db.relationship('Plan', foreign_keys=[from_plan_id])
    to_plan = db.relationship('Plan', foreign_keys=[to_plan_id])
    user = db.relationship('User', backref='subscription_history')

    def to_dict(self):
        return {
            'id': self.id,
            'action': self.action,
            'from_plan_name': self.from_plan.name if self.from_plan else None,
            'to_plan_name': self.to_plan.name if self.to_plan else None,
            'reason': self.reason,
            'effective_date': self.effective_date.isoformat() if self.effective_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))  # nullable if using Cognito
    role = db.Column(db.Enum("owner", "subuser", "job_seeker", "employee", "recruiter", "employer", "admin"), nullable=False)
    user_type = db.Column(db.String(50), nullable=True)  # Store the original user type for display
    company_name = db.Column(db.String(255), nullable=True)  # Company name for employers/recruiters
    linkedin_id = db.Column(db.String(255), nullable=True)  # LinkedIn ID for OAuth
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    search_logs = db.relationship('JDSearchLog', backref='user', lazy=True)
    candidate_profile = db.relationship('CandidateProfile', backref='user', uselist=False)

class JDSearchLog(db.Model):
    __tablename__ = "jd_search_logs"
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    job_description = db.Column(db.Text)  # Add missing field
    candidates_found = db.Column(db.Integer, default=0, nullable=False)  # Number of candidates found
    search_criteria = db.Column(db.Text)  # JSON string of search criteria used
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

class StafferlinkIntegration(db.Model):
    __tablename__ = "stafferlink_integrations"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    stafferlink_email = db.Column(db.String(255), nullable=False)
    stafferlink_api_key = db.Column(db.String(255), nullable=False)
    stafferlink_agency_id = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('stafferlink_integration', uselist=False)) 

class UserSocialLinks(db.Model):
    __tablename__ = "user_social_links"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    linkedin = db.Column(db.String(255))
    facebook = db.Column(db.String(255))
    x = db.Column(db.String(255))
    github = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('social_links', uselist=False)) 

class UserTrial(db.Model):
    __tablename__ = "user_trials"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    trial_start_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    trial_end_date = db.Column(db.DateTime, nullable=False)
    searches_used_today = db.Column(db.Integer, default=0, nullable=False)
    last_search_date = db.Column(db.Date, nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('trial', uselist=False))

    def is_trial_valid(self):
        """Check if trial is still valid (within 7 days and not expired)"""
        now = datetime.utcnow()
        return self.is_active and now <= self.trial_end_date

    def can_search_today(self):
        """Check if user can perform searches today (within daily limit)"""
        if not self.is_trial_valid():
            return False
        
        today = datetime.utcnow().date()
        if self.last_search_date != today:
            # New day, reset count automatically
            self.searches_used_today = 0
            self.last_search_date = today
            self.updated_at = datetime.utcnow()
            return True  # New day, reset count
        return self.searches_used_today < 5  # 5 searches per day limit

    def increment_search_count(self):
        """Increment the search count for today"""
        today = datetime.utcnow().date()
        if self.last_search_date != today:
            # New day, reset count
            self.searches_used_today = 1
            self.last_search_date = today
        else:
            # Same day, increment count
            self.searches_used_today += 1
        self.updated_at = datetime.utcnow()
        
    def get_daily_quota_status(self):
        """Get detailed daily quota status"""
        today = datetime.utcnow().date()
        if self.last_search_date != today:
            # New day, reset count
            self.searches_used_today = 0
            self.last_search_date = today
            self.updated_at = datetime.utcnow()
        
        return {
            'searches_used_today': self.searches_used_today,
            'searches_remaining_today': max(0, 5 - self.searches_used_today),
            'daily_limit': 5,
            'last_search_date': self.last_search_date,
            'is_new_day': self.last_search_date == today
        }

class CandidateProfile(db.Model):
    __tablename__ = "candidate_profiles"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    full_name = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(50))
    location = db.Column(db.String(255))
    summary = db.Column(db.Text)
    experience_years = db.Column(db.Integer)
    current_salary = db.Column(db.String(100))
    expected_salary = db.Column(db.String(100))
    availability = db.Column(db.String(100))  # e.g., "Immediate", "2 weeks notice", etc.
    resume_s3_key = db.Column(db.String(500))  # S3 key for the resume file
    resume_filename = db.Column(db.String(255))
    resume_upload_date = db.Column(db.DateTime)
    is_public = db.Column(db.Boolean, default=True)  # Whether profile is visible to employers
    visa_status = db.Column(db.String(100), nullable=True) # Visa status for job seekers
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    skills = db.relationship('CandidateSkill', backref='profile', lazy=True, cascade='all, delete-orphan')
    education = db.relationship('CandidateEducation', backref='profile', lazy=True, cascade='all, delete-orphan')
    experience = db.relationship('CandidateExperience', backref='profile', lazy=True, cascade='all, delete-orphan')
    certifications = db.relationship('CandidateCertification', backref='profile', lazy=True, cascade='all, delete-orphan')
    projects = db.relationship('CandidateProject', backref='profile', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        # Get social links for this user
        social_links = UserSocialLinks.query.filter_by(user_id=self.user_id).first()
        social_links_data = {
            'linkedin': social_links.linkedin if social_links else '',
            'facebook': social_links.facebook if social_links else '',
            'x': social_links.x if social_links else '',
            'github': social_links.github if social_links else ''
        }
        
        return {
            'id': self.id,
            'user_id': self.user_id,
            'full_name': self.full_name,
            'phone': self.phone,
            'location': self.location,
            'summary': self.summary,
            'experience_years': self.experience_years,
            'current_salary': self.current_salary,
            'expected_salary': self.expected_salary,
            'availability': self.availability,
            'resume_s3_key': self.resume_s3_key,
            'resume_filename': self.resume_filename,
            'resume_upload_date': self.resume_upload_date.isoformat() if self.resume_upload_date and hasattr(self.resume_upload_date, 'isoformat') and callable(getattr(self.resume_upload_date, 'isoformat', None)) else None,
            'is_public': self.is_public,
            'visa_status': self.visa_status,
            'social_links': social_links_data,
            'skills': [skill.to_dict() for skill in self.skills],
            'education': [edu.to_dict() for edu in self.education],
            'experience': [exp.to_dict() for exp in self.experience],
            'certifications': [cert.to_dict() for cert in self.certifications],
            'projects': [project.to_dict() for project in self.projects]
        }

class CandidateSkill(db.Model):
    __tablename__ = "candidate_skills"
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey("candidate_profiles.id"), nullable=False)
    skill_name = db.Column(db.String(255), nullable=False)
    proficiency_level = db.Column(db.String(50))  # e.g., "Beginner", "Intermediate", "Advanced", "Expert"
    years_experience = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'skill_name': self.skill_name,
            'proficiency_level': self.proficiency_level,
            'years_experience': self.years_experience
        }

class CandidateEducation(db.Model):
    __tablename__ = "candidate_education"
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey("candidate_profiles.id"), nullable=False)
    institution = db.Column(db.String(255), nullable=False)
    degree = db.Column(db.String(255), nullable=False)
    field_of_study = db.Column(db.String(255))
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    gpa = db.Column(db.Float)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'institution': self.institution,
            'degree': self.degree,
            'field_of_study': self.field_of_study,
            'start_date': self.start_date.isoformat() if self.start_date and hasattr(self.start_date, 'isoformat') and callable(getattr(self.start_date, 'isoformat', None)) else None,
            'end_date': self.end_date.isoformat() if self.end_date and hasattr(self.end_date, 'isoformat') and callable(getattr(self.end_date, 'isoformat', None)) else None,
            'gpa': self.gpa,
            'description': self.description
        }

class CandidateExperience(db.Model):
    __tablename__ = "candidate_experience"
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey("candidate_profiles.id"), nullable=False)
    company = db.Column(db.String(255), nullable=False)
    position = db.Column(db.String(255), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date)  # Null for current position
    is_current = db.Column(db.Boolean, default=False)
    description = db.Column(db.Text)
    achievements = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'company': self.company,
            'position': self.position,
            'start_date': self.start_date.isoformat() if self.start_date and hasattr(self.start_date, 'isoformat') and callable(getattr(self.start_date, 'isoformat', None)) else None,
            'end_date': self.end_date.isoformat() if self.end_date and hasattr(self.end_date, 'isoformat') and callable(getattr(self.end_date, 'isoformat', None)) else None,
            'is_current': self.is_current,
            'description': self.description,
            'achievements': self.achievements
        }

class CandidateCertification(db.Model):
    __tablename__ = "candidate_certifications"
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey("candidate_profiles.id"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    issuing_organization = db.Column(db.String(255))
    issue_date = db.Column(db.Date)
    expiry_date = db.Column(db.Date)
    credential_id = db.Column(db.String(255))
    credential_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'issuing_organization': self.issuing_organization,
            'issue_date': self.issue_date.isoformat() if self.issue_date and hasattr(self.issue_date, 'isoformat') and callable(getattr(self.issue_date, 'isoformat', None)) else None,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date and hasattr(self.expiry_date, 'isoformat') and callable(getattr(self.expiry_date, 'isoformat', None)) else None,
            'credential_id': self.credential_id,
            'credential_url': self.credential_url
        }

class CandidateProject(db.Model):
    __tablename__ = "candidate_projects"
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey("candidate_profiles.id"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    technologies = db.Column(db.String(500))  # Comma-separated technologies used
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    project_url = db.Column(db.String(500))
    github_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'technologies': self.technologies,
            'start_date': self.start_date.isoformat() if self.start_date and hasattr(self.start_date, 'isoformat') and callable(getattr(self.start_date, 'isoformat', None)) else None,
            'end_date': self.end_date.isoformat() if self.end_date and hasattr(self.end_date, 'isoformat') and callable(getattr(self.end_date, 'isoformat', None)) else None,
            'project_url': self.project_url,
            'github_url': self.github_url,
            'created_at': self.created_at.isoformat() if self.created_at and hasattr(self.created_at, 'isoformat') and callable(getattr(self.created_at, 'isoformat', None)) else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at and hasattr(self.updated_at, 'isoformat') and callable(getattr(self.updated_at, 'isoformat', None)) else None
        }

class SecretSection(db.Model):
    __tablename__ = "secret_section"
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(255), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PasswordResetOTP(db.Model):
    __tablename__ = 'password_reset_otps'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    otp = db.Column(db.String(10), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)
    used_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref='password_reset_otps')
    
    def __repr__(self):
        return f'<PasswordResetOTP {self.user_id}>'
    
    @property
    def is_expired(self):
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self):
        return not self.used and not self.is_expired 

class UnlimitedQuotaUser(db.Model):
    """Model for users with unlimited quota"""
    __tablename__ = 'unlimited_quota_users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    reason = db.Column(db.String(500), nullable=True)
    quota_limit = db.Column(db.Integer, default=-1)  # -1 means unlimited
    daily_limit = db.Column(db.Integer, default=-1)
    monthly_limit = db.Column(db.Integer, default=-1)
    added_by = db.Column(db.String(255), nullable=False)
    added_date = db.Column(db.DateTime, default=datetime.utcnow)
    expires = db.Column(db.DateTime, nullable=True)
    active = db.Column(db.Boolean, default=True)
    updated_by = db.Column(db.String(255), nullable=True)
    updated_date = db.Column(db.DateTime, nullable=True)
    removed_by = db.Column(db.String(255), nullable=True)
    removed_date = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<UnlimitedQuotaUser {self.email}>'
    
    def to_dict(self):
        return {
            'email': self.email,
            'reason': self.reason,
            'quota_limit': self.quota_limit,
            'daily_limit': self.daily_limit,
            'monthly_limit': self.monthly_limit,
            'added_by': self.added_by,
            'added_date': self.added_date.isoformat() if self.added_date and hasattr(self.added_date, 'isoformat') and callable(getattr(self.added_date, 'isoformat', None)) else None,
            'expires': self.expires.isoformat() if self.expires and hasattr(self.expires, 'isoformat') and callable(getattr(self.expires, 'isoformat', None)) else None,
            'active': self.active,
            'updated_by': self.updated_by,
            'updated_date': self.updated_date.isoformat() if self.updated_date and hasattr(self.updated_date, 'isoformat') and callable(getattr(self.updated_date, 'isoformat', None)) else None,
            'removed_by': self.removed_by,
            'removed_date': self.removed_date.isoformat() if self.removed_date and hasattr(self.removed_date, 'isoformat') and callable(getattr(self.removed_date, 'isoformat', None)) else None
        }

class UserImage(db.Model):
    """Model for storing user profile images"""
    __tablename__ = 'user_images'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    image_data = db.Column(db.Text(length=16777215), nullable=False)  # Base64 encoded image (MEDIUMTEXT equivalent)
    image_type = db.Column(db.String(50), nullable=False)  # e.g., 'image/jpeg', 'image/png'
    file_name = db.Column(db.String(255), nullable=True)
    file_size = db.Column(db.Integer, nullable=True)  # Size in bytes
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('profile_image', uselist=False))
    
    def __repr__(self):
        return f'<UserImage {self.user_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'image_type': self.image_type,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at and hasattr(self.uploaded_at, 'isoformat') and callable(getattr(self.uploaded_at, 'isoformat', None)) else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at and hasattr(self.updated_at, 'isoformat') and callable(getattr(self.updated_at, 'isoformat', None)) else None,
            'image_url': f'/api/user/{self.user_id}/image'  # URL to retrieve image
        }

# New KPI Models
class UserKPIs(db.Model):
    """Model for storing user-specific KPI data"""
    __tablename__ = 'user_kpis'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    role_fit_score = db.Column(db.Float, default=0.0)  # 0-100
    career_benchmark = db.Column(db.String(50), default='Top 50%')  # e.g., 'Top 20%', 'Top 50%'
    industry_targeting = db.Column(db.Integer, default=0)  # Number of industries targeted
    experience_level = db.Column(db.String(50), default='Average')  # e.g., '30% Below', 'Average', 'Above'
    skills_learned = db.Column(db.Integer, default=0)
    jobs_applied = db.Column(db.Integer, default=0)
    courses_completed = db.Column(db.Integer, default=0)
    learning_streak = db.Column(db.Integer, default=0)  # Days
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('kpis', uselist=False))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'role_fit_score': self.role_fit_score,
            'career_benchmark': self.career_benchmark,
            'industry_targeting': self.industry_targeting,
            'experience_level': self.experience_level,
            'skills_learned': self.skills_learned,
            'jobs_applied': self.jobs_applied,
            'courses_completed': self.courses_completed,
            'learning_streak': self.learning_streak,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class UserSkillGap(db.Model):
    """Model for storing user skill gap analysis"""
    __tablename__ = 'user_skill_gaps'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    skill_name = db.Column(db.String(255), nullable=False)
    current_level = db.Column(db.Integer, default=0)  # 0-100
    target_level = db.Column(db.Integer, default=0)  # 0-100
    priority = db.Column(db.String(20), default='Medium')  # High, Medium, Low
    role_target = db.Column(db.String(255), nullable=True)  # Target role for this skill
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('skill_gaps', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'skill_name': self.skill_name,
            'current_level': self.current_level,
            'target_level': self.target_level,
            'priority': self.priority,
            'role_target': self.role_target,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class UserLearningPath(db.Model):
    """Model for storing user learning pathway data"""
    __tablename__ = 'user_learning_paths'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    pathway_name = db.Column(db.String(255), nullable=False)
    pathway_description = db.Column(db.Text)
    total_duration = db.Column(db.String(50))  # e.g., '4-6 months'
    progress = db.Column(db.Float, default=0.0)  # 0-100
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('learning_paths', lazy=True))
    modules = db.relationship('LearningModule', backref='learning_path', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'pathway_name': self.pathway_name,
            'pathway_description': self.pathway_description,
            'total_duration': self.total_duration,
            'progress': self.progress,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'modules': [module.to_dict() for module in self.modules]
        }

class LearningModule(db.Model):
    """Model for storing learning modules within a pathway"""
    __tablename__ = 'learning_modules'
    
    id = db.Column(db.Integer, primary_key=True)
    learning_path_id = db.Column(db.Integer, db.ForeignKey('user_learning_paths.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), default='upcoming')  # completed, in-progress, upcoming
    duration = db.Column(db.String(50))  # e.g., '2 weeks'
    order_index = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    courses = db.relationship('LearningCourse', backref='module', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'learning_path_id': self.learning_path_id,
            'title': self.title,
            'status': self.status,
            'duration': self.duration,
            'order_index': self.order_index,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'courses': [course.to_dict() for course in self.courses]
        }

class LearningCourse(db.Model):
    """Model for storing individual courses within modules"""
    __tablename__ = 'learning_courses'
    
    id = db.Column(db.Integer, primary_key=True)
    module_id = db.Column(db.Integer, db.ForeignKey('learning_modules.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    provider = db.Column(db.String(255))  # e.g., 'Coursera', 'Udemy'
    rating = db.Column(db.Float)  # 0.0-5.0
    duration = db.Column(db.String(50))  # e.g., '3 hours'
    course_url = db.Column(db.String(500))
    completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'module_id': self.module_id,
            'name': self.name,
            'provider': self.provider,
            'rating': self.rating,
            'duration': self.duration,
            'course_url': self.course_url,
            'completed': self.completed,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class UserAchievement(db.Model):
    """Model for storing user achievements and milestones"""
    __tablename__ = 'user_achievements'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    type = db.Column(db.String(50))  # skill, profile, habit, course
    points = db.Column(db.Integer, default=0)
    achieved_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('achievements', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'type': self.type,
            'points': self.points,
            'achieved_at': self.achieved_at.isoformat() if self.achieved_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class UserGoal(db.Model):
    """Model for storing user goals and targets"""
    __tablename__ = 'user_goals'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    progress = db.Column(db.Float, default=0.0)  # 0-100
    deadline = db.Column(db.String(50))  # e.g., '2 days', '1 week'
    priority = db.Column(db.String(20), default='medium')  # high, medium, low
    is_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('goals', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'progress': self.progress,
            'deadline': self.deadline,
            'priority': self.priority,
            'is_completed': self.is_completed,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class UserSchedule(db.Model):
    """Model for storing user schedule and calendar events"""
    __tablename__ = 'user_schedules'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    event_date = db.Column(db.DateTime, nullable=False)
    duration_minutes = db.Column(db.Integer, default=60)
    event_type = db.Column(db.String(50))  # study, interview, networking, etc.
    is_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('schedules', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'event_date': self.event_date.isoformat() if self.event_date else None,
            'duration_minutes': self.duration_minutes,
            'event_type': self.event_type,
            'is_completed': self.is_completed,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class SavedCandidate(db.Model):
    """Model for storing saved candidates with 20-day visibility"""
    __tablename__ = "saved_candidates"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    candidate_data = db.Column(db.Text, nullable=False)  # JSON string of candidate data
    job_description = db.Column(db.Text)  # Job description when candidate was saved
    saved_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)  # 20 days from saved_at
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('saved_candidates', lazy=True))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set expiration date to 20 days from now
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(days=20)
    
    def is_expired(self):
        """Check if the saved candidate has expired (older than 20 days)"""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self):
        import json
        return {
            'id': self.id,
            'user_id': self.user_id,
            'candidate_data': json.loads(self.candidate_data) if self.candidate_data else None,
            'job_description': self.job_description,
            'saved_at': self.saved_at.isoformat() if self.saved_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'is_expired': self.is_expired(),
            'days_remaining': max(0, (self.expires_at - datetime.utcnow()).days) if self.expires_at else 0
        }

class SearchHistory(db.Model):
    """Model for storing chatbot search history with 5-day retention"""
    __tablename__ = "search_history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    search_query = db.Column(db.Text, nullable=False)
    search_results = db.Column(db.Text)  # JSON string of search results
    conversation_history = db.Column(db.Text)  # JSON string of conversation history
    search_type = db.Column(db.String(50), default='chatbot')  # 'chatbot', 'talent_search', etc.
    searched_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)  # 5 days from searched_at
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('search_history', lazy=True))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set expiration date to 5 days from now
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(days=5)
    
    def is_expired(self):
        """Check if the search history has expired (older than 5 days)"""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self):
        import json
        return {
            'id': self.id,
            'user_id': self.user_id,
            'search_query': self.search_query,
            'search_results': json.loads(self.search_results) if self.search_results else None,
            'conversation_history': json.loads(self.conversation_history) if self.conversation_history else None,
            'search_type': self.search_type,
            'searched_at': self.searched_at.isoformat() if self.searched_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'is_expired': self.is_expired(),
            'days_remaining': max(0, (self.expires_at - datetime.utcnow()).days) if self.expires_at else 0
        }

from .db import db
from datetime import datetime

class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    company_name = db.Column(db.String(100), nullable=False)
    employment_type = db.Column(db.String(50), nullable=False)  # full-time, part-time, contract, etc.
    experience_level = db.Column(db.String(50), nullable=False)  # entry, mid, senior, executive
    salary_min = db.Column(db.Integer, nullable=True)
    salary_max = db.Column(db.Integer, nullable=True)
    currency = db.Column(db.String(3), default='USD')
    remote_allowed = db.Column(db.Boolean, default=False)
    skills_required = db.Column(db.Text, nullable=True)  # JSON string of skills
    benefits = db.Column(db.Text, nullable=True)
    requirements = db.Column(db.Text, nullable=True)
    responsibilities = db.Column(db.Text, nullable=True)
    
    # Status and visibility
    status = db.Column(db.String(20), default='active')  # active, closed, draft
    is_public = db.Column(db.Boolean, default=True)
    views_count = db.Column(db.Integer, default=0)
    applications_count = db.Column(db.Integer, default=0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)
    
    # Foreign keys
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=False)
    
    # Relationships
    creator = db.relationship('User', backref='created_jobs', foreign_keys=[created_by])
    tenant = db.relationship('Tenant', backref='jobs')
    applications = db.relationship('JobApplication', backref='job', lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'location': self.location,
            'company_name': self.company_name,
            'employment_type': self.employment_type,
            'experience_level': self.experience_level,
            'salary_min': self.salary_min,
            'salary_max': self.salary_max,
            'currency': self.currency,
            'remote_allowed': self.remote_allowed,
            'skills_required': json.loads(self.skills_required) if self.skills_required else [],
            'benefits': self.benefits,
            'requirements': self.requirements,
            'responsibilities': self.responsibilities,
            'status': self.status,
            'is_public': self.is_public,
            'views_count': self.views_count,
            'applications_count': self.applications_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_by': self.created_by,
            'tenant_id': self.tenant_id
        }
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'location': self.location,
            'company_name': self.company_name,
            'employment_type': self.employment_type,
            'experience_level': self.experience_level,
            'salary_min': self.salary_min,
            'salary_max': self.salary_max,
            'currency': self.currency,
            'remote_allowed': self.remote_allowed,
            'skills_required': self.skills_required,
            'benefits': self.benefits,
            'requirements': self.requirements,
            'responsibilities': self.responsibilities,
            'status': self.status,
            'is_public': self.is_public,
            'views_count': self.views_count,
            'applications_count': self.applications_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_by': self.created_by,
            'tenant_id': self.tenant_id,
            'creator_name': self.creator.email if self.creator else None,
            'public_url': f"/jobs/{self.id}"
        }
    
    def is_active(self):
        """Check if job is active and not expired"""
        if self.status != 'active' or not self.is_public:
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True
    
    def increment_views(self):
        """Increment view count"""
        self.views_count += 1
        db.session.commit()
    
    def increment_applications(self):
        """Increment application count"""
        self.applications_count += 1
        db.session.commit()

class JobApplication(db.Model):
    __tablename__ = 'job_applications'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False)
    applicant_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Application details
    cover_letter = db.Column(db.Text, nullable=True)
    resume_s3_key = db.Column(db.String(500), nullable=True)
    resume_filename = db.Column(db.String(200), nullable=True)
    
    # Application status
    status = db.Column(db.String(20), default='submitted')  # submitted, reviewed, shortlisted, rejected, hired, interview_scheduled
    notes = db.Column(db.Text, nullable=True)  # Internal notes by employer
    
    # Interview details
    interview_scheduled = db.Column(db.Boolean, default=False)
    interview_date = db.Column(db.DateTime, nullable=True)
    interview_meeting_link = db.Column(db.String(500), nullable=True)
    interview_meeting_type = db.Column(db.String(20), nullable=True)  # zoom, google_meet, teams, other
    interview_notes = db.Column(db.Text, nullable=True)
    
    # Additional questions/answers (JSON)
    additional_answers = db.Column(db.Text, nullable=True)
    
    # Timestamps
    applied_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    applicant = db.relationship('User', backref='job_applications', foreign_keys=[applicant_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'applicant_id': self.applicant_id,
            'cover_letter': self.cover_letter,
            'resume_s3_key': self.resume_s3_key,
            'resume_filename': self.resume_filename,
            'status': self.status,
            'notes': self.notes,
            'interview_scheduled': self.interview_scheduled,
            'interview_date': self.interview_date.isoformat() if self.interview_date else None,
            'interview_meeting_link': self.interview_meeting_link,
            'interview_meeting_type': self.interview_meeting_type,
            'interview_notes': self.interview_notes,
            'additional_answers': self.additional_answers,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'applicant_name': self.applicant.email if self.applicant else None,
            'applicant_email': self.applicant.email if self.applicant else None,
            'job_title': self.job.title if self.job else None,
            'job': self.job.to_dict() if self.job else None
        }
