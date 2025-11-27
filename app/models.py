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
    organization_metadata = db.relationship('OrganizationMetadata', backref='tenant', uselist=False)

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
    employee_profile = db.relationship('EmployeeProfile', backref='user', uselist=False)
    timesheets = db.relationship('Timesheet', foreign_keys='Timesheet.user_id', backref='employee_user', lazy=True)

class UserBankAccount(db.Model):
    __tablename__ = "user_bank_accounts"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)

    # Primary Contacts
    owner_or_authorized_rep_name = db.Column(db.String(255), nullable=True)
    title_or_role = db.Column(db.String(255), nullable=True)
    contact_email = db.Column(db.String(255), nullable=True)
    contact_phone = db.Column(db.String(50), nullable=True)
    payroll_contact_person = db.Column(db.String(255), nullable=True)
    is_payroll_contact_different = db.Column(db.Boolean, default=False, nullable=False)
    account_holder_name = db.Column(db.String(255), nullable=True)

    # Banking Details
    bank_name = db.Column(db.String(255), nullable=True)
    routing_number = db.Column(db.String(50), nullable=True)
    account_number = db.Column(db.String(50), nullable=True)

    # Payroll Setup Preferences
    pay_frequency = db.Column(db.String(50), nullable=True)
    first_intended_pay_date = db.Column(db.Date, nullable=True)
    direct_deposit_or_check_preference = db.Column(db.String(50), nullable=True)
    third_party_integrations = db.Column(db.Text, nullable=True)

    # Tax and Compliance
    tax_filing_responsibility = db.Column(db.String(50), nullable=True)
    state_unemployment_account_info = db.Column(db.Text, nullable=True)
    workers_comp_carrier = db.Column(db.String(255), nullable=True)
    workers_comp_policy_number = db.Column(db.String(255), nullable=True)
    company_registration_states = db.Column(db.JSON, default=list, nullable=False)

    # Document URLs (optional if stored externally)
    voided_check_or_bank_letter_url = db.Column(db.String(512), nullable=True)
    signed_service_agreement_url = db.Column(db.String(512), nullable=True)
    power_of_attorney_url = db.Column(db.String(512), nullable=True)
    business_registration_certificate_url = db.Column(db.String(512), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    documents = db.relationship(
        'UserBankDocument',
        backref='bank_account',
        lazy=True,
        cascade='all, delete-orphan'
    )

    def to_dict(self, document_urls=None):
        return {
            'ownerOrAuthorizedRepName': self.owner_or_authorized_rep_name,
            'titleOrRole': self.title_or_role,
            'contactEmail': self.contact_email,
            'contactPhone': self.contact_phone,
            'payrollContactPerson': self.payroll_contact_person,
            'isPayrollContactDifferent': self.is_payroll_contact_different,
            'accountHolderName': self.account_holder_name,
            'bankName': self.bank_name,
            'routingNumber': self.routing_number,
            'accountNumber': self.account_number,
            'payFrequency': self.pay_frequency,
            'firstIntendedPayDate': self.first_intended_pay_date.isoformat() if self.first_intended_pay_date else '',
            'directDepositOrCheckPreference': self.direct_deposit_or_check_preference,
            'thirdPartyIntegrations': self.third_party_integrations or '',
            'taxFilingResponsibility': self.tax_filing_responsibility,
            'stateUnemploymentAccountInfo': self.state_unemployment_account_info or '',
            'workersCompCarrier': self.workers_comp_carrier,
            'workersCompPolicyNumber': self.workers_comp_policy_number,
            'companyRegistrationStates': self.company_registration_states or [],
            'voidedCheckOrBankLetterUrl': self.voided_check_or_bank_letter_url if self.voided_check_or_bank_letter_url else (document_urls or {}).get('voided_check'),
            'signedServiceAgreementUrl': self.signed_service_agreement_url if self.signed_service_agreement_url else (document_urls or {}).get('signed_service_agreement'),
            'powerOfAttorneyUrl': self.power_of_attorney_url if self.power_of_attorney_url else (document_urls or {}).get('power_of_attorney'),
            'businessRegistrationCertificateUrl': self.business_registration_certificate_url if self.business_registration_certificate_url else (document_urls or {}).get('business_registration_certificate'),
        }

class UserBankDocument(db.Model):
    __tablename__ = "user_bank_documents"
    id = db.Column(db.Integer, primary_key=True)
    bank_account_id = db.Column(db.Integer, db.ForeignKey("user_bank_accounts.id"), nullable=False)
    doc_type = db.Column(db.String(64), nullable=False)
    file_name = db.Column(db.String(255), nullable=True)
    content_type = db.Column(db.String(128), nullable=True)
    file_size = db.Column(db.Integer, nullable=True)
    data = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_metadata(self):
        return {
            'doc_type': self.doc_type,
            'file_name': self.file_name,
            'content_type': self.content_type,
            'file_size': self.file_size,
        }

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
    last_job_sync_at = db.Column(db.DateTime, nullable=True)
    user = db.relationship('User', backref=db.backref('stafferlink_integration', uselist=False))

class StafferlinkJob(db.Model):
    __tablename__ = "stafferlink_jobs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    order_id = db.Column(db.String(255), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    company = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255), nullable=True)
    salary = db.Column(db.String(255), nullable=True)
    duration = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    requirements = db.Column(db.Text, nullable=True)
    experience = db.Column(db.String(255), nullable=True)
    job_code = db.Column(db.String(255), nullable=True)
    posted_at = db.Column(db.DateTime, nullable=True)
    last_seen_at = db.Column(db.DateTime, default=datetime.utcnow)
    raw_payload = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'order_id', name='uq_stafferlink_job_user_order'),
    )

    user = db.relationship('User', backref=db.backref('stafferlink_jobs', lazy=True))

    def to_dict(self):
        return {
            'id': self.order_id or str(self.id),
            'title': self.title,
            'company': self.company,
            'location': self.location,
            'salary': self.salary,
            'duration': self.duration,
            'description': self.description or '',
            'requirements': self.requirements or '',
            'experience': self.experience or '',
            'job_code': self.job_code,
            'created_at': self.posted_at.isoformat() if self.posted_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_synced_at': self.last_seen_at.isoformat() if self.last_seen_at else None,
        }

class JobAdderIntegration(db.Model):
    __tablename__ = "jobadder_integrations"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    client_id = db.Column(db.String(255), nullable=False)
    client_secret = db.Column(db.String(255), nullable=False)  # Encrypted
    access_token = db.Column(db.Text, nullable=True)  # Current access token
    refresh_token = db.Column(db.Text, nullable=True)  # If provided
    token_expires_at = db.Column(db.DateTime, nullable=True)
    account_name = db.Column(db.String(255), nullable=True)  # Store account name
    account_email = db.Column(db.String(255), nullable=True)  # Store account email
    account_user_id = db.Column(db.String(255), nullable=True)  # Store user ID from JobAdder
    account_company_id = db.Column(db.String(255), nullable=True)  # Store company ID from JobAdder
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('jobadder_integration', uselist=False))

class LinkedInRecruiterIntegration(db.Model):
    __tablename__ = "linkedin_recruiter_integrations"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    client_id = db.Column(db.String(255), nullable=False)
    client_secret = db.Column(db.String(255), nullable=False)  # Encrypted
    company_id = db.Column(db.String(255), nullable=False)  # LinkedIn Company ID
    contract_id = db.Column(db.String(255), nullable=False)  # LinkedIn Contract ID
    access_token = db.Column(db.Text, nullable=True)  # Current access token
    refresh_token = db.Column(db.Text, nullable=True)  # If provided
    token_expires_at = db.Column(db.DateTime, nullable=True)
    account_name = db.Column(db.String(255), nullable=True)  # Store account name
    account_email = db.Column(db.String(255), nullable=True)  # Store account email
    account_user_id = db.Column(db.String(255), nullable=True)  # Store user ID from LinkedIn
    account_organization_id = db.Column(db.String(255), nullable=True)  # Store organization ID from LinkedIn
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('linkedin_recruiter_integration', uselist=False))

class IntegrationSubmission(db.Model):
    __tablename__ = "integration_submissions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    user_email = db.Column(db.String(255), nullable=False)
    integration_type = db.Column(db.String(255), nullable=False)  # e.g., 'jobadder', 'workday', etc.
    integration_name = db.Column(db.String(255), nullable=False)  # e.g., 'JobAdder', 'Workday Recruiting'
    status = db.Column(db.String(50), default='in_progress', nullable=False)  # 'in_progress', 'completed', 'failed', 'pending'
    data = db.Column(db.Text, nullable=True)  # JSON string of integration data/credentials
    callback_url = db.Column(db.String(500), nullable=True)
    source = db.Column(db.String(100), nullable=True)  # 'integration_overview', 'jobadder_integration_overview', etc.
    saved_to_server = db.Column(db.Boolean, default=True, nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('integration_submissions', lazy=True))
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'userId': self.user_email or str(self.user_id),
            'userEmail': self.user_email,
            'integrationType': self.integration_type,
            'integrationName': self.integration_name,
            'status': self.status,
            'submittedAt': self.submitted_at.isoformat() if self.submitted_at else None,
            'updatedAt': self.updated_at.isoformat() if self.updated_at else None,
            'data': json.loads(self.data) if isinstance(self.data, str) else (self.data if isinstance(self.data, dict) else {}),
            'savedToServer': self.saved_to_server,
            'callbackUrl': self.callback_url,
            'source': self.source
        } 

class UserSocialLinks(db.Model):
    __tablename__ = "user_social_links"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    linkedin = db.Column(db.String(255))
    facebook = db.Column(db.String(255))
    x = db.Column(db.String(255))
    github = db.Column(db.String(255))



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

# Onboarding models
class OnboardingFlag(db.Model):
    __tablename__ = 'onboarding_flags'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    required = db.Column(db.Boolean, default=False, nullable=False)
    completed = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('onboarding_flag', uselist=False))

class OnboardingSubmission(db.Model):
    __tablename__ = 'onboarding_submissions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    data = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref='onboarding_submissions')

class UserFunctionalityPreferences(db.Model):
    """Model for storing user functionality preferences"""
    __tablename__ = 'user_functionality_preferences'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    functionalities = db.Column(db.JSON, nullable=False, default=list)  # Array of functionality IDs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('functionality_preferences', uselist=False))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'functionalities': self.functionalities if isinstance(self.functionalities, list) else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
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
        # Check if job is expired
        is_expired = self.expires_at and self.expires_at < datetime.utcnow()
        # Check if job is actually active (status is active AND not expired AND is public)
        is_actually_active = self.is_active()
        
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
            'public_url': f"/jobs/{self.id}",
            'is_expired': is_expired,
            'is_actually_active': is_actually_active
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

# Admin Activity Logging Models
class AdminActivityLog(db.Model):
    __tablename__ = 'admin_activity_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Admin user information
    admin_email = db.Column(db.String(255), nullable=False, index=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    admin_role = db.Column(db.String(50), nullable=False)  # admin, owner
    
    # Activity details
    activity_type = db.Column(db.String(100), nullable=False, index=True)  # login, logout, action
    action = db.Column(db.String(200), nullable=True)  # specific action performed
    endpoint = db.Column(db.String(500), nullable=True)  # API endpoint accessed
    method = db.Column(db.String(10), nullable=True)  # GET, POST, PUT, DELETE
    
    # Request details
    ip_address = db.Column(db.String(45), nullable=True)  # IPv4 or IPv6
    user_agent = db.Column(db.Text, nullable=True)
    request_data = db.Column(db.Text, nullable=True)  # JSON string of request data
    
    # Response details
    status_code = db.Column(db.Integer, nullable=True)
    response_time_ms = db.Column(db.Integer, nullable=True)  # response time in milliseconds
    
    # Additional context
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=True)
    session_id = db.Column(db.String(255), nullable=True, index=True)
    error_message = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'admin_email': self.admin_email,
            'admin_id': self.admin_id,
            'admin_role': self.admin_role,
            'activity_type': self.activity_type,
            'action': self.action,
            'endpoint': self.endpoint,
            'method': self.method,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_data': self.request_data,
            'status_code': self.status_code,
            'response_time_ms': self.response_time_ms,
            'tenant_id': self.tenant_id,
            'session_id': self.session_id,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<AdminActivityLog {self.id}: {self.admin_email} - {self.activity_type} at {self.created_at}>'

class AdminSession(db.Model):
    __tablename__ = 'admin_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(255), unique=True, nullable=False, index=True)
    admin_email = db.Column(db.String(255), nullable=False, index=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    admin_role = db.Column(db.String(50), nullable=False)
    
    # Session details
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    login_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    logout_time = db.Column(db.DateTime, nullable=True)
    
    # Session status
    is_active = db.Column(db.Boolean, default=True, nullable=False, index=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'admin_email': self.admin_email,
            'admin_id': self.admin_id,
            'admin_role': self.admin_role,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'login_time': self.login_time.isoformat() if self.login_time else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'logout_time': self.logout_time.isoformat() if self.logout_time else None,
            'is_active': self.is_active,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<AdminSession {self.id}: {self.admin_email} - {self.session_id}>'

class CandidateSearchHistory(db.Model):
    __tablename__ = "candidate_search_history"
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.String(128), nullable=False)  # Cognito user ID
    user_email = db.Column(db.String(255))
    job_description = db.Column(db.Text, nullable=False)
    search_criteria = db.Column(db.Text)  # JSON string of search filters
    candidates_found = db.Column(db.Integer, default=0)
    search_status = db.Column(db.String(20), default='completed')  # 'completed', 'failed', 'in_progress'
    search_duration_ms = db.Column(db.Integer)  # Time taken for search
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=10))
    
    # Relationships
    tenant = db.relationship('Tenant', backref='candidate_searches')
    
    def to_dict(self):
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'user_email': self.user_email,
            'job_description': self.job_description,
            'search_criteria': json.loads(self.search_criteria) if self.search_criteria else None,
            'candidates_found': self.candidates_found,
            'search_status': self.search_status,
            'search_duration_ms': self.search_duration_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_expired': self.expires_at < datetime.utcnow() if self.expires_at else False
        }
    
    def extend_expiry(self, days=10):
        """Extend the search expiry by specified days"""
        if self.expires_at:
            self.expires_at = self.expires_at + timedelta(days=days)
        else:
            self.expires_at = datetime.utcnow() + timedelta(days=days)
        self.updated_at = datetime.utcnow()
        return self

class CandidateSearchResult(db.Model):
    __tablename__ = "candidate_search_results"
    
    id = db.Column(db.Integer, primary_key=True)
    search_history_id = db.Column(db.Integer, db.ForeignKey("candidate_search_history.id"), nullable=False)
    candidate_id = db.Column(db.String(128), nullable=False)  # External candidate ID
    candidate_name = db.Column(db.String(255), nullable=False)
    candidate_email = db.Column(db.String(255))
    candidate_phone = db.Column(db.String(50))
    candidate_location = db.Column(db.String(255))
    match_score = db.Column(db.Float, nullable=False)
    match_reasons = db.Column(db.Text)  # JSON string of match reasons/explanation
    candidate_data = db.Column(db.Text)  # JSON string of full candidate data
    is_saved = db.Column(db.Boolean, default=False)
    is_contacted = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    search_history = db.relationship('CandidateSearchHistory', backref='candidate_results')
    
    def to_dict(self):
        return {
            'id': self.id,
            'search_history_id': self.search_history_id,
            'candidate_id': self.candidate_id,
            'candidate_name': self.candidate_name,
            'candidate_email': self.candidate_email,
            'candidate_phone': self.candidate_phone,
            'candidate_location': self.candidate_location,
            'match_score': self.match_score,
            'match_reasons': json.loads(self.match_reasons) if self.match_reasons else None,
            'candidate_data': json.loads(self.candidate_data) if self.candidate_data else None,
            'is_saved': self.is_saved,
            'is_contacted': self.is_contacted,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class CandidateMatchLog(db.Model):
    """
    Long-term storage for candidate match logs
    Stores detailed information about why candidates were matched/retrieved
    This is kept for a long time (2+ years) for analytics and auditing purposes
    """
    __tablename__ = "candidate_match_logs"
    
    id = db.Column(db.Integer, primary_key=True)
    search_history_id = db.Column(db.Integer, db.ForeignKey("candidate_search_history.id"), nullable=True)
    candidate_result_id = db.Column(db.Integer, db.ForeignKey("candidate_search_results.id"), nullable=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.String(128), nullable=False)  # Cognito user ID
    user_email = db.Column(db.String(255))  # Email of the user who performed the search
    
    # Candidate information
    candidate_id = db.Column(db.String(128), nullable=False)
    candidate_name = db.Column(db.String(255), nullable=False)
    candidate_email = db.Column(db.String(255))
    
    # Search information
    job_description = db.Column(db.Text, nullable=False)
    search_query = db.Column(db.Text)  # The actual search query used
    search_criteria = db.Column(db.Text)  # JSON string of search filters
    
    # Match details
    match_score = db.Column(db.Float, nullable=False)
    match_reasons = db.Column(db.Text, nullable=False)  # JSON string of detailed match reasons
    match_explanation = db.Column(db.Text)  # Human-readable explanation
    match_details = db.Column(db.Text)  # JSON string of detailed scoring breakdown
    
    # Additional metadata
    algorithm_version = db.Column(db.String(50))  # Which algorithm was used
    search_duration_ms = db.Column(db.Integer)  # How long the search took
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    search_history = db.relationship('CandidateSearchHistory', backref='match_logs')
    candidate_result = db.relationship('CandidateSearchResult', backref='match_logs')
    tenant = db.relationship('Tenant', backref='candidate_match_logs')
    
    def to_dict(self):
        return {
            'id': self.id,
            'search_history_id': self.search_history_id,
            'candidate_result_id': self.candidate_result_id,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'user_email': self.user_email or (self.search_history.user_email if self.search_history else None),
            'candidate_id': self.candidate_id,
            'candidate_name': self.candidate_name,
            'candidate_email': self.candidate_email,
            'job_description': self.job_description,
            'search_query': self.search_query,
            'search_criteria': json.loads(self.search_criteria) if self.search_criteria else None,
            'match_score': self.match_score,
            'match_reasons': json.loads(self.match_reasons) if self.match_reasons else None,
            'match_explanation': self.match_explanation,
            'match_details': json.loads(self.match_details) if self.match_details else None,
            'algorithm_version': self.algorithm_version,
            'search_duration_ms': self.search_duration_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<CandidateMatchLog {self.id}: {self.candidate_name} - Score: {self.match_score}>'

# Communication Models for Twilio Integration
class MessageTemplate(db.Model):
    __tablename__ = "message_templates"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    channel = db.Column(db.Enum("email", "sms", "whatsapp"), nullable=False)
    subject = db.Column(db.String(500), nullable=True)  # For email only
    body = db.Column(db.Text, nullable=False)
    variables = db.Column(db.Text)  # JSON string of available variables
    is_default = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='message_templates')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'channel': self.channel,
            'subject': self.subject,
            'body': self.body,
            'variables': json.loads(self.variables) if self.variables else [],
            'is_default': self.is_default,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class CandidateCommunication(db.Model):
    __tablename__ = "candidate_communications"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    candidate_id = db.Column(db.String(255), nullable=False)
    candidate_name = db.Column(db.String(255), nullable=True)
    candidate_email = db.Column(db.String(255), nullable=True)
    candidate_phone = db.Column(db.String(50), nullable=True)
    channel = db.Column(db.Enum("email", "sms", "whatsapp"), nullable=False)
    template_id = db.Column(db.Integer, db.ForeignKey("message_templates.id"), nullable=True)
    message_subject = db.Column(db.String(500), nullable=True)  # For email
    message_body = db.Column(db.Text, nullable=False)
    twilio_message_sid = db.Column(db.String(100), nullable=True)  # For SMS/WhatsApp tracking
    sendgrid_message_id = db.Column(db.String(100), nullable=True)  # For email tracking
    status = db.Column(db.Enum("pending", "sent", "delivered", "failed", "read", "replied"), default="pending", nullable=False)
    delivery_status = db.Column(db.String(100), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    sent_at = db.Column(db.DateTime, nullable=True)
    delivered_at = db.Column(db.DateTime, nullable=True)
    read_at = db.Column(db.DateTime, nullable=True)
    replied_at = db.Column(db.DateTime, nullable=True)
    reply_content = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='candidate_communications')
    template = db.relationship('MessageTemplate', backref='communications')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'candidate_id': self.candidate_id,
            'candidate_name': self.candidate_name,
            'candidate_email': self.candidate_email,
            'candidate_phone': self.candidate_phone,
            'channel': self.channel,
            'template_id': self.template_id,
            'message_subject': self.message_subject,
            'message_body': self.message_body,
            'twilio_message_sid': self.twilio_message_sid,
            'sendgrid_message_id': self.sendgrid_message_id,
            'status': self.status,
            'delivery_status': self.delivery_status,
            'error_message': self.error_message,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'delivered_at': self.delivered_at.isoformat() if self.delivered_at else None,
            'read_at': self.read_at.isoformat() if self.read_at else None,
            'replied_at': self.replied_at.isoformat() if self.replied_at else None,
            'reply_content': self.reply_content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class CommunicationReply(db.Model):
    __tablename__ = "communication_replies"
    id = db.Column(db.Integer, primary_key=True)
    communication_id = db.Column(db.Integer, db.ForeignKey("candidate_communications.id"), nullable=False)
    candidate_phone = db.Column(db.String(50), nullable=True)  # For SMS/WhatsApp
    candidate_email = db.Column(db.String(255), nullable=True)  # For email
    reply_content = db.Column(db.Text, nullable=False)
    channel = db.Column(db.Enum("email", "sms", "whatsapp"), nullable=False)
    twilio_message_sid = db.Column(db.String(100), nullable=True)
    received_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    communication = db.relationship('CandidateCommunication', backref='replies')
    
    def to_dict(self):
        return {
            'id': self.id,
            'communication_id': self.communication_id,
            'candidate_phone': self.candidate_phone,
            'candidate_email': self.candidate_email,
            'reply_content': self.reply_content,
            'channel': self.channel,
            'twilio_message_sid': self.twilio_message_sid,
            'received_at': self.received_at.isoformat() if self.received_at else None
        }


class EmployeeProfile(db.Model):
    """Employee-specific profile data - separate from User model to avoid affecting existing flow"""
    __tablename__ = "employee_profiles"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    first_name = db.Column(db.String(255), nullable=True)
    last_name = db.Column(db.String(255), nullable=True)
    phone = db.Column(db.String(50), nullable=True)
    department = db.Column(db.String(255), nullable=True)
    location = db.Column(db.String(255), nullable=True)
    employment_type = db.Column(db.String(50), nullable=True)  # full_time, part_time, contractor
    category = db.Column(db.String(50), nullable=True)  # it, non_it, healthcare
    salary_amount = db.Column(db.Numeric(10, 2), nullable=True)
    salary_currency = db.Column(db.String(10), nullable=True, default='USD')
    salary_type = db.Column(db.String(20), nullable=True, default='monthly')  # 'monthly' or 'hourly'
    hire_date = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'department': self.department,
            'location': self.location,
            'employment_type': self.employment_type,
            'category': self.category,
            'salary_amount': float(self.salary_amount) if self.salary_amount else None,
            'salary_currency': self.salary_currency,
            'salary_type': self.salary_type or 'monthly',
            'hire_date': self.hire_date.isoformat() if self.hire_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class OrganizationMetadata(db.Model):
    """Organization metadata - separate from Tenant model to avoid affecting existing flow"""
    __tablename__ = "organization_metadata"
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False, unique=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)  # Track who created it
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'name': self.name,
            'description': self.description,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Timesheet(db.Model):
    """Employee timesheet entries with overtime, bonus, and holiday calculations"""
    __tablename__ = "timesheets"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    employee_profile_id = db.Column(db.Integer, db.ForeignKey("employee_profiles.id"), nullable=True)
    
    # Date and period
    date = db.Column(db.Date, nullable=False)
    week_start_date = db.Column(db.Date, nullable=True)  # Start of the week for grouping
    week_end_date = db.Column(db.Date, nullable=True)  # End of the week for grouping
    pay_period_start = db.Column(db.Date, nullable=True)
    pay_period_end = db.Column(db.Date, nullable=True)
    
    # Hours
    regular_hours = db.Column(db.Numeric(5, 2), default=0, nullable=False)  # Regular working hours
    overtime_hours = db.Column(db.Numeric(5, 2), default=0, nullable=False)  # Overtime hours (>8 hours/day or >40/week)
    holiday_hours = db.Column(db.Numeric(5, 2), default=0, nullable=False)  # Hours worked on holidays
    total_hours = db.Column(db.Numeric(5, 2), nullable=False)  # Total hours (regular + overtime + holiday)
    
    # Rates (stored for historical accuracy)
    regular_rate = db.Column(db.Numeric(10, 2), nullable=True)  # Regular hourly rate
    overtime_rate = db.Column(db.Numeric(10, 2), nullable=True)  # Overtime hourly rate (usually 1.5x)
    holiday_rate = db.Column(db.Numeric(10, 2), nullable=True)  # Holiday hourly rate (usually 2x)
    
    # Earnings
    regular_earnings = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    overtime_earnings = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    holiday_earnings = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    bonus_amount = db.Column(db.Numeric(10, 2), default=0, nullable=False)  # Bonus for this period
    total_earnings = db.Column(db.Numeric(10, 2), nullable=False)  # Total earnings for this entry
    
    # Status and notes
    status = db.Column(db.Enum("draft", "submitted", "approved", "rejected", "paid"), default="draft", nullable=False)
    notes = db.Column(db.Text, nullable=True)
    approved_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    approved_at = db.Column(db.DateTime, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    submitted_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    employee = db.relationship('User', foreign_keys=[user_id])  # No backref - User already has timesheets relationship
    approver = db.relationship('User', foreign_keys=[approved_by])
    employee_profile = db.relationship('EmployeeProfile', backref='timesheets')
    
    def calculate_earnings(self):
        """Calculate earnings based on hours and rates"""
        # Regular earnings
        if self.regular_rate and self.regular_hours:
            self.regular_earnings = float(self.regular_rate) * float(self.regular_hours)
        else:
            self.regular_earnings = 0
        
        # Overtime earnings (typically 1.5x regular rate)
        if self.overtime_rate and self.overtime_hours:
            self.overtime_earnings = float(self.overtime_rate) * float(self.overtime_hours)
        elif self.regular_rate and self.overtime_hours:
            # Default to 1.5x if overtime rate not set
            self.overtime_earnings = float(self.regular_rate) * 1.5 * float(self.overtime_hours)
        else:
            self.overtime_earnings = 0
        
        # Holiday earnings (typically 2x regular rate)
        if self.holiday_rate and self.holiday_hours:
            self.holiday_earnings = float(self.holiday_rate) * float(self.holiday_hours)
        elif self.regular_rate and self.holiday_hours:
            # Default to 2x if holiday rate not set
            self.holiday_earnings = float(self.regular_rate) * 2.0 * float(self.holiday_hours)
        else:
            self.holiday_earnings = 0
        
        # Total earnings
        self.total_earnings = self.regular_earnings + self.overtime_earnings + self.holiday_earnings + float(self.bonus_amount or 0)
        
        return self.total_earnings
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'employee_profile_id': self.employee_profile_id,
            'date': self.date.isoformat() if self.date else None,
            'week_start_date': self.week_start_date.isoformat() if self.week_start_date else None,
            'week_end_date': self.week_end_date.isoformat() if self.week_end_date else None,
            'pay_period_start': self.pay_period_start.isoformat() if self.pay_period_start else None,
            'pay_period_end': self.pay_period_end.isoformat() if self.pay_period_end else None,
            'regular_hours': float(self.regular_hours) if self.regular_hours else 0,
            'overtime_hours': float(self.overtime_hours) if self.overtime_hours else 0,
            'holiday_hours': float(self.holiday_hours) if self.holiday_hours else 0,
            'total_hours': float(self.total_hours) if self.total_hours else 0,
            'regular_rate': float(self.regular_rate) if self.regular_rate else None,
            'overtime_rate': float(self.overtime_rate) if self.overtime_rate else None,
            'holiday_rate': float(self.holiday_rate) if self.holiday_rate else None,
            'regular_earnings': float(self.regular_earnings) if self.regular_earnings else 0,
            'overtime_earnings': float(self.overtime_earnings) if self.overtime_earnings else 0,
            'holiday_earnings': float(self.holiday_earnings) if self.holiday_earnings else 0,
            'bonus_amount': float(self.bonus_amount) if self.bonus_amount else 0,
            'total_earnings': float(self.total_earnings) if self.total_earnings else 0,
            'status': self.status,
            'notes': self.notes,
            'approved_by': self.approved_by,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'period_type': 'monthly' if (
                self.pay_period_start
                and self.pay_period_end
                and self.pay_period_end != self.pay_period_start
            ) else 'daily'
        }


class Payslip(db.Model):
    """Stored payslips for employees"""
    __tablename__ = "payslips"
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    employee_profile_id = db.Column(db.Integer, db.ForeignKey("employee_profiles.id"), nullable=True)
    
    # Pay period
    pay_period_start = db.Column(db.Date, nullable=False)
    pay_period_end = db.Column(db.Date, nullable=False)
    pay_date = db.Column(db.Date, nullable=False)
    
    # Hours
    total_regular_hours = db.Column(db.Numeric(5, 2), default=0, nullable=False)
    total_overtime_hours = db.Column(db.Numeric(5, 2), default=0, nullable=False)
    total_holiday_hours = db.Column(db.Numeric(5, 2), default=0, nullable=False)
    total_hours = db.Column(db.Numeric(5, 2), nullable=False)
    
    # Earnings
    regular_earnings = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    overtime_earnings = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    holiday_earnings = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    bonus_amount = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    gross_earnings = db.Column(db.Numeric(10, 2), nullable=False)
    
    # Deductions
    tax_deduction = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    other_deductions = db.Column(db.Numeric(10, 2), default=0, nullable=False)
    total_deductions = db.Column(db.Numeric(10, 2), nullable=False)
    net_pay = db.Column(db.Numeric(10, 2), nullable=False)
    
    # Currency and status
    currency = db.Column(db.String(10), default='USD', nullable=False)
    status = db.Column(db.Enum("draft", "generated", "sent", "paid"), default="generated", nullable=False)
    
    # Email tracking
    email_sent = db.Column(db.Boolean, default=False, nullable=False)
    email_sent_at = db.Column(db.DateTime, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    employee = db.relationship('User', foreign_keys=[employee_id])
    employee_profile = db.relationship('EmployeeProfile', backref='payslips')
    
    def to_dict(self):
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'employee_profile_id': self.employee_profile_id,
            'pay_period_start': self.pay_period_start.isoformat() if self.pay_period_start else None,
            'pay_period_end': self.pay_period_end.isoformat() if self.pay_period_end else None,
            'pay_date': self.pay_date.isoformat() if self.pay_date else None,
            'total_regular_hours': float(self.total_regular_hours) if self.total_regular_hours else 0,
            'total_overtime_hours': float(self.total_overtime_hours) if self.total_overtime_hours else 0,
            'total_holiday_hours': float(self.total_holiday_hours) if self.total_holiday_hours else 0,
            'total_hours': float(self.total_hours) if self.total_hours else 0,
            'regular_earnings': float(self.regular_earnings) if self.regular_earnings else 0,
            'overtime_earnings': float(self.overtime_earnings) if self.overtime_earnings else 0,
            'holiday_earnings': float(self.holiday_earnings) if self.holiday_earnings else 0,
            'bonus_amount': float(self.bonus_amount) if self.bonus_amount else 0,
            'gross_earnings': float(self.gross_earnings) if self.gross_earnings else 0,
            'tax_deduction': float(self.tax_deduction) if self.tax_deduction else 0,
            'other_deductions': float(self.other_deductions) if self.other_deductions else 0,
            'total_deductions': float(self.total_deductions) if self.total_deductions else 0,
            'net_pay': float(self.net_pay) if self.net_pay else 0,
            'currency': self.currency,
            'status': self.status,
            'email_sent': self.email_sent,
            'email_sent_at': self.email_sent_at.isoformat() if self.email_sent_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class UserModuleAccess(db.Model):
    """Model for tracking which modules users have access to"""
    __tablename__ = "user_module_access"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    module_name = db.Column(db.String(100), nullable=False)  # e.g., 'payroll', 'talent_matchmaker', 'jobseeker'
    granted_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)  # Admin who granted access
    granted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id], backref='module_access')
    granted_by_user = db.relationship('User', foreign_keys=[granted_by])
    
    # Unique constraint: one user can have one access record per module
    __table_args__ = (db.UniqueConstraint('user_id', 'module_name', name='unique_user_module'),)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'module_name': self.module_name,
            'granted_by': self.granted_by,
            'granted_at': self.granted_at.isoformat() if self.granted_at else None,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# ============================================================================
# Jobvite Integration Models
# ============================================================================

class JobviteSettings(db.Model):
    """Jobvite integration settings per tenant"""
    __tablename__ = "jobvite_settings"
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    
    # Environment - stored as lowercase enum
    environment = db.Column(db.Enum("stage", "prod", name="jobvite_env"), nullable=False)
    
    # API v2 Credentials (encrypted at rest)
    api_key = db.Column(db.String(255), nullable=False)
    api_secret_encrypted = db.Column(db.Text, nullable=False)  # AES-256 encrypted
    company_id = db.Column(db.String(255), nullable=False, unique=True)  # Used for webhook tenant resolution
    # NOTE: company_id is globally unique (one Jobvite company = one Kempian tenant)
    # If you need to support the same Jobvite company across multiple tenants,
    # remove unique=True and add a composite unique constraint on (company_id, tenant_id)
    
    # Webhook Signing Key (encrypted at rest)
    webhook_signing_key_encrypted = db.Column(db.Text, nullable=True)
    
    # RSA Keys (for Onboarding API)
    our_public_rsa_key = db.Column(db.Text, nullable=True)  # PEM format
    our_private_rsa_key_encrypted = db.Column(db.Text, nullable=True)  # Encrypted PEM
    jobvite_public_rsa_key = db.Column(db.Text, nullable=True)  # PEM format
    
    # Service Account (for Onboarding API - if required)
    service_account_username = db.Column(db.String(255), nullable=True)
    service_account_password_encrypted = db.Column(db.Text, nullable=True)  # AES-256 encrypted
    
    # Sync Configuration (JSONB)
    sync_config = db.Column(db.JSON, nullable=True, default=dict)
    
    # Status
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    last_sync_at = db.Column(db.DateTime, nullable=True)
    last_sync_status = db.Column(db.String(50), nullable=True)  # 'success', 'partial', 'failed'
    last_error = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenant = db.relationship('Tenant', backref='jobvite_settings')
    user = db.relationship('User', backref='jobvite_settings')
    
    # Constraints and indexes
    __table_args__ = (
        db.UniqueConstraint('tenant_id', 'environment', name='uq_jobvite_tenant_env'),
        db.Index('idx_jobvite_settings_company_id', 'company_id'),  # For webhook lookup
        db.Index('idx_jobvite_settings_tenant', 'tenant_id'),
    )
    
    def decrypt_secret(self) -> str:
        """Decrypt API secret using application encryption key"""
        from app.jobvite.crypto import decrypt_at_rest
        return decrypt_at_rest(self.api_secret_encrypted)
    
    def decrypt_webhook_key(self) -> str:
        """Decrypt webhook signing key"""
        from app.jobvite.crypto import decrypt_at_rest
        if self.webhook_signing_key_encrypted:
            return decrypt_at_rest(self.webhook_signing_key_encrypted)
        return None
    
    def decrypt_private_key(self) -> str:
        """Decrypt RSA private key"""
        from app.jobvite.crypto import decrypt_at_rest
        if self.our_private_rsa_key_encrypted:
            return decrypt_at_rest(self.our_private_rsa_key_encrypted)
        return None
    
    def decrypt_service_account_password(self) -> str:
        """Decrypt service account password"""
        from app.jobvite.crypto import decrypt_at_rest
        if self.service_account_password_encrypted:
            return decrypt_at_rest(self.service_account_password_encrypted)
        return None

class JobviteJob(db.Model):
    """Jobvite jobs synced from Jobvite API"""
    __tablename__ = "jobvite_jobs"
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    jobvite_job_id = db.Column(db.String(255), nullable=False)
    requisition_id = db.Column(db.String(255), nullable=True)
    
    # Core fields
    title = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(100), nullable=True)
    department = db.Column(db.String(255), nullable=True)
    category = db.Column(db.String(255), nullable=True)
    
    # Recruiters/HMs
    primary_recruiter_email = db.Column(db.String(255), nullable=True)
    primary_hiring_manager_email = db.Column(db.String(255), nullable=True)
    
    # Location
    location_main = db.Column(db.String(500), nullable=True)
    region = db.Column(db.String(255), nullable=True)
    subsidiary = db.Column(db.String(255), nullable=True)
    
    # Salary
    salary_currency = db.Column(db.String(10), nullable=True)
    salary_min = db.Column(db.Numeric(10, 2), nullable=True)
    salary_max = db.Column(db.Numeric(10, 2), nullable=True)
    salary_frequency = db.Column(db.String(50), nullable=True)
    
    # Remote
    remote_type = db.Column(db.String(50), nullable=True)
    
    # Raw data
    raw_json = db.Column(db.JSON, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenant = db.relationship('Tenant', backref='jobvite_jobs')
    candidates = db.relationship('JobviteCandidate', back_populates='job', lazy=True)
    
    # Single __table_args__ with both constraints and indexes
    __table_args__ = (
        db.UniqueConstraint('tenant_id', 'jobvite_job_id', name='uq_jobvite_job_tenant'),
        db.Index('idx_jobvite_job_status', 'status'),
        db.Index('idx_jobvite_job_updated', 'updated_at'),
        db.Index('idx_jobvite_job_tenant', 'tenant_id'),
    )

class JobviteCandidate(db.Model):
    """Jobvite candidates synced from Jobvite API"""
    __tablename__ = "jobvite_candidates"
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    jobvite_candidate_id = db.Column(db.String(255), nullable=False)
    jobvite_application_id = db.Column(db.String(255), nullable=True)
    
    # Link to job
    jobvite_job_id = db.Column(db.String(255), nullable=True)
    job_id = db.Column(db.Integer, db.ForeignKey("jobvite_jobs.id"), nullable=True)
    
    # Core fields
    email = db.Column(db.String(255), nullable=True)
    first_name = db.Column(db.String(255), nullable=True)
    last_name = db.Column(db.String(255), nullable=True)
    workflow_state = db.Column(db.String(100), nullable=True)
    personal_data_processing_status = db.Column(db.String(100), nullable=True)
    
    # Raw data
    raw_json = db.Column(db.JSON, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenant = db.relationship('Tenant', backref='jobvite_candidates')
    job = db.relationship('JobviteJob', back_populates='candidates')
    documents = db.relationship('JobviteCandidateDocument', backref='candidate', cascade='all, delete-orphan')
    
    # Single __table_args__ with both constraints and indexes
    __table_args__ = (
        db.UniqueConstraint('tenant_id', 'jobvite_candidate_id', name='uq_jobvite_candidate_tenant'),
        db.Index('idx_jobvite_candidate_email', 'email'),
        db.Index('idx_jobvite_candidate_workflow', 'workflow_state'),
        db.Index('idx_jobvite_candidate_job', 'job_id'),
        db.Index('idx_jobvite_candidate_tenant', 'tenant_id'),
    )

class JobviteCandidateDocument(db.Model):
    """Documents associated with Jobvite candidates"""
    __tablename__ = "jobvite_candidate_documents"
    
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("jobvite_candidates.id"), nullable=False)
    
    doc_type = db.Column(db.Enum('resume', 'cover_letter', 'attachment', name='doc_type'), nullable=False)
    filename = db.Column(db.String(500), nullable=False)
    mime_type = db.Column(db.String(100), nullable=True)
    
    # Storage: either S3 path or local path
    storage_path = db.Column(db.String(1000), nullable=True)
    external_url = db.Column(db.String(1000), nullable=True)
    
    size_bytes = db.Column(db.Integer, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        db.Index('idx_jobvite_doc_candidate', 'candidate_id'),
        db.Index('idx_jobvite_doc_type', 'doc_type'),
    )

class JobviteOnboardingProcess(db.Model):
    """Onboarding processes from Jobvite Onboarding API"""
    __tablename__ = "jobvite_onboarding_processes"
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    jobvite_process_id = db.Column(db.String(255), nullable=False)
    jobvite_new_hire_id = db.Column(db.String(255), nullable=True)
    jobvite_candidate_id = db.Column(db.String(255), nullable=True)  # Link to candidate
    
    # Status
    status = db.Column(db.String(100), nullable=True)
    
    # Dates
    hire_date = db.Column(db.Date, nullable=True)
    kickoff_date = db.Column(db.Date, nullable=True)
    
    # Milestones and tasks summary (JSONB)
    milestone_status_json = db.Column(db.JSON, nullable=True)
    tasks_summary_json = db.Column(db.JSON, nullable=True)
    
    # Raw data
    raw_json = db.Column(db.JSON, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenant = db.relationship('Tenant', backref='jobvite_onboarding_processes')
    tasks = db.relationship('JobviteOnboardingTask', backref='process', cascade='all, delete-orphan')
    
    # Unique constraint
    __table_args__ = (
        db.UniqueConstraint('tenant_id', 'jobvite_process_id', name='uq_jobvite_process_tenant'),
        db.Index('idx_jobvite_process_tenant', 'tenant_id'),
        db.Index('idx_jobvite_process_status', 'status'),
    )

class JobviteOnboardingTask(db.Model):
    """Onboarding tasks from Jobvite Onboarding API"""
    __tablename__ = "jobvite_onboarding_tasks"
    
    id = db.Column(db.Integer, primary_key=True)
    process_id = db.Column(db.Integer, db.ForeignKey("jobvite_onboarding_processes.id"), nullable=False)
    jobvite_task_id = db.Column(db.String(255), nullable=False)
    
    name = db.Column(db.String(500), nullable=True)
    type = db.Column(db.Enum('W4', 'I9', 'DOC', 'FORM', 'CUSTOM', name='task_type'), nullable=True)
    status = db.Column(db.String(100), nullable=True)
    due_date = db.Column(db.Date, nullable=True)
    
    # File reference (if task has attached file)
    file_storage_path = db.Column(db.String(1000), nullable=True)
    
    # Raw data
    raw_json = db.Column(db.JSON, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint
    __table_args__ = (
        db.UniqueConstraint('process_id', 'jobvite_task_id', name='uq_jobvite_task_process'),
        db.Index('idx_jobvite_task_process', 'process_id'),
        db.Index('idx_jobvite_task_status', 'status'),
    )

class JobviteWebhookLog(db.Model):
    """Webhook events received from Jobvite"""
    __tablename__ = "jobvite_webhook_logs"
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey("tenants.id"), nullable=False)
    
    event_type = db.Column(db.String(100), nullable=False)
    source = db.Column(db.Enum('candidate', 'job', 'onboarding', name='webhook_source'), nullable=False)
    jobvite_entity_id = db.Column(db.String(255), nullable=True)
    
    payload = db.Column(db.JSON, nullable=False)
    signature = db.Column(db.String(500), nullable=True)
    signature_valid = db.Column(db.Boolean, default=False, nullable=False)
    
    processed = db.Column(db.Boolean, default=False, nullable=False)
    error_message = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    tenant = db.relationship('Tenant', backref='jobvite_webhook_logs')
    
    # Indexes
    __table_args__ = (
        db.Index('idx_jobvite_webhook_event', 'event_type'),
        db.Index('idx_jobvite_webhook_source', 'source'),
        db.Index('idx_jobvite_webhook_entity', 'jobvite_entity_id'),
        db.Index('idx_jobvite_webhook_processed', 'processed'),
        db.Index('idx_jobvite_webhook_created', 'created_at'),
        db.Index('idx_jobvite_webhook_tenant', 'tenant_id'),
    )
