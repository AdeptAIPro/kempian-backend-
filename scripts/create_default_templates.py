"""
Script to create default message templates for communications
Run this after database migration to populate default templates
"""
from app import create_app, db
from app.models import MessageTemplate, User
from datetime import datetime
import json

def create_default_templates():
    """Create default templates for all users or a specific user"""
    app = create_app()
    
    with app.app_context():
        # Get all users (or filter by specific user)
        users = User.query.filter(User.role.in_(['employer', 'recruiter', 'admin', 'owner'])).all()
        
        if not users:
            print("No users found. Creating templates for admin users only.")
            return
        
        templates_created = 0
        
        for user in users:
            # Check if user already has templates
            existing_templates = MessageTemplate.query.filter_by(user_id=user.id).count()
            if existing_templates > 0:
                print(f"User {user.email} already has templates. Skipping...")
                continue
            
            print(f"Creating default templates for user: {user.email}")
            
            # Template 1: Email - Initial Contact
            template1 = MessageTemplate(
                user_id=user.id,
                name="Initial Contact - Email",
                channel="email",
                subject="Exciting Opportunity - {{job_title}}",
                body="""Hello {{candidate_name}},

We found your profile matching our {{job_title}} position at {{company_name}}. We'd love to discuss this opportunity with you.

Your qualifications align well with what we're looking for, and we believe you could be a great fit for our team.

Best regards,
{{sender_name}}
{{company_name}}""",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name', 'sender_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template1)
            
            # Template 2: Email - Interview Invitation
            template2 = MessageTemplate(
                user_id=user.id,
                name="Interview Invitation - Email",
                channel="email",
                subject="Interview Invitation - {{job_title}}",
                body="""Hi {{candidate_name}},

Congratulations! We'd like to invite you for an interview for the {{job_title}} position at {{company_name}}.

We're impressed by your qualifications and would love to learn more about you.

Please reply to confirm your availability, and we'll coordinate the next steps.

Best regards,
{{sender_name}}
{{company_name}}""",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name', 'sender_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template2)
            
            # Template 3: Email - Follow-up
            template3 = MessageTemplate(
                user_id=user.id,
                name="Follow-up - Email",
                channel="email",
                subject="Following Up - {{job_title}}",
                body="""Hello {{candidate_name}},

Just following up on our previous message about the {{job_title}} opportunity at {{company_name}}.

Please let us know if you're interested, and we can schedule a time to discuss this further.

Best regards,
{{sender_name}}
{{company_name}}""",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name', 'sender_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template3)
            
            # Template 4: SMS - Initial Contact
            template4 = MessageTemplate(
                user_id=user.id,
                name="Initial Contact - SMS",
                channel="sms",
                subject=None,
                body="Hello {{candidate_name}}, we found your profile matching our {{job_title}} position. We'd love to discuss this opportunity. Reply YES to learn more. - {{company_name}}",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template4)
            
            # Template 5: SMS - Interview Invitation
            template5 = MessageTemplate(
                user_id=user.id,
                name="Interview Invitation - SMS",
                channel="sms",
                subject=None,
                body="Hi {{candidate_name}}, congratulations! We'd like to invite you for an interview for the {{job_title}} position. Please reply to confirm. - {{company_name}}",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template5)
            
            # Template 6: SMS - Follow-up
            template6 = MessageTemplate(
                user_id=user.id,
                name="Follow-up - SMS",
                channel="sms",
                subject=None,
                body="Hi {{candidate_name}}, just following up on the {{job_title}} opportunity at {{company_name}}. Are you still interested? Reply YES or NO. - {{sender_name}}",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name', 'sender_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template6)
            
            # Template 7: WhatsApp - Initial Contact
            template7 = MessageTemplate(
                user_id=user.id,
                name="Initial Contact - WhatsApp",
                channel="whatsapp",
                subject=None,
                body="""Hello {{candidate_name}},

We found your profile matching our {{job_title}} position at {{company_name}}. We'd love to discuss this opportunity with you.

Reply YES to learn more!

Best regards,
{{sender_name}}""",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name', 'sender_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template7)
            
            # Template 8: WhatsApp - Interview Invitation
            template8 = MessageTemplate(
                user_id=user.id,
                name="Interview Invitation - WhatsApp",
                channel="whatsapp",
                subject=None,
                body="""Hi {{candidate_name}},

Congratulations! We'd like to invite you for an interview for the {{job_title}} position at {{company_name}}.

Please reply to confirm your availability.

Best regards,
{{sender_name}}""",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name', 'sender_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template8)
            
            # Template 9: WhatsApp - Follow-up
            template9 = MessageTemplate(
                user_id=user.id,
                name="Follow-up - WhatsApp",
                channel="whatsapp",
                subject=None,
                body="""Hi {{candidate_name}},

Just following up on the {{job_title}} opportunity at {{company_name}}.

Are you still interested? Please let us know!

Best regards,
{{sender_name}}""",
                variables=json.dumps(['candidate_name', 'job_title', 'company_name', 'sender_name']),
                is_default=True,
                is_active=True
            )
            db.session.add(template9)
            
            templates_created += 9
        
        try:
            db.session.commit()
            print(f"\n[SUCCESS] Successfully created {templates_created} default templates!")
            print(f"   - 3 Email templates")
            print(f"   - 3 SMS templates")
            print(f"   - 3 WhatsApp templates")
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error creating templates: {str(e)}")
            raise

if __name__ == '__main__':
    import json
    create_default_templates()

