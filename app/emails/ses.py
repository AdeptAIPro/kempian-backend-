import boto3
import os
from flask import render_template_string

SES_REGION = os.getenv('SES_REGION')
SES_FROM_EMAIL = os.getenv('SES_FROM_EMAIL')
ses_client = boto3.client('ses', region_name=SES_REGION)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

def load_template(filename):
    with open(os.path.join(TEMPLATES_DIR, filename), 'r') as f:
        return f.read()

def send_invite_email(to_email, invite_link):
    subject = "You're invited to Talent-Match!"
    template = load_template('invite.html')
    body_html = render_template_string(template, invite_link=invite_link)
    ses_client.send_email(
        Source=SES_FROM_EMAIL,
        Destination={'ToAddresses': [to_email]},
        Message={
            'Subject': {'Data': subject},
            'Body': {'Html': {'Data': body_html}}
        }
    )

def send_quota_alert_email(to_email, percent):
    subject = f"Talent-Match: {percent}% Quota Used"
    template = load_template('quota_alert.html')
    body_html = render_template_string(template, percent=percent)
    ses_client.send_email(
        Source=SES_FROM_EMAIL,
        Destination={'ToAddresses': [to_email]},
        Message={
            'Subject': {'Data': subject},
            'Body': {'Html': {'Data': body_html}}
        }
    ) 