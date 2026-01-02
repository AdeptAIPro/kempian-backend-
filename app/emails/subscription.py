"""
Subscription-related email functions
"""
from app.simple_logger import get_logger
from .smtp import send_email_via_smtp
from .ses import get_ses_client, SES_FROM_EMAIL
from flask import render_template_string
import os

logger = get_logger('emails')

def send_subscription_purchase_receipt(to_email, user_name, plan_name, amount_display, transaction_id, 
                                     payment_method, purchase_date, billing_cycle, next_billing_date,
                                     jd_quota, max_subaccounts, dashboard_url, invoice_number=None,
                                     invoice_url=None, receipt_url=None):
    """Send purchase receipt email to user with invoice/bill details"""
    logger.info(f"[EMAIL] send_subscription_purchase_receipt called with:")
    logger.info(f"   - to_email: {to_email}")
    logger.info(f"   - user_name: {user_name}")
    logger.info(f"   - plan_name: {plan_name}")
    logger.info(f"   - amount: {amount_display}")
    logger.info(f"   - transaction_id: {transaction_id}")
    logger.info(f"   - invoice_number: {invoice_number}")
    logger.info(f"   - invoice_url: {invoice_url}")
    logger.info(f"   - receipt_url: {receipt_url}")
    
    # Try SMTP first (primary method)
    logger.info(f"[SMTP] Attempting to send purchase receipt via SMTP (primary method)...")
    try:
        smtp_result = send_subscription_purchase_receipt_smtp(
            to_email, user_name, plan_name, amount_display, transaction_id,
            payment_method, purchase_date, billing_cycle, next_billing_date,
            jd_quota, max_subaccounts, dashboard_url, invoice_number, invoice_url, receipt_url
        )
        if smtp_result:
            logger.info(f"[SUCCESS] Purchase receipt email sent successfully via SMTP to {to_email}")
            return True
        else:
            logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
    except Exception as smtp_error:
        logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
    # Fallback to AWS SES
    logger.info(f"[SES] Attempting to send purchase receipt via AWS SES (fallback method)...")
    try:
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for purchase receipt email")
            return False
            
        subject = f"Purchase Confirmed - {plan_name} Subscription"
        
        # Load template with error handling
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'subscription_purchase_receipt.html')
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
        except FileNotFoundError:
            logger.error(f"[SES_ERROR] Template file not found: {template_path}")
            logger.error(f"[SES_ERROR] Cannot send email without template. Please check template file exists.")
            return False
        except Exception as template_error:
            logger.error(f"[SES_ERROR] Failed to load template file {template_path}: {str(template_error)}")
            return False
        
        body_html = render_template_string(template,
            subject=subject,
            user_name=user_name,
            plan_name=plan_name,
            amount_display=amount_display,
            transaction_id=transaction_id,
            payment_method=payment_method,
            purchase_date=purchase_date,
            billing_cycle=billing_cycle,
            next_billing_date=next_billing_date,
            jd_quota=jd_quota,
            max_subaccounts=max_subaccounts,
            dashboard_url=dashboard_url,
            invoice_number=invoice_number or transaction_id,
            invoice_url=invoice_url,
            receipt_url=receipt_url
        )
        
        # Create text version
        invoice_display = invoice_number or transaction_id
        text_body = f"""
{subject}

Dear {user_name},

Thank you for subscribing to {plan_name}! Your payment has been processed successfully.

Receipt Details:
- Invoice/Bill Number: {invoice_display}
- Transaction ID: {transaction_id}
- Plan: {plan_name}
- Amount: {amount_display}
- Payment Method: {payment_method}
- Date: {purchase_date}
- Billing Cycle: {billing_cycle}
{f"- Next Billing Date: {next_billing_date}" if next_billing_date else ""}
{f"- Invoice URL: {invoice_url}" if invoice_url else ""}
{f"- Receipt URL: {receipt_url}" if receipt_url else ""}

What's Included:
- {jd_quota} Job Description searches per month
- {max_subaccounts} Sub-accounts
- Priority support
- Advanced analytics

Access your dashboard: {dashboard_url}

Best regards,
The Talent-Match Team
        """
        
        logger.info(f"[SES] Sending purchase receipt email via SES...")
        logger.info(f"   - From: {SES_FROM_EMAIL}")
        logger.info(f"   - To: {to_email}")
        logger.info(f"   - Subject: {subject}")
        
        client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {
                    'Html': {'Data': body_html},
                    'Text': {'Data': text_body}
                }
            }
        )
        
        logger.info(f"[SUCCESS] Purchase receipt email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ERROR] Both SMTP and AWS SES failed to send purchase receipt email to {to_email}")
        logger.error(f"[SES_ERROR] AWS SES error: {error_msg}")
        return False

def send_subscription_plan_changed(to_email, user_name, change_type, from_plan_name, to_plan_name,
                                 effective_date, proration_amount, reason, new_jd_quota, 
                                 new_max_subaccounts, dashboard_url):
    """Send plan change notification email to user"""
    logger.info(f"[EMAIL] send_subscription_plan_changed called with:")
    logger.info(f"   - to_email: {to_email}")
    logger.info(f"   - user_name: {user_name}")
    logger.info(f"   - change_type: {change_type}")
    logger.info(f"   - from_plan: {from_plan_name} -> to_plan: {to_plan_name}")
    
    # Try SMTP first (primary method)
    logger.info(f"[SMTP] Attempting to send plan change notification via SMTP (primary method)...")
    try:
        smtp_result = send_subscription_plan_changed_smtp(
            to_email, user_name, change_type, from_plan_name, to_plan_name,
            effective_date, proration_amount, reason, new_jd_quota, 
            new_max_subaccounts, dashboard_url
        )
        if smtp_result:
            logger.info(f"[SUCCESS] Plan change email sent successfully via SMTP to {to_email}")
            return True
        else:
            logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
    except Exception as smtp_error:
        logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
    # Fallback to AWS SES
    logger.info(f"[SES] Attempting to send plan change notification via AWS SES (fallback method)...")
    try:
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for plan change email")
            return False
            
        subject = f"Plan {change_type.title()}d - {to_plan_name} Subscription"
        
        # Load template
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'subscription_plan_changed.html')
        with open(template_path, 'r') as f:
            template = f.read()
        
        body_html = render_template_string(template,
            subject=subject,
            user_name=user_name,
            change_type=change_type,
            from_plan_name=from_plan_name,
            to_plan_name=to_plan_name,
            effective_date=effective_date,
            proration_amount=proration_amount,
            reason=reason,
            new_jd_quota=new_jd_quota,
            new_max_subaccounts=new_max_subaccounts,
            dashboard_url=dashboard_url
        )
        
        # Create text version
        text_body = f"""
{subject}

Dear {user_name},

Your subscription plan has been successfully {change_type}d.

Plan Change Details:
- Change Type: {change_type.title()}
- From Plan: {from_plan_name}
- To Plan: {to_plan_name}
- Effective Date: {effective_date}
{f"- Proration Amount: {proration_amount}" if proration_amount else ""}
{f"- Reason: {reason}" if reason else ""}

New Plan Features:
- {new_jd_quota} Job Description searches per month
- {new_max_subaccounts} Sub-accounts
{f"- Access to advanced features" if change_type == 'upgrade' else ""}
{f"- Priority support" if change_type == 'upgrade' else ""}
{f"- Enhanced analytics" if change_type == 'upgrade' else ""}

Access your dashboard: {dashboard_url}

Best regards,
The Talent-Match Team
        """
        
        logger.info(f"[SES] Sending plan change email via SES...")
        logger.info(f"   - From: {SES_FROM_EMAIL}")
        logger.info(f"   - To: {to_email}")
        logger.info(f"   - Subject: {subject}")
        
        client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {
                    'Html': {'Data': body_html},
                    'Text': {'Data': text_body}
                }
            }
        )
        
        logger.info(f"[SUCCESS] Plan change email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ERROR] Both SMTP and AWS SES failed to send plan change email to {to_email}")
        logger.error(f"[SES_ERROR] AWS SES error: {error_msg}")
        return False

def send_subscription_cancelled(to_email, user_name, plan_name, cancellation_date, effective_date,
                              reason, refund_amount, refund_status, feedback_url, reactivate_url, dashboard_url):
    """Send subscription cancellation email to user"""
    logger.info(f"[EMAIL] send_subscription_cancelled called with:")
    logger.info(f"   - to_email: {to_email}")
    logger.info(f"   - user_name: {user_name}")
    logger.info(f"   - plan_name: {plan_name}")
    logger.info(f"   - cancellation_date: {cancellation_date}")
    
    # Try SMTP first (primary method)
    logger.info(f"[SMTP] Attempting to send cancellation notification via SMTP (primary method)...")
    try:
        smtp_result = send_subscription_cancelled_smtp(
            to_email, user_name, plan_name, cancellation_date, effective_date,
            reason, refund_amount, refund_status, feedback_url, reactivate_url, dashboard_url
        )
        if smtp_result:
            logger.info(f"[SUCCESS] Cancellation email sent successfully via SMTP to {to_email}")
            return True
        else:
            logger.warning(f"[SMTP_FAILED] SMTP failed, trying AWS SES fallback...")
    except Exception as smtp_error:
        logger.warning(f"[SMTP_ERROR] SMTP failed: {str(smtp_error)}, trying AWS SES fallback...")
    
    # Fallback to AWS SES
    logger.info(f"[SES] Attempting to send cancellation notification via AWS SES (fallback method)...")
    try:
        client = get_ses_client()
        if not client:
            logger.error("SES client not available for cancellation email")
            return False
            
        subject = f"Subscription Cancelled - {plan_name}"
        
        # Load template
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'subscription_cancelled.html')
        with open(template_path, 'r') as f:
            template = f.read()
        
        body_html = render_template_string(template,
            subject=subject,
            user_name=user_name,
            plan_name=plan_name,
            cancellation_date=cancellation_date,
            effective_date=effective_date,
            reason=reason,
            refund_amount=refund_amount,
            refund_status=refund_status,
            feedback_url=feedback_url,
            reactivate_url=reactivate_url,
            dashboard_url=dashboard_url
        )
        
        # Create text version
        text_body = f"""
{subject}

Dear {user_name},

We've received your request to cancel your subscription. Your {plan_name} plan has been cancelled 
{"and will remain active until " + effective_date + "." if effective_date else "immediately."}

Cancellation Details:
- Cancelled Plan: {plan_name}
- Cancellation Date: {cancellation_date}
{f"- Access Until: {effective_date}" if effective_date else ""}
{f"- Reason: {reason}" if reason else ""}
{f"- Refund Amount: {refund_amount}" if refund_amount else ""}
{f"- Refund Status: {refund_status}" if refund_status else ""}

{"Important: You will retain access to all features until " + effective_date + ". After this date, your account will be downgraded to the free tier." if effective_date else "Your access to premium features has been immediately revoked. You can still use the free tier features."}

Share your feedback: {feedback_url}
Reactivate subscription: {reactivate_url}
Access your dashboard: {dashboard_url}

Best regards,
The Talent-Match Team
        """
        
        logger.info(f"[SES] Sending cancellation email via SES...")
        logger.info(f"   - From: {SES_FROM_EMAIL}")
        logger.info(f"   - To: {to_email}")
        logger.info(f"   - Subject: {subject}")
        
        client.send_email(
            Source=SES_FROM_EMAIL,
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {
                    'Html': {'Data': body_html},
                    'Text': {'Data': text_body}
                }
            }
        )
        
        logger.info(f"[SUCCESS] Cancellation email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ERROR] Both SMTP and AWS SES failed to send cancellation email to {to_email}")
        logger.error(f"[SES_ERROR] AWS SES error: {error_msg}")
        return False

# SMTP implementations
def send_subscription_purchase_receipt_smtp(to_email, user_name, plan_name, amount_display, transaction_id,
                                          payment_method, purchase_date, billing_cycle, next_billing_date,
                                          jd_quota, max_subaccounts, dashboard_url, invoice_number=None,
                                          invoice_url=None, receipt_url=None):
    """Send purchase receipt email via SMTP with invoice/bill details"""
    subject = f"Purchase Confirmed - {plan_name} Subscription"
    invoice_display = invoice_number or transaction_id
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{subject}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: #28a745; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">Purchase Confirmed</h1>
            <p style="margin: 10px 0 0 0; font-size: 16px;">Thank you for your subscription</p>
        </div>
        
        <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; border: 1px solid #dee2e6;">
            <p style="font-size: 16px; margin-bottom: 20px;">Dear {user_name},</p>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                Thank you for subscribing to {plan_name}! Your payment has been processed successfully.
            </p>
            
            <div style="background: white; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #28a745;">
                <h3 style="margin-top: 0; color: #28a745;">Invoice & Receipt Details</h3>
                <p><strong>Invoice/Bill Number:</strong> <span style="font-size: 18px; font-weight: bold; color: #007bff;">{invoice_display}</span></p>
                <p><strong>Transaction ID:</strong> {transaction_id}</p>
                <p><strong>Plan:</strong> <span style="font-size: 20px; font-weight: bold; color: #007bff;">{plan_name}</span></p>
                <p><strong>Amount:</strong> <span style="font-size: 24px; font-weight: bold; color: #28a745;">{amount_display}</span></p>
                <p><strong>Payment Method:</strong> {payment_method}</p>
                <p><strong>Date:</strong> {purchase_date}</p>
                <p><strong>Billing Cycle:</strong> {billing_cycle}</p>
                {f'<p><strong>Next Billing Date:</strong> {next_billing_date}</p>' if next_billing_date else ''}
                {f'<p style="margin-top: 15px;"><a href="{invoice_url}" style="color: #007bff; text-decoration: none; font-weight: bold;">View Invoice</a></p>' if invoice_url else ''}
                {f'<p><a href="{receipt_url}" style="color: #28a745; text-decoration: none; font-weight: bold;">Download Receipt</a></p>' if receipt_url else ''}
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 6px; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #1976d2;">What's Included</h4>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li><span style="color: #28a745; font-weight: bold;">{jd_quota}</span> Job Description searches per month</li>
                    <li><span style="color: #28a745; font-weight: bold;">{max_subaccounts}</span> Sub-accounts</li>
                    <li>Priority support</li>
                    <li>Advanced analytics</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="{dashboard_url}" style="display: inline-block; background: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 10px 0;">Access Your Dashboard</a>
            </div>
            
            <p style="font-size: 16px; margin-bottom: 0;">
                Best regards,<br>
                The Talent-Match Team
            </p>
        </div>
    </body>
    </html>
    """
    
    return send_email_via_smtp(to_email, subject, html_body)

def send_subscription_plan_changed_smtp(to_email, user_name, change_type, from_plan_name, to_plan_name,
                                      effective_date, proration_amount, reason, new_jd_quota, 
                                      new_max_subaccounts, dashboard_url):
    """Send plan change notification email via SMTP"""
    subject = f"Plan {change_type.title()}d - {to_plan_name} Subscription"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{subject}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: #007bff; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">Plan Change Confirmed</h1>
            <p style="margin: 10px 0 0 0; font-size: 16px;">Your subscription has been updated</p>
        </div>
        
        <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; border: 1px solid #dee2e6;">
            <p style="font-size: 16px; margin-bottom: 20px;">Dear {user_name},</p>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                Your subscription plan has been successfully {change_type}d.
            </p>
            
            <div style="background: white; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #007bff;">
                <h3 style="margin-top: 0; color: #007bff;">Plan Change Details</h3>
                <p><strong>Change Type:</strong> <span style="font-size: 18px; font-weight: bold; color: #007bff;">{change_type.title()}</span></p>
                <p><strong>From Plan:</strong> {from_plan_name}</p>
                <p><strong>To Plan:</strong> <span style="font-size: 20px; font-weight: bold; color: #007bff;">{to_plan_name}</span></p>
                <p><strong>Effective Date:</strong> {effective_date}</p>
                {f'<p><strong>Proration Amount:</strong> <span style="color: #28a745; font-weight: bold;">{proration_amount}</span></p>' if proration_amount else ''}
                {f'<p><strong>Reason:</strong> {reason}</p>' if reason else ''}
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 6px; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #1976d2;">New Plan Features</h4>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li><span style="color: #28a745; font-weight: bold;">{new_jd_quota}</span> Job Description searches per month</li>
                    <li><span style="color: #28a745; font-weight: bold;">{new_max_subaccounts}</span> Sub-accounts</li>
                    {f'<li>Access to advanced features</li>' if change_type == 'upgrade' else ''}
                    {f'<li>Priority support</li>' if change_type == 'upgrade' else ''}
                    {f'<li>Enhanced analytics</li>' if change_type == 'upgrade' else ''}
                </ul>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="{dashboard_url}" style="display: inline-block; background: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 10px 0;">View Your Dashboard</a>
            </div>
            
            <p style="font-size: 16px; margin-bottom: 0;">
                Best regards,<br>
                The Talent-Match Team
            </p>
        </div>
    </body>
    </html>
    """
    
    return send_email_via_smtp(to_email, subject, html_body)

def send_subscription_cancelled_smtp(to_email, user_name, plan_name, cancellation_date, effective_date,
                                   reason, refund_amount, refund_status, feedback_url, reactivate_url, dashboard_url):
    """Send subscription cancellation email via SMTP"""
    subject = f"Subscription Cancelled - {plan_name}"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{subject}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: #dc3545; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">Subscription Cancelled</h1>
            <p style="margin: 10px 0 0 0; font-size: 16px;">We're sorry to see you go</p>
        </div>
        
        <div style="background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; border: 1px solid #dee2e6;">
            <p style="font-size: 16px; margin-bottom: 20px;">Dear {user_name},</p>
            
            <p style="font-size: 16px; margin-bottom: 20px;">
                We've received your request to cancel your subscription. Your {plan_name} plan has been cancelled 
                {"and will remain active until " + effective_date + "." if effective_date else "immediately."}
            </p>
            
            <div style="background: white; padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #dc3545;">
                <h3 style="margin-top: 0; color: #dc3545;">Cancellation Details</h3>
                <p><strong>Cancelled Plan:</strong> <span style="font-size: 20px; font-weight: bold; color: #dc3545;">{plan_name}</span></p>
                <p><strong>Cancellation Date:</strong> {cancellation_date}</p>
                {f'<p><strong>Access Until:</strong> <span style="color: #dc3545; font-weight: bold;">{effective_date}</span></p>' if effective_date else ''}
                {f'<p><strong>Reason:</strong> {reason}</p>' if reason else ''}
                {f'<p><strong>Refund Amount:</strong> <span style="color: #dc3545; font-weight: bold;">{refund_amount}</span></p>' if refund_amount else ''}
                {f'<p><strong>Refund Status:</strong> {refund_status}</p>' if refund_status else ''}
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="{dashboard_url}" style="display: inline-block; background: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 10px 0;">Access Your Dashboard</a>
            </div>
            
            <p style="font-size: 16px; margin-bottom: 0;">
                Best regards,<br>
                The Talent-Match Team
            </p>
        </div>
    </body>
    </html>
    """
    
    return send_email_via_smtp(to_email, subject, html_body)
