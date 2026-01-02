import os
import random
import string
from datetime import datetime, timedelta
from app.simple_logger import get_logger
from flask import Blueprint, request, jsonify, current_app
from flask_mail import Mail, Message
from sqlalchemy import and_
from app import db
from app.models import User, PasswordResetOTP
from app.utils import decode_jwt
import logging

logger = get_logger("auth")

password_reset_bp = Blueprint('password_reset', __name__)

# Initialize Flask-Mail
mail = Mail()

def generate_otp(length=6):
    """Generate a random OTP of specified length"""
    return ''.join(random.choices(string.digits, k=length))

def send_otp_email(email, otp, user_name):
    """Send OTP email to user"""
    try:
        subject = "Password Reset OTP - Kempian"
        body = f"""
        Hello {user_name},
        
        You have requested to reset your password. Please use the following OTP to proceed:
        
        OTP: {otp}
        
        This OTP is valid for 10 minutes only.
        
        If you didn't request this password reset, please ignore this email.
        
        Best regards,
        Kempian Team
        """
        
        msg = Message(
            subject=subject,
            recipients=[email],
            body=body,
            sender=current_app.config.get('MAIL_DEFAULT_SENDER', 'noreply@kempian.com')
        )
        
        mail.send(msg)
        logger.info(f"OTP email sent successfully to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send OTP email to {email}: {e}")
        return False

@password_reset_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Request password reset OTP"""
    try:
        data = request.get_json()
        if not data or not data.get('email'):
            return jsonify({'error': 'Email is required'}), 400
        
        email = data['email'].lower().strip()
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        if not user:
            # Don't reveal if user exists or not for security
            return jsonify({'message': 'If an account with this email exists, a password reset OTP has been sent'}), 200
        
        # Check if there's already a valid OTP
        existing_otp = PasswordResetOTP.query.filter(
            and_(
                PasswordResetOTP.user_id == user.id,
                PasswordResetOTP.expires_at > datetime.utcnow(),
                PasswordResetOTP.used == False
            )
        ).first()
        
        if existing_otp:
            # If OTP exists and is still valid, don't create a new one
            time_remaining = (existing_otp.expires_at - datetime.utcnow()).total_seconds()
            if time_remaining > 60:  # More than 1 minute remaining
                return jsonify({
                    'message': 'An OTP has already been sent. Please check your email or wait before requesting another.',
                    'can_resend_in': int(time_remaining - 60)
                }), 200
        
        # Generate new OTP
        otp = generate_otp(6)
        expires_at = datetime.utcnow() + timedelta(minutes=10)
        
        # Invalidate any existing OTPs for this user
        PasswordResetOTP.query.filter_by(user_id=user.id).update({'used': True})
        
        # Create new OTP record
        otp_record = PasswordResetOTP(
            user_id=user.id,
            otp=otp,
            expires_at=expires_at,
            used=False,
            created_at=datetime.utcnow()
        )
        
        db.session.add(otp_record)
        db.session.commit()
        
        # Send OTP email
        if send_otp_email(email, otp, user.first_name or user.email):
            logger.info(f"Password reset OTP created for user {email}")
            return jsonify({
                'message': 'Password reset OTP has been sent to your email',
                'expires_in': 600  # 10 minutes in seconds
            }), 200
        else:
            # Rollback if email sending fails
            db.session.rollback()
            return jsonify({'error': 'Failed to send OTP email. Please try again.'}), 500
            
    except Exception as e:
        logger.error(f"Error in forgot_password: {e}")
        db.session.rollback()
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@password_reset_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP for password reset"""
    try:
        data = request.get_json()
        if not data or not data.get('email') or not data.get('otp'):
            return jsonify({'error': 'Email and OTP are required'}), 400
        
        email = data['email'].lower().strip()
        otp = data['otp'].strip()
        
        # Find user
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'Invalid email or OTP'}), 400
        
        # Find valid OTP
        otp_record = PasswordResetOTP.query.filter(
            and_(
                PasswordResetOTP.user_id == user.id,
                PasswordResetOTP.otp == otp,
                PasswordResetOTP.expires_at > datetime.utcnow(),
                PasswordResetOTP.used == False
            )
        ).first()
        
        if not otp_record:
            return jsonify({'error': 'Invalid or expired OTP'}), 400
        
        # Mark OTP as used
        otp_record.used = True
        otp_record.used_at = datetime.utcnow()
        db.session.commit()
        
        # Generate reset token (you can use JWT or a simple token)
        reset_token = generate_reset_token(user.id, otp_record.id)
        
        logger.info(f"OTP verified successfully for user {email}")
        return jsonify({
            'message': 'OTP verified successfully',
            'reset_token': reset_token,
            'user_id': user.id
        }), 200
        
    except Exception as e:
        logger.error(f"Error in verify_otp: {e}")
        db.session.rollback()
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@password_reset_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset password using reset token"""
    try:
        data = request.get_json()
        if not data or not data.get('reset_token') or not data.get('new_password'):
            return jsonify({'error': 'Reset token and new password are required'}), 400
        
        reset_token = data['reset_token']
        new_password = data['new_password']
        
        # Validate password strength
        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Decode reset token to get user_id and otp_id
        try:
            token_data = decode_reset_token(reset_token)
            user_id = token_data.get('user_id')
            otp_id = token_data.get('otp_id')
        except:
            return jsonify({'error': 'Invalid reset token'}), 400
        
        # Verify OTP was used
        otp_record = PasswordResetOTP.query.filter(
            and_(
                PasswordResetOTP.id == otp_id,
                PasswordResetOTP.user_id == user_id,
                PasswordResetOTP.used == True
            )
        ).first()
        
        if not otp_record:
            return jsonify({'error': 'Invalid reset token'}), 400
        
        # Update user password
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 400
        
        # Hash and set new password (assuming you have a password hashing method)
        user.set_password(new_password)
        user.updated_at = datetime.utcnow()
        
        # Invalidate all OTPs for this user
        PasswordResetOTP.query.filter_by(user_id=user_id).update({'used': True})
        
        db.session.commit()
        
        logger.info(f"Password reset successfully for user {user.email}")
        return jsonify({'message': 'Password reset successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error in reset_password: {e}")
        db.session.rollback()
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@password_reset_bp.route('/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP if expired or not received"""
    try:
        data = request.get_json()
        if not data or not data.get('email'):
            return jsonify({'error': 'Email is required'}), 400
        
        email = data['email'].lower().strip()
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        if not user:
            # Don't reveal if user exists or not for security
            return jsonify({'message': 'If an account with this email exists, a new OTP has been sent'}), 200
        
        # Check if there's a recent OTP request (within last 1 minute)
        recent_otp = PasswordResetOTP.query.filter(
            and_(
                PasswordResetOTP.user_id == user.id,
                PasswordResetOTP.created_at > datetime.utcnow() - timedelta(minutes=1)
            )
        ).first()
        
        if recent_otp:
            return jsonify({
                'error': 'Please wait at least 1 minute before requesting another OTP',
                'can_resend_in': 60
            }), 429
        
        # Generate new OTP
        otp = generate_otp(6)
        expires_at = datetime.utcnow() + timedelta(minutes=10)
        
        # Invalidate any existing OTPs for this user
        PasswordResetOTP.query.filter_by(user_id=user.id).update({'used': True})
        
        # Create new OTP record
        otp_record = PasswordResetOTP(
            user_id=user.id,
            otp=otp,
            expires_at=expires_at,
            used=False,
            created_at=datetime.utcnow()
        )
        
        db.session.add(otp_record)
        db.session.commit()
        
        # Send OTP email
        if send_otp_email(email, otp, user.first_name or user.email):
            logger.info(f"OTP resent successfully for user {email}")
            return jsonify({
                'message': 'New OTP has been sent to your email',
                'expires_in': 600  # 10 minutes in seconds
            }), 200
        else:
            # Rollback if email sending fails
            db.session.rollback()
            return jsonify({'error': 'Failed to send OTP email. Please try again.'}), 500
            
    except Exception as e:
        logger.error(f"Error in resend_otp: {e}")
        db.session.rollback()
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

def generate_reset_token(user_id, otp_id):
    """Generate a simple reset token (you can enhance this with JWT)"""
    import hashlib
    import time
    
    # Create a simple hash-based token
    token_data = f"{user_id}:{otp_id}:{int(time.time())}"
    token_hash = hashlib.sha256(token_data.encode()).hexdigest()[:32]
    
    return token_hash

def decode_reset_token(token):
    """Decode reset token (simple implementation - enhance with JWT)"""
    # This is a simple implementation - you should use JWT for production
    # For now, we'll store the token data in a temporary way
    # In production, use JWT with proper signing and expiration
    
    # This is a placeholder - implement proper JWT decoding
    return {'user_id': None, 'otp_id': None}

# Add this to your app initialization
def init_password_reset(app):
    """Initialize password reset system"""
    mail.init_app(app)
    app.register_blueprint(password_reset_bp, url_prefix='/auth')
