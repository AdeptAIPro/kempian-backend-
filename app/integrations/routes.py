from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import db, User
from app.utils import get_current_user
from datetime import datetime
import json

logger = get_logger("integrations")

integrations_bp = Blueprint('integrations', __name__)

# Import IntegrationSubmission model
from app.models import IntegrationSubmission


@integrations_bp.route('/integrations/submit', methods=['POST'])
def submit_integration():
    """Submit a new integration request"""
    try:
        # Get JWT token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header in submit_integration")
            return jsonify({'error': 'Unauthorized - Missing or invalid Authorization header'}), 401
        
        user_jwt = get_current_user()
        if not user_jwt:
            logger.warning("get_current_user returned None in submit_integration")
            return jsonify({'error': 'Unauthorized - Invalid or expired token'}), 401
        
        if not user_jwt.get('email'):
            logger.warning(f"JWT token missing email field in submit_integration: {list(user_jwt.keys())}")
            return jsonify({'error': 'Unauthorized - Token missing email'}), 401
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get user from database
        user = User.query.filter_by(email=user_jwt['email']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Extract integration data
        integration_type = data.get('integrationType')
        integration_name = data.get('integrationName', integration_type)
        user_id = data.get('userId', user_jwt.get('email', user_jwt.get('sub')))
        user_email = data.get('userEmail', user_jwt.get('email'))
        submission_data = data.get('data', {})
        status = data.get('status', 'in_progress')
        callback_url = data.get('callbackUrl')
        source = data.get('source', 'integration_overview')
        
        # Ensure all data from the form is included (including customFields and additionalNotes)
        # This ensures nothing is lost - all form data is captured
        logger.info(f"Storing integration data for {integration_name}: {len(submission_data)} fields")
        
        # Import model here to avoid circular imports
        from app.models import IntegrationSubmission
        
        # Create integration submission
        # Store ALL data as JSON string (including customFields and additionalNotes)
        # This ensures complete data persistence - nothing is lost
        data_json = json.dumps(submission_data) if submission_data else None
        
        integration = IntegrationSubmission(
            user_id=user.id,
            user_email=user_email,
            integration_type=integration_type,
            integration_name=integration_name,
            status=status,
            data=data_json,  # Contains ALL form data including custom fields and notes
            callback_url=callback_url,
            source=source,
            saved_to_server=True
        )
        
        db.session.add(integration)
        db.session.commit()
        
        # Log data storage confirmation
        field_count = len(submission_data) if submission_data else 0
        has_custom = 'customFields' in submission_data if submission_data else False
        has_notes = 'additionalNotes' in submission_data if submission_data else False
        logger.info(f"Integration submitted: {integration_name} by user {user_email} - {field_count} fields stored (customFields: {has_custom}, notes: {has_notes})")
        
        return jsonify({
            'success': True,
            'id': integration.id,
            'message': f'{integration_name} integration submitted successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Error submitting integration: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@integrations_bp.route('/integrations/user', methods=['GET'])
def get_user_integrations():
    """Get integrations for the current user"""
    try:
        # Get JWT token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header in get_user_integrations")
            return jsonify({'error': 'Unauthorized - Missing or invalid Authorization header'}), 401
        
        user_jwt = get_current_user()
        if not user_jwt:
            logger.warning("get_current_user returned None in get_user_integrations")
            return jsonify({'error': 'Unauthorized - Invalid or expired token'}), 401
        
        if not user_jwt.get('email'):
            logger.warning(f"JWT token missing email field in get_user_integrations: {list(user_jwt.keys())}")
            return jsonify({'error': 'Unauthorized - Token missing email'}), 401
        
        # Get user from database
        user = User.query.filter_by(email=user_jwt['email']).first()
        if not user:
            logger.warning(f"User not found in database: {user_jwt.get('email')}")
            return jsonify({'error': 'User not found'}), 404
        
        # Import model here to avoid circular imports
        from app.models import IntegrationSubmission
        
        # Get integrations for this user
        integrations = IntegrationSubmission.query.filter_by(
            user_id=user.id
        ).order_by(
            IntegrationSubmission.submitted_at.desc()
        ).all()
        
        integrations_list = []
        for integration in integrations:
            # Parse data field (stored as JSON string)
            data_dict = {}
            if integration.data:
                try:
                    if isinstance(integration.data, str):
                        data_dict = json.loads(integration.data)
                    elif isinstance(integration.data, dict):
                        data_dict = integration.data
                except (json.JSONDecodeError, TypeError):
                    data_dict = {}
            
            integrations_list.append({
                'id': integration.id,
                'userId': integration.user_email or str(integration.user_id),
                'userEmail': integration.user_email,
                'integrationType': integration.integration_type,
                'integrationName': integration.integration_name,
                'status': integration.status,
                'submittedAt': integration.submitted_at.isoformat() if integration.submitted_at else None,
                'updatedAt': integration.updated_at.isoformat() if integration.updated_at else None,
                'data': data_dict,
                'savedToServer': integration.saved_to_server,
                'callbackUrl': integration.callback_url,
                'source': integration.source
            })
        
        logger.info(f"Retrieved {len(integrations_list)} integrations for user {user.email}")
        
        return jsonify({
            'success': True,
            'integrations': integrations_list,
            'count': len(integrations_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching user integrations: {str(e)}")
        return jsonify({'error': str(e)}), 500


@integrations_bp.route('/integrations/all', methods=['GET'])
def get_all_integrations():
    """Get all integration submissions (admin only)"""
    try:
        # Get JWT token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header")
            return jsonify({'error': 'Unauthorized - Missing or invalid Authorization header'}), 401
        
        user_jwt = get_current_user()
        if not user_jwt:
            logger.warning("get_current_user returned None - JWT token may be invalid or expired")
            return jsonify({'error': 'Unauthorized - Invalid or expired token'}), 401
        
        if not user_jwt.get('email'):
            logger.warning(f"JWT token missing email field: {list(user_jwt.keys())}")
            return jsonify({'error': 'Unauthorized - Token missing email'}), 401
        
        # Check if user is admin
        user = User.query.filter_by(email=user_jwt['email']).first()
        if not user:
            logger.warning(f"User not found in database: {user_jwt.get('email')}")
            return jsonify({'error': 'User not found'}), 404
        
        # Allow both 'admin' and 'owner' roles to access admin endpoints
        if user.role not in ['admin', 'owner']:
            logger.warning(f"Non-admin user attempted to access admin endpoint: {user.email} (role: {user.role})")
            return jsonify({'error': 'Admin or Owner access required'}), 403
        
        # Import model here to avoid circular imports
        from app.models import IntegrationSubmission
        
        # Get all integrations
        integrations = IntegrationSubmission.query.order_by(
            IntegrationSubmission.submitted_at.desc()
        ).all()
        
        integrations_list = []
        for integration in integrations:
            # Parse data field (stored as JSON string)
            data_dict = {}
            if integration.data:
                try:
                    if isinstance(integration.data, str):
                        data_dict = json.loads(integration.data)
                    elif isinstance(integration.data, dict):
                        data_dict = integration.data
                except (json.JSONDecodeError, TypeError):
                    data_dict = {}
            
            integrations_list.append({
                'id': integration.id,
                'userId': integration.user_email or str(integration.user_id),
                'userEmail': integration.user_email,
                'integrationType': integration.integration_type,
                'integrationName': integration.integration_name,
                'status': integration.status,
                'submittedAt': integration.submitted_at.isoformat() if integration.submitted_at else None,
                'updatedAt': integration.updated_at.isoformat() if integration.updated_at else None,
                'data': data_dict,
                'savedToServer': integration.saved_to_server
            })
        
        return jsonify({
            'success': True,
            'integrations': integrations_list,
            'count': len(integrations_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching integrations: {str(e)}")
        return jsonify({'error': str(e)}), 500


@integrations_bp.route('/integrations/<int:integration_id>/update', methods=['PUT'])
def update_integration(integration_id):
    """Update an integration submission"""
    try:
        # Get JWT token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header in update_integration")
            return jsonify({'error': 'Unauthorized - Missing or invalid Authorization header'}), 401
        
        user_jwt = get_current_user()
        if not user_jwt:
            logger.warning("get_current_user returned None in update_integration")
            return jsonify({'error': 'Unauthorized - Invalid or expired token'}), 401
        
        if not user_jwt.get('email'):
            logger.warning(f"JWT token missing email field in update_integration: {list(user_jwt.keys())}")
            return jsonify({'error': 'Unauthorized - Token missing email'}), 401
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Import model here to avoid circular imports
        from app.models import IntegrationSubmission
        
        # Get integration
        integration = IntegrationSubmission.query.get(integration_id)
        if not integration:
            return jsonify({'error': 'Integration not found'}), 404
        
        # Check if user is admin or the owner
        user = User.query.filter_by(email=user_jwt['email']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Allow both 'admin' and 'owner' roles to update any integration, or users can update their own
        if user.role not in ['admin', 'owner'] and integration.user_id != user.id:
            return jsonify({'error': 'Unauthorized to update this integration'}), 403
        
        # Update integration data
        if 'data' in data:
            # Store data as JSON string
            integration.data = json.dumps(data['data']) if data['data'] else None
        
        if 'status' in data:
            integration.status = data['status']
        
        integration.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(f"Integration updated: {integration.integration_name} (ID: {integration_id})")
        
        return jsonify({
            'success': True,
            'message': 'Integration updated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating integration: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@integrations_bp.route('/integrations/send-email', methods=['POST'])
def send_integration_email():
    """Send email notification for integration submission"""
    try:
        # Get JWT token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header in send_integration_email")
            return jsonify({'error': 'Unauthorized - Missing or invalid Authorization header'}), 401
        
        user_jwt = get_current_user()
        if not user_jwt:
            logger.warning("get_current_user returned None in send_integration_email")
            return jsonify({'error': 'Unauthorized - Invalid or expired token'}), 401
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        integration_data = data.get('integrationData', {})
        recipient_email = data.get('recipientEmail', 'nishant@adeptaipro.com')
        subject = data.get('subject', 'New Integration Request')
        template = data.get('template', 'integration_request')
        
        # Try to send email using SMTP
        try:
            from app.emails.smtp import send_email
            from app.simple_logger import get_logger
            
            email_logger = get_logger("integration_email")
            
            # Prepare email content
            integration_name = integration_data.get('integrationName', 'Unknown')
            user_email = integration_data.get('userEmail', 'Unknown')
            integration_type = integration_data.get('integrationType', 'Unknown')
            
            email_body = f"""
            New Integration Request
            
            Integration Name: {integration_name}
            Integration Type: {integration_type}
            User Email: {user_email}
            Submitted At: {integration_data.get('submittedAt', 'N/A')}
            
            Please review the integration request in the admin panel.
            """
            
            send_email(
                to_email=recipient_email,
                subject=subject,
                body=email_body
            )
            
            email_logger.info(f"Integration email sent to {recipient_email} for {integration_name}")
            
            return jsonify({
                'success': True,
                'message': 'Email sent successfully'
            }), 200
            
        except ImportError:
            logger.warning("SMTP email service not available")
            return jsonify({
                'success': False,
                'message': 'Email service not available'
            }), 503
        except Exception as email_error:
            logger.error(f"Error sending email: {str(email_error)}")
            return jsonify({
                'success': False,
                'message': f'Failed to send email: {str(email_error)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Error in send_integration_email: {str(e)}")
        return jsonify({'error': str(e)}), 500

