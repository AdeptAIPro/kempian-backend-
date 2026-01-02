from flask import Blueprint, jsonify, request
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, Tenant, EmployeeProfile, UserBankAccount, db
from app.auth.cognito import cognito_admin_create_user
from app.emails.ses import send_invite_email
from datetime import datetime
from decimal import Decimal
from urllib.parse import quote
import os
import secrets
import string
import boto3


hr_employees_bp = Blueprint('hr_employees', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


def _can_view_full_directory(db_user):
    role = (db_user.role or '').lower()
    user_type = (db_user.user_type or '').lower()
    return role in ['admin', 'owner'] or user_type in ['admin', 'employer', 'recruiter']


@hr_employees_bp.route('/', methods=['GET'], strict_slashes=False)
def list_employees():
    user, error = _auth_or_401()
    if error:
        return error
    
    # Debug: Log user info
    print(f"[DEBUG] list_employees - User email: {user.get('email') if user else 'None'}")
    
    # Get user from database to check role and get organizations they created
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = get_current_tenant_id()
    print(f"[DEBUG] list_employees - Tenant ID: {tenant_id}")
    print(f"[DEBUG] list_employees - User role: {db_user.role}, user_type: {db_user.user_type}")
    
    # Build query for employees
    from sqlalchemy.orm import joinedload
    from app.models import OrganizationMetadata
    
    query = User.query.filter((User.user_type == 'employee') | (User.role == 'employee'))
    
    is_employee = (db_user.role == 'employee') or (db_user.user_type == 'employee')
    can_view_all = _can_view_full_directory(db_user)
    
    if is_employee:
        # Employees can only view their own record
        query = query.filter(User.id == db_user.id)
        print(f"[DEBUG] list_employees - Employee restricted to self view (user_id={db_user.id})")
    elif not can_view_all:
        print(f"[WARNING] list_employees - User lacks permission to view employee directory")
        return jsonify({'error': 'You do not have permission to view employee records'}), 403
    
    # Filter by tenant_id based on user role
    if tenant_id:
        if db_user.role in ['owner', 'admin']:
            # For owners/admins: show employees from their current org AND any orgs they created
            # Get organization IDs that this user created
            created_org_ids = db.session.query(OrganizationMetadata.tenant_id).filter_by(created_by=db_user.id).all()
            created_org_ids = [org[0] for org in created_org_ids]
            
            # Include current tenant_id and all created organization IDs
            allowed_tenant_ids = [tenant_id] + created_org_ids
            print(f"[DEBUG] list_employees - Allowed tenant IDs (owner/admin): {allowed_tenant_ids}")
            query = query.filter(User.tenant_id.in_(allowed_tenant_ids))
        else:
            # For other roles: only show employees from their current organization
            query = query.filter_by(tenant_id=tenant_id)
            print(f"[DEBUG] list_employees - Filtering by tenant_id: {tenant_id}")
    else:
        print(f"[DEBUG] list_employees - No tenant_id filter applied")
    
    # Eager load employee_profile relationship
    employees = query.options(joinedload(User.employee_profile)).all()
    
    print(f"[DEBUG] list_employees - Found {len(employees)} employees")
    for idx, emp in enumerate(employees):
        print(f"[DEBUG] Employee {idx + 1}: id={emp.id}, email={emp.email}, role={emp.role}, user_type={emp.user_type}, tenant_id={emp.tenant_id}, has_profile={emp.employee_profile is not None}")
    
    results = []
    for u in employees:
        employee_data = {
            'id': u.id,
            'email': u.email,
            'role': u.role,
            'user_type': u.user_type,
            'tenant_id': u.tenant_id,
            'company_name': u.company_name,
            'created_at': u.created_at.isoformat() if u.created_at else None
        }
        # Get employee profile data if exists
        if u.employee_profile:
            profile = u.employee_profile
            employee_data.update({
                'first_name': profile.first_name,
                'last_name': profile.last_name,
                'phone': profile.phone,
                'department': profile.department,
                'location': profile.location,
                'employment_type': profile.employment_type,
                'category': profile.category,
                'salary_amount': float(profile.salary_amount) if profile.salary_amount else None,
                'salary_currency': profile.salary_currency,
                'salary_type': profile.salary_type or 'monthly',
                'hire_date': profile.hire_date.isoformat() if profile.hire_date else None
            })
        results.append(employee_data)
    
    print(f"[DEBUG] list_employees - Returning {len(results)} employee results")
    print(f"[DEBUG] list_employees - Results: {results}")
    
    return jsonify({'employees': results}), 200


@hr_employees_bp.route('/', methods=['POST'], strict_slashes=False)
def create_employee():
    """Create a new employee and send invite link - for admin, employer, recruiter"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Explicitly deny employees from creating other employees
    if db_user.role == 'employee' or db_user.user_type == 'employee':
        print(f"[ERROR] create_employee - Employee attempted to create another employee: {db_user.email}")
        return jsonify({'error': 'Employees cannot create other employees'}), 403
    
    # Only admin, employer, recruiter, and owner can create employees
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        print(f"[ERROR] create_employee - Permission denied for user: {db_user.role}/{db_user.user_type}")
        return jsonify({'error': 'You do not have permission to create employees'}), 403
    
    data = request.get_json()
    email = data.get('email')
    first_name = data.get('first_name', '')
    last_name = data.get('last_name', '')
    phone = data.get('phone')
    department = data.get('department')
    location = data.get('location')
    employment_type = data.get('employment_type')
    category = data.get('category')
    organization_id = data.get('organization_id')  # Optional: specific organization
    salary_amount = data.get('salary_amount')
    salary_currency = data.get('salary_currency', 'USD')
    salary_type = data.get('salary_type', 'monthly')  # 'monthly' or 'hourly'
    hire_date = data.get('hire_date')
    
    # Debug logging
    print(f"[DEBUG] Creating employee with data: email={email}, organization_id={organization_id}, employment_type={employment_type}, category={category}, first_name={first_name}, last_name={last_name}")
    
    # Validate required fields with detailed error messages
    validation_errors = []
    if not email:
        validation_errors.append('Email is required')
    if not first_name:
        validation_errors.append('First name is required')
    if not last_name:
        validation_errors.append('Last name is required')
    if not employment_type:
        validation_errors.append('Employment type is required')
    if not category:
        validation_errors.append('Category is required')
    
    if validation_errors:
        error_message = 'Missing required fields: ' + ', '.join(validation_errors)
        print(f"[ERROR] Validation failed: {error_message}")
        return jsonify({'error': error_message, 'missing_fields': validation_errors}), 400
    
    # Check if user already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        # Check if it's already an employee - if so, return the existing employee
        if (existing_user.role == 'employee' or existing_user.user_type == 'employee'):
            # Employee already exists, return existing employee data
            employee_data = {
                'id': existing_user.id,
                'email': existing_user.email,
                'tenant_id': existing_user.tenant_id,
                'created_at': existing_user.created_at.isoformat() if existing_user.created_at else None
            }
            if existing_user.employee_profile:
                profile = existing_user.employee_profile
                employee_data.update({
                    'first_name': profile.first_name,
                    'last_name': profile.last_name,
                })
            return jsonify({
                'employee': employee_data,
                'message': 'Employee already exists',
                'invite_link': None,
                'email_sent': False
            }), 200
        else:
            return jsonify({'error': 'User with this email already exists with a different role'}), 400
    
    try:
        # Determine tenant_id
        # Handle organization_id: if it's "none", None, empty string, or 0, use user's tenant_id
        if organization_id and organization_id != "none" and str(organization_id).strip():
            try:
                org_id_int = int(organization_id)
                # Verify organization exists
                org = Tenant.query.get(org_id_int)
                if not org:
                    return jsonify({'error': 'Organization not found'}), 404
                tenant_id = org_id_int
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid organization ID format'}), 400
        else:
            # Use user's current tenant_id
            tenant_id = db_user.tenant_id
            if not tenant_id:
                return jsonify({'error': 'User does not have an associated organization'}), 400
        
        # Create user in database (only core fields - no employee-specific data)
        new_employee = User(
            tenant_id=tenant_id,
            email=email,
            role='employee',
            user_type='employee',
            company_name=getattr(db_user, 'company_name', None)
        )
        db.session.add(new_employee)
        db.session.flush()  # Flush to get the user ID
        
        # Create employee profile with employee-specific data
        employee_profile = EmployeeProfile(
            user_id=new_employee.id,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            department=department,
            location=location,
            employment_type=employment_type,
            category=category
        )
        
        # Add salary information if provided
        if salary_amount is not None:
            try:
                employee_profile.salary_amount = Decimal(str(salary_amount))
                employee_profile.salary_currency = salary_currency or 'USD'
            except (ValueError, TypeError):
                pass  # Skip if invalid salary amount
        
        # Always persist salary type if provided
        if salary_type:
            employee_profile.salary_type = salary_type or 'monthly'
        
        # Add hire date if provided
        if hire_date:
            try:
                employee_profile.hire_date = datetime.strptime(hire_date, '%Y-%m-%d').date()
            except (ValueError, TypeError):
                pass  # Skip if invalid date format
        db.session.add(employee_profile)
        db.session.commit()
        
        print(f"[DEBUG] create_employee - Employee created successfully:")
        print(f"[DEBUG]   - User ID: {new_employee.id}")
        print(f"[DEBUG]   - Email: {new_employee.email}")
        print(f"[DEBUG]   - Role: {new_employee.role}")
        print(f"[DEBUG]   - User Type: {new_employee.user_type}")
        print(f"[DEBUG]   - Tenant ID: {new_employee.tenant_id}")
        print(f"[DEBUG]   - Has Profile: {employee_profile is not None}")
        
        # Generate temporary password and create in Cognito
        temp_password = None
        invite_link = None
        email_sent = False
        
        try:
            from botocore.exceptions import ClientError
            temp_password, _ = cognito_admin_create_user(email, tenant_id, role='employee')
            print(f"[DEBUG] create_employee - Cognito user created successfully, temp_password: {'Yes' if temp_password else 'No'}")
        except ClientError as cognito_error:
            # Handle Cognito-specific errors
            error_code = cognito_error.response.get('Error', {}).get('Code', '')
            error_msg = str(cognito_error)
            print(f"[WARNING] create_employee - Cognito ClientError: {error_code} - {error_msg}")
            
            if error_code == 'UsernameExistsException':
                # User already exists in Cognito - reset their password to get a real temp password
                print(f"[DEBUG] create_employee - User already exists in Cognito, resetting password to get temp password")
                try:
                    from app.auth.cognito import COGNITO_USER_POOL_ID, cognito_client, cognito_admin_update_user_attributes
                    import random
                    new_temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12)) + '1!A'
                    
                    # Reset the user's password to the new temp password
                    cognito_client.admin_set_user_password(
                        UserPoolId=COGNITO_USER_POOL_ID,
                        Username=email,
                        Password=new_temp_password,
                        Permanent=False  # Keep as temporary so user must change it
                    )
                    
                    # Update user attributes to ensure role and user_type are set correctly for employee
                    try:
                        cognito_admin_update_user_attributes(email, {
                            'custom:role': 'employee',
                            'custom:user_type': 'employee',
                            'custom:tenant_id': str(tenant_id)
                        })
                        print(f"[DEBUG] create_employee - Updated Cognito attributes: role=employee, user_type=employee")
                    except Exception as attr_error:
                        print(f"[WARNING] create_employee - Failed to update Cognito attributes: {str(attr_error)}")
                    
                    temp_password = new_temp_password
                    print(f"[DEBUG] create_employee - Password reset successfully, new temp password generated")
                except Exception as reset_error:
                    print(f"[WARNING] create_employee - Failed to reset password: {str(reset_error)}")
                    # Fallback: generate a temp password (won't work if user is already activated)
                    temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12)) + '1!A'
                    print(f"[DEBUG] create_employee - Generated fallback temp password for invite link")
            else:
                # Other Cognito errors - log but continue (user can be invited later)
                import traceback
                error_trace = traceback.format_exc()
                print(f"[ERROR] create_employee - Cognito error: {error_code} - {error_msg}")
                print(f"[ERROR] create_employee - Cognito error traceback: {error_trace}")
                print(f"[WARNING] create_employee - Employee created in database but not in Cognito. Use invite endpoint to create Cognito user.")
        except Exception as cognito_error:
            # Other unexpected errors - log but continue
            import traceback
            error_trace = traceback.format_exc()
            print(f"[WARNING] create_employee - Failed to create user in Cognito: {str(cognito_error)}")
            print(f"[WARNING] create_employee - Cognito error traceback: {error_trace}")
            print(f"[WARNING] create_employee - Employee created in database but not in Cognito. Use invite endpoint to create Cognito user.")
        
        # Generate invite link if we have a temp password
        if temp_password:
            # Use temporary password as the code in invite link (frontend expects it as temp_password)
            # The invite link uses 'code' parameter which the frontend uses as temp_password
            # URL encode the code and email to handle special characters like !
            # Get frontend URL (local for development, production for production)
            def get_frontend_url():
                frontend_url = os.getenv('FRONTEND_URL')
                if frontend_url:
                    return frontend_url
                flask_env = os.getenv('FLASK_ENV', '').lower()
                flask_debug = os.getenv('FLASK_DEBUG', '').lower()
                is_development = (
                    flask_env == 'development' or 
                    flask_debug == 'true' or 
                    flask_debug == '1' or
                    os.getenv('ENVIRONMENT', '').lower() == 'development' or
                    os.getenv('ENV', '').lower() == 'development'
                )
                if is_development:
                    local_port = os.getenv('FRONTEND_PORT', '5173')
                    return f'http://localhost:{local_port}'
                else:
                    return 'https://kempian.ai'
            
            frontend_url = get_frontend_url()
            encoded_code = quote(temp_password, safe='')
            encoded_email = quote(email, safe='')
            invite_link = (
                f"{frontend_url}/invite?"
                f"email={encoded_email}&username={encoded_email}&code={encoded_code}&type=employee"
            )
            
            # Send invite email
            try:
                print(f"[DEBUG] create_employee - Attempting to send invite email to: {email}")
                print(f"[DEBUG] create_employee - Invite link: {invite_link}")
                email_sent = send_invite_email(email, invite_link)
                print(f"[DEBUG] create_employee - Email send result: {email_sent}")
            except Exception as email_error:
                # Log email error but continue
                import traceback
                error_trace = traceback.format_exc()
                print(f"[ERROR] create_employee - Failed to send invite email: {str(email_error)}")
                print(f"[ERROR] create_employee - Email error traceback: {error_trace}")
                email_sent = False
        else:
            print(f"[WARNING] create_employee - No temp password available, invite link not generated. Use invite endpoint to create Cognito user.")
        
        return jsonify({
            'employee': {
                'id': new_employee.id,
                'email': new_employee.email,
                'first_name': employee_profile.first_name,
                'last_name': employee_profile.last_name,
                'tenant_id': new_employee.tenant_id,
                'created_at': new_employee.created_at.isoformat() if new_employee.created_at else None
            },
            'invite_link': invite_link,
            'email_sent': email_sent
        }), 201
    except Exception as e:
        db.session.rollback()
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Failed to create employee: {str(e)}")
        print(f"[ERROR] Traceback: {error_trace}")
        return jsonify({'error': f'Failed to create employee: {str(e)}'}), 500


@hr_employees_bp.route('/export', methods=['GET'])
def export_employees():
    """Export employees list to CSV"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, employer, recruiter, and owner can export
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to export employees'}), 403
    
    tenant_id = get_current_tenant_id()
    query = User.query
    if tenant_id:
        query = query.filter_by(tenant_id=tenant_id)
    employees = query.filter((User.user_type == 'employee') | (User.role == 'employee')).all()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Employee ID', 'Email', 'First Name', 'Last Name', 'Phone', 'Department',
        'Location', 'Employment Type', 'Category', 'Salary Amount', 'Currency',
        'Hire Date', 'Organization', 'Created At'
    ])
    
    # Write data
    for u in employees:
        profile = u.employee_profile
        org_name = "Default"
        if u.tenant_id:
            from app.models import OrganizationMetadata
            org_meta = OrganizationMetadata.query.filter_by(tenant_id=u.tenant_id).first()
            if org_meta:
                org_name = org_meta.name
        
        writer.writerow([
            u.id,
            u.email,
            profile.first_name if profile else '',
            profile.last_name if profile else '',
            profile.phone if profile else '',
            profile.department if profile else '',
            profile.location if profile else '',
            profile.employment_type if profile else '',
            profile.category if profile else '',
            float(profile.salary_amount) if profile and profile.salary_amount else '',
            profile.salary_currency if profile else '',
            profile.hire_date.isoformat() if profile and profile.hire_date else '',
            org_name,
            u.created_at.isoformat() if u.created_at else ''
        ])
    
    output.seek(0)
    
    # Create response
    response = send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'employees_export_{datetime.now().strftime("%Y%m%d")}.csv'
    )
    
    return response


@hr_employees_bp.route('/<int:employee_id>/invite', methods=['POST'])
def send_employee_invite(employee_id):
    """Send invite link to an existing employee"""
    user, error = _auth_or_401()
    if error:
        return error
    
    print(f"[DEBUG] send_employee_invite - Request for employee_id: {employee_id}")
    print(f"[DEBUG] send_employee_invite - User email: {user.get('email') if user else 'None'}")
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        print(f"[ERROR] send_employee_invite - User not found in database")
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, employer, recruiter, and owner can send invites
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        print(f"[ERROR] send_employee_invite - Permission denied for user: {db_user.role}/{db_user.user_type}")
        return jsonify({'error': 'You do not have permission to send invites'}), 403
    
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        print(f"[ERROR] send_employee_invite - Employee not found: {employee_id}")
        return jsonify({'error': 'Employee not found'}), 404
    
    print(f"[DEBUG] send_employee_invite - Employee found: {employee.email}, role: {employee.role}, user_type: {employee.user_type}")
    
    if employee.role != 'employee' and employee.user_type != 'employee':
        print(f"[ERROR] send_employee_invite - User is not an employee: role={employee.role}, user_type={employee.user_type}")
        return jsonify({'error': 'User is not an employee'}), 400
    
    try:
        # Get or create temporary password for the employee
        # First check if user exists in Cognito and get temp password
        from app.auth.cognito import cognito_admin_create_user
        from botocore.exceptions import ClientError
        temp_password = None
        try:
            print(f"[DEBUG] send_employee_invite - Attempting to create/get user in Cognito: {employee.email}")
            # Try to get temp password (this will create user if doesn't exist)
            temp_password, _ = cognito_admin_create_user(employee.email, employee.tenant_id, role='employee')
            print(f"[DEBUG] send_employee_invite - Cognito user created successfully, temp_password: {'Yes' if temp_password else 'No'}")
        except ClientError as cognito_error:
            # If user already exists, we can't get temp password - generate one for invite link
            error_code = cognito_error.response.get('Error', {}).get('Code', '')
            print(f"[WARNING] send_employee_invite - Cognito ClientError: {error_code} - {str(cognito_error)}")
            
            if error_code == 'UsernameExistsException':
                # User already exists in Cognito - reset their password to get a real temp password
                print(f"[DEBUG] send_employee_invite - User already exists in Cognito, resetting password to get temp password")
                try:
                    # Import Cognito client
                    from app.auth.cognito import COGNITO_USER_POOL_ID, cognito_client, cognito_admin_update_user_attributes
                    
                    # Generate a new temporary password
                    import random
                    new_temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12)) + '1!A'
                    
                    # Reset the user's password to the new temp password
                    cognito_client.admin_set_user_password(
                        UserPoolId=COGNITO_USER_POOL_ID,
                        Username=employee.email,
                        Password=new_temp_password,
                        Permanent=False  # Keep as temporary so user must change it
                    )
                    
                    # Update user attributes to ensure role and user_type are set correctly for employee
                    try:
                        cognito_admin_update_user_attributes(employee.email, {
                            'custom:role': 'employee',
                            'custom:user_type': 'employee',
                            'custom:tenant_id': str(employee.tenant_id)
                        })
                        print(f"[DEBUG] send_employee_invite - Updated Cognito attributes: role=employee, user_type=employee")
                    except Exception as attr_error:
                        print(f"[WARNING] send_employee_invite - Failed to update Cognito attributes: {str(attr_error)}")
                    
                    temp_password = new_temp_password
                    print(f"[DEBUG] send_employee_invite - Password reset successfully, new temp password generated")
                except Exception as reset_error:
                    print(f"[WARNING] send_employee_invite - Failed to reset password: {str(reset_error)}")
                    # Fallback: generate a temp password (won't work if user is already activated)
                    temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12)) + '1!A'
                    print(f"[DEBUG] send_employee_invite - Generated fallback temp password for invite link")
            else:
                # Other Cognito errors
                import traceback
                error_trace = traceback.format_exc()
                print(f"[ERROR] send_employee_invite - Cognito error: {error_code} - {str(cognito_error)}")
                print(f"[ERROR] send_employee_invite - Cognito error traceback: {error_trace}")
        except Exception as cognito_error:
            # Other unexpected errors
            import traceback
            error_trace = traceback.format_exc()
            print(f"[ERROR] send_employee_invite - Unexpected Cognito error: {str(cognito_error)}")
            print(f"[ERROR] send_employee_invite - Error traceback: {error_trace}")
        
        # Always generate a temp password if we don't have one (fallback)
        if not temp_password:
            print(f"[WARNING] send_employee_invite - No temp password from Cognito, trying to create user")
            try:
                # Try one more time to create the user
                temp_password, _ = cognito_admin_create_user(employee.email, employee.tenant_id, role='employee')
                print(f"[DEBUG] send_employee_invite - User created successfully on retry")
            except Exception as retry_error:
                print(f"[ERROR] send_employee_invite - Failed to create user on retry: {str(retry_error)}")
                # Last resort: generate a temp password (won't work, but at least we send the invite)
                temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12)) + '1!A'
                print(f"[DEBUG] send_employee_invite - Generated fallback temp password (this won't work for activation)")
        
        # Use temporary password as the code in invite link
        # URL encode the code to handle special characters like !
        frontend_url = os.getenv('FRONTEND_URL', 'https://kempian.ai')
        encoded_code = quote(temp_password, safe='')
        encoded_email = quote(employee.email, safe='')
        invite_link = (
            f"{frontend_url}/invite?"
            f"email={encoded_email}&username={encoded_email}&code={encoded_code}&type=employee"
        )
        
        print(f"[DEBUG] send_employee_invite - Invite link generated: {invite_link}")
        
        # Check if email should be sent (default to True for backward compatibility)
        data = request.get_json() or {}
        send_email = data.get('send_email', True)
        
        # Send invite email only if requested
        email_sent = False
        if send_email:
            try:
                print(f"[DEBUG] send_employee_invite - Attempting to send invite email to: {employee.email}")
                email_sent = send_invite_email(employee.email, invite_link)
                print(f"[DEBUG] send_employee_invite - Email send result: {email_sent}")
            except Exception as email_error:
                # Log email error but continue
                import traceback
                error_trace = traceback.format_exc()
                print(f"[ERROR] send_employee_invite - Failed to send invite email: {str(email_error)}")
                print(f"[ERROR] send_employee_invite - Email error traceback: {error_trace}")
                email_sent = False
        else:
            print(f"[DEBUG] send_employee_invite - Email sending skipped (manual mode)")
        
        return jsonify({
            'invite_link': invite_link,
            'email_sent': email_sent,
            'employee_email': employee.email
        }), 200
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] send_employee_invite - Unexpected error: {str(e)}")
        print(f"[ERROR] send_employee_invite - Error traceback: {error_trace}")
        return jsonify({'error': f'Failed to send invite: {str(e)}'}), 500


@hr_employees_bp.route('/<int:employee_id>', methods=['PUT'], strict_slashes=False)
def update_employee(employee_id):
    """Update employee details - for admin and owner"""
    user, error = _auth_or_401()
    if error:
        return error
    
    print(f"[DEBUG] update_employee - Request for employee_id: {employee_id}")
    print(f"[DEBUG] update_employee - User email: {user.get('email') if user else 'None'}")
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        print(f"[ERROR] update_employee - User not found in database")
        return jsonify({'error': 'User not found'}), 404
    
    # Explicitly deny employees from updating other employees
    if db_user.role == 'employee' or db_user.user_type == 'employee':
        print(f"[ERROR] update_employee - Employee attempted to update another employee: {db_user.email}")
        return jsonify({'error': 'Employees cannot update other employees'}), 403
    
    # Only admin and owner can update employees
    if db_user.role not in ['admin', 'owner']:
        print(f"[ERROR] update_employee - Permission denied for user: {db_user.role}/{db_user.user_type}")
        return jsonify({'error': 'You do not have permission to update employees'}), 403
    
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        print(f"[ERROR] update_employee - Employee not found: {employee_id}")
        return jsonify({'error': 'Employee not found'}), 404
    
    if employee.role != 'employee' and employee.user_type != 'employee':
        print(f"[ERROR] update_employee - User is not an employee: role={employee.role}, user_type={employee.user_type}")
        return jsonify({'error': 'User is not an employee'}), 400
    
    data = request.get_json()
    print(f"[DEBUG] update_employee - Update data: {data}")
    
    try:
        # Get or create employee profile
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        if not employee_profile:
            # Create profile if it doesn't exist
            employee_profile = EmployeeProfile(user_id=employee_id)
            db.session.add(employee_profile)
        
        # Update employee profile fields
        if 'first_name' in data:
            employee_profile.first_name = data['first_name']
        if 'last_name' in data:
            employee_profile.last_name = data['last_name']
        if 'phone' in data:
            employee_profile.phone = data['phone']
        if 'department' in data:
            employee_profile.department = data['department']
        if 'location' in data:
            employee_profile.location = data['location']
        if 'employment_type' in data:
            employee_profile.employment_type = data['employment_type']
        if 'category' in data:
            employee_profile.category = data['category']
        
        # Update salary information
        if 'salary_amount' in data:
            try:
                if data['salary_amount']:
                    employee_profile.salary_amount = Decimal(str(data['salary_amount']))
                else:
                    employee_profile.salary_amount = None
            except (ValueError, TypeError):
                pass  # Skip if invalid salary amount
        
        if 'salary_currency' in data:
            employee_profile.salary_currency = data['salary_currency'] or 'USD'
        
        if 'salary_type' in data:
            employee_profile.salary_type = data['salary_type'] or 'monthly'
        
        # Update hire date
        if 'hire_date' in data:
            try:
                if data['hire_date']:
                    employee_profile.hire_date = datetime.strptime(data['hire_date'], '%Y-%m-%d').date()
                else:
                    employee_profile.hire_date = None
            except (ValueError, TypeError):
                pass  # Skip if invalid date format
        
        # Update email if provided
        if 'email' in data and data['email']:
            new_email = data['email'].strip().lower()
            if new_email != employee.email:
                # Check if email is already taken by another user
                existing_user = User.query.filter_by(email=new_email).first()
                if existing_user and existing_user.id != employee.id:
                    return jsonify({'error': 'Email already exists'}), 400
                employee.email = new_email
        
        # Update organization/tenant_id if provided
        if 'organization_id' in data and data['organization_id']:
            org_id = data['organization_id']
            if org_id != "none" and str(org_id).strip():
                try:
                    org_id_int = int(org_id)
                    # Verify organization exists
                    org = Tenant.query.get(org_id_int)
                    if not org:
                        return jsonify({'error': 'Organization not found'}), 404
                    employee.tenant_id = org_id_int
                except (ValueError, TypeError):
                    return jsonify({'error': 'Invalid organization ID format'}), 400
        
        db.session.commit()
        
        print(f"[DEBUG] update_employee - Employee updated successfully:")
        print(f"[DEBUG]   - Employee ID: {employee.id}")
        print(f"[DEBUG]   - Email: {employee.email}")
        print(f"[DEBUG]   - Tenant ID: {employee.tenant_id}")
        
        # Generate invite link if requested
        invite_link = None
        email_sent = False
        if data.get('send_invite', False):
            print(f"[DEBUG] update_employee - Generating invite link for employee: {employee.email}")
            try:
                from botocore.exceptions import ClientError
                temp_password = None
                try:
                    print(f"[DEBUG] update_employee - Attempting to create/get user in Cognito: {employee.email}")
                    temp_password, _ = cognito_admin_create_user(employee.email, employee.tenant_id, role='employee')
                    print(f"[DEBUG] update_employee - Cognito user created successfully, temp_password: {'Yes' if temp_password else 'No'}")
                except ClientError as cognito_error:
                    error_code = cognito_error.response.get('Error', {}).get('Code', '')
                    print(f"[WARNING] update_employee - Cognito ClientError: {error_code} - {str(cognito_error)}")
                    
                    if error_code == 'UsernameExistsException':
                        print(f"[DEBUG] update_employee - User already exists in Cognito, generating temp password for invite link")
                        temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12)) + '1!A'
                        print(f"[DEBUG] update_employee - Generated temp password for invite link")
                    else:
                        import traceback
                        error_trace = traceback.format_exc()
                        print(f"[ERROR] update_employee - Cognito error: {error_code} - {str(cognito_error)}")
                        print(f"[ERROR] update_employee - Cognito error traceback: {error_trace}")
                
                if not temp_password:
                    print(f"[WARNING] update_employee - No temp password from Cognito, generating fallback")
                    temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12)) + '1!A'
                    print(f"[DEBUG] update_employee - Generated fallback temp password")
                
                # Generate invite link
                # Get frontend URL (local for development, production for production)
            def get_frontend_url():
                frontend_url = os.getenv('FRONTEND_URL')
                if frontend_url:
                    return frontend_url
                flask_env = os.getenv('FLASK_ENV', '').lower()
                flask_debug = os.getenv('FLASK_DEBUG', '').lower()
                is_development = (
                    flask_env == 'development' or 
                    flask_debug == 'true' or 
                    flask_debug == '1' or
                    os.getenv('ENVIRONMENT', '').lower() == 'development' or
                    os.getenv('ENV', '').lower() == 'development'
                )
                if is_development:
                    local_port = os.getenv('FRONTEND_PORT', '5173')
                    return f'http://localhost:{local_port}'
                else:
                    return 'https://kempian.ai'
            
            frontend_url = get_frontend_url()
                invite_link = f"{frontend_url}/invite?email={employee.email}&code={temp_password}&type=employee"
                
                print(f"[DEBUG] update_employee - Invite link generated: {invite_link}")
                
                # Send invite email
                try:
                    print(f"[DEBUG] update_employee - Attempting to send invite email to: {employee.email}")
                    email_sent = send_invite_email(employee.email, invite_link)
                    print(f"[DEBUG] update_employee - Email send result: {email_sent}")
                except Exception as email_error:
                    import traceback
                    error_trace = traceback.format_exc()
                    print(f"[ERROR] update_employee - Failed to send invite email: {str(email_error)}")
                    print(f"[ERROR] update_employee - Email error traceback: {error_trace}")
                    email_sent = False
            except Exception as invite_error:
                import traceback
                error_trace = traceback.format_exc()
                print(f"[ERROR] update_employee - Failed to generate invite: {str(invite_error)}")
                print(f"[ERROR] update_employee - Invite error traceback: {error_trace}")
        
        # Return updated employee data
        employee_data = {
            'id': employee.id,
            'email': employee.email,
            'role': employee.role,
            'user_type': employee.user_type,
            'tenant_id': employee.tenant_id,
            'company_name': employee.company_name,
            'created_at': employee.created_at.isoformat() if employee.created_at else None
        }
        
        if employee_profile:
            employee_data.update({
                'first_name': employee_profile.first_name,
                'last_name': employee_profile.last_name,
                'phone': employee_profile.phone,
                'department': employee_profile.department,
                'location': employee_profile.location,
                'employment_type': employee_profile.employment_type,
                'category': employee_profile.category,
                'salary_amount': float(employee_profile.salary_amount) if employee_profile.salary_amount else None,
                'salary_currency': employee_profile.salary_currency,
                'hire_date': employee_profile.hire_date.isoformat() if employee_profile.hire_date else None
            })
        
        response_data = {'employee': employee_data}
        if invite_link:
            response_data['invite_link'] = invite_link
            response_data['email_sent'] = email_sent
        
        return jsonify(response_data), 200
        
    except Exception as e:
        db.session.rollback()
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] update_employee - Failed to update employee: {str(e)}")
        print(f"[ERROR] update_employee - Error traceback: {error_trace}")
        return jsonify({'error': f'Failed to update employee: {str(e)}'}), 500


@hr_employees_bp.route('/<int:employee_id>', methods=['DELETE'], strict_slashes=False)
def delete_employee(employee_id):
    """Delete an employee - for admin and owner only"""
    user, error = _auth_or_401()
    if error:
        return error
    
    print(f"[DEBUG] delete_employee - Request for employee_id: {employee_id}")
    print(f"[DEBUG] delete_employee - User email: {user.get('email') if user else 'None'}")
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        print(f"[ERROR] delete_employee - User not found in database")
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin and owner can delete employees
    if db_user.role not in ['admin', 'owner']:
        print(f"[ERROR] delete_employee - Permission denied for user: {db_user.role}/{db_user.user_type}")
        return jsonify({'error': 'You do not have permission to delete employees'}), 403
    
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        print(f"[ERROR] delete_employee - Employee not found: {employee_id}")
        return jsonify({'error': 'Employee not found'}), 404
    
    if employee.role != 'employee' and employee.user_type != 'employee':
        print(f"[ERROR] delete_employee - User is not an employee: role={employee.role}, user_type={employee.user_type}")
        return jsonify({'error': 'User is not an employee'}), 400
    
    try:
        # Delete employee profile first
        employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
        if employee_profile:
            db.session.delete(employee_profile)
        
        # Delete related records that might have foreign key constraints
        from app.models import Timesheet, Payslip, UserBankAccount
        
        # Delete timesheets
        timesheets = Timesheet.query.filter_by(user_id=employee_id).all()
        for timesheet in timesheets:
            db.session.delete(timesheet)
        
        # Set approved_by to NULL for timesheets where this user is the approver
        timesheets_approved = Timesheet.query.filter_by(approved_by=employee_id).all()
        for timesheet in timesheets_approved:
            timesheet.approved_by = None
        
        # Delete payslips
        payslips = Payslip.query.filter_by(employee_id=employee_id).all()
        for payslip in payslips:
            db.session.delete(payslip)
        
        # Delete bank account
        bank_account = UserBankAccount.query.filter_by(user_id=employee_id).first()
        if bank_account:
            db.session.delete(bank_account)
        
        # Delete the user
        db.session.delete(employee)
        db.session.commit()
        
        print(f"[DEBUG] delete_employee - Employee {employee_id} deleted successfully")
        return jsonify({'message': 'Employee deleted successfully'}), 200
            
    except Exception as e:
        db.session.rollback()
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] delete_employee - Failed to delete employee: {str(e)}")
        print(f"[ERROR] delete_employee - Error traceback: {error_trace}")
        return jsonify({'error': f'Failed to delete employee: {str(e)}'}), 500


@hr_employees_bp.route('/<int:employee_id>/bank-account', methods=['GET'], strict_slashes=False)
def get_employee_bank_account(employee_id):
    """Get bank account for a specific employee - for admin and owner"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, employer, recruiter, and owner can view employee bank accounts
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view employee bank accounts'}), 403
    
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    if employee.role != 'employee' and employee.user_type != 'employee':
        return jsonify({'error': 'User is not an employee'}), 400
    
    # Get bank account
    account = UserBankAccount.query.filter_by(user_id=employee_id).first()
    if not account:
        return jsonify({'data': None}), 200
    
    # Serialize account data
    account_data = {
        'id': account.id,
        'accountHolderName': account.account_holder_name,
        'bankName': account.bank_name,
        'routingNumber': account.routing_number,
        'accountNumber': account.account_number,
        'contactEmail': account.contact_email,
        'contactPhone': account.contact_phone,
        'payFrequency': account.pay_frequency,
        'directDepositOrCheckPreference': account.direct_deposit_or_check_preference,
    }
    
    return jsonify({'data': account_data}), 200


@hr_employees_bp.route('/<int:employee_id>/verify-bank-account', methods=['POST'], strict_slashes=False)
def verify_bank_account(employee_id):
    """
    Verify employee bank account using penny-drop (END-TO-END)
    
    BEHAVIOR:
    - Triggers penny-drop verification via Razorpay
    - Stores verification_reference_id, verification_status, verified_at
    - Enforces: No payroll if verification failed
    - Enforces: 72-hour cooldown after bank change
    - Returns human-readable failure reasons
    
    Returns:
    {
        "verified": bool,
        "verification_id": string,
        "account_name_match": bool,
        "error": string (if failed),
        "cooldown_active": bool,
        "cooldown_until": datetime (if cooldown active)
    }
    """
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Check permissions
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to verify bank accounts'}), 403
    
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    # Check tenant
    tenant_id = get_current_tenant_id()
    if employee.tenant_id != tenant_id and db_user.role not in ['admin', 'owner']:
        return jsonify({'error': 'You do not have permission to verify this employee\'s bank account'}), 403
    
    # Get bank account
    bank_account = UserBankAccount.query.filter_by(user_id=employee_id).first()
    if not bank_account:
        return jsonify({'error': 'Bank account not found for this employee'}), 404
    
    # Check 72-hour cooldown
    if bank_account.bank_change_cooldown_until:
        cooldown_until = bank_account.bank_change_cooldown_until
        if datetime.utcnow() < cooldown_until:
            hours_remaining = (cooldown_until - datetime.utcnow()).total_seconds() / 3600
            return jsonify({
                'verified': False,
                'error': f'Bank account changed recently. 72-hour cooldown period active. Please wait {hours_remaining:.1f} more hours.',
                'cooldown_active': True,
                'cooldown_until': cooldown_until.isoformat(),
                'hours_remaining': round(hours_remaining, 2)
            }), 400
    
    data = request.get_json() or {}
    account_number = data.get('account_number') or bank_account.account_number
    ifsc_code = data.get('ifsc_code') or bank_account.ifsc_code
    account_holder_name = data.get('account_holder_name') or bank_account.account_holder_name
    
    if not account_number or not ifsc_code or not account_holder_name:
        return jsonify({'error': 'Missing required bank account details: account_number, ifsc_code, and account_holder_name are required'}), 400
    
    try:
        from app.services.penny_drop_verification import PennyDropVerification
        
        verification_service = PennyDropVerification(tenant_id=tenant_id)
        result = verification_service.verify_account(
            employee_id=employee_id,
            account_number=account_number,
            ifsc_code=ifsc_code,
            account_holder_name=account_holder_name
        )
        
        if result['verified']:
            # Update bank account with verification details
            bank_account.verified_by_penny_drop = True
            bank_account.verification_reference_id = result.get('verification_id')
            bank_account.verification_date = datetime.utcnow()
            bank_account.consent_given_at = datetime.utcnow()
            bank_account.consent_ip = request.remote_addr
            bank_account.last_updated_by = db_user.id
            # Clear cooldown if verification successful
            bank_account.bank_change_cooldown_until = None
            
            db.session.commit()
            
            # Audit log
            from app.utils.payment_security import PaymentAuditLogger
            audit_logger = PaymentAuditLogger()
            audit_logger.log_security_event('bank_account_verified', {
                'employee_id': employee_id,
                'verification_id': result.get('verification_id'),
                'account_name_match': result.get('account_name_match', False)
            }, db_user.id)
            
            return jsonify({
                'verified': True,
                'verification_id': result.get('verification_id'),
                'account_name_match': result.get('account_name_match', False),
                'verification_date': bank_account.verification_date.isoformat(),
                'message': 'Bank account verified successfully via penny-drop',
                'cooldown_active': False
            }), 200
        else:
            # Store verification attempt (even if failed)
            bank_account.verification_reference_id = result.get('verification_id')
            bank_account.verification_date = datetime.utcnow()
            bank_account.consent_given_at = datetime.utcnow()
            bank_account.consent_ip = request.remote_addr
            bank_account.last_updated_by = db_user.id
            
            db.session.commit()
            
            # Human-readable error messages
            error_msg = result.get('error', 'Verification failed')
            human_readable_error = error_msg
            
            # Map common Razorpay errors to human-readable messages
            if 'invalid' in error_msg.lower() or 'not found' in error_msg.lower():
                human_readable_error = 'Invalid bank account details. Please verify account number and IFSC code.'
            elif 'network' in error_msg.lower() or 'timeout' in error_msg.lower():
                human_readable_error = 'Network error during verification. Please try again.'
            elif 'unauthorized' in error_msg.lower() or 'authentication' in error_msg.lower():
                human_readable_error = 'Payment gateway authentication failed. Please contact support.'
            
            return jsonify({
                'verified': False,
                'error': human_readable_error,
                'technical_error': error_msg,
                'account_name_match': result.get('account_name_match', False),
                'verification_id': result.get('verification_id'),
                'cooldown_active': False
            }), 400
            
    except ValueError as e:
        # Service-level validation errors
        return jsonify({
            'verified': False,
            'error': str(e),
            'cooldown_active': False
        }), 400
    except Exception as e:
        logger.error(f"Error verifying bank account for employee {employee_id}: {str(e)}")
        return jsonify({
            'verified': False,
            'error': 'An unexpected error occurred during verification. Please try again or contact support.',
            'technical_error': str(e) if logger.level <= 10 else None,  # Only in debug mode
            'cooldown_active': False
        }), 500


@hr_employees_bp.route('/bank-accounts', methods=['GET'], strict_slashes=False)
def get_all_employee_bank_accounts():
    """Get bank accounts for all employees - for admin and owner"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, employer, recruiter, and owner can view employee bank accounts
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to view employee bank accounts'}), 403
    
    tenant_id = get_current_tenant_id()
    
    # Build query for employees
    from sqlalchemy.orm import joinedload
    from app.models import OrganizationMetadata
    
    query = User.query.filter((User.user_type == 'employee') | (User.role == 'employee'))
    
    # Filter by tenant_id based on user role
    if tenant_id:
        if db_user.role in ['owner', 'admin']:
            # For owners/admins: show employees from their current org AND any orgs they created
            created_org_ids = db.session.query(OrganizationMetadata.tenant_id).filter_by(created_by=db_user.id).all()
            created_org_ids = [org[0] for org in created_org_ids]
            allowed_tenant_ids = [tenant_id] + created_org_ids
            query = query.filter(User.tenant_id.in_(allowed_tenant_ids))
        else:
            query = query.filter_by(tenant_id=tenant_id)
    
    employees = query.all()
    
    # Get bank accounts for all employees
    results = []
    for emp in employees:
        account = UserBankAccount.query.filter_by(user_id=emp.id).first()
        employee_data = {
            'id': emp.id,
            'email': emp.email,
            'first_name': emp.employee_profile.first_name if emp.employee_profile else None,
            'last_name': emp.employee_profile.last_name if emp.employee_profile else None,
            'bank_account': None
        }
        
        if account:
            employee_data['bank_account'] = {
                'id': account.id,
                'accountHolderName': account.account_holder_name,
                'bankName': account.bank_name,
                'routingNumber': account.routing_number,
                'accountNumber': account.account_number,
                'contactEmail': account.contact_email,
                'contactPhone': account.contact_phone,
                'payFrequency': account.pay_frequency,
                'directDepositOrCheckPreference': account.direct_deposit_or_check_preference,
            }
        
        results.append(employee_data)
    
    return jsonify({'employees': results}), 200


