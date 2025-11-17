from flask import Blueprint, jsonify, request, send_file
from app.auth_utils import get_current_user_flexible, get_current_user, get_current_tenant_id
from app.models import User, Timesheet, EmployeeProfile, db, OrganizationMetadata
from datetime import datetime, date, timedelta
from decimal import Decimal
import csv
import io


hr_timesheets_bp = Blueprint('hr_timesheets', __name__)


def _auth_or_401():
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get('email'):
        return None, (jsonify({'error': 'Authentication required'}), 401)
    return user, None


def get_week_dates(target_date):
    """Get start and end dates of the week for a given date"""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    # Monday is 0, Sunday is 6
    days_since_monday = target_date.weekday()
    week_start = target_date - timedelta(days=days_since_monday)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end


@hr_timesheets_bp.route('/', methods=['GET'], strict_slashes=False)
def list_timesheets():
    """List timesheets - can filter by employee, date range, status"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Debug: Log user info
    print(f"[DEBUG] list_timesheets - User email: {user.get('email') if user else 'None'}")
    
    # Get user from database to check role and get organizations they created
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get query parameters
    employee_id = request.args.get('employee_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    status = request.args.get('status')
    week_start = request.args.get('week_start')
    
    # Build query
    query = Timesheet.query
    
    # IMPORTANT: If user is an employee, they can ONLY see their own timesheets
    if db_user.role == 'employee' or db_user.user_type == 'employee':
        print(f"[DEBUG] list_timesheets - Employee user detected, restricting to own timesheets only")
        # Force filter to only show timesheets for this employee
        query = query.filter(Timesheet.user_id == db_user.id)
    else:
        # Filter by tenant for non-employees
        tenant_id = get_current_tenant_id()
        print(f"[DEBUG] list_timesheets - Tenant ID: {tenant_id}")
        print(f"[DEBUG] list_timesheets - User role: {db_user.role}, user_type: {db_user.user_type}")
        
        if tenant_id:
            # For owners/admins: show timesheets from their current org AND any orgs they created
            if db_user.role in ['owner', 'admin']:
                # Get organization IDs that this user created
                created_org_ids = db.session.query(OrganizationMetadata.tenant_id).filter_by(created_by=db_user.id).all()
                created_org_ids = [org[0] for org in created_org_ids]
                
                # Include current tenant_id and all created organization IDs
                allowed_tenant_ids = [tenant_id] + created_org_ids
                print(f"[DEBUG] list_timesheets - Allowed tenant IDs (owner/admin): {allowed_tenant_ids}")
                # Explicitly join using the employee relationship (user_id foreign key)
                query = query.join(Timesheet.employee).filter(User.tenant_id.in_(allowed_tenant_ids))
            else:
                # For other roles: only show timesheets from their current organization
                print(f"[DEBUG] list_timesheets - Filtering by tenant_id: {tenant_id}")
                query = query.join(Timesheet.employee).filter(User.tenant_id == tenant_id)
        else:
            print(f"[DEBUG] list_timesheets - No tenant_id filter applied")
        
        # Filter by employee (only for non-employees)
        if employee_id:
            query = query.filter(Timesheet.user_id == employee_id)
    
    # Filter by date range
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            query = query.filter(Timesheet.date >= start)
        except ValueError:
            pass
    
    if end_date:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            query = query.filter(Timesheet.date <= end)
        except ValueError:
            pass
    
    # Filter by week
    if week_start:
        try:
            week_start_date = datetime.strptime(week_start, '%Y-%m-%d').date()
            week_end_date = week_start_date + timedelta(days=6)
            query = query.filter(
                Timesheet.week_start_date == week_start_date,
                Timesheet.week_end_date == week_end_date
            )
        except ValueError:
            pass
    
    # Filter by status
    if status:
        query = query.filter(Timesheet.status == status)
    
    # Order by date descending
    timesheets = query.order_by(Timesheet.date.desc(), Timesheet.created_at.desc()).all()
    
    print(f"[DEBUG] list_timesheets - Found {len(timesheets)} timesheets")
    for idx, ts in enumerate(timesheets):
        print(f"[DEBUG] Timesheet {idx + 1}: id={ts.id}, employee_id={ts.user_id}, date={ts.date}, status={ts.status}")
    
    results = [ts.to_dict() for ts in timesheets]
    print(f"[DEBUG] list_timesheets - Returning {len(results)} timesheet results")
    return jsonify({'timesheets': results}), 200


@hr_timesheets_bp.route('/', methods=['POST'], strict_slashes=False)
def create_timesheet():
    """Create a new timesheet entry"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    
    # Required fields
    employee_id = data.get('employee_id') or data.get('user_id')
    if not employee_id:
        return jsonify({'error': 'Employee ID is required'}), 400
    
    period_type = data.get('period_type', 'weekly')
    pay_period_start_str = data.get('pay_period_start')
    pay_period_end_str = data.get('pay_period_end')

    timesheet_date_str = data.get('date')

    if period_type == 'monthly':
        if not pay_period_start_str or not pay_period_end_str:
            return jsonify({'error': 'Pay period start and end are required for monthly timesheets'}), 400
        try:
            pay_period_start_date = datetime.strptime(pay_period_start_str, '%Y-%m-%d').date()
            pay_period_end_date = datetime.strptime(pay_period_end_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid pay period date format. Use YYYY-MM-DD'}), 400

        if pay_period_end_date < pay_period_start_date:
            return jsonify({'error': 'Pay period end must be on or after the start date'}), 400

        timesheet_date = pay_period_start_date
        week_start = None
        week_end = None
    elif period_type == 'weekly':
        # Weekly timesheet - 7 days
        if not timesheet_date_str and pay_period_start_str:
            timesheet_date_str = pay_period_start_str

        if not timesheet_date_str:
            return jsonify({'error': 'Week start date is required'}), 400

        try:
            timesheet_date = datetime.strptime(timesheet_date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

        # Calculate week start (Monday) and week end (Sunday) - 7 days
        week_start, week_end = get_week_dates(timesheet_date)
        
        # Use week dates as pay period for weekly timesheets
        pay_period_start_date = week_start
        pay_period_end_date = week_end
        
        # Set timesheet_date to week_start for consistency
        timesheet_date = week_start
    else:
        # Legacy daily support (kept for backward compatibility)
        if not timesheet_date_str and pay_period_start_str:
            timesheet_date_str = pay_period_start_str

        if not timesheet_date_str:
            return jsonify({'error': 'Date is required'}), 400

        try:
            timesheet_date = datetime.strptime(timesheet_date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

        try:
            pay_period_start_date = datetime.strptime(pay_period_start_str, '%Y-%m-%d').date() if pay_period_start_str else timesheet_date
        except ValueError:
            return jsonify({'error': 'Invalid pay period start format. Use YYYY-MM-DD'}), 400

        try:
            pay_period_end_date = datetime.strptime(pay_period_end_str, '%Y-%m-%d').date() if pay_period_end_str else timesheet_date
        except ValueError:
            return jsonify({'error': 'Invalid pay period end format. Use YYYY-MM-DD'}), 400

        if pay_period_end_date < pay_period_start_date:
            return jsonify({'error': 'Pay period end must be on or after the start date'}), 400

        week_start, week_end = get_week_dates(timesheet_date)
 
    # Get employee
    employee = User.query.get(employee_id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    
    if employee.role != 'employee' and employee.user_type != 'employee':
        return jsonify({'error': 'User is not an employee'}), 400
    
    # Permission checks
    organization_metadata = OrganizationMetadata.query.filter_by(tenant_id=employee.tenant_id if employee else None).first() if employee else None

    if db_user.role == 'admin':
        has_permission = True
    elif db_user.role == 'owner':
        has_permission = organization_metadata is not None and organization_metadata.created_by == db_user.id
    elif db_user.role == 'employee' or db_user.user_type == 'employee':
        has_permission = (db_user.id == employee_id)
    else:
        has_permission = False

    if not has_permission:
        return jsonify({'error': 'You do not have permission to create this timesheet'}), 403

    # Get employee profile for rates
    employee_profile = EmployeeProfile.query.filter_by(user_id=employee_id).first()
    
    # Calculate hourly rate from salary if not provided
    regular_rate = data.get('regular_rate')
    if not regular_rate and employee_profile and employee_profile.salary_amount:
        salary_amount = float(employee_profile.salary_amount)
        salary_type = (employee_profile.salary_type or 'monthly').lower()
        if salary_type == 'hourly':
            regular_rate = salary_amount
        elif salary_type == 'yearly':
            # Assume 40 hours/week, 52 weeks/year
            regular_rate = salary_amount / (40 * 52)
        else:
            # Default monthly -> convert to hourly (40 hours/week, 4.33 weeks/month)
            regular_rate = salary_amount / (40 * 4.33)
    elif not regular_rate:
        regular_rate = 0
    
    # Hours
    regular_hours = Decimal(str(data.get('regular_hours', 0)))
    overtime_hours = Decimal(str(data.get('overtime_hours', 0)))
    holiday_hours = Decimal(str(data.get('holiday_hours', 0)))
    is_holiday = data.get('is_holiday', False)
    
    # Auto-calculate overtime if hours exceed 8 per day or 40 per week
    if regular_hours > 8:
        excess = regular_hours - 8
        overtime_hours += excess
        regular_hours = 8
    
    # Check weekly hours if needed
    if period_type != 'monthly':
        week_timesheets = Timesheet.query.filter(
            Timesheet.user_id == employee_id,
            Timesheet.week_start_date == week_start,
            Timesheet.status.in_(['draft', 'submitted', 'approved'])
        ).all()

        week_total = sum(float(ts.regular_hours or 0) + float(ts.overtime_hours or 0) for ts in week_timesheets)
        week_total += float(regular_hours) + float(overtime_hours)

        if week_total > 40:
            excess_weekly = Decimal(str(week_total - 40))
            if regular_hours > 0:
                if regular_hours >= excess_weekly:
                    regular_hours -= excess_weekly
                    overtime_hours += excess_weekly
                else:
                    overtime_hours += regular_hours
                    regular_hours = 0
    
    # If marked as holiday, move hours to holiday
    if is_holiday:
        holiday_hours += regular_hours + overtime_hours
        regular_hours = 0
        overtime_hours = 0
    
    total_hours = regular_hours + overtime_hours + holiday_hours
    
    # Rates
    overtime_rate = data.get('overtime_rate') or (float(regular_rate) * 1.5)
    holiday_rate = data.get('holiday_rate') or (float(regular_rate) * 2.0)
    
    # Bonus
    bonus_amount = Decimal(str(data.get('bonus_amount', 0)))
    
    # Create timesheet
    timesheet = Timesheet(
        user_id=employee_id,
        employee_profile_id=employee_profile.id if employee_profile else None,
        date=timesheet_date,
        week_start_date=week_start,
        week_end_date=week_end,
        pay_period_start=pay_period_start_date,
        pay_period_end=pay_period_end_date,
        regular_hours=regular_hours,
        overtime_hours=overtime_hours,
        holiday_hours=holiday_hours,
        total_hours=total_hours,
        regular_rate=Decimal(str(regular_rate)),
        overtime_rate=Decimal(str(overtime_rate)),
        holiday_rate=Decimal(str(holiday_rate)),
        bonus_amount=bonus_amount,
        status=data.get('status', 'draft'),
        notes=data.get('notes')
    )
    
    # Calculate earnings
    timesheet.calculate_earnings()
    
    db.session.add(timesheet)
    db.session.commit()
    
    print(f"[DEBUG] create_timesheet - Timesheet created successfully:")
    print(f"[DEBUG]   - Timesheet ID: {timesheet.id}")
    print(f"[DEBUG]   - Employee ID: {timesheet.user_id}")
    print(f"[DEBUG]   - Date: {timesheet.date}")
    print(f"[DEBUG]   - Status: {timesheet.status}")
    print(f"[DEBUG]   - Total Hours: {timesheet.total_hours}")
    print(f"[DEBUG]   - Employee tenant_id: {employee.tenant_id}")
    
    return jsonify({'timesheet': timesheet.to_dict()}), 201


@hr_timesheets_bp.route('/<int:timesheet_id>', methods=['GET'])
def get_timesheet(timesheet_id):
    """Get a specific timesheet"""
    user, error = _auth_or_401()
    if error:
        return error
    
    timesheet = Timesheet.query.get(timesheet_id)
    if not timesheet:
        return jsonify({'error': 'Timesheet not found'}), 404
    
    return jsonify({'timesheet': timesheet.to_dict()}), 200


@hr_timesheets_bp.route('/<int:timesheet_id>', methods=['PUT'])
def update_timesheet(timesheet_id):
    """Update a timesheet"""
    user, error = _auth_or_401()
    if error:
        return error
 
    timesheet = Timesheet.query.get(timesheet_id)
    if not timesheet:
        return jsonify({'error': 'Timesheet not found'}), 404
 
    # Only allow updates if status is draft or submitted
    if timesheet.status in ['approved', 'paid']:
        return jsonify({'error': 'Cannot update approved or paid timesheet'}), 400
 
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404

    # Explicitly deny employees from updating timesheets
    if db_user.role == 'employee' or db_user.user_type == 'employee':
        print(f"[ERROR] update_timesheet - Employee attempted to update timesheet: {db_user.email}")
        return jsonify({'error': 'Employees cannot update timesheets. You can only create new timesheets.'}), 403

    employee = User.query.get(timesheet.user_id)
    organization_metadata = OrganizationMetadata.query.filter_by(
        tenant_id=employee.tenant_id if employee else None
    ).first() if employee else None

    if db_user.role == 'admin':
        has_permission = True
    elif db_user.role == 'owner':
        has_permission = organization_metadata is not None and organization_metadata.created_by == db_user.id
    else:
        has_permission = False

    if not has_permission:
        return jsonify({'error': 'You do not have permission to update this timesheet'}), 403

    data = request.get_json()
 
    # Update fields
    if 'date' in data:
        try:
            timesheet.date = datetime.strptime(data['date'], '%Y-%m-%d').date()
            week_start, week_end = get_week_dates(timesheet.date)
            timesheet.week_start_date = week_start
            timesheet.week_end_date = week_end
        except ValueError:
            return jsonify({'error': 'Invalid date format'}), 400
    
    if 'regular_hours' in data:
        timesheet.regular_hours = Decimal(str(data['regular_hours']))
    if 'overtime_hours' in data:
        timesheet.overtime_hours = Decimal(str(data['overtime_hours']))
    if 'holiday_hours' in data:
        timesheet.holiday_hours = Decimal(str(data['holiday_hours']))
    if 'regular_rate' in data:
        timesheet.regular_rate = Decimal(str(data['regular_rate']))
    if 'overtime_rate' in data:
        timesheet.overtime_rate = Decimal(str(data['overtime_rate']))
    if 'holiday_rate' in data:
        timesheet.holiday_rate = Decimal(str(data['holiday_rate']))
    if 'bonus_amount' in data:
        timesheet.bonus_amount = Decimal(str(data['bonus_amount']))
    if 'notes' in data:
        timesheet.notes = data['notes']
    if 'pay_period_start' in data:
        if data['pay_period_start']:
            try:
                timesheet.pay_period_start = datetime.strptime(data['pay_period_start'], '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'error': 'Invalid pay period start format. Use YYYY-MM-DD'}), 400
        else:
            timesheet.pay_period_start = None
    if 'pay_period_end' in data:
        if data['pay_period_end']:
            try:
                timesheet.pay_period_end = datetime.strptime(data['pay_period_end'], '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'error': 'Invalid pay period end format. Use YYYY-MM-DD'}), 400
        else:
            timesheet.pay_period_end = None

    if timesheet.pay_period_start and timesheet.pay_period_end and timesheet.pay_period_end < timesheet.pay_period_start:
        return jsonify({'error': 'Pay period end must be on or after the start date'}), 400

    period_type_update = data.get('period_type')
    is_monthly = period_type_update == 'monthly' or (
        timesheet.pay_period_start
        and timesheet.pay_period_end
        and timesheet.pay_period_end != timesheet.pay_period_start
    )

    if is_monthly:
        timesheet.date = timesheet.pay_period_start or timesheet.date
        timesheet.week_start_date = None
        timesheet.week_end_date = None
    elif 'date' in data:
        # If switching back to daily ensure week context recalculated
        week_start, week_end = get_week_dates(timesheet.date)
        timesheet.week_start_date = week_start
        timesheet.week_end_date = week_end
 
    if 'status' in data:
        timesheet.status = data['status']
        if data['status'] == 'submitted' and not timesheet.submitted_at:
            timesheet.submitted_at = datetime.utcnow()
    
    # Recalculate total hours and earnings
    timesheet.total_hours = (timesheet.regular_hours or 0) + (timesheet.overtime_hours or 0) + (timesheet.holiday_hours or 0)
    timesheet.calculate_earnings()
    
    db.session.commit()
    
    return jsonify({'timesheet': timesheet.to_dict()}), 200


@hr_timesheets_bp.route('/<int:timesheet_id>', methods=['DELETE'])
def delete_timesheet(timesheet_id):
    """Delete a timesheet"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Explicitly deny employees from deleting timesheets
    if db_user.role == 'employee' or db_user.user_type == 'employee':
        print(f"[ERROR] delete_timesheet - Employee attempted to delete timesheet: {db_user.email}")
        return jsonify({'error': 'Employees cannot delete timesheets'}), 403
    
    timesheet = Timesheet.query.get(timesheet_id)
    if not timesheet:
        return jsonify({'error': 'Timesheet not found'}), 404
    
    # Check permissions - owners/admins can delete any timesheet, others need additional checks
    employee = User.query.get(timesheet.user_id)
    organization_metadata = OrganizationMetadata.query.filter_by(
        tenant_id=employee.tenant_id if employee else None
    ).first() if employee else None

    if db_user.role == 'admin':
        has_permission = True
    elif db_user.role == 'owner':
        has_permission = organization_metadata is not None and organization_metadata.created_by == db_user.id
    else:
        has_permission = False

    if not has_permission:
        return jsonify({'error': 'You do not have permission to delete this timesheet'}), 403
    
    # Only allow deletion if status is draft (for safety, but admins/owners can override if needed)
    # For now, we'll allow deletion of draft timesheets only
    if timesheet.status != 'draft':
        return jsonify({'error': 'Can only delete draft timesheets'}), 400
    
    db.session.delete(timesheet)
    db.session.commit()
    
    print(f"[DEBUG] delete_timesheet - Timesheet {timesheet_id} deleted by {db_user.email}")
    return jsonify({'message': 'Timesheet deleted successfully'}), 200


@hr_timesheets_bp.route('/<int:timesheet_id>/approve', methods=['POST'])
def approve_timesheet(timesheet_id):
    """Approve a timesheet"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    timesheet = Timesheet.query.get(timesheet_id)
    if not timesheet:
        return jsonify({'error': 'Timesheet not found'}), 404

    employee = User.query.get(timesheet.user_id)
    organization_metadata = OrganizationMetadata.query.filter_by(tenant_id=employee.tenant_id if employee else None).first() if employee else None

    if db_user.role == 'admin':
        has_permission = True
    elif db_user.role == 'owner':
        has_permission = organization_metadata is not None and organization_metadata.created_by == db_user.id
    else:
        has_permission = False

    if not has_permission:
        return jsonify({'error': 'You do not have permission to approve timesheets'}), 403

    if timesheet.status != 'submitted':
        return jsonify({'error': 'Can only approve submitted timesheets'}), 400
    
    timesheet.status = 'approved'
    timesheet.approved_by = db_user.id
    timesheet.approved_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({'timesheet': timesheet.to_dict()}), 200


@hr_timesheets_bp.route('/<int:timesheet_id>/reject', methods=['POST'])
def reject_timesheet(timesheet_id):
    """Reject a timesheet"""
    user, error = _auth_or_401()
    if error:
        return error
    
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    timesheet = Timesheet.query.get(timesheet_id)
    if not timesheet:
        return jsonify({'error': 'Timesheet not found'}), 404

    employee = User.query.get(timesheet.user_id)
    organization_metadata = OrganizationMetadata.query.filter_by(tenant_id=employee.tenant_id if employee else None).first() if employee else None

    if db_user.role == 'admin':
        has_permission = True
    elif db_user.role == 'owner':
        has_permission = organization_metadata is not None and organization_metadata.created_by == db_user.id
    else:
        has_permission = False

    if not has_permission:
        return jsonify({'error': 'You do not have permission to reject timesheets'}), 403

    if timesheet.status != 'submitted':
        return jsonify({'error': 'Can only reject submitted timesheets'}), 400
    
    data = request.get_json() or {}
    rejection_notes = data.get('notes', '')
    
    timesheet.status = 'rejected'
    timesheet.notes = (timesheet.notes or '') + f'\nRejected: {rejection_notes}' if rejection_notes else timesheet.notes
    
    db.session.commit()
    
    return jsonify({'timesheet': timesheet.to_dict()}), 200


@hr_timesheets_bp.route('/summary', methods=['GET'])
def get_timesheet_summary():
    """Get timesheet summary for a period"""
    user, error = _auth_or_401()
    if error:
        return error
    
    employee_id = request.args.get('employee_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    week_start = request.args.get('week_start')
    
    if not (start_date and end_date) and not week_start:
        return jsonify({'error': 'Date range or week_start is required'}), 400

    valid_statuses = ['draft', 'submitted', 'approved', 'paid', 'rejected']
    query = Timesheet.query.filter(Timesheet.status.in_(valid_statuses))
    
    if employee_id:
        query = query.filter(Timesheet.user_id == employee_id)
    
    if week_start:
        try:
            week_start_date = datetime.strptime(week_start, '%Y-%m-%d').date()
            week_end_date = week_start_date + timedelta(days=6)
            query = query.filter(
                Timesheet.week_start_date == week_start_date,
                Timesheet.week_end_date == week_end_date
            )
        except ValueError:
            return jsonify({'error': 'Invalid week_start format'}), 400
    elif start_date and end_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            query = query.filter(Timesheet.date >= start, Timesheet.date <= end)
        except ValueError:
            return jsonify({'error': 'Invalid date format'}), 400
    
    tenant_id = get_current_tenant_id()
    if tenant_id:
        employee_ids = [u.id for u in User.query.filter_by(tenant_id=tenant_id).all()]
        if employee_ids:
            query = query.filter(Timesheet.user_id.in_(employee_ids))
        else:
            # No employees for this tenant - return empty summary
            return jsonify({'summary': {
                'total_regular_hours': 0,
                'total_overtime_hours': 0,
                'total_holiday_hours': 0,
                'total_hours': 0,
                'total_regular_earnings': 0,
                'total_overtime_earnings': 0,
                'total_holiday_earnings': 0,
                'total_bonus': 0,
                'total_earnings': 0,
                'timesheet_count': 0,
                'status_counts': {},
                'pending_approval_count': 0,
                'approved_count': 0,
                'paid_count': 0,
                'rejected_count': 0,
                'draft_count': 0,
                'unique_employee_count': 0
            }}), 200

    timesheets = query.all()

    status_counts = {}
    unique_employee_ids = set()
    for ts in timesheets:
        status = (ts.status or 'unknown').lower()
        status_counts[status] = status_counts.get(status, 0) + 1
        if ts.user_id:
            unique_employee_ids.add(ts.user_id)

    summary = {
        'total_regular_hours': sum(float(ts.regular_hours or 0) for ts in timesheets),
        'total_overtime_hours': sum(float(ts.overtime_hours or 0) for ts in timesheets),
        'total_holiday_hours': sum(float(ts.holiday_hours or 0) for ts in timesheets),
        'total_hours': sum(float(ts.total_hours or 0) for ts in timesheets),
        'total_regular_earnings': sum(float(ts.regular_earnings or 0) for ts in timesheets),
        'total_overtime_earnings': sum(float(ts.overtime_earnings or 0) for ts in timesheets),
        'total_holiday_earnings': sum(float(ts.holiday_earnings or 0) for ts in timesheets),
        'total_bonus': sum(float(ts.bonus_amount or 0) for ts in timesheets),
        'total_earnings': sum(float(ts.total_earnings or 0) for ts in timesheets),
        'timesheet_count': len(timesheets),
        'status_counts': status_counts,
        'pending_approval_count': status_counts.get('submitted', 0),
        'approved_count': status_counts.get('approved', 0),
        'paid_count': status_counts.get('paid', 0),
        'rejected_count': status_counts.get('rejected', 0),
        'draft_count': status_counts.get('draft', 0),
        'unique_employee_count': len(unique_employee_ids)
    }

    return jsonify({'summary': summary}), 200


@hr_timesheets_bp.route('/export', methods=['GET'])
def export_timesheets():
    """Export timesheets to CSV"""
    user, error = _auth_or_401()
    if error:
        return error
    
    # Get user from database to check role
    db_user = User.query.filter_by(email=user.get('email')).first()
    if not db_user:
        return jsonify({'error': 'User not found'}), 404
    
    # Only admin, employer, recruiter, and owner can export
    if db_user.role not in ['admin', 'owner'] and db_user.user_type not in ['employer', 'recruiter', 'admin']:
        return jsonify({'error': 'You do not have permission to export timesheets'}), 403
    
    # Get filters
    employee_id = request.args.get('employee_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    status = request.args.get('status')
    
    # Build query
    query = Timesheet.query
    tenant_id = get_current_tenant_id()
    
    if tenant_id:
        # Filter by tenant employees
        employee_ids = [u.id for u in User.query.filter_by(tenant_id=tenant_id).all()]
        query = query.filter(Timesheet.user_id.in_(employee_ids))
    
    if employee_id:
        query = query.filter_by(user_id=employee_id)
    if start_date:
        query = query.filter(Timesheet.date >= datetime.strptime(start_date, '%Y-%m-%d').date())
    if end_date:
        query = query.filter(Timesheet.date <= datetime.strptime(end_date, '%Y-%m-%d').date())
    if status:
        query = query.filter_by(status=status)
    
    timesheets = query.order_by(Timesheet.date.desc()).all()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'ID', 'Employee ID', 'Employee Email', 'Date', 'Regular Hours', 'Overtime Hours',
        'Holiday Hours', 'Total Hours', 'Regular Earnings', 'Overtime Earnings',
        'Holiday Earnings', 'Bonus', 'Total Earnings', 'Status', 'Notes', 'Created At'
    ])
    
    # Write data
    for ts in timesheets:
        employee = User.query.get(ts.user_id)
        writer.writerow([
            ts.id,
            ts.user_id,
            employee.email if employee else '',
            ts.date.isoformat() if ts.date else '',
            float(ts.regular_hours) if ts.regular_hours else 0,
            float(ts.overtime_hours) if ts.overtime_hours else 0,
            float(ts.holiday_hours) if ts.holiday_hours else 0,
            float(ts.total_hours) if ts.total_hours else 0,
            float(ts.regular_earnings) if ts.regular_earnings else 0,
            float(ts.overtime_earnings) if ts.overtime_earnings else 0,
            float(ts.holiday_earnings) if ts.holiday_earnings else 0,
            float(ts.bonus_amount) if ts.bonus_amount else 0,
            float(ts.total_earnings) if ts.total_earnings else 0,
            ts.status,
            ts.notes or '',
            ts.created_at.isoformat() if ts.created_at else ''
        ])
    
    output.seek(0)
    
    # Create response
    response = send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'timesheets_export_{datetime.now().strftime("%Y%m%d")}.csv'
    )
    
    return response


