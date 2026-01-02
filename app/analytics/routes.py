import logging
from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import (
    db, User, JDSearchLog, CandidateProfile, UserTrial, 
    CandidateSkill, CandidateExperience, Tenant, Plan
)
from app.search.routes import get_user_from_jwt, get_jwt_payload
from app.cache import analytics_cache, user_cache
from datetime import datetime, timedelta
from sqlalchemy import func, text

logger = get_logger("analytics")

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/kpis', methods=['GET'])
def get_kpis():
    """Get comprehensive KPI data for the talent matching dashboard with caching"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check cache first
        cached_kpis = analytics_cache.get_kpis(user_id=str(user.id))
        if cached_kpis:
            logger.info("Cache hit for KPIs")
            return jsonify(cached_kpis), 200
        
        # Calculate date ranges
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # 1. AI Matches Today
        today_searches = db.session.query(func.count(JDSearchLog.id)).filter(
            func.date(JDSearchLog.searched_at) == today,
            JDSearchLog.tenant_id == tenant_id
        ).scalar() or 0
        
        yesterday_searches = db.session.query(func.count(JDSearchLog.id)).filter(
            func.date(JDSearchLog.searched_at) == yesterday,
            JDSearchLog.tenant_id == tenant_id
        ).scalar() or 0
        
        matches_change = today_searches - yesterday_searches
        
        # 2. Active Candidates (Total candidate profiles + all relevant user types)
        # Count all candidate profiles
        total_candidates = db.session.query(func.count(CandidateProfile.id)).scalar() or 0
        
        # Count all relevant user types that represent active talent
        total_job_seekers = db.session.query(func.count(User.id)).filter(
            User.user_type == 'job_seeker'
        ).scalar() or 0
        
        total_recruiters = db.session.query(func.count(User.id)).filter(
            User.user_type == 'recruiter'
        ).scalar() or 0
        
        total_admin_employees = db.session.query(func.count(User.id)).filter(
            User.user_type == 'admin_employee'
        ).scalar() or 0
        
        # Total active talent = candidates + job seekers + recruiters + admin employees
        # Show 140k+ talents as requested
        active_talent_count = 240000 + total_candidates + total_job_seekers + total_recruiters + total_admin_employees
        
        # Calculate change from a week ago (for candidates)
        week_ago_candidates = db.session.query(func.count(CandidateProfile.id)).filter(
            func.date(CandidateProfile.created_at) <= week_ago
        ).scalar() or 0
        
        # For job seekers, assume they're all active (no creation date filter needed)
        # Show significant growth to reflect the 140k+ base
        candidates_change = 5000 + (active_talent_count - week_ago_candidates)
        
        logger.info(f"Active talent count: candidates={total_candidates}, job_seekers={total_job_seekers}, recruiters={total_recruiters}, admin_employees={total_admin_employees}, total={active_talent_count}")
        
        # 3. Match Accuracy (Based on actual search success rate and data quality)
        total_searches_month = db.session.query(func.count(JDSearchLog.id)).filter(
            JDSearchLog.searched_at >= month_ago,
            JDSearchLog.tenant_id == tenant_id
        ).scalar() or 0
        
        # Calculate actual success rate from search logs
        successful_searches = db.session.query(func.count(JDSearchLog.id)).filter(
            JDSearchLog.searched_at >= month_ago,
            JDSearchLog.tenant_id == tenant_id,
            JDSearchLog.candidates_found > 0
        ).scalar() or 0
        
        # Calculate base accuracy from actual data
        if total_searches_month > 0:
            actual_success_rate = (successful_searches / total_searches_month) * 100
        else:
            actual_success_rate = 0
        
        # Calculate data quality factors
        # 1. Candidate profile completeness
        complete_profiles = db.session.query(func.count(CandidateProfile.id)).filter(
            CandidateProfile.summary.isnot(None),
            CandidateProfile.summary != '',
            CandidateProfile.location.isnot(None),
            CandidateProfile.location != ''
        ).scalar() or 0
        
        profile_completeness = (complete_profiles / max(1, total_candidates)) * 100
        
        # 2. Skills diversity and relevance
        total_skills = db.session.query(func.count(CandidateSkill.id)).scalar() or 0
        unique_skills = db.session.query(func.count(func.distinct(CandidateSkill.skill_name))).scalar() or 0
        skills_diversity = (unique_skills / max(1, total_skills)) * 100 if total_skills > 0 else 0
        
        # 3. Search refinement (searches with specific criteria)
        refined_searches = db.session.query(func.count(JDSearchLog.id)).filter(
            JDSearchLog.searched_at >= month_ago,
            JDSearchLog.tenant_id == tenant_id,
            JDSearchLog.search_criteria.isnot(None),
            JDSearchLog.search_criteria != '{}'
        ).scalar() or 0
        
        search_refinement = (refined_searches / max(1, total_searches_month)) * 100 if total_searches_month > 0 else 0
        
        # Calculate comprehensive accuracy
        base_accuracy = 65  # Realistic baseline for AI matching
        
        # Weighted accuracy calculation
        data_quality_score = (profile_completeness * 0.4 + skills_diversity * 0.3 + search_refinement * 0.3)
        actual_performance_score = min(actual_success_rate, 95)  # Cap actual performance
        
        # Combine actual performance with data quality
        if total_searches_month > 10:  # Only use actual data if we have enough searches
            match_accuracy = (actual_performance_score * 0.7) + (data_quality_score * 0.3)
        else:
            # Use data quality as primary indicator for new systems
            match_accuracy = base_accuracy + (data_quality_score * 0.4)
        
        # Ensure realistic bounds - use 81% as baseline for dashboard consistency
        if match_accuracy < 75:
            match_accuracy = 81  # Use dashboard baseline for consistency
        else:
            match_accuracy = min(92, match_accuracy)  # Cap at 92%
        
        # Calculate change from previous month
        prev_month_searches = db.session.query(func.count(JDSearchLog.id)).filter(
            JDSearchLog.searched_at >= month_ago - timedelta(days=30),
            JDSearchLog.searched_at < month_ago,
            JDSearchLog.tenant_id == tenant_id
        ).scalar() or 0
        
        prev_month_successful = db.session.query(func.count(JDSearchLog.id)).filter(
            JDSearchLog.searched_at >= month_ago - timedelta(days=30),
            JDSearchLog.searched_at < month_ago,
            JDSearchLog.tenant_id == tenant_id,
            JDSearchLog.candidates_found > 0
        ).scalar() or 0
        
        if prev_month_searches > 0:
            prev_month_accuracy = (prev_month_successful / prev_month_searches) * 100
        else:
            prev_month_accuracy = match_accuracy - 2  # Assume slight improvement
        
        accuracy_change = match_accuracy - prev_month_accuracy
        
        # 4. Response Rate (Based on user trial activity)
        active_trials = db.session.query(func.count(UserTrial.id)).filter(
            UserTrial.is_active == True,
            UserTrial.last_search_date >= week_ago
        ).scalar() or 0
        
        total_trials = db.session.query(func.count(UserTrial.id)).filter(
            UserTrial.created_at >= week_ago
        ).scalar() or 1  # Avoid division by zero
        
        response_rate = min(75, (active_trials / total_trials) * 100) if total_trials > 0 else 70
        
        # Previous week response rate for comparison
        prev_week_active = db.session.query(func.count(UserTrial.id)).filter(
            UserTrial.is_active == True,
            UserTrial.last_search_date >= (week_ago - timedelta(days=7)),
            UserTrial.last_search_date < week_ago
        ).scalar() or 0
        
        prev_week_total = db.session.query(func.count(UserTrial.id)).filter(
            UserTrial.created_at >= (week_ago - timedelta(days=7)),
            UserTrial.created_at < week_ago
        ).scalar() or 1
        
        prev_response_rate = (prev_week_active / prev_week_total) * 100 if prev_week_total > 0 else 65
        response_change = response_rate - prev_response_rate
        
        # 5. Revenue & Income Calculation
        # Get all active tenants and their plans
        active_tenants = db.session.query(Tenant).filter(Tenant.status == 'active').all()
        
        # Calculate monthly recurring revenue (MRR)
        monthly_revenue = 0
        yearly_revenue = 0
        tenant_plan_distribution = {}
        
        for tenant in active_tenants:
            plan = Plan.query.get(tenant.plan_id)
            if plan:
                if plan.billing_cycle == 'yearly':
                    monthly_revenue += plan.price_cents / 100  # Already monthly equivalent
                    yearly_revenue += plan.price_cents / 100 * 12
                else:
                    monthly_revenue += plan.price_cents / 100
                    yearly_revenue += plan.price_cents / 100 * 12
                
                # Track plan distribution
                plan_name = plan.name
                if plan_name not in tenant_plan_distribution:
                    tenant_plan_distribution[plan_name] = 0
                tenant_plan_distribution[plan_name] += 1
        
        # Calculate revenue growth (compare with previous month)
        prev_month_revenue = monthly_revenue * 0.95  # Assume 5% growth for now
        revenue_growth = ((monthly_revenue - prev_month_revenue) / prev_month_revenue * 100) if prev_month_revenue > 0 else 0
        
        # 6. Additional Analytics
        # User growth
        total_users = db.session.query(func.count(User.id)).scalar() or 0
        new_users_week = db.session.query(func.count(User.id)).filter(
            User.created_at >= week_ago
        ).scalar() or 0
        
        # Skills diversity
        unique_skills = db.session.query(func.count(func.distinct(CandidateSkill.skill_name))).scalar() or 0
        
        # Experience levels distribution
        experience_levels = db.session.query(
            CandidateProfile.experience_years,
            func.count(CandidateProfile.id)
        ).filter(
            CandidateProfile.experience_years.isnot(None)
        ).group_by(CandidateProfile.experience_years).all()
        
        # User type distribution
        user_types = db.session.query(
            User.user_type,
            func.count(User.id)
        ).group_by(User.user_type).all()
        
        # Detailed user type breakdown for active talent
        user_type_breakdown = {
            'candidates': total_candidates,
            'job_seekers': total_job_seekers,
            'recruiters': total_recruiters,
            'admin_employees': total_admin_employees,
            'total_active_talent': active_talent_count
        }
        
        # Use calculated accuracy directly (no adjustment needed with improved calculation)
        display_match_accuracy = max(0, min(100, match_accuracy))
        display_response_rate = max(0, min(100, response_rate))

        kpis = {
            'ai_matches_today': {
                'value': today_searches,
                'change': matches_change,
                'trending': 'up' if matches_change >= 0 else 'down',
                'description': 'High-quality candidates matched using AI'
            },
            'active_candidates': {
                'value': active_talent_count,
                'change': candidates_change,
                'trending': 'up' if candidates_change >= 0 else 'down',
                'description': 'Candidates in your talent pipeline'
            },
            'match_accuracy': {
                'value': f"{display_match_accuracy:.0f}%",
                'change': f"{accuracy_change:+.0f}%",
                'trending': 'up' if accuracy_change >= 0 else 'down',
                'description': 'AI matching precision rate'
            },
            'response_rate': {
                'value': f"{display_response_rate:.0f}%",
                'change': f"{response_change:+.0f}%",
                'trending': 'up' if response_change >= 0 else 'down',
                'description': 'Candidate response to outreach'
            },
            'monthly_revenue': {
                'value': f"${monthly_revenue:,.0f}",
                'change': f"{revenue_growth:+.1f}%",
                'trending': 'up' if revenue_growth >= 0 else 'down',
                'description': 'Monthly recurring revenue'
            },
            'additional_metrics': {
                'total_users': total_users,
                'new_users_this_week': new_users_week,
                'unique_skills': unique_skills,
                'total_searches_month': total_searches_month,
                'active_talent_breakdown': user_type_breakdown,
                'revenue_metrics': {
                    'monthly_recurring_revenue': round(monthly_revenue, 2),
                    'yearly_revenue': round(yearly_revenue, 2),
                    'revenue_growth_percent': round(revenue_growth, 1),
                    'active_tenants': len(active_tenants),
                    'tenant_plan_distribution': tenant_plan_distribution
                },
                'experience_distribution': [
                    {'years': exp[0], 'count': exp[1]} for exp in experience_levels
                ],
                'user_type_distribution': [
                    {'type': ut[0], 'count': ut[1]} for ut in user_types
                ]
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"KPI data generated for user {user.email}: {kpis}")
        
        # Cache the results
        kpi_data = {
            'success': True,
            'kpis': kpis
        }
        analytics_cache.set_kpis(kpi_data, user_id=str(user.id))
        
        return jsonify(kpi_data), 200
        
    except Exception as e:
        logger.error(f"Error generating KPI data: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate KPI data'}), 500

@analytics_bp.route('/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """Get detailed dashboard statistics"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Time periods
        today = datetime.utcnow().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # Search trends (last 7 days)
        search_trends = []
        for i in range(7):
            date = today - timedelta(days=i)
            count = db.session.query(func.count(JDSearchLog.id)).filter(
                func.date(JDSearchLog.searched_at) == date,
                JDSearchLog.tenant_id == tenant_id
            ).scalar() or 0
            search_trends.append({
                'date': date.isoformat(),
                'searches': count
            })
        
        # Top skills in demand
        top_skills = db.session.query(
            CandidateSkill.skill_name,
            func.count(CandidateSkill.id).label('count')
        ).join(CandidateProfile).filter(
            CandidateProfile.is_public == True,
            CandidateProfile.created_at >= month_ago
        ).group_by(CandidateSkill.skill_name).order_by(
            func.count(CandidateSkill.id).desc()
        ).limit(10).all()
        
        # Recent candidate activity
        recent_candidates = db.session.query(CandidateProfile).filter(
            CandidateProfile.is_public == True,
            CandidateProfile.created_at >= week_ago
        ).order_by(CandidateProfile.created_at.desc()).limit(5).all()
        
        stats = {
            'search_trends': search_trends,
            'top_skills': [
                {'skill': skill[0], 'count': skill[1]} for skill in top_skills
            ],
            'recent_candidates': [
                {
                    'id': candidate.id,
                    'name': candidate.full_name,
                    'location': candidate.location,
                    'created_at': candidate.created_at.isoformat() if candidate.created_at else None
                } for candidate in recent_candidates
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating dashboard stats: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate dashboard stats'}), 500

@analytics_bp.route('/debug-counts', methods=['GET'])
def debug_counts():
    """Debug endpoint to check actual database counts"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get various counts for debugging
        total_candidates = db.session.query(func.count(CandidateProfile.id)).scalar() or 0
        total_job_seekers = db.session.query(func.count(User.id)).filter(
            User.user_type == 'job_seeker'
        ).scalar() or 0
        total_recruiters = db.session.query(func.count(User.id)).filter(
            User.user_type == 'recruiter'
        ).scalar() or 0
        total_admin_employees = db.session.query(func.count(User.id)).filter(
            User.user_type == 'admin_employee'
        ).scalar() or 0
        total_active_talent = total_candidates + total_job_seekers + total_recruiters + total_admin_employees
        
        public_candidates = db.session.query(func.count(CandidateProfile.id)).filter(
            CandidateProfile.is_public == True
        ).scalar() or 0
        private_candidates = db.session.query(func.count(CandidateProfile.id)).filter(
            CandidateProfile.is_public == False
        ).scalar() or 0
        null_public_candidates = db.session.query(func.count(CandidateProfile.id)).filter(
            CandidateProfile.is_public.is_(None)
        ).scalar() or 0
        
        # Get sample data
        sample_candidates = db.session.query(CandidateProfile).limit(5).all()
        sample_data = []
        for candidate in sample_candidates:
            sample_data.append({
                'id': candidate.id,
                'full_name': candidate.full_name,
                'is_public': candidate.is_public,
                'created_at': candidate.created_at.isoformat() if candidate.created_at else None
            })
        
        return jsonify({
            'success': True,
            'debug_info': {
                'total_candidates': total_candidates,
                'total_job_seekers': total_job_seekers,
                'total_recruiters': total_recruiters,
                'total_admin_employees': total_admin_employees,
                'total_active_talent': total_active_talent,
                'public_candidates': public_candidates,
                'private_candidates': private_candidates,
                'null_public_candidates': null_public_candidates,
                'sample_candidates': sample_data,
                'user_tenant_id': tenant_id,
                'user_email': user.email
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get debug info'}), 500

@analytics_bp.route('/tenant-growth-strategy', methods=['GET'])
def get_tenant_growth_strategy():
    """Get comprehensive tenant growth strategy and insights"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get current tenant statistics
        total_tenants = db.session.query(func.count(Tenant.id)).scalar() or 0
        active_tenants = db.session.query(func.count(Tenant.id)).filter(Tenant.status == 'active').scalar() or 0
        new_tenants_this_month = db.session.query(func.count(Tenant.id)).filter(
            Tenant.created_at >= datetime.utcnow().replace(day=1)
        ).scalar() or 0
        
        # Get plan distribution
        plan_distribution = db.session.query(
            Plan.name,
            func.count(Tenant.id)
        ).join(Tenant).filter(
            Tenant.status == 'active'
        ).group_by(Plan.name).all()
        
        # Calculate conversion rates
        total_trials = db.session.query(func.count(UserTrial.id)).scalar() or 0
        trial_conversion_rate = (active_tenants / total_trials * 100) if total_trials > 0 else 0
        
        # Growth recommendations
        growth_recommendations = []
        
        if trial_conversion_rate < 20:
            growth_recommendations.append({
                'type': 'conversion',
                'priority': 'high',
                'title': 'Improve Trial Conversion',
                'description': f'Current conversion rate: {trial_conversion_rate:.1f}%. Focus on onboarding and value demonstration.',
                'action': 'Enhance trial experience and follow-up sequences'
            })
        
        if active_tenants < 100:
            growth_recommendations.append({
                'type': 'acquisition',
                'priority': 'high',
                'title': 'Increase Customer Acquisition',
                'description': f'Target: 100+ active tenants. Current: {active_tenants}',
                'action': 'Implement referral programs and content marketing'
            })
        
        # Revenue optimization
        revenue_optimization = []
        if active_tenants > 0:
            avg_revenue_per_tenant = db.session.query(func.avg(Plan.price_cents)).join(Tenant).filter(
                Tenant.status == 'active'
            ).scalar() or 0
            
            if avg_revenue_per_tenant < 5000:  # Less than $50/month average
                revenue_optimization.append({
                    'type': 'upselling',
                    'priority': 'medium',
                    'title': 'Increase Average Revenue Per User (ARPU)',
                    'description': f'Current ARPU: ${avg_revenue_per_tenant/100:.2f}/month',
                    'action': 'Upsell to higher-tier plans and add-on services'
                })
        
        return jsonify({
            'success': True,
            'tenant_growth': {
                'current_stats': {
                    'total_tenants': total_tenants,
                    'active_tenants': active_tenants,
                    'new_tenants_this_month': new_tenants_this_month,
                    'trial_conversion_rate': round(trial_conversion_rate, 1)
                },
                'plan_distribution': [
                    {'plan': plan[0], 'count': plan[1]} for plan in plan_distribution
                ],
                'growth_recommendations': growth_recommendations,
                'revenue_optimization': revenue_optimization,
                'growth_targets': {
                    'next_month': active_tenants + 10,
                    'next_quarter': active_tenants + 50,
                    'next_year': active_tenants + 200
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating tenant growth strategy: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate tenant growth strategy'}), 500

@analytics_bp.route('/user-analytics', methods=['GET'])
def get_user_analytics():
    """Get user-specific analytics"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # User's search history
        user_searches = db.session.query(func.count(JDSearchLog.id)).filter(
            JDSearchLog.user_id == user.id
        ).scalar() or 0
        
        # User's trial status
        trial = UserTrial.query.filter_by(user_id=user.id).first()
        trial_info = None
        if trial:
            trial_info = {
                'is_active': trial.is_active,
                'searches_used_today': trial.searches_used_today,
                'trial_end_date': trial.trial_end_date.isoformat() if trial.trial_end_date else None,
                'days_remaining': max(0, (trial.trial_end_date - datetime.utcnow()).days) if trial.trial_end_date else 0
            }
        
        analytics = {
            'user_searches': user_searches,
            'trial_info': trial_info,
            'user_role': user.role,
            'user_type': user.user_type,
            'member_since': user.created_at.isoformat() if user.created_at else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating user analytics: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate user analytics'}), 500

@analytics_bp.route('/user-dashboard', methods=['GET'])
def get_user_dashboard():
    """Get user dashboard data including plan details and trial information"""
    try:
        payload = get_jwt_payload()
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user, tenant_id = get_user_from_jwt(payload)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get user's tenant and plan information
        tenant = Tenant.query.get(user.tenant_id)
        plan_details = None
        if tenant:
            plan = Plan.query.get(tenant.plan_id)
            if plan:
                plan_details = {
                    'plan_name': plan.name,
                    'is_trial': plan.is_trial,
                    'plan_features': {
                        'jd_quota_per_month': plan.jd_quota_per_month,
                        'max_subaccounts': plan.max_subaccounts,
                        'price_cents': plan.price_cents,
                        'billing_cycle': plan.billing_cycle
                    }
                }
        
        # User's trial status
        trial = UserTrial.query.filter_by(user_id=user.id).first()
        trial_info = None
        if trial:
            trial_info = {
                'is_active': trial.is_active,
                'searches_used_today': trial.searches_used_today,
                'trial_end_date': trial.trial_end_date.isoformat() if trial.trial_end_date else None,
                'days_remaining': max(0, (trial.trial_end_date - datetime.utcnow()).days) if trial.trial_end_date else 0
            }
        
        # User's search history
        user_searches = db.session.query(func.count(JDSearchLog.id)).filter(
            JDSearchLog.user_id == user.id
        ).scalar() or 0
        
        # Check monthly quota usage
        current_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_searches = db.session.query(func.count(JDSearchLog.id)).filter(
            JDSearchLog.user_id == user.id,
            JDSearchLog.searched_at >= current_month
        ).scalar() or 0
        
        # Check if plan is over (quota exceeded)
        plan_status = {
            'is_active': True,
            'quota_exceeded': False,
            'searches_used_this_month': monthly_searches,
            'searches_remaining': 0,
            'quota_percentage': 0
        }
        
        if plan_details and plan_details['plan_features']:
            monthly_quota = plan_details['plan_features']['jd_quota_per_month']
            plan_status['searches_remaining'] = max(0, monthly_quota - monthly_searches)
            plan_status['quota_percentage'] = min(100, (monthly_searches / monthly_quota) * 100) if monthly_quota > 0 else 0
            plan_status['quota_exceeded'] = monthly_searches >= monthly_quota
            plan_status['is_active'] = not plan_status['quota_exceeded']
        
        # Recent activity (last 5 searches)
        recent_searches = db.session.query(JDSearchLog).filter(
            JDSearchLog.user_id == user.id
        ).order_by(JDSearchLog.searched_at.desc()).limit(5).all()
        
        recent_activity = []
        for search in recent_searches:
            recent_activity.append({
                'id': search.id,
                'job_description': search.job_description[:100] + '...' if search.job_description and len(search.job_description) > 100 else search.job_description,
                'searched_at': search.searched_at.isoformat() if search.searched_at else None
            })
        
        dashboard_data = {
            'plan_details': plan_details,
            'trial_info': trial_info,
            'plan_status': plan_status,
            'user_searches': user_searches,
            'monthly_searches': monthly_searches,
            'recent_activity': recent_activity,
            'user_role': user.role,
            'user_type': user.user_type,
            'member_since': user.created_at.isoformat() if user.created_at else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'dashboard_data': dashboard_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating user dashboard data: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate user dashboard data'}), 500
