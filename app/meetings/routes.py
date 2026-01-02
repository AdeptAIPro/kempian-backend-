from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from typing import List

from app.simple_logger import get_logger
from app.utils import get_current_user_flexible, get_current_user

logger = get_logger("meetings")

meeting_bp = Blueprint("meeting_bp", __name__)


def _get_authed_user():
    """
    Resolve the current authenticated user from JWT / Cognito token.
    Returns (user_dict, error_response, status_code).
    """
    user = get_current_user_flexible() or get_current_user()
    if not user or not user.get("email"):
        return None, jsonify({"error": "Unauthorized"}), 401
    return user, None, None


def _parse_iso_datetime(value: str) -> datetime:
    """
    Parse an ISO datetime string, handling trailing 'Z' as UTC.
    """
    if not isinstance(value, str):
        raise ValueError("Invalid datetime format")
    # Handle common "2025-01-01T10:00:00Z" format
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value)


@meeting_bp.route("/meetings/schedule", methods=["POST"])
def schedule_meeting():
    """
    Schedule an interview/meeting and send email invites.

    This endpoint is used by:
      - MatchingResultsContainer.tsx
      - ScheduleInterviews.tsx

    Expected JSON body (minimal contract):
      - platform: 'zoom' | 'google_meet' | 'teams' | 'phone' | 'in-person' | 'other'
      - title: string
      - description: string (optional)
      - start_time: ISO datetime string
      - duration_minutes: number
      - attendees: string[] (email addresses preferred)
      - timezone: string (IANA, e.g. 'America/Los_Angeles')
      - notes: string (optional)
      - location: string (optional, for in-person)
      - meeting_link: string (optional for phone/in-person, required for video)
    """
    user, resp, code = _get_authed_user()
    if resp:
        return resp, code

    data = request.get_json(silent=True) or {}

    required_fields = ["platform", "title", "start_time", "duration_minutes", "attendees"]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    platform = str(data.get("platform")).lower()
    if platform not in ["zoom", "google_meet", "teams", "phone", "in-person", "in_person", "other"]:
        return jsonify({"error": "Invalid platform"}), 400

    # Normalise in_person variant
    if platform == "in_person":
        platform = "in-person"

    attendees: List[str] = data.get("attendees") or []
    if not isinstance(attendees, list) or len(attendees) == 0:
        return jsonify({"error": "attendees must be a non-empty array"}), 400

    try:
        start_time = _parse_iso_datetime(data.get("start_time"))
    except Exception:
        return jsonify({"error": "Invalid start_time format. Use ISO 8601 datetime string."}), 400

    try:
        duration_minutes = int(data.get("duration_minutes"))
    except (TypeError, ValueError):
        return jsonify({"error": "duration_minutes must be a number"}), 400

    end_time = start_time + timedelta(minutes=duration_minutes)

    title = data.get("title") or "Interview"
    description = data.get("description") or ""
    notes = data.get("notes") or ""
    timezone = data.get("timezone") or "UTC"
    location = data.get("location") or "Remote"
    meeting_link = data.get("meeting_link") or ""

    logger.info("[MEETINGS] /api/meetings/schedule called")
    logger.info(f"  User: {user.get('email')}")
    logger.info(f"  Title: {title}")
    logger.info(f"  Platform: {platform}")
    logger.info(f"  Start: {start_time.isoformat()}  Duration: {duration_minutes} minutes")
    logger.info(f"  Attendees: {attendees}")

    # Best-effort: send interview invitation email to each attendee
    # Uses existing robust email helper in app.emails.interview
    try:
        from app.emails.interview import send_interview_invitation_email
    except Exception as import_err:
        logger.error(f"[MEETINGS] Failed to import interview email helper: {import_err}")
        # We still return success for scheduling even if email helper import fails
        send_interview_invitation_email = None  # type: ignore

    if send_interview_invitation_email:
        # Determine company name / interviewer name from user context where possible
        company_name = user.get("custom:company_name") or user.get("company_name") or "Kempian"
        interviewer_name = user.get("name") or user.get("custom:name") or user.get("email", "").split("@")[0]

        for attendee in attendees:
            to_email = attendee
            if not isinstance(to_email, str) or "@" not in to_email:
                # Skip non-email attendees silently to avoid breaking existing UI that might
                # send names instead of emails in some flows.
                logger.warning(f"[MEETINGS] Skipping non-email attendee value: {attendee!r}")
                continue

            candidate_name = data.get("candidate_name") or to_email.split("@")[0]
            job_title = data.get("job_title") or title
            job_location = location

            try:
                logger.info(f"[MEETINGS] Sending interview invitation email to {to_email}")
                email_sent = send_interview_invitation_email(
                    to_email=to_email,
                    candidate_name=candidate_name,
                    job_title=job_title,
                    company_name=company_name,
                    job_location=job_location,
                    interview_date=start_time,
                    meeting_link=meeting_link,
                    meeting_type=platform,
                    interviewer_name=interviewer_name,
                    interview_notes=notes or description,
                )
                if email_sent:
                    logger.info(f"[MEETINGS] Interview email sent successfully to {to_email}")
                else:
                    logger.warning(f"[MEETINGS] Failed to send interview email to {to_email}")
            except Exception as e:
                logger.error(f"[MEETINGS] Error sending interview email to {to_email}: {e}", exc_info=True)
                # Do not fail the scheduling just because email failed

    # For now we don't persist meetings; return a synthetic meeting payload
    meeting_payload = {
        "id": None,
        "title": title,
        "description": description,
        "platform": platform,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_minutes": duration_minutes,
        "timezone": timezone,
        "attendees": attendees,
        "meeting_link": meeting_link,
        "location": location,
        "notes": notes,
        "status": "scheduled",
    }

    return jsonify({"message": "Interview scheduled successfully", "meeting": meeting_payload}), 200


@meeting_bp.route("/meetings", methods=["GET"])
def list_meetings():
    """
    Basic stub endpoint so the MeetingManager UI can load without errors.
    Currently returns an empty list; extend with real persistence if needed.
    """
    user, resp, code = _get_authed_user()
    if resp:
        return resp, code

    logger.info(f"[MEETINGS] List meetings for user {user.get('email')}")
    return jsonify({"meetings": []}), 200


@meeting_bp.route("/meetings/<int:meeting_id>/cancel", methods=["POST"])
def cancel_meeting(meeting_id: int):
    """
    Basic stub endpoint to 'cancel' a meeting.
    No persistence yet; responds success so UI can update optimistically.
    """
    user, resp, code = _get_authed_user()
    if resp:
        return resp, code

    logger.info(f"[MEETINGS] Cancel meeting {meeting_id} requested by {user.get('email')}")

    return jsonify({"message": "Meeting cancelled", "meeting_id": meeting_id, "status": "cancelled"}), 200


