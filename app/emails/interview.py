"""
Interview invitation email functions
"""
from app.simple_logger import get_logger
import os
from typing import Optional

logger = get_logger('emails')


def _clean_job_title(job_title: Optional[str]) -> str:
    """
    Normalize job title for email display.
    - Remove long job description text accidentally appended (e.g. starting with 'We are seeking')
    - Keep only the first line
    - Trim to a reasonable length
    """
    if not job_title:
        return "Interview"

    title = str(job_title)

    # If a full JD was appended (common phrasing), cut it off
    cut_markers = [
        "We are seeking",
        "We are looking",
        "The ideal candidate",
        "Responsibilities:",
        "Requirements:",
    ]
    for marker in cut_markers:
        if marker in title:
            title = title.split(marker, 1)[0].strip()
            break

    # Only keep the first line
    title = title.splitlines()[0].strip()

    # Avoid empty title
    if not title:
        title = "Interview"

    # Hard cap length to avoid ugly subjects/position fields
    if len(title) > 140:
        title = title[:137].rstrip() + "..."

    return title


def _clean_interview_notes(notes: Optional[str]) -> str:
    """
    Normalize interview notes so they don't contain the entire job description.
    - Remove repeated JD markers if present
    - Trim to a reasonable length
    """
    if not notes:
        return ""

    text = str(notes).strip()

    # If notes clearly contain a pasted JD, cut at common markers
    cut_markers = [
        "Requirements:",
        "Responsibilities:",
        "We are seeking",
        "The ideal candidate",
    ]
    for marker in cut_markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
            break

    # Limit to a few sentences / 400 chars max
    if len(text) > 400:
        text = text[:397].rstrip() + "..."

    return text


def send_interview_invitation_email(
    to_email,
    candidate_name,
    job_title,
    company_name,
    job_location,
    interview_date,
    meeting_link,
    meeting_type,
    interviewer_name,
    interview_notes,
):
    """
    Send interview invitation email to candidate using Hostinger SMTP only.

    This function is used by multiple routes (e.g. /jobs, /api/meetings/schedule)
    and should rely exclusively on our SMTP configuration (Hostinger) instead
    of falling back to AWS SES. Any SMTP failure is logged and returned as False.
    """
    # Clean up noisy inputs so emails don't contain the full job description everywhere
    cleaned_job_title = _clean_job_title(job_title)
    cleaned_notes = _clean_interview_notes(interview_notes)

    logger.info(f"[EMAIL] send_interview_invitation_email called with:")
    logger.info(f"   - to_email: {to_email}")
    logger.info(f"   - candidate_name: {candidate_name}")
    logger.info(f"   - job_title: {cleaned_job_title}")
    logger.info(f"   - company_name: {company_name}")
    logger.info(f"   - job_location: {job_location}")
    logger.info(f"   - interview_date: {interview_date}")
    logger.info(f"   - meeting_link: {meeting_link}")
    logger.info(f"   - meeting_type: {meeting_type}")
    logger.info(f"   - interviewer_name: {interviewer_name}")

    logger.info("[SMTP] Attempting to send interview invitation via Hostinger SMTP...")
    try:
        from .smtp import send_interview_invitation_email_smtp

        smtp_result = send_interview_invitation_email_smtp(
            to_email,
            candidate_name,
            cleaned_job_title,
            company_name,
            job_location,
            interview_date,
            meeting_link,
            meeting_type,
            interviewer_name,
            cleaned_notes,
        )

        if smtp_result:
            logger.info(f"[SUCCESS] Interview invitation email sent successfully via SMTP to {to_email}")
            return True

        logger.warning(f"[SMTP_FAILED] Hostinger SMTP reported failure for {to_email}")
        return False

    except Exception as smtp_error:
        logger.error(f"[SMTP_ERROR] Hostinger SMTP error while sending to {to_email}: {smtp_error}")
        return False
