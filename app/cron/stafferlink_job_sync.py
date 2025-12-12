#!/usr/bin/env python3
"""
Daily Stafferlink Job Sync
Fetches the latest Stafferlink orders for every connected user and stores them locally.
"""
import logging
from datetime import datetime

from app import create_app
from app.models import StafferlinkIntegration, User
from app.simple_logger import get_logger
from app.stafferlink.routes import sync_stafferlink_jobs_for_integration

logger = get_logger("stafferlink_job_sync")


def sync_all_stafferlink_jobs(last_modified_minutes: int = 1440):
    """Sync Stafferlink jobs for every connected integration."""
    app = create_app()

    with app.app_context():
        integrations = StafferlinkIntegration.query.all()
        total_saved = 0
        processed = 0

        for integration in integrations:
            user = User.query.get(integration.user_id)
            if not user:
                logger.warning(f"Skipping integration {integration.id}: user not found")
                continue

            processed += 1
            ok, result = sync_stafferlink_jobs_for_integration(
                user,
                user.tenant_id,
                integration,
                last_modified_minutes=last_modified_minutes,
            )

            if ok:
                saved = result.get("saved", 0) if isinstance(result, dict) else 0
                logger.info(
                    f"Synced {saved} Stafferlink jobs for user_id={user.id}, tenant_id={user.tenant_id}"
                )
                total_saved += saved
            else:
                logger.error(
                    f"Failed to sync Stafferlink jobs for user_id={user.id}: {result}"
                )

        logger.info(
            f"Completed Stafferlink sync run at {datetime.utcnow().isoformat()} "
            f"for {processed} integrations. Total jobs processed: {total_saved}"
        )
        return total_saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Stafferlink daily sync job...")
    sync_all_stafferlink_jobs()


