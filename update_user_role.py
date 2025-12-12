import argparse
import os
import sys
from typing import Optional

# Ensure the backend package is importable when the script is run from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.app import create_app  # type: ignore  # pylint: disable=wrong-import-position
from backend.app.models import User  # type: ignore  # pylint: disable=wrong-import-position
from backend.app.db import db  # type: ignore  # pylint: disable=wrong-import-position

VALID_ROLES = {
    "owner",
    "subuser",
    "job_seeker",
    "employee",
    "recruiter",
    "employer",
    "admin",
}


def update_user_role(
    email: str, role: str, user_type: Optional[str] = None, dry_run: bool = False
) -> dict:
    """Update a user's role (and optionally user_type) by email."""
    normalized_email = email.strip().lower()
    desired_role = role.strip().lower()

    if desired_role not in VALID_ROLES:
        raise ValueError(
            f"Invalid role '{desired_role}'. Allowed values: {', '.join(sorted(VALID_ROLES))}"
        )

    user = User.query.filter_by(email=normalized_email).first()
    if not user:
        return {"found": False, "email": normalized_email}

    changes = {
        "found": True,
        "email": user.email,
        "previous_role": user.role,
        "new_role": desired_role,
        "previous_user_type": user.user_type,
        "new_user_type": user_type if user_type is not None else user.user_type,
        "dry_run": dry_run,
    }

    if dry_run:
        return changes

    user.role = desired_role
    if user_type is not None:
        user.user_type = user_type

    db.session.commit()
    db.session.refresh(user)
    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the role (and optionally user_type) for a user by email."
    )
    parser.add_argument("email", help="Email address of the user to update")
    parser.add_argument(
        "role",
        help=f"New role to assign. Allowed values: {', '.join(sorted(VALID_ROLES))}",
    )
    parser.add_argument(
        "--user-type",
        help="Optional user_type field to set after changing the role",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without committing to the database",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    app = create_app()
    with app.app_context():
        try:
            result = update_user_role(
                email=args.email,
                role=args.role,
                user_type=args.user_type,
                dry_run=args.dry_run,
            )
        except ValueError as exc:
            print(str(exc))
            sys.exit(1)
        finally:
            db.session.remove()

    if not result.get("found"):
        print(f"No user found with email: {args.email.strip().lower()}")
        sys.exit(1)

    if result.get("dry_run"):
        print(
            "Dry run complete. No changes saved.\n"
            f"Current role: {result['previous_role']} -> {result['new_role']}\n"
            f"Current user_type: {result['previous_user_type']} -> {result['new_user_type']}"
        )
        return

    print(
        f"Successfully updated {result['email']}:\n"
        f"- role: {result['previous_role']} -> {result['new_role']}\n"
        f"- user_type: {result['previous_user_type']} -> {result['new_user_type']}"
    )


if __name__ == "__main__":
    main()

