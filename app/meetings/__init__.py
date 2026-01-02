from flask import Blueprint

# This blueprint is registered in app.__init__ as "meeting_bp" with url_prefix="/api"
from .routes import meeting_bp  # noqa: F401


