from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr

# Re-export existing validation utilities and models
from .validation_simple import (
	SearchRequest,
	CandidateData,
	JobDescription,
	validate_input,
	sanitize_string,
	validate_file_upload,
)


class UserFeedback(BaseModel):
	"""Minimal user feedback model for testing purposes."""
	user_id: Optional[str] = Field(default=None, description="User identifier")
	feedback_text: str = Field(..., min_length=1, max_length=2000, description="Feedback content")
	rating: Optional[int] = Field(default=None, ge=1, le=5, description="Optional rating 1-5")
	metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class APIKeyRequest(BaseModel):
	"""Minimal API key request model for testing purposes."""
	email: EmailStr = Field(..., description="Requester email")
	use_case: str = Field(..., min_length=5, max_length=500, description="Intended use case")
	organization: Optional[str] = Field(default=None, max_length=200, description="Organization name")
	scopes: Optional[List[str]] = Field(default=None, description="Requested API scopes")

