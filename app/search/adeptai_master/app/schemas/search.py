from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any


class SearchRequest(BaseModel):
    """Unified Pydantic model for search requests (validation and DTO)"""
    query: str = Field(..., min_length=1, max_length=5000, description="Search query (supports full job descriptions)")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    include_explanation: bool = Field(default=False, description="Include explanation for results")
    include_behavioural_analysis: bool = Field(default=False, description="Run behavioral analysis and include in results")
    enable_domain_filtering: bool = Field(default=True, description="Enable domain filtering when supported")
    
    # Pagination parameters
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(10, ge=1, le=100, description="Number of results per page")
    
    class Config:
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment
    
    @validator("query")
    def strip_query(cls, v: str) -> str:
        """Strip and validate query"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator("filters")
    def validate_filters(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate filter keys"""
        if v is None:
            return v
        
        allowed_keys = {
            'location', 'experience_level', 'skills', 'industry',
            'salary_min', 'salary_max', 'company_size', 'remote'
        }
        
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid filter key: {key}")
        
        return v
    
    def get_offset(self) -> int:
        """Calculate offset for pagination"""
        return (self.page - 1) * self.page_size
    
    def get_limit(self) -> int:
        """Get limit for pagination (use top_k or page_size, whichever is smaller)"""
        return min(self.top_k, self.page_size)


class SearchResponse(BaseModel):
    """Response model for search results"""
    status: str = Field("ok", description="Response status")
    data: Dict[str, Any] = Field(..., description="Search results data")
    pagination: Optional[Dict[str, Any]] = Field(None, description="Pagination metadata")
    trace_id: Optional[str] = Field(None, description="Request trace ID")
    request_id: Optional[str] = Field(None, description="Request ID")


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details for JSON error response"""
    type: str = Field(..., description="A URI reference that identifies the problem type")
    title: str = Field(..., description="A short, human-readable summary of the problem")
    status: int = Field(..., description="HTTP status code")
    detail: str = Field(..., description="A human-readable explanation specific to this occurrence")
    instance: Optional[str] = Field(None, description="A URI reference that identifies the specific occurrence")
    trace_id: Optional[str] = Field(None, description="Request trace ID for correlation")
    error_code: Optional[str] = Field(None, description="Application-specific error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "https://api.adeptai.com/problems/validation-error",
                "title": "Validation Error",
                "status": 400,
                "detail": "Invalid search request: query is required",
                "instance": "/api/search",
                "trace_id": "abc123",
                "error_code": "INVALID_SEARCH_REQUEST",
                "details": {"validation_errors": []}
            }
        }


