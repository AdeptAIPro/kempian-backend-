from flask import Blueprint, jsonify, request, current_app
from pydantic import ValidationError as PydanticValidationError
from typing import Dict, Any, Optional, Union
import logging
import uuid

from ..schemas.search import SearchRequest, SearchResponse, ErrorResponse
from ..exceptions import (
    SearchSystemUnavailableError, 
    SearchQueryError, 
    ValidationError as AdeptAIValidationError
)
from ..exceptions import get_http_status_code
from ..services import get_service

logger = logging.getLogger(__name__)

search_bp = Blueprint("search", __name__)

# Problem type URIs for RFC 7807 compliance
PROBLEM_TYPES = {
    "validation_error": "https://api.adeptai.com/problems/validation-error",
    "search_error": "https://api.adeptai.com/problems/search-error",
    "service_unavailable": "https://api.adeptai.com/problems/service-unavailable",
    "internal_error": "https://api.adeptai.com/problems/internal-error",
}


def get_trace_id() -> Optional[str]:
    """Extract or generate trace ID from request headers"""
    # Check for common trace ID headers
    trace_id = (
        request.headers.get("X-Request-ID") or
        request.headers.get("X-Trace-ID") or
        request.headers.get("X-Correlation-ID") or
        request.headers.get("Traceparent") or
        None
    )
    
    # Generate new trace ID if not present
    if not trace_id:
        trace_id = str(uuid.uuid4())
    
    return trace_id


def create_problem_response(
    exception: Union[AdeptAIValidationError, SearchQueryError, SearchSystemUnavailableError, Exception],
    trace_id: Optional[str] = None,
    instance: Optional[str] = None
) -> tuple[Dict[str, Any], int]:
    """Create RFC 7807 Problem Details JSON response"""
    
    # Determine problem type and status code
    if isinstance(exception, AdeptAIValidationError):
        problem_type = PROBLEM_TYPES["validation_error"]
        title = "Validation Error"
        status = 400
    elif isinstance(exception, SearchQueryError):
        problem_type = PROBLEM_TYPES["search_error"]
        title = "Search Error"
        status = 400
    elif isinstance(exception, SearchSystemUnavailableError):
        problem_type = PROBLEM_TYPES["service_unavailable"]
        title = "Service Unavailable"
        status = 503
    else:
        problem_type = PROBLEM_TYPES["internal_error"]
        title = "Internal Server Error"
        status = 500
    
    # Get status code from exception mapping if available
    if hasattr(exception, 'error_code'):
        status = get_http_status_code(exception)
    
    # Build error response
    error_detail = str(exception)
    error_code = getattr(exception, 'error_code', None)
    error_details = getattr(exception, 'details', None)
    
    problem = ErrorResponse(
        type=problem_type,
        title=title,
        status=status,
        detail=error_detail,
        instance=instance or request.path,
        trace_id=trace_id,
        error_code=error_code,
        details=error_details
    )
    
    return problem.dict(exclude_none=True), status


@search_bp.post("/search")
def search() -> tuple[Dict[str, Any], int]:
    """
    Search for candidates based on query.
    
    Supports:
    - Unified Pydantic validation and DTOs
    - Pagination (page, page_size)
    - Request tracing (X-Request-ID, X-Trace-ID headers)
    - RFC 7807 Problem Details error responses
    
    Returns:
        Tuple of (response_data, status_code)
        
    Raises:
        SearchSystemUnavailableError: If search system is not available
        SearchQueryError: If search query is invalid
        AdeptAIValidationError: If request validation fails
    """
    trace_id = get_trace_id()
    
    try:
        # Validate request method and content type
        if request.method != 'POST':
            problem = ErrorResponse(
                type=PROBLEM_TYPES["validation_error"],
                title="Method Not Allowed",
                status=405,
                detail="POST method is required for this endpoint",
                instance=request.path,
                trace_id=trace_id,
                error_code="METHOD_NOT_ALLOWED"
            )
            return jsonify(problem.dict(exclude_none=True)), 405
            
        if not request.is_json:
            problem = ErrorResponse(
                type=PROBLEM_TYPES["validation_error"],
                title="Invalid Content Type",
                status=400,
                detail="Content-Type must be application/json",
                instance=request.path,
                trace_id=trace_id,
                error_code="INVALID_CONTENT_TYPE"
            )
            return jsonify(problem.dict(exclude_none=True)), 400
            
        payload = request.get_json(silent=True) or {}
        
        # Basic input validation
        if not payload:
            problem = ErrorResponse(
                type=PROBLEM_TYPES["validation_error"],
                title="Missing Request Body",
                status=400,
                detail="Request body is required",
                instance=request.path,
                trace_id=trace_id,
                error_code="MISSING_REQUEST_BODY"
            )
            return jsonify(problem.dict(exclude_none=True)), 400
        
        # Validate request using unified Pydantic model
        try:
            req = SearchRequest(**payload)
        except PydanticValidationError as ve:
            logger.warning(f"Validation error in search: {ve.errors()}", extra={"trace_id": trace_id})
            raise AdeptAIValidationError(
                f"Invalid search request: {ve.errors()}",
                error_code="INVALID_SEARCH_REQUEST",
                details={"validation_errors": ve.errors()}
            )
        
        # Get search system
        search_system = get_service("search_system")
        if not search_system:
            raise SearchSystemUnavailableError(
                "Search system is not available",
                error_code="SEARCH_SYSTEM_UNAVAILABLE"
            )
        
        # Perform search
        try:
            # Pass feature flags through to search
            results = search_system.search(
                req.query,
                top_k=req.top_k,
                include_behavioural_analysis=req.include_behavioural_analysis,
                enable_domain_filtering=req.enable_domain_filtering
            )
            
            # Handle pagination
            # Extract results list from different response formats
            if isinstance(results, list):
                results_list = results
                results_dict = None
            elif isinstance(results, dict):
                results_list = results.get("results", [])
                results_dict = results
            else:
                results_list = []
                results_dict = None
            
            # Apply pagination if we have a list of results
            if isinstance(results_list, list) and len(results_list) > 0:
                total_count = len(results_list)
                
                # Calculate pagination
                offset = req.get_offset()
                limit = req.get_limit()
                
                # Apply pagination
                paginated_results = results_list[offset:offset + limit]
                
                # Build pagination metadata
                total_pages = (total_count + req.page_size - 1) // req.page_size if req.page_size > 0 else 0
                pagination = {
                    "page": req.page,
                    "page_size": req.page_size,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_next": offset + limit < total_count,
                    "has_previous": req.page > 1
                }
                
                # Update results with paginated data
                if isinstance(results, list):
                    results = paginated_results
                elif isinstance(results, dict):
                    results["results"] = paginated_results
                    results["pagination"] = pagination
            else:
                # No pagination if results are not in list format or empty
                pagination = None
            
            # Build response with tracing
            # Ensure data is properly formatted
            response_data = results
            if isinstance(results, dict) and "pagination" in results:
                # Remove pagination from data if it was added there
                response_data = {k: v for k, v in results.items() if k != "pagination"}
            
            response = SearchResponse(
                status="ok",
                data=response_data,
                pagination=pagination,
                trace_id=trace_id,
                request_id=trace_id
            )
            
            # Add trace ID to response headers
            response_headers = {
                "X-Request-ID": trace_id,
                "X-Trace-ID": trace_id
            }
            
            return jsonify(response.dict(exclude_none=True)), 200, response_headers
            
        except Exception as e:
            logger.error(f"Search execution failed: {str(e)}", exc_info=True, extra={"trace_id": trace_id})
            raise SearchQueryError(
                f"Search failed: {str(e)}",
                error_code="SEARCH_EXECUTION_FAILED",
                details={"error": str(e)}
            )
            
    except AdeptAIValidationError as e:
        logger.warning(f"Validation error in search: {str(e)}", extra={"trace_id": trace_id})
        return create_problem_response(e, trace_id=trace_id)
    except SearchSystemUnavailableError as e:
        logger.error(f"Search system unavailable: {str(e)}", extra={"trace_id": trace_id})
        return create_problem_response(e, trace_id=trace_id)
    except SearchQueryError as e:
        logger.error(f"Search query error: {str(e)}", extra={"trace_id": trace_id})
        return create_problem_response(e, trace_id=trace_id)
    except Exception as e:
        logger.error(f"Unexpected search error: {str(e)}", exc_info=True, extra={"trace_id": trace_id})
        search_error = SearchQueryError(
            f"Unexpected search error: {str(e)}",
            error_code="UNEXPECTED_SEARCH_ERROR",
            details={"error": str(e)}
        )
        return create_problem_response(search_error, trace_id=trace_id)


@search_bp.get("/api/search/performance")
def search_performance() -> tuple[Dict[str, Any], int]:
    """
    Get search system performance statistics.
    
    Supports:
    - Request tracing (X-Request-ID, X-Trace-ID headers)
    - RFC 7807 Problem Details error responses
    
    Returns:
        Tuple of (performance_data, status_code)
        
    Raises:
        SearchSystemUnavailableError: If search system is not available
    """
    trace_id = get_trace_id()
    
    try:
        search_system = get_service("search_system")
        if not search_system:
            raise SearchSystemUnavailableError(
                "Search system is not available",
                error_code="SEARCH_SYSTEM_UNAVAILABLE"
            )
        
        stats = search_system.get_performance_stats()
        
        # Build response with tracing
        response = {
            "status": "ok",
            "data": stats,
            "trace_id": trace_id,
            "request_id": trace_id
        }
        
        # Add trace ID to response headers
        response_headers = {
            "X-Request-ID": trace_id,
            "X-Trace-ID": trace_id
        }
        
        return jsonify(response), 200, response_headers
        
    except SearchSystemUnavailableError as e:
        logger.error(f"Failed to get performance stats: {str(e)}", extra={"trace_id": trace_id})
        return create_problem_response(e, trace_id=trace_id)
    except Exception as e:
        logger.error(f"Unexpected error getting performance stats: {str(e)}", exc_info=True, extra={"trace_id": trace_id})
        search_error = SearchSystemUnavailableError(
            f"Failed to get performance stats: {str(e)}",
            error_code="PERFORMANCE_STATS_FAILED",
            details={"error": str(e)}
        )
        return create_problem_response(search_error, trace_id=trace_id)


