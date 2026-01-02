import os
from decimal import Decimal
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

from app.simple_logger import get_logger

try:
    # Re‑use the same DynamoDB table that the search service uses for candidates
    from app.search.service import table as search_table  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    search_table = None  # type: ignore

logger = get_logger("candidates_api")

candidates_bp = Blueprint("candidates_bp", __name__)


def _convert_dynamodb_types(value: Any) -> Any:
    """
    Recursively convert DynamoDB types (Decimal, etc.) into plain JSON‑serializable types.
    """
    if isinstance(value, Decimal):
        # Cast Decimals to int when they are whole numbers, otherwise to float
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, list):
        return [_convert_dynamodb_types(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_dynamodb_types(v) for k, v in value.items()}
    return value


@candidates_bp.route("/candidates/all", methods=["GET"])
def get_all_candidates():
    """
    Fetch a list of candidates from DynamoDB for the talent matching UI.

    Frontend: CandidateSearchWindow.tsx calls GET /api/candidates/all and then
    performs its own mapping / filtering, so we return a light wrapper with a
    "candidates" array.

    Query params:
      - limit (int, optional): maximum number of candidates to return (default 500, max 5000)
    """
    table = search_table

    if table is None:
        logger.error("DynamoDB table is not configured for candidates API")
        return (
            jsonify(
                {
                    "candidates": [],
                    "error": "DynamoDB not configured on backend",
                }
            ),
            500,
        )

    # Safety limit to avoid scanning the entire table in one request
    try:
        limit_param = request.args.get("limit")
        if limit_param is not None:
            requested_limit = int(limit_param)
        else:
            requested_limit = 500
    except ValueError:
        requested_limit = 500

    max_limit = int(os.getenv("CANDIDATES_API_MAX_LIMIT", "5000"))
    limit = max(1, min(requested_limit, max_limit))

    items: List[Dict[str, Any]] = []
    last_evaluated_key: Dict[str, Any] | None = None

    try:
        while len(items) < limit:
            page_limit = min(1000, limit - len(items))
            scan_kwargs: Dict[str, Any] = {"Limit": page_limit}
            if last_evaluated_key:
                scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

            response = table.scan(**scan_kwargs)
            page_items = response.get("Items", []) or []
            items.extend(page_items)

            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break

        logger.info(
            "Loaded %s candidates from DynamoDB for /api/candidates/all (requested_limit=%s)",
            len(items),
            limit,
        )
    except Exception as e:
        logger.error("Error scanning DynamoDB for candidates: %s", e, exc_info=True)
        return (
            jsonify(
                {
                    "candidates": [],
                    "error": "Failed to load candidates from DynamoDB",
                }
            ),
            500,
        )

    clean_items = [_convert_dynamodb_types(item) for item in items]

    return jsonify({"candidates": clean_items}), 200


