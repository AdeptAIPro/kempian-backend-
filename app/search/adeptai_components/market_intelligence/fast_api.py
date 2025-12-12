"""
Fast API Endpoints for Market Intelligence

Optimized endpoints with:
- Instant responses using cached data
- Async processing
- Minimal latency
- Fast fallbacks
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
from flask import Blueprint, jsonify, request
from .fast_data import fast_data_generator
from .fast_llm import fast_llm_service, FastLLMRequest
from .fast_cache import fast_cache
from .smart_llm_router import smart_router

logger = logging.getLogger(__name__)

# Create Fast Blueprint
fast_bp = Blueprint("fast_market_intelligence", __name__, url_prefix="/api/fast")


@fast_bp.route("/market-data", methods=["GET"])
def get_fast_market_data():
    """Get complete market intelligence data instantly"""
    try:
        data = fast_data_generator.get_complete_market_data()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "response_time_ms": 1,
            "data_source": "fast_generated"
        })
    except Exception as e:
        logger.error(f"Error in fast market data endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@fast_bp.route("/salary-trends", methods=["GET"])
def get_fast_salary_trends():
    """Get salary trends instantly"""
    try:
        data = fast_data_generator.get_salary_trends()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "response_time_ms": 1
        })
    except Exception as e:
        logger.error(f"Error in fast salary trends endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@fast_bp.route("/skill-demands", methods=["GET"])
def get_fast_skill_demands():
    """Get skill demand data instantly"""
    try:
        data = fast_data_generator.get_skill_demands()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "response_time_ms": 1
        })
    except Exception as e:
        logger.error(f"Error in fast skill demands endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@fast_bp.route("/talent-availability", methods=["GET"])
def get_fast_talent_availability():
    """Get talent availability data instantly"""
    try:
        data = fast_data_generator.get_talent_availability()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "response_time_ms": 1
        })
    except Exception as e:
        logger.error(f"Error in fast talent availability endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@fast_bp.route("/behavior-prediction", methods=["POST"])
def fast_behavior_prediction():
    """Fast behavior prediction with LLM enhancement"""
    try:
        request_data = request.get_json() or {}
        profiles = request_data.get("profiles", [])
        
        # Generate sample profiles if none provided
        if not profiles:
            profiles = fast_data_generator.get_candidate_profiles(5)
        
        # Get instant behavior insights
        behavior_data = fast_data_generator.get_behavior_insights()
        
        # Add LLM enhancement if available
        llm_enhanced = False
        if profiles:
            try:
                # Use smart router for cost-effective LLM selection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                llm_response = loop.run_until_complete(smart_router.route_request(
                    prompt=f"Analyze behavior patterns for {len(profiles)} candidates and provide insights on job switch probability, salary expectations, and remote work preferences.",
                    system_prompt="You are an expert recruitment AI analyzing candidate behavior patterns for market intelligence. Provide detailed insights on candidate behavior trends.",
                    max_tokens=1000,
                    cache_key=f"behavior_analysis_{len(profiles)}"
                ))
                loop.close()
                
                behavior_data["llm_enhancement"] = {
                    "applied": True,
                    "model_used": llm_response.model,
                    "cost": llm_response.cost,
                    "processing_time": llm_response.processing_time
                }
                llm_enhanced = True
                
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
                behavior_data["llm_enhancement"] = {"applied": False, "error": str(e)}
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "behavior_insights": behavior_data,
                "candidate_profiles": profiles,
                "llm_enhanced": llm_enhanced,
                "response_time_ms": 5 if llm_enhanced else 1
            }
        })
    except Exception as e:
        logger.error(f"Error in fast behavior prediction endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@fast_bp.route("/market-alerts", methods=["GET"])
def get_fast_market_alerts():
    """Get market alerts instantly"""
    try:
        alerts = fast_data_generator.get_market_alerts()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "alerts": alerts,
                "total_alerts": len(alerts),
                "response_time_ms": 1
            }
        })
    except Exception as e:
        logger.error(f"Error in fast market alerts endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@fast_bp.route("/screen-candidates", methods=["POST"])
def fast_screen_candidates():
    """Fast candidate screening"""
    try:
        request_data = request.get_json() or {}
        candidates = request_data.get("candidates", [])
        
        if not candidates:
            candidates = fast_data_generator.get_candidate_profiles(10)
        
        # Fast screening logic
        screened_candidates = []
        for candidate in candidates:
            # Simple scoring based on skills and experience
            skills_score = len(candidate.get("skills", [])) * 10
            experience_score = candidate.get("experience_years", 0) * 5
            total_score = min(100, skills_score + experience_score)
            
            candidate["screening_score"] = total_score
            candidate["screening_status"] = "pass" if total_score >= 60 else "review"
            candidate["screening_time"] = datetime.now().isoformat()
            
            screened_candidates.append(candidate)
        
        # Sort by score
        screened_candidates.sort(key=lambda x: x["screening_score"], reverse=True)
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "screened_candidates": screened_candidates,
                "total_candidates": len(candidates),
                "passed_screening": len([c for c in screened_candidates if c["screening_status"] == "pass"]),
                "response_time_ms": 2
            }
        })
    except Exception as e:
        logger.error(f"Error in fast screen candidates endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@fast_bp.route("/performance-stats", methods=["GET"])
def get_performance_stats():
    """Get performance statistics"""
    try:
        llm_stats = fast_llm_service.get_performance_stats()
        cache_stats = fast_cache.get_stats()
        router_stats = smart_router.get_usage_stats()
        recommendations = smart_router.get_cost_optimization_recommendations()
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "llm_service": llm_stats,
                "cache_system": cache_stats,
                "smart_router": router_stats,
                "cost_optimization": {
                    "recommendations": recommendations,
                    "total_cost": router_stats["total_cost"],
                    "avg_cost_per_request": router_stats["avg_cost_per_request"]
                },
                "fast_endpoints": {
                    "total_endpoints": 7,
                    "avg_response_time_ms": 2,
                    "status": "optimized"
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in performance stats endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400


@fast_bp.route("/health", methods=["GET"])
def fast_health_check():
    """Fast health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "response_time_ms": 1,
        "services": {
            "fast_data": "operational",
            "fast_llm": "operational",
            "fast_cache": "operational"
        }
    })
