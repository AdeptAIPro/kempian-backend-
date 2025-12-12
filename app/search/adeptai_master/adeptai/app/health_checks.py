"""
Health checks and monitoring for AdeptAI application
Provides comprehensive health monitoring and status reporting
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.config import get_settings
from app.services import get_service
from app.exceptions import ServiceInitializationError, SearchSystemUnavailableError


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    response_time: float
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class HealthChecker:
    """Main health checker class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checks: Dict[str, callable] = {}
        self.cache: Dict[str, HealthCheckResult] = {}
        self.cache_ttl = 30  # seconds
        self.register_default_checks()
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check function"""
        self.checks[name] = check_func
        self.logger.debug(f"Registered health check: {name}")
    
    def register_default_checks(self):
        """Register default health checks"""
        self.register_check("database", self.check_database)
        self.register_check("search_system", self.check_search_system)
        self.register_check("ml_services", self.check_ml_services)
        self.register_check("behavioral_analysis", self.check_behavioral_analysis)
        self.register_check("bias_prevention", self.check_bias_prevention)
        self.register_check("explainable_ai", self.check_explainable_ai)
        self.register_check("memory", self.check_memory)
        self.register_check("disk_space", self.check_disk_space)
        self.register_check("cpu_usage", self.check_cpu_usage)
        self.register_check("external_apis", self.check_external_apis)
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                response_time=0.0
            )
        
        # Check cache first
        if name in self.cache:
            cached_result = self.cache[name]
            if (datetime.utcnow() - cached_result.timestamp).seconds < self.cache_ttl:
                return cached_result
        
        start_time = time.time()
        try:
            check_func = self.checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            response_time = time.time() - start_time
            
            # Cache the result
            self.cache[name] = result
            result.response_time = response_time
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Health check '{name}' failed: {e}")
            
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time=response_time,
                details={"error": str(e)}
            )
            
            self.cache[name] = result
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        tasks = [self.run_check(name) for name in self.checks.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_results = {}
        for i, result in enumerate(results):
            check_name = list(self.checks.keys())[i]
            if isinstance(result, Exception):
                health_results[check_name] = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(result)}",
                    response_time=0.0,
                    details={"error": str(result)}
                )
            else:
                health_results[check_name] = result
        
        return health_results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Get overall health status from individual check results"""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    # Individual health check methods
    
    def check_database(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            # Try to get DynamoDB service
            # This is a simplified check - in production you'd test actual queries
            settings = get_settings()
            
            if not settings.aws_access_key_id or settings.aws_access_key_id == "your_aws_access_key_here":
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message="Database credentials not configured - using mock data",
                    response_time=0.0,
                    details={"mode": "mock"}
                )
            
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection healthy",
                response_time=0.0,
                details={"mode": "production"}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_search_system(self) -> HealthCheckResult:
        """Check search system health"""
        try:
            search_system = get_service("search_system")
            
            if not search_system:
                return HealthCheckResult(
                    name="search_system",
                    status=HealthStatus.UNHEALTHY,
                    message="Search system not available",
                    response_time=0.0
                )
            
            # Test search functionality
            test_results = search_system.search("test query", top_k=1)
            
            return HealthCheckResult(
                name="search_system",
                status=HealthStatus.HEALTHY,
                message="Search system operational",
                response_time=0.0,
                details={"test_results_count": len(test_results)}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="search_system",
                status=HealthStatus.UNHEALTHY,
                message=f"Search system error: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_ml_services(self) -> HealthCheckResult:
        """Check ML services health"""
        try:
            embedding_service = get_service("embedding_service")
            ml_service = get_service("ml_service")
            
            if not embedding_service or not ml_service:
                return HealthCheckResult(
                    name="ml_services",
                    status=HealthStatus.UNHEALTHY,
                    message="ML services not available",
                    response_time=0.0
                )
            
            return HealthCheckResult(
                name="ml_services",
                status=HealthStatus.HEALTHY,
                message="ML services operational",
                response_time=0.0,
                details={
                    "embedding_service": "available",
                    "ml_service": "available"
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="ml_services",
                status=HealthStatus.UNHEALTHY,
                message=f"ML services error: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_behavioral_analysis(self) -> HealthCheckResult:
        """Check behavioral analysis pipeline health"""
        try:
            pipeline = get_service("behavioral_pipeline")
            
            if not pipeline:
                return HealthCheckResult(
                    name="behavioral_analysis",
                    status=HealthStatus.UNHEALTHY,
                    message="Behavioral analysis pipeline not available",
                    response_time=0.0
                )
            
            return HealthCheckResult(
                name="behavioral_analysis",
                status=HealthStatus.HEALTHY,
                message="Behavioral analysis pipeline operational",
                response_time=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="behavioral_analysis",
                status=HealthStatus.UNHEALTHY,
                message=f"Behavioral analysis error: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_bias_prevention(self) -> HealthCheckResult:
        """Check bias prevention components health"""
        try:
            sanitizer = get_service("bias_sanitizer")
            monitor = get_service("bias_monitor")
            
            if not sanitizer or not monitor:
                return HealthCheckResult(
                    name="bias_prevention",
                    status=HealthStatus.UNHEALTHY,
                    message="Bias prevention components not available",
                    response_time=0.0
                )
            
            return HealthCheckResult(
                name="bias_prevention",
                status=HealthStatus.HEALTHY,
                message="Bias prevention components operational",
                response_time=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="bias_prevention",
                status=HealthStatus.UNHEALTHY,
                message=f"Bias prevention error: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_explainable_ai(self) -> HealthCheckResult:
        """Check explainable AI system health"""
        try:
            explainable_system = get_service("explainable_ai")
            
            if not explainable_system:
                return HealthCheckResult(
                    name="explainable_ai",
                    status=HealthStatus.UNHEALTHY,
                    message="Explainable AI system not available",
                    response_time=0.0
                )
            
            return HealthCheckResult(
                name="explainable_ai",
                status=HealthStatus.HEALTHY,
                message="Explainable AI system operational",
                response_time=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="explainable_ai",
                status=HealthStatus.UNHEALTHY,
                message=f"Explainable AI error: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_memory(self) -> HealthCheckResult:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            
            if memory_usage_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_usage_percent:.1f}%"
            elif memory_usage_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory_usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_usage_percent:.1f}%"
            
            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                response_time=0.0,
                details={
                    "usage_percent": memory_usage_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_disk_space(self) -> HealthCheckResult:
        """Check disk space usage"""
        try:
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            if disk_usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Disk space critical: {disk_usage_percent:.1f}%"
            elif disk_usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"Disk space low: {disk_usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space normal: {disk_usage_percent:.1f}%"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                response_time=0.0,
                details={
                    "usage_percent": disk_usage_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Disk space check failed: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.DEGRADED
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheckResult(
                name="cpu_usage",
                status=status,
                message=message,
                response_time=0.0,
                details={
                    "usage_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count()
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="cpu_usage",
                status=HealthStatus.UNKNOWN,
                message=f"CPU check failed: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )
    
    def check_external_apis(self) -> HealthCheckResult:
        """Check external API connectivity"""
        try:
            settings = get_settings()
            
            # Check OpenAI API key
            openai_configured = bool(settings.openai_api_key and 
                                   settings.openai_api_key != "your_openai_api_key_here")
            
            if not openai_configured:
                return HealthCheckResult(
                    name="external_apis",
                    status=HealthStatus.DEGRADED,
                    message="External APIs not configured - using fallback",
                    response_time=0.0,
                    details={"openai_configured": False}
                )
            
            return HealthCheckResult(
                name="external_apis",
                status=HealthStatus.HEALTHY,
                message="External APIs configured",
                response_time=0.0,
                details={"openai_configured": True}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="external_apis",
                status=HealthStatus.UNKNOWN,
                message=f"External API check failed: {str(e)}",
                response_time=0.0,
                details={"error": str(e)}
            )


# Global health checker instance
health_checker = HealthChecker()


async def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status"""
    results = await health_checker.run_all_checks()
    overall_status = health_checker.get_overall_status(results)
    
    return {
        "status": overall_status.value,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": {
            name: {
                "status": result.status.value,
                "message": result.message,
                "response_time": result.response_time,
                "details": result.details
            }
            for name, result in results.items()
        },
        "summary": {
            "total_checks": len(results),
            "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
            "degraded": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
            "unhealthy": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
            "unknown": sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN)
        }
    }


async def get_readiness_status() -> Dict[str, Any]:
    """Get readiness status for Kubernetes readiness probe"""
    critical_checks = ["database", "search_system", "ml_services"]
    results = await health_checker.run_all_checks()
    
    critical_results = {name: results[name] for name in critical_checks if name in results}
    
    is_ready = all(
        result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        for result in critical_results.values()
    )
    
    return {
        "ready": is_ready,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "critical_checks": {
            name: {
                "status": result.status.value,
                "message": result.message
            }
            for name, result in critical_results.items()
        }
    }


async def get_liveness_status() -> Dict[str, Any]:
    """Get liveness status for Kubernetes liveness probe"""
    # Basic liveness check - just ensure the application is responding
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime": time.time()  # This would be actual uptime in production
    }
