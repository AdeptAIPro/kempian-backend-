"""
Module Health Monitoring
Provides endpoints to check the health status of all backend modules.
"""

from flask import Blueprint, jsonify
from app.module_isolation import get_isolation_manager
from app.simple_logger import get_logger

logger = get_logger("module_health")

module_health_bp = Blueprint('module_health', __name__)


@module_health_bp.route('/health/modules', methods=['GET'])
def get_modules_health():
    """Get health status of all modules"""
    try:
        manager = get_isolation_manager()
        status = manager.get_all_status()
        
        # Calculate overall health
        total_modules = len(status)
        healthy_modules = sum(1 for s in status.values() if s['status'] == 'healthy')
        degraded_modules = sum(1 for s in status.values() if s['status'] == 'degraded')
        unhealthy_modules = sum(1 for s in status.values() if s['status'] in ['unhealthy', 'circuit_open'])
        
        overall_status = 'healthy'
        if unhealthy_modules > 0:
            overall_status = 'unhealthy'
        elif degraded_modules > 0:
            overall_status = 'degraded'
            
        return jsonify({
            'overall_status': overall_status,
            'summary': {
                'total_modules': total_modules,
                'healthy': healthy_modules,
                'degraded': degraded_modules,
                'unhealthy': unhealthy_modules
            },
            'modules': status
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting module health: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to get module health status',
            'message': str(e)
        }), 500


@module_health_bp.route('/health/modules/<module_name>', methods=['GET'])
def get_module_health(module_name: str):
    """Get health status of a specific module"""
    try:
        manager = get_isolation_manager()
        health = manager.get_module_health(module_name)
        status = health.get_status()
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Error getting health for module {module_name}: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Failed to get health for module {module_name}',
            'message': str(e)
        }), 500


@module_health_bp.route('/health/modules/<module_name>/reset', methods=['POST'])
def reset_module_health(module_name: str):
    """Reset health status of a specific module"""
    try:
        manager = get_isolation_manager()
        manager.reset_module(module_name)
        
        return jsonify({
            'message': f'Module {module_name} health status reset',
            'module': module_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error resetting health for module {module_name}: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Failed to reset health for module {module_name}',
            'message': str(e)
        }), 500

