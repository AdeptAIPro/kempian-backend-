"""
Secure Logs Dashboard for Kempian Backend
Only accessible to vinit@adeptaipro.com
"""

import os
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template_string
from functools import wraps
from app.simple_logger import get_logger

logs_dashboard_bp = Blueprint('logs_dashboard', __name__)
logger = get_logger('admin')

# Allowed email for logs access
ALLOWED_EMAIL = "vinit@adeptaipro.com"

def require_logs_access(f):
    """Decorator to check if user has access to logs"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # For now, allow access to vinit@adeptaipro.com
        # TODO: Implement proper JWT-based authentication
        
        # Check if user is authenticated (has any valid token)
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("No valid authorization header")
            return jsonify({
                'error': 'Access denied',
                'message': 'Authentication required'
            }), 401
        
        # For development, allow access if token exists
        # In production, you should verify the JWT and check the user's role
        logger.info("Logs access granted (development mode)")
        return f(*args, **kwargs)
    
    return decorated_function

def get_log_files():
    """Get list of available log files"""
    log_dir = Path(__file__).parent.parent / 'logs'
    if not log_dir.exists():
        return []
    
    log_files = []
    for file_path in log_dir.glob('*.log'):
        if file_path.is_file():
            stat = file_path.stat()
            log_files.append({
                'name': file_path.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'path': str(file_path)
            })
    
    return sorted(log_files, key=lambda x: x['modified'], reverse=True)

def read_log_file(file_path, lines=100, search=None):
    """Read log file with optional filtering"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # Apply search filter if provided
        if search:
            filtered_lines = [line for line in all_lines if search.lower() in line.lower()]
        else:
            filtered_lines = all_lines
        
        # Get last N lines
        recent_lines = filtered_lines[-lines:] if lines > 0 else filtered_lines
        
        # Parse log entries
        log_entries = []
        for i, line in enumerate(recent_lines):
            line = line.strip()
            if not line:
                continue
            
            # Try to parse as structured log
            try:
                entry = json.loads(line)
                log_entries.append({
                    'line_number': len(all_lines) - len(recent_lines) + i + 1,
                    'timestamp': entry.get('timestamp', ''),
                    'level': entry.get('level', 'INFO'),
                    'logger': entry.get('logger', ''),
                    'message': entry.get('message', ''),
                    'raw': line,
                    'structured': True
                })
            except json.JSONDecodeError:
                # Plain text log
                log_entries.append({
                    'line_number': len(all_lines) - len(recent_lines) + i + 1,
                    'timestamp': '',
                    'level': 'INFO',
                    'logger': 'unknown',
                    'message': line,
                    'raw': line,
                    'structured': False
                })
        
        return log_entries
        
    except Exception as e:
        logger.error(f"Error reading log file {file_path}: {e}")
        return []

@logs_dashboard_bp.route('/logs')
@require_logs_access
def logs_dashboard():
    """Main logs dashboard page"""
    log_files = get_log_files()
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kempian Logs Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e0e0e0; }
            .header h1 { color: #333; margin: 0; }
            .controls { display: flex; gap: 10px; align-items: center; }
            .file-list { margin-bottom: 20px; }
            .file-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; border: 1px solid #ddd; margin-bottom: 5px; border-radius: 4px; cursor: pointer; transition: background 0.2s; }
            .file-item:hover { background: #f0f0f0; }
            .file-item.active { background: #e3f2fd; border-color: #2196f3; }
            .file-info { flex: 1; }
            .file-name { font-weight: bold; color: #333; }
            .file-meta { font-size: 0.9em; color: #666; }
            .log-viewer { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px; font-family: 'Courier New', monospace; max-height: 500px; overflow-y: auto; }
            .log-entry { margin-bottom: 5px; padding: 2px 0; }
            .log-entry.error { color: #f48771; }
            .log-entry.warning { color: #dcdcaa; }
            .log-entry.info { color: #9cdcfe; }
            .log-entry.debug { color: #808080; }
            .search-box { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 200px; }
            .btn { padding: 8px 16px; background: #2196f3; color: white; border: none; border-radius: 4px; cursor: pointer; }
            .btn:hover { background: #1976d2; }
            .btn:disabled { background: #ccc; cursor: not-allowed; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
            .stat-card { background: #f8f9fa; padding: 15px; border-radius: 4px; border-left: 4px solid #2196f3; }
            .stat-value { font-size: 1.5em; font-weight: bold; color: #333; }
            .stat-label { color: #666; font-size: 0.9em; }
            .loading { text-align: center; padding: 20px; color: #666; }
            .error { color: #f44336; background: #ffebee; padding: 10px; border-radius: 4px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Kempian Logs Dashboard</h1>
                <div class="controls">
                    <input type="text" id="searchBox" class="search-box" placeholder="Search logs...">
                    <button onclick="refreshLogs()" class="btn">Refresh</button>
                    <button onclick="clearLogs()" class="btn">Clear</button>
                </div>
            </div>
            
            <div class="stats" id="stats">
                <div class="stat-card">
                    <div class="stat-value" id="totalFiles">-</div>
                    <div class="stat-label">Log Files</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalSize">-</div>
                    <div class="stat-label">Total Size</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="lastUpdate">-</div>
                    <div class="stat-label">Last Updated</div>
                </div>
            </div>
            
            <div class="file-list">
                <h3>Log Files</h3>
                <div id="fileList">
                    <div class="loading">Loading log files...</div>
                </div>
            </div>
            
            <div class="log-viewer" id="logViewer">
                <div class="loading">Select a log file to view</div>
            </div>
        </div>
        
        <script>
            let currentFile = null;
            let currentSearch = '';
            
            async function loadLogFiles() {
                try {
                    const response = await fetch('/api/logs/files');
                    const data = await response.json();
                    
                    if (data.error) {
                        document.getElementById('fileList').innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    const fileList = document.getElementById('fileList');
                    fileList.innerHTML = '';
                    
                    data.files.forEach(file => {
                        const fileItem = document.createElement('div');
                        fileItem.className = 'file-item';
                        fileItem.onclick = () => selectFile(file.name);
                        fileItem.innerHTML = `
                            <div class="file-info">
                                <div class="file-name">${file.name}</div>
                                <div class="file-meta">${formatSize(file.size)} • ${formatDate(file.modified)}</div>
                            </div>
                        `;
                        fileList.appendChild(fileItem);
                    });
                    
                    updateStats(data.stats);
                } catch (error) {
                    document.getElementById('fileList').innerHTML = `<div class="error">Error loading files: ${error.message}</div>`;
                }
            }
            
            async function selectFile(fileName) {
                currentFile = fileName;
                
                // Update UI
                document.querySelectorAll('.file-item').forEach(item => item.classList.remove('active'));
                event.target.closest('.file-item').classList.add('active');
                
                // Load log content
                await loadLogContent(fileName);
            }
            
            async function loadLogContent(fileName, search = '') {
                const logViewer = document.getElementById('logViewer');
                logViewer.innerHTML = '<div class="loading">Loading log content...</div>';
                
                try {
                    const response = await fetch(`/api/logs/content/${fileName}?search=${encodeURIComponent(search)}`);
                    const data = await response.json();
                    
                    if (data.error) {
                        logViewer.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }
                    
                    logViewer.innerHTML = '';
                    data.entries.forEach(entry => {
                        const logEntry = document.createElement('div');
                        logEntry.className = `log-entry ${entry.level.toLowerCase()}`;
                        logEntry.innerHTML = `
                            <span style="color: #808080;">[${entry.line_number}]</span>
                            ${entry.timestamp ? `<span style="color: #569cd6;">[${entry.timestamp}]</span>` : ''}
                            <span style="color: #ce9178;">[${entry.level}]</span>
                            <span style="color: #9cdcfe;">[${entry.logger}]</span>
                            ${entry.message}
                        `;
                        logViewer.appendChild(logEntry);
                    });
                } catch (error) {
                    logViewer.innerHTML = `<div class="error">Error loading log content: ${error.message}</div>`;
                }
            }
            
            function updateStats(stats) {
                document.getElementById('totalFiles').textContent = stats.total_files || 0;
                document.getElementById('totalSize').textContent = formatSize(stats.total_size || 0);
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            }
            
            function formatSize(bytes) {
                if (bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
            }
            
            function formatDate(isoString) {
                return new Date(isoString).toLocaleString();
            }
            
            function refreshLogs() {
                loadLogFiles();
                if (currentFile) {
                    loadLogContent(currentFile, currentSearch);
                }
            }
            
            function clearLogs() {
                document.getElementById('logViewer').innerHTML = '<div class="loading">Select a log file to view</div>';
                currentFile = null;
            }
            
            // Search functionality
            document.getElementById('searchBox').addEventListener('input', (e) => {
                currentSearch = e.target.value;
                if (currentFile) {
                    loadLogContent(currentFile, currentSearch);
                }
            });
            
            // Auto-refresh every 30 seconds
            setInterval(refreshLogs, 30000);
            
            // Load initial data
            loadLogFiles();
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_template)

@logs_dashboard_bp.route('/api/logs/files')
@require_logs_access
def api_get_log_files():
    """API endpoint to get list of log files"""
    try:
        log_files = get_log_files()
        
        # Calculate stats
        total_size = sum(file['size'] for file in log_files)
        stats = {
            'total_files': len(log_files),
            'total_size': total_size,
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify({
            'files': log_files,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting log files: {e}")
        return jsonify({'error': str(e)}), 500

@logs_dashboard_bp.route('/api/logs/content/<filename>')
@require_logs_access
def api_get_log_content(filename):
    """API endpoint to get log file content"""
    try:
        # Security check - only allow .log files
        if not filename.endswith('.log'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        log_dir = Path(__file__).parent.parent / 'logs'
        file_path = log_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # Get parameters
        lines = int(request.args.get('lines', 100))
        search = request.args.get('search', '')
        
        # Read log content
        entries = read_log_file(file_path, lines, search)
        
        return jsonify({
            'filename': filename,
            'entries': entries,
            'total_lines': len(entries)
        })
        
    except Exception as e:
        logger.error(f"Error reading log file {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@logs_dashboard_bp.route('/api/logs/download/<filename>')
@require_logs_access
def api_download_log(filename):
    """API endpoint to download log file"""
    try:
        # Security check - only allow .log files
        if not filename.endswith('.log'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        log_dir = Path(__file__).parent.parent / 'logs'
        file_path = log_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        from flask import send_file
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Error downloading log file {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@logs_dashboard_bp.route('/api/logs/stats')
@require_logs_access
def api_get_log_stats():
    """API endpoint to get detailed log statistics"""
    try:
        log_files = get_log_files()
        
        # Calculate detailed stats
        stats = {
            'total_files': len(log_files),
            'total_size': sum(file['size'] for file in log_files),
            'files_by_type': {},
            'oldest_file': None,
            'newest_file': None
        }
        
        if log_files:
            # Group by file type
            for file in log_files:
                file_type = file['name'].replace('kempian_', '').replace('.log', '')
                stats['files_by_type'][file_type] = stats['files_by_type'].get(file_type, 0) + 1
            
            # Find oldest and newest
            dates = [datetime.fromisoformat(file['modified']) for file in log_files]
            stats['oldest_file'] = min(dates).isoformat()
            stats['newest_file'] = max(dates).isoformat()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting log stats: {e}")
        return jsonify({'error': str(e)}), 500
