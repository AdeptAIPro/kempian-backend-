"""
Log Management Utilities
Provides log rotation, cleanup, and monitoring functionality.
"""

import os
import glob
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging
from app.logging_config import get_logger
from .logging_config import get_logger

class LogManager:
    """Manages log files including rotation, compression, and cleanup"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or os.path.join(os.path.dirname(__file__), '..', 'logs')
        self.logger = get_logger('app')
        
    def compress_old_logs(self, days_old: int = 7):
        """Compress log files older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            compressed_count = 0
            
            # Find all log files
            log_files = glob.glob(os.path.join(self.log_dir, "*.log"))
            
            for log_file in log_files:
                file_path = Path(log_file)
                
                # Skip already compressed files
                if file_path.suffix == '.gz':
                    continue
                    
                # Check if file is old enough
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    # Compress the file
                    compressed_file = f"{log_file}.gz"
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original file
                    os.remove(log_file)
                    compressed_count += 1
                    
                    self.logger.info(f"Compressed log file: {file_path.name}")
            
            self.logger.info(f"Compressed {compressed_count} log files")
            return compressed_count
            
        except Exception as e:
            self.logger.error(f"Error compressing logs: {str(e)}")
            return 0
    
    def cleanup_old_logs(self, days_old: int = 30):
        """Delete compressed log files older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            # Find all compressed log files
            compressed_files = glob.glob(os.path.join(self.log_dir, "*.log.gz"))
            
            for log_file in compressed_files:
                file_path = Path(log_file)
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    os.remove(log_file)
                    deleted_count += 1
                    self.logger.info(f"Deleted old log file: {file_path.name}")
            
            self.logger.info(f"Deleted {deleted_count} old log files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up logs: {str(e)}")
            return 0
    
    def get_log_stats(self):
        """Get statistics about log files"""
        try:
            stats = {
                'total_files': 0,
                'total_size': 0,
                'compressed_files': 0,
                'compressed_size': 0,
                'uncompressed_files': 0,
                'uncompressed_size': 0,
                'oldest_file': None,
                'newest_file': None,
                'files_by_type': {}
            }
            
            # Find all log files
            all_files = glob.glob(os.path.join(self.log_dir, "*"))
            
            for file_path in all_files:
                if os.path.isfile(file_path):
                    file_stat = os.stat(file_path)
                    file_size = file_stat.st_size
                    file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    stats['total_files'] += 1
                    stats['total_size'] += file_size
                    
                    # Track oldest and newest files
                    if stats['oldest_file'] is None or file_mtime < stats['oldest_file']:
                        stats['oldest_file'] = file_mtime
                    if stats['newest_file'] is None or file_mtime > stats['newest_file']:
                        stats['newest_file'] = file_mtime
                    
                    # Categorize by file type
                    file_name = os.path.basename(file_path)
                    if file_name.endswith('.gz'):
                        stats['compressed_files'] += 1
                        stats['compressed_size'] += file_size
                    else:
                        stats['uncompressed_files'] += 1
                        stats['uncompressed_size'] += file_size
                    
                    # Categorize by log type
                    if 'app' in file_name:
                        stats['files_by_type']['app'] = stats['files_by_type'].get('app', 0) + 1
                    elif 'error' in file_name:
                        stats['files_by_type']['error'] = stats['files_by_type'].get('error', 0) + 1
                    elif 'access' in file_name:
                        stats['files_by_type']['access'] = stats['files_by_type'].get('access', 0) + 1
                    elif 'performance' in file_name:
                        stats['files_by_type']['performance'] = stats['files_by_type'].get('performance', 0) + 1
                    elif 'database' in file_name:
                        stats['files_by_type']['database'] = stats['files_by_type'].get('database', 0) + 1
                    elif 'security' in file_name:
                        stats['files_by_type']['security'] = stats['files_by_type'].get('security', 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting log stats: {str(e)}")
            return {}
    
    def format_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def print_log_stats(self):
        """Print log statistics to console"""
        stats = self.get_log_stats()
        
        if not stats:
            print("No log statistics available")
            return
        
        print("\n" + "="*50)
        print("KEMPIAN BACKEND LOG STATISTICS")
        print("="*50)
        print(f"Total Files: {stats['total_files']}")
        print(f"Total Size: {self.format_size(stats['total_size'])}")
        print(f"Uncompressed Files: {stats['uncompressed_files']} ({self.format_size(stats['uncompressed_size'])})")
        print(f"Compressed Files: {stats['compressed_files']} ({self.format_size(stats['compressed_size'])})")
        
        if stats['oldest_file']:
            print(f"Oldest File: {stats['oldest_file'].strftime('%Y-%m-%d %H:%M:%S')}")
        if stats['newest_file']:
            print(f"Newest File: {stats['newest_file'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nFiles by Type:")
        for log_type, count in stats['files_by_type'].items():
            print(f"  {log_type}: {count} files")
        
        print("="*50)
    
    def run_maintenance(self, compress_days: int = 7, cleanup_days: int = 30):
        """Run complete log maintenance"""
        self.logger.info("Starting log maintenance")
        
        # Compress old logs
        compressed = self.compress_old_logs(compress_days)
        
        # Clean up very old logs
        deleted = self.cleanup_old_logs(cleanup_days)
        
        # Print statistics
        self.print_log_stats()
        
        self.logger.info(f"Log maintenance completed: {compressed} compressed, {deleted} deleted")
        
        return {
            'compressed': compressed,
            'deleted': deleted
        }

def setup_log_rotation():
    """Setup automatic log rotation using cron or similar"""
    # This would typically be called from a cron job or scheduled task
    log_manager = LogManager()
    return log_manager.run_maintenance()

# CLI interface for log management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kempian Log Management")
    parser.add_argument("--compress", type=int, default=7, help="Compress logs older than N days")
    parser.add_argument("--cleanup", type=int, default=30, help="Delete logs older than N days")
    parser.add_argument("--stats", action="store_true", help="Show log statistics")
    parser.add_argument("--maintenance", action="store_true", help="Run full maintenance")
    
    args = parser.parse_args()
    
    log_manager = LogManager()
    
    if args.stats:
        log_manager.print_log_stats()
    elif args.maintenance:
        log_manager.run_maintenance(args.compress, args.cleanup)
    else:
        print("Use --help for available options")
