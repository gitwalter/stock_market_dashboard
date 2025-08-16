#!/usr/bin/env python3
"""
Logging and monitoring utilities for Stock Market Dashboard
Provides structured logging and performance monitoring
"""

import logging
import json
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps


class DashboardLogger:
    """Structured logging for the dashboard"""
    
    def __init__(self, name: str = 'stock_dashboard', level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Add console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_data_download(self, tickers: list, success: bool, duration: float, 
                         error: Optional[str] = None):
        """Log data download events"""
        log_entry = {
            'event': 'data_download',
            'tickers': tickers,
            'success': success,
            'duration': duration,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_strategy_execution(self, strategy_name: str, success: bool, 
                             duration: float, error: Optional[str] = None):
        """Log strategy execution events"""
        log_entry = {
            'event': 'strategy_execution',
            'strategy': strategy_name,
            'success': success,
            'duration': duration,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_chart_generation(self, chart_type: str, success: bool, 
                           duration: float, error: Optional[str] = None):
        """Log chart generation events"""
        log_entry = {
            'event': 'chart_generation',
            'chart_type': chart_type,
            'success': success,
            'duration': duration,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict] = None):
        """Log error events"""
        log_entry = {
            'event': 'error',
            'error_type': error_type,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.error(json.dumps(log_entry))
    
    def log_performance(self, operation: str, duration: float, 
                       memory_usage: Optional[float] = None):
        """Log performance metrics"""
        log_entry = {
            'event': 'performance',
            'operation': operation,
            'duration': duration,
            'memory_usage': memory_usage,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_entry))


class PerformanceMonitor:
    """Monitor application performance"""
    
    def __init__(self):
        self.metrics = {}
        self.logger = DashboardLogger()
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {
            'start': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        if operation in self.metrics:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - self.metrics[operation]['start']
            memory_usage = end_memory - self.metrics[operation]['start_memory']
            
            self.metrics[operation].update({
                'duration': duration,
                'end_memory': end_memory,
                'memory_usage': memory_usage,
                'end': end_time
            })
            
            # Log performance
            self.logger.log_performance(operation, duration, memory_usage)
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.metrics
    
    def get_operation_metrics(self, operation: str) -> Optional[Dict]:
        """Get metrics for a specific operation"""
        return self.metrics.get(operation)
    
    def clear_metrics(self):
        """Clear all metrics"""
        self.metrics.clear()


def monitor_performance(operation_name: Optional[str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            monitor = PerformanceMonitor()
            
            try:
                monitor.start_timer(op_name)
                result = func(*args, **kwargs)
                monitor.end_timer(op_name)
                return result
            except Exception as e:
                monitor.end_timer(op_name)
                raise
        return wrapper
    return decorator


# Global instances
logger = DashboardLogger()
performance_monitor = PerformanceMonitor()
