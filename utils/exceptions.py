#!/usr/bin/env python3
"""
Custom exceptions for Stock Market Dashboard
Provides specific exception types for different error scenarios
"""


class DashboardError(Exception):
    """Base exception for dashboard errors"""
    pass


class DataDownloadError(DashboardError):
    """Raised when data download fails"""
    pass


class ValidationError(DashboardError):
    """Raised when input validation fails"""
    pass


class ConfigurationError(DashboardError):
    """Raised when configuration is invalid"""
    pass


class StrategyError(DashboardError):
    """Raised when strategy execution fails"""
    pass


class AnalysisError(DashboardError):
    """Raised when analysis fails"""
    pass


class ChartError(DashboardError):
    """Raised when chart generation fails"""
    pass


class NetworkError(DashboardError):
    """Raised when network operations fail"""
    pass


class RateLimitError(DashboardError):
    """Raised when API rate limits are exceeded"""
    pass


class DataProcessingError(DashboardError):
    """Raised when data processing fails"""
    pass
