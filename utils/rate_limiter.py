#!/usr/bin/env python3
"""
Rate limiting utilities for Stock Market Dashboard
Provides rate limiting for API calls and data downloads
"""

import time
from typing import Dict, List, Optional
from utils.exceptions import RateLimitError


class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
    
    def can_call(self) -> bool:
        """Check if API call is allowed"""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call for call in self.calls if now - call < self.time_window]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record an API call"""
        self.calls.append(time.time())
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        if not self.can_call():
            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = self.time_window - (time.time() - oldest_call)
            
            if wait_time > 0:
                time.sleep(wait_time)
    
    def call(self, func, *args, **kwargs):
        """Execute function with rate limiting"""
        self.wait_if_needed()
        
        if not self.can_call():
            raise RateLimitError(f"Rate limit exceeded: {self.max_calls} calls per {self.time_window} seconds")
        
        self.record_call()
        return func(*args, **kwargs)


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts based on API responses"""
    
    def __init__(self, max_calls: int, time_window: int, 
                 backoff_factor: float = 2.0, max_backoff: int = 300):
        """
        Initialize adaptive rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
            backoff_factor: Factor to increase wait time on errors
            max_backoff: Maximum backoff time in seconds
        """
        super().__init__(max_calls, time_window)
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.current_backoff = 0
        self.error_count = 0
    
    def handle_error(self):
        """Handle API error by increasing backoff"""
        self.error_count += 1
        self.current_backoff = min(
            self.current_backoff * self.backoff_factor,
            self.max_backoff
        )
    
    def handle_success(self):
        """Handle successful API call by reducing backoff"""
        if self.error_count > 0:
            self.error_count -= 1
            self.current_backoff = max(
                self.current_backoff / self.backoff_factor,
                0
            )
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded, including backoff"""
        super().wait_if_needed()
        
        if self.current_backoff > 0:
            time.sleep(self.current_backoff)


class YahooFinanceRateLimiter(AdaptiveRateLimiter):
    """Rate limiter specifically for Yahoo Finance API"""
    
    def __init__(self):
        # Yahoo Finance has rate limits, so be conservative
        super().__init__(max_calls=100, time_window=60)  # 100 calls per minute
    
    def call(self, func, *args, **kwargs):
        """Execute function with Yahoo Finance specific rate limiting"""
        try:
            result = super().call(func, *args, **kwargs)
            self.handle_success()
            return result
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                self.handle_error()
                raise RateLimitError(f"Yahoo Finance rate limit exceeded: {e}")
            raise


# Global rate limiters
yahoo_finance_limiter = YahooFinanceRateLimiter()
general_limiter = RateLimiter(max_calls=1000, time_window=60)  # 1000 calls per minute
