#!/usr/bin/env python3
"""
Input validation utilities for Stock Market Dashboard
Provides validation functions for user inputs and data
"""

import re
import pandas as pd
from typing import List, Union, Optional
from datetime import datetime, timedelta
from utils.exceptions import ValidationError


def validate_ticker_symbol(ticker: str) -> bool:
    """
    Validate ticker symbol format
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic ticker pattern: 1-5 uppercase letters, numbers, and some special chars
    pattern = r'^[A-Z0-9^=]{1,10}$'
    return bool(re.match(pattern, ticker.strip()))


def validate_ticker_list(tickers: List[str]) -> List[str]:
    """
    Validate a list of ticker symbols
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        List of valid tickers
        
    Raises:
        ValidationError: If no valid tickers found
    """
    if not tickers:
        raise ValidationError("Ticker list cannot be empty")
    
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        if validate_ticker_symbol(ticker):
            valid_tickers.append(ticker.strip())
        else:
            invalid_tickers.append(ticker)
    
    if invalid_tickers:
        print(f"⚠️  Invalid tickers ignored: {invalid_tickers}")
    
    if not valid_tickers:
        raise ValidationError("No valid tickers found in the list")
    
    return valid_tickers


def validate_date_range(start_date: Union[str, datetime], 
                       end_date: Union[str, datetime]) -> tuple[datetime, datetime]:
    """
    Validate date range
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
        
    Raises:
        ValidationError: If dates are invalid
    """
    try:
        # Convert to datetime if strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Validate date types
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValidationError("Invalid date format")
        
        # Check if start is before end
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")
        
        # Check if dates are not too far in the future
        max_future_date = datetime.now() + timedelta(days=365)
        if start_date > max_future_date or end_date > max_future_date:
            raise ValidationError("Dates cannot be more than 1 year in the future")
        
        return start_date, end_date
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Invalid date format: {e}")


def validate_dataframe(data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    Validate DataFrame structure and content
    
    Args:
        data: DataFrame to validate
        required_columns: List of required columns
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If DataFrame is invalid
    """
    if not isinstance(data, pd.DataFrame):
        raise ValidationError("Data must be a pandas DataFrame")
    
    if data.empty:
        raise ValidationError("DataFrame cannot be empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
    
    return True


def sanitize_user_input(user_input: str) -> str:
    """
    Sanitize user input to prevent injection attacks
    
    Args:
        user_input: Raw user input
        
    Returns:
        Sanitized input
    """
    if not user_input:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', user_input.strip())
    
    # Limit length
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000]
    
    return sanitized


def validate_numeric_range(value: Union[int, float], 
                          min_val: Optional[Union[int, float]] = None,
                          max_val: Optional[Union[int, float]] = None,
                          name: str = "value") -> Union[int, float]:
    """
    Validate numeric value is within range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be at least {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be at most {max_val}")
    
    return value


def validate_strategy_parameters(params: dict) -> dict:
    """
    Validate strategy parameters
    
    Args:
        params: Strategy parameters dictionary
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    validated = {}
    
    # Validate common parameters
    if 'commission' in params:
        validated['commission'] = validate_numeric_range(
            params['commission'], 0, 1, "commission"
        )
    
    if 'slippage' in params:
        validated['slippage'] = validate_numeric_range(
            params['slippage'], 0, 1, "slippage"
        )
    
    if 'cash' in params:
        validated['cash'] = validate_numeric_range(
            params['cash'], 0, None, "cash"
        )
    
    return validated


def validate_chart_options(options: dict) -> dict:
    """
    Validate chart display options
    
    Args:
        options: Chart options dictionary
        
    Returns:
        Validated options
    """
    validated = {}
    
    # Validate chart dimensions
    if 'height' in options:
        validated['height'] = validate_numeric_range(
            options['height'], 100, 2000, "chart height"
        )
    
    if 'width' in options:
        validated['width'] = validate_numeric_range(
            options['width'], 100, 2000, "chart width"
        )
    
    # Validate boolean options
    for key in ['show_volume', 'show_indicators', 'interactive']:
        if key in options:
            validated[key] = bool(options[key])
    
    return validated
