#!/usr/bin/env python3
"""
Test script for cloud compatibility of the downloader
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datafeed.downloader import CloudCompatibleDownloader, is_streamlit_cloud
from utils.logging import logger


def test_cloud_detection():
    """Test if cloud detection works"""
    print("Testing cloud detection...")
    is_cloud = is_streamlit_cloud()
    print(f"Running on Streamlit Cloud: {is_cloud}")
    return is_cloud


def test_cloud_downloader():
    """Test the cloud-compatible downloader"""
    print("\nTesting cloud-compatible downloader...")
    
    # Test parameters
    tickers = ['AAPL', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    interval = '1d'
    fields = ['Open', 'Low', 'High', 'Close', 'Volume']
    
    try:
        result = CloudCompatibleDownloader.download_with_fallbacks(
            tickers, start_date, end_date, interval, fields
        )
        
        if not result.empty:
            print(f"✅ Success! Downloaded data shape: {result.shape}")
            print(f"Columns: {result.columns.tolist()}")
            print(f"Index range: {result.index.min()} to {result.index.max()}")
            return True
        else:
            print("❌ Downloaded data is empty")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def test_single_ticker():
    """Test single ticker download"""
    print("\nTesting single ticker download...")
    
    tickers = ['AAPL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    interval = '1d'
    fields = ['Open', 'Low', 'High', 'Close', 'Volume']
    
    try:
        result = CloudCompatibleDownloader.download_with_fallbacks(
            tickers, start_date, end_date, interval, fields
        )
        
        if not result.empty:
            print(f"✅ Success! Downloaded data shape: {result.shape}")
            return True
        else:
            print("❌ Downloaded data is empty")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def test_intraday_fallback():
    """Test intraday data fallback"""
    print("\nTesting intraday data fallback...")
    
    tickers = ['AAPL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    interval = '1m'  # This will likely fail and trigger fallback
    fields = ['Open', 'Low', 'High', 'Close', 'Volume']
    
    try:
        result = CloudCompatibleDownloader.download_with_fallbacks(
            tickers, start_date, end_date, interval, fields
        )
        
        if not result.empty:
            print(f"✅ Success! Downloaded data shape: {result.shape}")
            print(f"Interval used: {result.index.freq if hasattr(result.index, 'freq') else 'Unknown'}")
            return True
        else:
            print("❌ Downloaded data is empty")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Cloud Compatibility")
    print("=" * 50)
    
    # Test cloud detection
    cloud_detection_ok = test_cloud_detection()
    
    # Test basic download
    basic_download_ok = test_cloud_downloader()
    
    # Test single ticker
    single_ticker_ok = test_single_ticker()
    
    # Test intraday fallback
    intraday_fallback_ok = test_intraday_fallback()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Cloud Detection: {'✅ PASS' if cloud_detection_ok else '❌ FAIL'}")
    print(f"Basic Download: {'✅ PASS' if basic_download_ok else '❌ FAIL'}")
    print(f"Single Ticker: {'✅ PASS' if single_ticker_ok else '❌ FAIL'}")
    print(f"Intraday Fallback: {'✅ PASS' if intraday_fallback_ok else '❌ FAIL'}")
    
    all_passed = all([basic_download_ok, single_ticker_ok, intraday_fallback_ok])
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    main()
