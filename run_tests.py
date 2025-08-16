#!/usr/bin/env python3
"""
Test runner for Stock Market Dashboard
Run all tests and generate a comprehensive report
"""

import subprocess
import sys
import os
from datetime import datetime


def run_tests():
    """Run all tests and generate a report"""
    print("=" * 80)
    print("STOCK MARKET DASHBOARD - TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run pytest with coverage
    try:
        # Install pytest-cov if not available
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest-cov"], 
                      capture_output=True, check=False)
        
        # Run tests with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--cov=datafeed",
            "--cov=analyzer", 
            "--cov=strategy",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ], capture_output=True, text=True)
        
        print("TEST RESULTS:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("WARNINGS/ERRORS:")
            print("-" * 40)
            print(result.stderr)
        
        print("=" * 80)
        if result.returncode == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
        
        print(f"Coverage report generated in: htmlcov/index.html")
        print("=" * 80)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_specific_test(test_file):
    """Run a specific test file"""
    print(f"Running specific test: {test_file}")
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        f"tests/{test_file}", 
        "-v"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Run specific test
        test_file = sys.argv[1]
        success = run_specific_test(test_file)
    else:
        # Run all tests
        success = run_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
