#!/usr/bin/env python3
"""
Test script to verify strategy imports and functionality
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_strategy_imports():
    """Test if all strategies can be imported"""
    print("🧪 Testing Strategy Imports")
    print("=" * 40)
    
    try:
        # Test individual imports
        from strategy.BuyAndHold import BuyAndHold
        print("✅ BuyAndHold imported successfully")
        
        from strategy.RSIStrategy import RSIStrategy
        print("✅ RSIStrategy imported successfully")
        
        from strategy.MinerviniMomentum import MinerviniMomentum
        print("✅ MinerviniMomentum imported successfully")
        
        from strategy.SmaCross import SmaCross
        print("✅ SmaCross imported successfully")
        
        from strategy.TrailingStopLoss import TrailingStopLoss
        print("✅ TrailingStopLoss imported successfully")
        
        # Test package import
        from strategy import BuyAndHold, RSIStrategy, MinerviniMomentum, SmaCross, TrailingStopLoss
        print("✅ All strategies imported from package successfully")
        
        # Test globals access (like in the main app)
        strategies = {
            'BuyAndHold': BuyAndHold,
            'RSIStrategy': RSIStrategy,
            'MinerviniMomentum': MinerviniMomentum,
            'SmaCross': SmaCross,
            'TrailingStopLoss': TrailingStopLoss
        }
        
        print("\n📋 Available Strategies:")
        for name, strategy_class in strategies.items():
            print(f"  - {name}: {strategy_class.__name__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_config_strategies():
    """Test if config strategy names match available strategies"""
    print("\n🔧 Testing Config Strategy Names")
    print("=" * 40)
    
    try:
        from config import config
        
        print(f"Config strategy names: {config.strategy_names}")
        
        # Check if all config strategies exist
        from strategy import BuyAndHold, RSIStrategy, MinerviniMomentum, SmaCross, TrailingStopLoss
        
        available_strategies = {
            'BuyAndHold': BuyAndHold,
            'RSIStrategy': RSIStrategy,
            'MinerviniMomentum': MinerviniMomentum,
            'SmaCross': SmaCross,
            'TrailingStopLoss': TrailingStopLoss
        }
        
        missing_strategies = []
        for strategy_name in config.strategy_names:
            if strategy_name in available_strategies:
                print(f"✅ {strategy_name} found")
            else:
                print(f"❌ {strategy_name} not found")
                missing_strategies.append(strategy_name)
        
        if missing_strategies:
            print(f"\n⚠️  Missing strategies: {missing_strategies}")
            return False
        else:
            print("\n✅ All config strategies are available")
            return True
            
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Strategy Import Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_strategy_imports()
    
    # Test config
    config_ok = test_config_strategies()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Strategy Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Config Validation: {'✅ PASS' if config_ok else '❌ FAIL'}")
    
    all_passed = imports_ok and config_ok
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Strategies are ready for use in the dashboard!")
    else:
        print("\n⚠️  Please fix the issues above before using strategies in the dashboard.")
    
    return all_passed


if __name__ == "__main__":
    main()
