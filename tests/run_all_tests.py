#!/usr/bin/env python3
"""
Unified Test Runner for Monopoly RL Project

This script runs all tests with proper path setup to avoid import issues.
It can be run from anywhere in the project and will automatically set up
the correct Python paths.

Usage:
    python tests/run_all_tests.py
    # or from the tests directory:
    python run_all_tests.py
"""

import sys
import os
import unittest
from pathlib import Path

def setup_paths():
    """Set up Python paths for imports."""
    # Get the project root directory (parent of this file's directory)
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent
    
    # Add both the project root and MARL+IPPO directory to Python path
    paths_to_add = [
        str(project_root),
        str(project_root / "MARL+IPPO"),
        str(current_dir),  # Add tests directory
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"✅ Set up Python paths:")
    for path in paths_to_add:
        print(f"   - {path}")
    
    # Also add current working directory if not already there
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        print(f"   - {cwd} (current working directory)")

def run_tests():
    """Run all test modules."""
    print("\n🧪 Running Monopoly RL Test Suite")
    print("=" * 50)
    
    # Set up paths
    setup_paths()
    
    # Test modules to run
    test_modules = [
        'tests.monopoly_core_test',
        'tests.monopoly_env_test', 
        'tests.monopoly_reward_test'
    ]
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load tests from each module
    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTest(tests)
            print(f"✅ Loaded tests from {module}")
        except Exception as e:
            print(f"❌ Failed to load tests from {module}: {e}")
    
    # Run the tests
    print(f"\n🚀 Running {suite.countTestCases()} tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ Some tests failed. Check the output above.")
    
    return success

def test_imports():
    """Test that all critical imports work."""
    print("\n🔍 Testing critical imports...")
    
    imports_to_test = [
        ("monopoly_env.core.board", "Board"),
        ("monopoly_env.core.player", "Player"),
        ("monopoly_env.core.game_logic", "GameLogic"),
        ("monopoly_env.envs.monopoly_env", "MonopolyEnv"),
        ("monopoly_env.utils.reward_calculator", "RewardCalculator"),
    ]
    
    failed_imports = []
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            failed_imports.append((module_name, class_name, str(e)))
    
    if failed_imports:
        print(f"\n⚠️ Failed imports:")
        for module, class_name, error in failed_imports:
            print(f"   - {module}.{class_name}: {error}")
        return False
    else:
        print(f"\n✅ All imports successful!")
        return True

def main():
    """Main test runner."""
    print("🧪 Monopoly RL Test Suite")
    print("=" * 40)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed. Cannot proceed with unit tests.")
        return False
    
    # Run unit tests
    success = run_tests()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)