#!/usr/bin/env python3
"""Test script for the auto-researching AI system."""

import os
import sys
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from research_agents.base_agent import ResearchExperiment, BaseAgent
        print("✅ Base agent imports successful")
    except ImportError as e:
        print(f"❌ Base agent import failed: {e}")
        return False
    
    try:
        from research_agents.suggestion_agent import SuggestionAgent
        print("✅ Suggestion agent import successful")
    except ImportError as e:
        print(f"❌ Suggestion agent import failed: {e}")
        return False
    
    try:
        from research_queue.experiment_queue import ExperimentQueue
        print("✅ Experiment queue import successful")
    except ImportError as e:
        print(f"❌ Experiment queue import failed: {e}")
        return False
    
    try:
        from research_data.database import ResearchDatabase
        print("✅ Database import successful")
    except ImportError as e:
        print(f"❌ Database import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality without API calls."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test database creation
        from research_data.database import ResearchDatabase
        db = ResearchDatabase("test_research.db")
        print("✅ Database creation successful")
        
        # Test queue creation
        from research_queue.experiment_queue import ExperimentQueue
        queue = ExperimentQueue("test_queue.json")
        print("✅ Queue creation successful")
        
        # Test experiment creation
        from research_agents.base_agent import ResearchExperiment
        exp = ResearchExperiment(
            id="test-001",
            title="Test Experiment",
            description="A test experiment",
            hypothesis="This will work",
            parameters={"test_param": "test_value"},
            priority=5,
            estimated_duration=60
        )
        print("✅ Experiment creation successful")
        
        # Test adding to queue
        exp_id = queue.add_experiment(exp)
        print(f"✅ Experiment added to queue: {exp_id}")
        
        # Test saving to database
        success = db.save_experiment(exp)
        print(f"✅ Experiment saved to database: {success}")
        
        # Test queue status
        status = queue.get_queue_status()
        print(f"✅ Queue status retrieved: {status['total_experiments']} experiments")
        
        # Cleanup
        if os.path.exists("test_research.db"):
            os.remove("test_research.db")
        if os.path.exists("test_queue.json"):
            os.remove("test_queue.json")
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        # Test coordinator config
        config_path = "configs/research_coordinator_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ Coordinator config loaded successfully")
            print(f"   Auto review threshold: {config.get('auto_review_threshold', 'N/A')}")
        else:
            print("⚠️  Coordinator config file not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🔬 Blueberry LLM Auto-Researching AI System - Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test config loading
    if test_config_loading():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Set your OpenRouter API key: export OPENROUTER_API_KEY='your_key'")
        print("2. Run: python run_research.py")
        print("3. Or use CLI: python research_cli.py suggest --count 3")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
