#!/usr/bin/env python3
"""
Quick test of the momentum warmup experiment setup
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_experiment_setup():
    """Test that the experiment can be created and configured"""
    print("🧪 Testing experiment setup...")
    
    try:
        from research_agents.momentum_warmup_experiment import MomentumWarmupExperiment
        from research_agents.momentum_warmup_config import QUICK_EXPERIMENT
        
        # Create experiment
        experiment = MomentumWarmupExperiment(QUICK_EXPERIMENT)
        
        # Test configuration access
        assert hasattr(experiment.config, 'momentum_schedules'), "Config should have momentum_schedules"
        assert len(experiment.config.momentum_schedules) > 0, "Should have momentum schedules"
        
        # Test schedule structure
        schedule = experiment.config.momentum_schedules[0]
        assert 'name' in schedule, "Schedule should have name"
        assert 'type' in schedule, "Schedule should have type"
        
        print(f"✅ Experiment setup successful")
        print(f"   - Experiment name: {experiment.config.experiment_name}")
        print(f"   - Number of schedules: {len(experiment.config.momentum_schedules)}")
        print(f"   - Max steps: {experiment.config.training_config['max_steps']}")
        print(f"   - Runs per schedule: {experiment.config.training_config['num_runs_per_schedule']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment setup error: {e}")
        return False

def test_momentum_scheduler():
    """Test momentum scheduler with new configuration"""
    print("\n🧪 Testing momentum scheduler...")
    
    try:
        from research_agents.momentum_warmup_experiment import MomentumWarmupScheduler
        
        # Test linear warmup schedule
        schedule_config = {
            'name': 'linear_warmup',
            'type': 'linear',
            'start_momentum': 0.0,
            'end_momentum': 0.95,
            'warmup_steps': 100
        }
        
        scheduler = MomentumWarmupScheduler(schedule_config, max_steps=500)
        
        # Test momentum values
        assert scheduler.get_momentum(0) == 0.0, "Initial momentum should be 0.0"
        assert scheduler.get_momentum(50) == 0.475, "Mid-warmup momentum should be 0.475"
        assert scheduler.get_momentum(100) == 0.95, "End warmup momentum should be 0.95"
        assert scheduler.get_momentum(200) == 0.95, "Post-warmup momentum should be 0.95"
        
        print("✅ Momentum scheduler working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Momentum scheduler error: {e}")
        return False

def main():
    """Run tests"""
    print("🧪 Momentum Warmup Experiment Test")
    print("=" * 50)
    
    tests = [
        test_experiment_setup,
        test_momentum_scheduler,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Experiment setup is ready.")
        print("\nYou can now run:")
        print("python research_agents/run_momentum_experiment.py quick")
    else:
        print("❌ Some tests failed. Please fix issues before running experiments.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
