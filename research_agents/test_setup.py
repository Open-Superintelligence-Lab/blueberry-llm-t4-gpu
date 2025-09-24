#!/usr/bin/env python3
"""
Test script for Momentum Warmup Research Setup

This script validates that all components are working correctly
before running the full experiments.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from research_agents.momentum_warmup_experiment import MomentumWarmupExperiment
        from research_agents.momentum_warmup_config import MomentumWarmupExperimentConfig
        from research_agents.muon_warmup_optimizer import MuonWithMomentumWarmup, MomentumScheduler
        from research_agents.momentum_warmup_analysis import MomentumWarmupEvaluator
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_momentum_scheduler():
    """Test momentum scheduler functionality"""
    print("\n🧪 Testing momentum scheduler...")
    
    try:
        from research_agents.muon_warmup_optimizer import MomentumScheduler
        
        # Test linear warmup
        config = {
            'type': 'linear',
            'start_momentum': 0.0,
            'end_momentum': 0.95,
            'warmup_steps': 100
        }
        
        scheduler = MomentumScheduler(config, max_steps=500)
        
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

def test_muon_optimizer():
    """Test Muon optimizer with momentum warmup"""
    print("\n🧪 Testing Muon optimizer...")
    
    try:
        from research_agents.muon_warmup_optimizer import MuonWithMomentumWarmup, MomentumScheduler
        
        # Create test parameters
        params = [torch.randn(10, 10, requires_grad=True)]
        
        # Test standard Muon
        optimizer = MuonWithMomentumWarmup(params, lr=0.01)
        assert optimizer.get_current_momentum() == 0.95, "Default momentum should be 0.95"
        
        # Test Muon with warmup
        config = {
            'type': 'linear',
            'start_momentum': 0.0,
            'end_momentum': 0.95,
            'warmup_steps': 10
        }
        scheduler = MomentumScheduler(config, max_steps=50)
        optimizer_warmup = MuonWithMomentumWarmup(params, lr=0.01, momentum_scheduler=scheduler)
        
        # Test step counting
        assert optimizer_warmup.get_step_count() == 0, "Initial step count should be 0"
        
        print("✅ Muon optimizer working correctly")
        return True
    except Exception as e:
        print(f"❌ Muon optimizer error: {e}")
        return False

def test_experiment_config():
    """Test experiment configuration"""
    print("\n🧪 Testing experiment configuration...")
    
    try:
        from research_agents.momentum_warmup_config import MomentumWarmupExperimentConfig
        
        config = MomentumWarmupExperimentConfig()
        
        # Test configuration structure
        assert hasattr(config, 'experiment_name'), "Config should have experiment_name"
        assert hasattr(config, 'momentum_schedules'), "Config should have momentum_schedules"
        assert len(config.momentum_schedules) > 0, "Config should have momentum schedules"
        
        # Test schedule structure
        schedule = config.momentum_schedules[0]
        assert 'name' in schedule, "Schedule should have name"
        assert 'type' in schedule, "Schedule should have type"
        
        print("✅ Experiment configuration working correctly")
        return True
    except Exception as e:
        print(f"❌ Experiment configuration error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and memory"""
    print("\n🧪 Testing GPU availability...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - experiments will run on CPU")
        return True
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU available: {gpu_name}")
    print(f"✅ GPU memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 10:
        print("⚠️  GPU memory may be insufficient for full experiments")
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\n🧪 Testing data loading...")
    
    try:
        from data.loader import load_and_cache_data
        from configs.moe_config import MoEModelConfig
        
        # Test data loading
        config = MoEModelConfig()
        texts, tokenizer, tokens = load_and_cache_data(config)
        
        assert len(texts) > 0, "Should load some texts"
        assert len(tokens) > 0, "Should tokenize texts"
        assert config.vocab_size > 0, "Should have valid vocab size"
        
        print(f"✅ Data loading successful: {len(texts)} texts, vocab size {config.vocab_size}")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🧪 Testing model creation...")
    
    try:
        from models.moe_llm import MoEMinimalLLM
        from configs.moe_config import MoEModelConfig
        
        # Create small test model
        config = MoEModelConfig(
            d_model=64,
            n_layers=2,
            num_experts=2,
            vocab_size=1000
        )
        
        model = MoEMinimalLLM(config)
        
        # Test forward pass
        batch_size = 4
        seq_len = 32
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            logits, aux_loss = model(x, return_aux_loss=True)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size), "Logits shape incorrect"
        
        print("✅ Model creation and forward pass successful")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Momentum Warmup Research Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_momentum_scheduler,
        test_muon_optimizer,
        test_experiment_config,
        test_gpu_availability,
        test_data_loading,
        test_model_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run experiments.")
        print("\nNext steps:")
        print("1. Run quick experiment: python research_agents/run_momentum_experiment.py quick")
        print("2. Analyze results: python research_agents/momentum_warmup_analysis.py experiments/quick_momentum_test/")
    else:
        print("❌ Some tests failed. Please fix issues before running experiments.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
