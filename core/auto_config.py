#!/usr/bin/env python3
"""
Auto-configuration for Blueberry LLM
Detects hardware and automatically configures optimal training setup
"""

import os
import sys
import torch
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

@dataclass
class AutoConfig:
    """Auto-detected configuration"""
    # Hardware
    num_gpus: int
    gpu_memory_gb: float
    
    # Model (auto-scaled)
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    num_experts: int
    
    # Training (auto-optimized)
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    max_seq_len: int
    
    # Performance
    use_amp: bool

class BlueberryAutoConfigurator:
    """One class that does everything"""
    
    def __init__(self):
        self.config = self._detect_and_configure()
    
    def _detect_and_configure(self) -> AutoConfig:
        """Main auto-configuration logic for single T4 GPU"""
        
        # Detect hardware - expect single T4 GPU
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus == 0:
            print("❌ No CUDA GPU detected. This version requires a T4 GPU.")
            raise RuntimeError("T4 GPU required but not found")
        
        if num_gpus > 1:
            print(f"⚠️ Multiple GPUs detected ({num_gpus}), but this version is optimized for single T4 GPU only")
            print("   Using only the first GPU")
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Check for T4 GPU
        device_name = torch.cuda.get_device_name(0).lower()
        if "tesla t4" in device_name or "t4" in device_name:
            print("🚀 Tesla T4 detected - using optimized configuration")
        else:
            print(f"⚠️ Non-T4 GPU detected: {device_name}")
            print("   This version is optimized for T4 GPUs, but will attempt to run")
        
        return self._t4_optimized_config(1, gpu_memory_gb)  # Always use T4 config
    
    def _t4_optimized_config(self, num_gpus: int, gpu_memory_gb: float) -> AutoConfig:
        """Optimized config for single Tesla T4 GPU - balanced for memory efficiency"""
        return AutoConfig(
            num_gpus=1,  # Force single GPU
            gpu_memory_gb=gpu_memory_gb,
            d_model=384,  # Moderate increase from 256 (was 512)
            n_layers=6,   # Moderate increase from 4 (was 8)
            n_heads=8,    # Increased from 4
            d_ff=1536,    # Moderate increase from 1024 (was 2048)
            num_experts=8,  # Increased from 4
            batch_size=12,  # Moderate increase from 8 (was 16)
            gradient_accumulation_steps=3,  # Balanced
            max_steps=2000,  # Increased from 1000
            learning_rate=0.01,
            max_seq_len=1024,  # Moderate increase from 512 (was 1024)
            use_amp=True
        )
    
    def _cpu_config(self) -> AutoConfig:
        """Minimal config for CPU-only systems"""
        return AutoConfig(
            num_gpus=0, gpu_memory_gb=0,
            d_model=128, n_layers=2, n_heads=4, d_ff=512, num_experts=2,
            batch_size=4, gradient_accumulation_steps=8, max_steps=1000,
            learning_rate=0.001, max_seq_len=256,
            use_amp=False
        )
    
    def print_config(self):
        """Print detected configuration"""
        print("🫐 Blueberry LLM Auto-Configuration")
        print("=" * 50)
        
        if self.config.num_gpus == 0:
            print("🖥️  Mode: CPU Training (Limited)")
        else:
            print(f"🚀 Mode: GPU Training ({self.config.num_gpus} GPUs)")
            print(f"   Memory: {self.config.gpu_memory_gb:.1f} GB per GPU")
        
        print(f"📏 Model: {self.config.d_model}d × {self.config.n_layers}L × {self.config.n_heads}H")
        print(f"🧠 Experts: {self.config.num_experts}")
        print(f"📊 Batch: {self.config.batch_size} (accum: {self.config.gradient_accumulation_steps})")
        print(f"📝 Sequence: {self.config.max_seq_len}")
        print(f"⚡ Mixed Precision: {'Yes' if self.config.use_amp else 'No'}")
        
        print(f"🌐 Single T4 GPU Training")
        
        print("=" * 50)
    
    def get_model_config(self):
        """Convert to MoEModelConfig format"""
        from legacy.llm import MoEModelConfig
        
        return MoEModelConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            batch_size=self.config.batch_size,
            max_steps=self.config.max_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            muon_lr=self.config.learning_rate,
            max_seq_len=self.config.max_seq_len,
            num_experts=self.config.num_experts,
            use_amp=self.config.use_amp,
        )

def auto_configure() -> BlueberryAutoConfigurator:
    """One function call to auto-configure everything"""
    return BlueberryAutoConfigurator()

if __name__ == "__main__":
    # Demo the auto-configuration
    configurator = auto_configure()
    configurator.print_config()
