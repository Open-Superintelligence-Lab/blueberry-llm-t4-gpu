#!/usr/bin/env python3
"""
Simple runner for the Training Speedrun Challenge
"""

import sys
import subprocess
import os

def main():
    """Run the speedrun challenge."""
    print("🏁 Training Speedrun Challenge")
    print("=" * 50)
    print("📊 Challenge: Baseline vs Memory Optimized")
    print("⏱️  Steps per experiment: 200 (quadrupled from 50)")
    print("🎯 Goal: Establish performance measurement standards")
    print()
    
    # Run the speedrun challenge
    print("🚀 Starting speedrun challenge...")
    result = subprocess.run([
        sys.executable, "speedrun_challenge.py", 
        "--output", "speedrun_challenge_results.json"
    ], timeout=7200)  # 2 hour timeout
    
    if result.returncode == 0:
        print("\n✅ Speedrun challenge completed successfully!")
        print("📁 Results saved to: speedrun_challenge_results.json")
    else:
        print(f"\n❌ Speedrun challenge failed with return code: {result.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
