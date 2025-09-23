#!/usr/bin/env python3
"""
Run Training Speed Experiments with Record Tracking

This script runs the training speed experiments using the same record tracking
technique as reference_measurement.py for comprehensive performance analysis.
"""

import os
import sys
import subprocess
import time

def main():
    """Run the training speed experiments with record tracking."""
    print("🚀 Starting Training Speed Experiments with Record Tracking")
    print("=" * 70)
    
    # Change to the project directory
    project_dir = "/Users/vukrosic/AI Science Projects/blueberry-llm-t4-gpu"
    os.chdir(project_dir)
    
    # Run the experiments
    print("📊 Running 6 experiments (baseline + 5 optimizations)...")
    print("⏱️  Each experiment will run for 200 steps")
    print("🔍 Detailed timing measurements will be collected")
    print("📈 Records will be tracked and saved to logs/")
    print()
    
    start_time = time.time()
    
    try:
        # Run the experiments script
        result = subprocess.run([
            sys.executable, "experiments.py", 
            "--output", "training_speed_results.json"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Experiments completed successfully!")
            print("\n📊 Results:")
            print(result.stdout)
            
            if result.stderr:
                print("\n⚠️  Warnings/Info:")
                print(result.stderr)
        else:
            print("❌ Experiments failed!")
            print("Error output:")
            print(result.stderr)
            print("Standard output:")
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⏰ Experiments timed out after 1 hour")
    except Exception as e:
        print(f"❌ Error running experiments: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total experiment time: {total_time/60:.1f} minutes")
    
    # Check if results file was created
    results_file = os.path.join(project_dir, "training_speed_results.json")
    if os.path.exists(results_file):
        print(f"📁 Results saved to: {results_file}")
        
        # Show a quick summary
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"\n📈 Quick Summary:")
            print(f"   Experiments completed: {len(results)}")
            
            if results:
                # Find fastest
                fastest = max(results, key=lambda x: x['steps_per_second'])
                print(f"   Fastest: {fastest['config']['name']} ({fastest['steps_per_second']:.2f} steps/sec)")
                
                # Find baseline
                baseline = next((r for r in results if r['config']['name'] == 'baseline'), None)
                if baseline and fastest['config']['name'] != 'baseline':
                    speedup = fastest['steps_per_second'] / baseline['steps_per_second']
                    print(f"   Speedup vs baseline: {speedup:.2f}x")
        except Exception as e:
            print(f"   Could not parse results: {e}")
    else:
        print("❌ Results file not found")
    
    # Check for record files
    logs_dir = os.path.join(project_dir, "logs")
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.txt')]
        record_files = [f for f in os.listdir(logs_dir) if f.endswith('_record.json')]
        
        print(f"\n📊 Record Tracking:")
        print(f"   Log files created: {len(log_files)}")
        print(f"   Record files created: {len(record_files)}")
        
        if record_files:
            print(f"   Latest record: {max(record_files)}")
        
        # Check for master records
        master_records = os.path.join(logs_dir, "speed_records.json")
        if os.path.exists(master_records):
            print(f"   Master records: speed_records.json")
            try:
                with open(master_records, 'r') as f:
                    records = json.load(f)
                print(f"   Total records in master file: {len(records)}")
            except:
                pass

if __name__ == "__main__":
    main()
