#!/usr/bin/env python3
"""Simple script to run the auto-researching AI system."""

import os
import sys
import time
from research_coordinator import ResearchCoordinator


def main():
    """Main entry point for running research."""
    print("🔬 Blueberry LLM Auto-Researching AI System")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("   You can get an API key from: https://openrouter.ai/")
        return
    
    # Initialize coordinator
    print("🚀 Initializing research coordinator...")
    coordinator = ResearchCoordinator(api_key)
    
    # Show initial status
    print("\n📊 Initial System Status:")
    status = coordinator.get_system_status()
    print(f"   GPU: {status['runner_status']['gpu_name']} ({status['runner_status']['gpu_memory_gb']:.1f}GB)")
    print(f"   Queue: {status['queue_status']['total_experiments']} experiments")
    
    # Run automated research cycle
    print("\n🤖 Starting automated research cycle...")
    results = coordinator.start_auto_research_cycle(
        domain="moe_architecture",  # Focus on MoE improvements
        max_cycles=3
    )
    
    print(f"\n✅ Research cycle completed:")
    print(f"   Generated: {results['total_suggestions']} suggestions")
    print(f"   Reviewed: {results['total_reviews']} experiments")
    print(f"   Added to queue: {results['experiments_added']} experiments")
    
    # Start experiment runner if experiments were added
    if results['experiments_added'] > 0:
        print("\n🏃 Starting experiment runner...")
        coordinator.start_experiment_runner()
        
        print("\n📊 Running experiments... (Press Ctrl+C to stop)")
        try:
            while True:
                time.sleep(30)  # Check every 30 seconds
                status = coordinator.get_system_status()
                
                if not status['runner_status']['is_running']:
                    print("❌ Experiment runner stopped")
                    break
                
                current_exp = status['runner_status']['current_experiment_title']
                if current_exp:
                    print(f"   Currently running: {current_exp}")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping experiment runner...")
            coordinator.stop_experiment_runner()
    
    # Show final status
    print("\n📊 Final System Status:")
    final_status = coordinator.get_system_status()
    print(f"   Total experiments: {final_status['queue_status']['total_experiments']}")
    print(f"   Completed: {final_status['database_statistics'].get('total_experiments', 0)}")
    
    # Export results
    export_file = coordinator.export_research_data()
    print(f"\n📤 Research data exported to: {export_file}")
    
    print("\n🎉 Research session completed!")


if __name__ == '__main__':
    main()
