#!/usr/bin/env python3
"""Command-line interface for the auto-researching AI system."""

import argparse
import json
import os
import sys
import logging
from typing import Dict, Any, List

from research_coordinator import ResearchCoordinator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('research_data/research.log')
        ]
    )


def get_openrouter_key() -> str:
    """Get OpenRouter API key from environment or prompt user."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        print("OpenRouter API key not found in environment variables.")
        api_key = input("Please enter your OpenRouter API key: ").strip()
        
        if not api_key:
            print("Error: API key is required")
            sys.exit(1)
    
    return api_key


def print_status(status: Dict[str, Any]):
    """Print system status in a formatted way."""
    print("\n" + "="*60)
    print("🔬 AUTO-RESEARCHING AI SYSTEM STATUS")
    print("="*60)
    
    # Coordinator status
    print(f"📊 Coordinator Status: {status['coordinator_status']}")
    
    # Runner status
    runner_status = status['runner_status']
    print(f"🏃 Runner Status: {'Running' if runner_status['is_running'] else 'Stopped'}")
    if runner_status['current_experiment']:
        print(f"   Current Experiment: {runner_status['current_experiment_title']}")
    print(f"   GPU: {runner_status['gpu_name']} ({runner_status['gpu_memory_gb']:.1f}GB)")
    
    # Queue status
    queue_status = status['queue_status']
    print(f"📋 Queue Status:")
    print(f"   Total Experiments: {queue_status['total_experiments']}")
    print(f"   Pending in Queue: {queue_status['queue_length']}")
    print(f"   Currently Running: {queue_status['currently_running'] or 'None'}")
    
    # Database statistics
    db_stats = status['database_statistics']
    print(f"💾 Database Statistics:")
    print(f"   Total Experiments: {db_stats.get('total_experiments', 0)}")
    print(f"   Success Rate: {db_stats.get('success_rate', 0):.1%}")
    print(f"   Average Review Score: {db_stats.get('average_review_score', 0):.1f}/10")
    
    # GPU utilization
    gpu_info = status['gpu_utilization']
    if gpu_info.get('available'):
        print(f"🎮 GPU Utilization:")
        print(f"   Memory Usage: {gpu_info['memory_usage_percent']:.1f}%")
        if gpu_info.get('utilization_percent'):
            print(f"   GPU Usage: {gpu_info['utilization_percent']}%")
    
    print("="*60)


def print_experiments(experiments: List[Dict[str, Any]], limit: int = 10):
    """Print list of experiments."""
    print(f"\n📋 EXPERIMENTS (showing first {limit})")
    print("-" * 80)
    
    for i, exp in enumerate(experiments[:limit]):
        print(f"{i+1}. {exp['title']}")
        print(f"   Status: {exp['status']} | Priority: {exp['priority']}/10 | Duration: {exp['estimated_duration']}min")
        print(f"   Description: {exp['description'][:100]}...")
        print()


def cmd_suggest(args, coordinator: ResearchCoordinator):
    """Generate research suggestions."""
    print(f"🔬 Generating {args.count} research suggestions...")
    
    suggestions = coordinator.generate_research_suggestions(
        domain=args.domain,
        focus_area=args.focus_area,
        count=args.count
    )
    
    print(f"✅ Generated {len(suggestions)} suggestions:")
    print_experiments([suggestion.__dict__ for suggestion in suggestions])
    
    if args.add_to_queue:
        exp_ids = coordinator.add_experiments_to_queue(suggestions)
        print(f"📋 Added {len(exp_ids)} experiments to queue")


def cmd_review(args, coordinator: ResearchCoordinator):
    """Review experiments."""
    print("🔍 Reviewing experiments...")
    
    # Get experiments from queue
    experiments = coordinator.queue.list_experiments(limit=args.limit)
    
    if not experiments:
        print("No experiments to review")
        return
    
    # Review experiments
    reviews = coordinator.review_experiments(experiments)
    
    print(f"✅ Completed {len(reviews)} reviews:")
    for exp, review in zip(experiments, reviews):
        print(f"   {exp.title}: Score {review['weighted_score']:.1f}/10 - {review['recommendation']}")


def cmd_improve(args, coordinator: ResearchCoordinator):
    """Improve experiments based on reviews."""
    print("🔧 Improving experiments...")
    
    # Get low-scoring experiments
    experiments = coordinator.queue.list_experiments()
    reviews = []
    
    for exp in experiments:
        exp_reviews = coordinator.database.get_experiment_reviews(exp.id)
        if exp_reviews:
            reviews.append(exp_reviews[0])  # Get latest review
    
    if not reviews:
        print("No reviews found to improve experiments")
        return
    
    # Filter low-scoring experiments
    low_score_experiments = [
        exp for exp, review in zip(experiments, reviews)
        if review['weighted_score'] < 6.0
    ]
    
    if not low_score_experiments:
        print("No low-scoring experiments to improve")
        return
    
    # Improve experiments
    improved = coordinator.improve_experiments(
        low_score_experiments, 
        [r for r in reviews if r['weighted_score'] < 6.0]
    )
    
    print(f"✅ Improved {len(improved)} experiments")


def cmd_run(args, coordinator: ResearchCoordinator):
    """Run experiments."""
    if args.single:
        print(f"🏃 Running single experiment: {args.single}")
        success, results = coordinator.run_single_experiment(args.single)
        if success:
            print("✅ Experiment completed successfully")
            print(f"Results: {results}")
        else:
            print(f"❌ Experiment failed: {results.get('error_message', 'Unknown error')}")
    else:
        print("🏃 Starting experiment runner...")
        coordinator.start_experiment_runner()
        
        if args.daemon:
            print("🔄 Running in daemon mode. Press Ctrl+C to stop.")
            try:
                while True:
                    import time
                    time.sleep(10)
                    status = coordinator.get_system_status()
                    if not status['runner_status']['is_running']:
                        print("❌ Runner stopped unexpectedly")
                        break
            except KeyboardInterrupt:
                print("\n🛑 Stopping experiment runner...")
                coordinator.stop_experiment_runner()


def cmd_status(args, coordinator: ResearchCoordinator):
    """Show system status."""
    status = coordinator.get_system_status()
    print_status(status)


def cmd_list(args, coordinator: ResearchCoordinator):
    """List experiments."""
    experiments = coordinator.queue.list_experiments(
        status=args.status,
        limit=args.limit
    )
    
    if not experiments:
        print("No experiments found")
        return
    
    print_experiments([exp.__dict__ for exp in experiments], args.limit)


def cmd_export(args, coordinator: ResearchCoordinator):
    """Export research data."""
    output_file = coordinator.export_research_data(args.output)
    print(f"📤 Exported research data to: {output_file}")


def cmd_auto_research(args, coordinator: ResearchCoordinator):
    """Run automated research cycle."""
    print("🤖 Starting automated research cycle...")
    
    results = coordinator.start_auto_research_cycle(
        domain=args.domain,
        focus_area=args.focus_area,
        max_cycles=args.max_cycles
    )
    
    print("✅ Automated research cycle completed:")
    print(f"   Cycles completed: {results['cycles_completed']}")
    print(f"   Suggestions generated: {results['total_suggestions']}")
    print(f"   Experiments reviewed: {results['total_reviews']}")
    print(f"   Experiments improved: {results['total_improvements']}")
    print(f"   Experiments added to queue: {results['experiments_added']}")
    
    if args.start_runner:
        print("🏃 Starting experiment runner...")
        coordinator.start_experiment_runner()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-Researching AI System for MoE LLM Training on T4 GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate research suggestions
  python research_cli.py suggest --count 5 --domain moe_architecture
  
  # Run automated research cycle
  python research_cli.py auto-research --max-cycles 3 --start-runner
  
  # Start experiment runner in daemon mode
  python research_cli.py run --daemon
  
  # Show system status
  python research_cli.py status
  
  # List pending experiments
  python research_cli.py list --status pending
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Suggest command
    suggest_parser = subparsers.add_parser('suggest', help='Generate research suggestions')
    suggest_parser.add_argument('--count', type=int, default=5,
                               help='Number of suggestions to generate')
    suggest_parser.add_argument('--domain', type=str,
                               help='Research domain (moe_architecture, optimization, evaluation, efficiency)')
    suggest_parser.add_argument('--focus-area', type=str,
                               help='Specific focus area within domain')
    suggest_parser.add_argument('--add-to-queue', action='store_true',
                               help='Add suggestions directly to queue')
    
    # Review command
    review_parser = subparsers.add_parser('review', help='Review experiments')
    review_parser.add_argument('--limit', type=int, default=10,
                              help='Maximum number of experiments to review')
    
    # Improve command
    improve_parser = subparsers.add_parser('improve', help='Improve low-scoring experiments')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run experiments')
    run_group = run_parser.add_mutually_exclusive_group()
    run_group.add_argument('--single', type=str,
                          help='Run a single experiment by ID')
    run_group.add_argument('--daemon', action='store_true',
                          help='Run experiment runner in daemon mode')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--status', type=str,
                            help='Filter by status (pending, running, completed, failed)')
    list_parser.add_argument('--limit', type=int, default=10,
                            help='Maximum number of experiments to show')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export research data')
    export_parser.add_argument('--output', type=str,
                              help='Output file path')
    
    # Auto-research command
    auto_parser = subparsers.add_parser('auto-research', help='Run automated research cycle')
    auto_parser.add_argument('--domain', type=str,
                            help='Research domain to focus on')
    auto_parser.add_argument('--focus-area', type=str,
                            help='Specific focus area')
    auto_parser.add_argument('--max-cycles', type=int, default=5,
                            help='Maximum number of research cycles')
    auto_parser.add_argument('--start-runner', action='store_true',
                            help='Start experiment runner after cycle')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Get API key
    api_key = get_openrouter_key()
    
    # Initialize coordinator
    coordinator = ResearchCoordinator(api_key)
    
    # Execute command
    try:
        if args.command == 'suggest':
            cmd_suggest(args, coordinator)
        elif args.command == 'review':
            cmd_review(args, coordinator)
        elif args.command == 'improve':
            cmd_improve(args, coordinator)
        elif args.command == 'run':
            cmd_run(args, coordinator)
        elif args.command == 'status':
            cmd_status(args, coordinator)
        elif args.command == 'list':
            cmd_list(args, coordinator)
        elif args.command == 'export':
            cmd_export(args, coordinator)
        elif args.command == 'auto-research':
            cmd_auto_research(args, coordinator)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
