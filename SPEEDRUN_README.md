# Training Speedrun Challenge

## Overview
Minimal training speed comparison between baseline and memory-optimized configurations.

## Challenge Details
- **Baseline**: Standard T4-optimized configuration
- **Memory Optimized**: Smaller batch size (8) with higher gradient accumulation (4)
- **Steps per experiment**: 1000
- **Goal**: Establish performance measurement standards

## Performance Metrics
- **Primary**: Steps per second
- **Secondary**: Total time, Average step time
- **Memory**: Peak GPU memory usage
- **Quality**: Final validation loss

## Usage

### Run the Challenge
```bash
python run_speedrun.py
```

### Direct Execution
```bash
python speedrun_challenge.py --output results.json
```

## Expected Results
Based on previous testing, memory-optimized configuration should show:
- ~77% speed improvement over baseline
- Better memory efficiency
- Maintained model quality

## Files
- `speedrun_challenge.py` - Main challenge implementation
- `run_speedrun.py` - Simple runner script
- `record_tracker.py` - Performance tracking system
- `timed_training.py` - Training with detailed timing
- `timing.py` - Timing measurement utilities
