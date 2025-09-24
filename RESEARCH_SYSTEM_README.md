# 🔬 Blueberry LLM Auto-Researching AI System

An intelligent multi-agent research system that automatically generates, reviews, improves, and executes research experiments for MoE LLM training on T4 GPUs.

## 🎯 Overview

This system transforms the Blueberry LLM repository into an autonomous research platform with multiple AI agents working together to:

- **Generate** novel research ideas and experiment suggestions
- **Review** experiments for quality, feasibility, and impact
- **Improve** experiments based on feedback
- **Execute** experiments on your T4 GPU
- **Store** all results in a comprehensive database

## 🤖 Multi-Agent Architecture

### 1. Suggestion Agent
- Generates innovative research ideas
- Focuses on MoE architecture improvements
- Considers T4 GPU constraints
- Avoids duplicate suggestions

### 2. Review Agent
- Evaluates experiments on multiple criteria
- Provides detailed feedback and scoring
- Compares experiments for prioritization
- Ensures research quality

### 3. Improvement Agent
- Refines experiments based on reviews
- Optimizes for T4 GPU performance
- Suggests alternative approaches
- Enhances methodology

### 4. Experiment Runner
- Executes experiments on T4 GPU
- Monitors GPU utilization
- Handles failures gracefully
- Collects detailed results

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up your OpenRouter API key
export OPENROUTER_API_KEY="your_api_key_here"
```

### 2. Run Simple Research Session

```bash
# Run a complete automated research cycle
python run_research.py
```

### 3. Use CLI for Advanced Control

```bash
# Generate research suggestions
python research_cli.py suggest --count 5 --domain moe_architecture

# Start automated research cycle
python research_cli.py auto-research --max-cycles 3 --start-runner

# Run experiments in daemon mode
python research_cli.py run --daemon

# Check system status
python research_cli.py status
```

## 📋 CLI Commands

### Generate Research Suggestions
```bash
python research_cli.py suggest [OPTIONS]

Options:
  --count INT              Number of suggestions to generate (default: 5)
  --domain STR             Research domain (moe_architecture, optimization, evaluation, efficiency)
  --focus-area STR         Specific focus area within domain
  --add-to-queue          Add suggestions directly to queue
```

### Review Experiments
```bash
python research_cli.py review [OPTIONS]

Options:
  --limit INT              Maximum number of experiments to review (default: 10)
```

### Run Experiments
```bash
python research_cli.py run [OPTIONS]

Options:
  --single EXPERIMENT_ID   Run a single experiment by ID
  --daemon                 Run experiment runner in daemon mode
```

### System Management
```bash
# Show system status
python research_cli.py status

# List experiments
python research_cli.py list [--status STATUS] [--limit INT]

# Export research data
python research_cli.py export [--output FILE]

# Run automated research cycle
python research_cli.py auto-research [--domain DOMAIN] [--max-cycles INT] [--start-runner]
```

## 🔧 Configuration

### Research Coordinator Config (`configs/research_coordinator_config.json`)
```json
{
  "auto_review_threshold": 7.0,
  "auto_improve_threshold": 5.0,
  "max_experiments_per_suggestion": 5,
  "auto_start_runner": false,
  "gpu_optimization": {
    "max_batch_size": 32,
    "max_sequence_length": 1024,
    "max_experts": 16,
    "memory_safety_margin": 0.8
  }
}
```

### Environment Variables
```bash
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional
OPENROUTER_SITE_URL=https://github.com/your-username/blueberry-llm-t4-gpu
OPENROUTER_SITE_NAME=Blueberry LLM T4 GPU
```

## 📊 Database Schema

The system uses SQLite to store:

- **Experiments**: Research experiment definitions and status
- **Results**: Detailed experiment results and metrics
- **Reviews**: Agent reviews and evaluations
- **Agent Interactions**: All agent communications
- **Research Insights**: Discovered patterns and insights

## 🎮 GPU Optimization

The system automatically optimizes experiments for T4 GPU constraints:

- **Memory Management**: Limits batch sizes and model dimensions
- **Sequence Length**: Caps at 1024 tokens for memory efficiency
- **Expert Count**: Limits to 16 experts maximum
- **Mixed Precision**: Uses FP16 for training efficiency
- **Gradient Accumulation**: Optimizes for memory usage

## 📈 Research Domains

### 1. MoE Architecture (`moe_architecture`)
- Expert routing mechanisms
- Load balancing strategies
- Expert specialization
- Memory efficiency
- Training stability

### 2. Optimization (`optimization`)
- Learning rate schedules
- Optimizer improvements
- Gradient accumulation
- Mixed precision training
- Memory optimization

### 3. Evaluation (`evaluation`)
- Evaluation metrics
- Benchmark datasets
- Performance analysis
- Comparative studies
- Ablation studies

### 4. Efficiency (`efficiency`)
- Model compression
- Quantization
- Pruning
- Distributed training
- Inference optimization

## 🔄 Research Workflow

1. **Suggestion Phase**: AI agents generate novel research ideas
2. **Review Phase**: Experiments are evaluated for quality and feasibility
3. **Improvement Phase**: Low-scoring experiments are refined
4. **Queue Management**: Approved experiments are prioritized and queued
5. **Execution Phase**: Experiments run automatically on T4 GPU
6. **Analysis Phase**: Results are stored and analyzed for insights

## 📁 File Structure

```
blueberry-llm-t4-gpu/
├── research_agents/           # AI agents
│   ├── base_agent.py         # Base agent class
│   ├── suggestion_agent.py   # Research suggestion agent
│   ├── review_agent.py       # Experiment review agent
│   └── improvement_agent.py  # Experiment improvement agent
├── research_queue/           # Experiment queue management
│   └── experiment_queue.py   # Queue system
├── research_data/            # Database and storage
│   └── database.py           # SQLite database layer
├── research_execution/       # Experiment execution
│   └── experiment_runner.py  # T4 GPU experiment runner
├── configs/                  # Configuration files
│   └── research_coordinator_config.json
├── research_coordinator.py   # Main coordinator
├── research_cli.py           # Command-line interface
├── run_research.py           # Simple execution script
└── env_example.txt           # Environment variables example
```

## 🚨 Error Handling

The system includes comprehensive error handling:

- **GPU Memory**: Automatic batch size adjustment
- **Training Failures**: Graceful failure recovery
- **Agent Errors**: Fallback responses for LLM failures
- **Queue Management**: Robust queue state management
- **Database Integrity**: Transaction-based data storage

## 📊 Monitoring and Logging

- **Real-time Status**: Monitor system status via CLI
- **GPU Utilization**: Track GPU usage and memory
- **Experiment Progress**: Live experiment tracking
- **Agent Interactions**: Full conversation logging
- **Performance Metrics**: Detailed result analysis

## 🔒 Security and Privacy

- **API Key Management**: Secure environment variable storage
- **Local Database**: All data stored locally in SQLite
- **No Data Sharing**: Experiments and results remain private
- **Configurable Models**: Use different LLM models as needed

## 🎯 Use Cases

### Research Automation
- Generate hundreds of research ideas automatically
- Continuously improve experiment quality
- Run experiments 24/7 on your T4 GPU
- Build a comprehensive research database

### Academic Research
- Systematic exploration of MoE architectures
- Reproducible experimental protocols
- Detailed performance analysis
- Publication-ready results

### Industry Applications
- Optimize models for specific hardware
- Systematic hyperparameter search
- Performance benchmarking
- Cost-effective research scaling

## 🛠️ Customization

### Adding New Research Domains
1. Update `research_domains` in suggestion agent
2. Add domain-specific prompts
3. Configure domain weights in coordinator config

### Custom Experiment Types
1. Extend `ResearchExperiment` class
2. Add new parameter types
3. Implement custom execution logic

### Additional Agents
1. Inherit from `BaseAgent` class
2. Implement `process()` method
3. Register agent in coordinator

## 📞 Support

For issues, questions, or contributions:

1. Check the logs in `research_data/research.log`
2. Use `python research_cli.py status` for system diagnostics
3. Export data with `python research_cli.py export` for analysis
4. Review configuration files for customization

## 🎉 Getting Started

1. **Set up your API key**: `export OPENROUTER_API_KEY="your_key"`
2. **Run your first research cycle**: `python run_research.py`
3. **Monitor progress**: `python research_cli.py status`
4. **Explore results**: `python research_cli.py list`

The system will automatically generate, review, improve, and execute research experiments, building a comprehensive knowledge base of MoE LLM training optimizations for T4 GPUs.

Happy researching! 🔬🚀
