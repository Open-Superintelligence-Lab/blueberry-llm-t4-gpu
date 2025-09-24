"""Research Suggestion Agent - generates new research ideas and experiments."""

import json
import uuid
import os
from typing import List, Dict, Any
from datetime import datetime

from .base_agent import BaseAgent, ResearchExperiment


class SuggestionAgent(BaseAgent):
    """Agent responsible for generating research suggestions and experiment ideas."""
    
    def __init__(self, openrouter_api_key: str, model: str = "x-ai/grok-4-fast:free"):
        super().__init__("suggestion_agent", openrouter_api_key, model)
        
        # Load research domain knowledge
        self.research_domains = self._load_research_domains()
        self.previous_suggestions = self._load_previous_suggestions()
    
    def _load_research_domains(self) -> Dict[str, Any]:
        """Load research domain knowledge."""
        return {
            "moe_architecture": {
                "description": "Mixture of Experts architecture improvements",
                "focus_areas": [
                    "Expert routing mechanisms",
                    "Load balancing strategies",
                    "Expert specialization",
                    "Memory efficiency",
                    "Training stability"
                ]
            },
            "optimization": {
                "description": "Training and optimization improvements",
                "focus_areas": [
                    "Learning rate schedules",
                    "Optimizer improvements",
                    "Gradient accumulation",
                    "Mixed precision training",
                    "Memory optimization"
                ]
            },
            "evaluation": {
                "description": "Model evaluation and benchmarking",
                "focus_areas": [
                    "Evaluation metrics",
                    "Benchmark datasets",
                    "Performance analysis",
                    "Comparative studies",
                    "Ablation studies"
                ]
            },
            "efficiency": {
                "description": "Computational and memory efficiency",
                "focus_areas": [
                    "Model compression",
                    "Quantization",
                    "Pruning",
                    "Distributed training",
                    "Inference optimization"
                ]
            }
        }
    
    def _load_previous_suggestions(self) -> List[Dict[str, Any]]:
        """Load previously generated suggestions to avoid duplicates."""
        try:
            with open("research_data/previous_suggestions.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _save_suggestion(self, suggestion: Dict[str, Any]):
        """Save suggestion to avoid duplicates."""
        os.makedirs("research_data", exist_ok=True)
        self.previous_suggestions.append(suggestion)
        
        with open("research_data/previous_suggestions.json", "w") as f:
            json.dump(self.previous_suggestions, f, indent=2)
    
    def generate_research_suggestions(self, 
                                    domain: str = None, 
                                    focus_area: str = None,
                                    count: int = 5) -> List[ResearchExperiment]:
        """Generate research experiment suggestions."""
        
        system_prompt = f"""
You are a research scientist specializing in Machine Learning and Mixture of Experts (MoE) architectures.
You have deep knowledge of:
- Transformer architectures and MoE models
- Training optimization techniques
- GPU optimization for T4 hardware
- Model evaluation and benchmarking

Your task is to generate innovative research experiment suggestions for improving MoE LLM training on T4 GPUs.

Focus on:
1. Novel approaches that haven't been extensively explored
2. Practical improvements for T4 GPU constraints
3. Measurable hypotheses with clear success criteria
4. Experiments that can be run efficiently on single T4 GPU

Current research domains available:
{json.dumps(self.research_domains, indent=2)}

Generate {count} research experiment suggestions. Each should include:
- A clear, specific title
- Detailed description of the approach
- Testable hypothesis
- Specific parameters to vary
- Estimated duration
- Priority level (1-10, 10 being highest)

Avoid suggesting experiments that are too similar to previous ones.
"""
        
        # Prepare context about previous suggestions
        previous_context = ""
        if self.previous_suggestions:
            recent_suggestions = self.previous_suggestions[-10:]  # Last 10 suggestions
            previous_context = "\nRecent previous suggestions to avoid duplicating:\n"
            for suggestion in recent_suggestions:
                previous_context += f"- {suggestion.get('title', 'Unknown')}: {suggestion.get('description', 'No description')[:100]}...\n"
        
        messages = [{
            "role": "user",
            "content": f"""
Generate {count} research experiment suggestions for MoE LLM training on T4 GPUs.

Focus domain: {domain or 'any relevant domain'}
Specific focus area: {focus_area or 'any relevant focus area'}

{previous_context}

Please provide the suggestions in the following JSON format:
{{
  "suggestions": [
    {{
      "title": "Clear, specific experiment title",
      "description": "Detailed description of what this experiment tests and why it's important",
      "hypothesis": "Specific, testable hypothesis about expected outcomes",
      "parameters": {{
        "param1": "value1",
        "param2": "value2"
      }},
      "estimated_duration_minutes": 120,
      "priority": 8,
      "domain": "domain_name",
      "focus_area": "specific_area"
    }}
  ]
}}
"""
        }]
        
        response = self.call_llm(messages, system_prompt)
        
        try:
            # Parse JSON response
            data = json.loads(response)
            suggestions = []
            
            for suggestion_data in data.get("suggestions", []):
                experiment = ResearchExperiment(
                    id=str(uuid.uuid4()),
                    title=suggestion_data["title"],
                    description=suggestion_data["description"],
                    hypothesis=suggestion_data["hypothesis"],
                    parameters=suggestion_data["parameters"],
                    priority=suggestion_data["priority"],
                    estimated_duration=suggestion_data["estimated_duration_minutes"]
                )
                suggestions.append(experiment)
                
                # Save suggestion to avoid duplicates
                self._save_suggestion({
                    "title": suggestion_data["title"],
                    "description": suggestion_data["description"],
                    "domain": suggestion_data.get("domain", domain),
                    "focus_area": suggestion_data.get("focus_area", focus_area),
                    "created_at": datetime.now().isoformat()
                })
            
            self.logger.info(f"Generated {len(suggestions)} research suggestions")
            return suggestions
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Fallback: create a simple suggestion
            return [self._create_fallback_suggestion()]
    
    def _create_fallback_suggestion(self) -> ResearchExperiment:
        """Create a fallback suggestion if LLM parsing fails."""
        return ResearchExperiment(
            id=str(uuid.uuid4()),
            title="MoE Expert Load Balancing Optimization",
            description="Investigate different load balancing strategies for MoE experts on T4 GPU",
            hypothesis="Custom load balancing will improve training stability and convergence",
            parameters={
                "load_balancing_weight": [0.01, 0.05, 0.1],
                "expert_top_k": [2, 4],
                "num_experts": [8, 16]
            },
            priority=7,
            estimated_duration=180
        )
    
    def suggest_parameter_variations(self, base_experiment: ResearchExperiment) -> List[ResearchExperiment]:
        """Generate parameter variations for a base experiment."""
        
        system_prompt = """
You are an expert in experimental design for machine learning research.
Your task is to suggest meaningful parameter variations for a given experiment.

Consider:
- Parameter ranges that are likely to show meaningful differences
- Computational constraints of T4 GPU
- Statistical significance requirements
- Avoiding redundant or uninformative variations

For each variation, suggest 2-3 different parameter sets that test different hypotheses.
"""
        
        messages = [{
            "role": "user",
            "content": f"""
Base experiment:
Title: {base_experiment.title}
Description: {base_experiment.description}
Current parameters: {json.dumps(base_experiment.parameters, indent=2)}

Suggest 3-5 parameter variations that would test different aspects of this experiment.
Each variation should have a clear hypothesis about what it tests.

Return in JSON format:
{{
  "variations": [
    {{
      "title": "Variation title",
      "hypothesis": "What this variation tests",
      "parameters": {{"param": "value"}},
      "priority": 8
    }}
  ]
}}
"""
        }]
        
        response = self.call_llm(messages, system_prompt)
        
        try:
            data = json.loads(response)
            variations = []
            
            for var_data in data.get("variations", []):
                variation = ResearchExperiment(
                    id=str(uuid.uuid4()),
                    title=f"{base_experiment.title} - {var_data['title']}",
                    description=f"Variation of {base_experiment.description}. {var_data['hypothesis']}",
                    hypothesis=var_data["hypothesis"],
                    parameters=var_data["parameters"],
                    priority=var_data["priority"],
                    estimated_duration=base_experiment.estimated_duration
                )
                variations.append(variation)
            
            return variations
            
        except json.JSONDecodeError:
            return [base_experiment]  # Return original if parsing fails
    
    def process(self, input_data: Any) -> Any:
        """Process input to generate research suggestions."""
        if isinstance(input_data, dict):
            domain = input_data.get("domain")
            focus_area = input_data.get("focus_area")
            count = input_data.get("count", 5)
            return self.generate_research_suggestions(domain, focus_area, count)
        
        return self.generate_research_suggestions()
