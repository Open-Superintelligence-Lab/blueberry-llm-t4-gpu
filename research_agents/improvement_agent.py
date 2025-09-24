"""Research Improvement Agent - improves and refines research experiments."""

import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_agent import BaseAgent, ResearchExperiment


class ImprovementAgent(BaseAgent):
    """Agent responsible for improving and refining research experiments based on reviews."""
    
    def __init__(self, openrouter_api_key: str, model: str = "x-ai/grok-4-fast:free"):
        super().__init__("improvement_agent", openrouter_api_key, model)
        
        # Improvement strategies
        self.improvement_strategies = {
            "hypothesis_refinement": "Improve hypothesis clarity and testability",
            "parameter_optimization": "Optimize parameter ranges and values",
            "methodology_enhancement": "Enhance experimental methodology",
            "resource_optimization": "Optimize for T4 GPU constraints",
            "evaluation_improvement": "Improve evaluation metrics and procedures"
        }
    
    def improve_experiment(self, 
                          experiment: ResearchExperiment, 
                          review: Dict[str, Any]) -> ResearchExperiment:
        """Improve an experiment based on review feedback."""
        
        system_prompt = f"""
You are an expert research methodologist specializing in Machine Learning experiments.
Your task is to improve research experiments based on reviewer feedback.

You excel at:
- Refining hypotheses to be more testable and specific
- Optimizing experimental parameters for better results
- Improving methodologies for T4 GPU constraints
- Enhancing evaluation procedures
- Balancing novelty with feasibility

Available improvement strategies:
{json.dumps(self.improvement_strategies, indent=2)}

When improving an experiment, consider:
1. Address all reviewer concerns systematically
2. Maintain the core research question while improving methodology
3. Ensure feasibility on T4 GPU hardware
4. Make improvements measurable and specific
5. Preserve the original novelty and impact potential
"""
        
        messages = [{
            "role": "user",
            "content": f"""
Please improve this research experiment based on the review feedback:

**Original Experiment:**
Title: {experiment.title}
Description: {experiment.description}
Hypothesis: {experiment.hypothesis}
Parameters: {json.dumps(experiment.parameters, indent=2)}
Priority: {experiment.priority}
Estimated Duration: {experiment.estimated_duration} minutes

**Review Feedback:**
Overall Score: {review.get('weighted_score', 'N/A')}/10
Recommendation: {review.get('recommendation', 'N/A')}
Success Probability: {review.get('success_probability', 'N/A')}

**Detailed Scores:**
{json.dumps(review.get('scores', {}), indent=2)}

**Key Strengths:**
{json.dumps(review.get('key_strengths', []), indent=2)}

**Areas for Improvement:**
{json.dumps(review.get('areas_for_improvement', []), indent=2)}

**Suggested Modifications:**
{json.dumps(review.get('suggested_modifications', []), indent=2)}

**Risks:**
{json.dumps(review.get('risks', []), indent=2)}

**Detailed Feedback:**
{review.get('detailed_feedback', 'No detailed feedback available')}

Create an improved version of this experiment that addresses the reviewer's concerns
while maintaining the core research value. Return in this JSON format:

{{
  "improved_experiment": {{
    "title": "Improved experiment title",
    "description": "Enhanced description addressing reviewer concerns",
    "hypothesis": "Refined, more testable hypothesis",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "priority": 8,
    "estimated_duration_minutes": 150
  }},
  "improvements_made": [
    "List of specific improvements made"
  ],
  "improvement_rationale": "Explanation of why these improvements were made",
  "expected_impact": "Expected impact of improvements on experiment success"
}}
"""
        }]
        
        response = self.call_llm(messages, system_prompt)
        
        try:
            improvement_data = json.loads(response)
            improved_exp = improvement_data["improved_experiment"]
            
            # Create improved experiment
            improved_experiment = ResearchExperiment(
                id=str(uuid.uuid4()),
                title=improved_exp["title"],
                description=improved_exp["description"],
                hypothesis=improved_exp["hypothesis"],
                parameters=improved_exp["parameters"],
                priority=improved_exp["priority"],
                estimated_duration=improved_exp["estimated_duration_minutes"]
            )
            
            # Add improvement metadata
            improved_experiment.metadata = {
                "original_experiment_id": experiment.id,
                "improvements_made": improvement_data.get("improvements_made", []),
                "improvement_rationale": improvement_data.get("improvement_rationale", ""),
                "expected_impact": improvement_data.get("expected_impact", ""),
                "improved_at": datetime.now().isoformat(),
                "original_review_score": review.get("weighted_score", 0)
            }
            
            return improved_experiment
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse improvement response: {e}")
            return self._create_fallback_improvement(experiment, review)
    
    def _create_fallback_improvement(self, 
                                   experiment: ResearchExperiment, 
                                   review: Dict[str, Any]) -> ResearchExperiment:
        """Create a fallback improvement if LLM parsing fails."""
        improved_experiment = ResearchExperiment(
            id=str(uuid.uuid4()),
            title=f"{experiment.title} (Improved)",
            description=f"{experiment.description} [Improved based on review feedback]",
            hypothesis=experiment.hypothesis,
            parameters=experiment.parameters,
            priority=min(experiment.priority + 1, 10),  # Slightly increase priority
            estimated_duration=experiment.estimated_duration
        )
        
        improved_experiment.metadata = {
            "original_experiment_id": experiment.id,
            "improvements_made": ["Basic improvement due to parsing error"],
            "improvement_rationale": "Unable to parse LLM response, applying basic improvements",
            "expected_impact": "Minimal improvement due to parsing error",
            "improved_at": datetime.now().isoformat(),
            "original_review_score": review.get("weighted_score", 0)
        }
        
        return improved_experiment
    
    def generate_alternative_approaches(self, 
                                      experiment: ResearchExperiment) -> List[ResearchExperiment]:
        """Generate alternative approaches to test the same research question."""
        
        system_prompt = """
You are an expert in experimental design and research methodology.
Your task is to generate alternative experimental approaches to test the same research question.

Consider:
- Different methodologies that could test the same hypothesis
- Alternative parameter configurations
- Different evaluation approaches
- Complementary experiments that could provide additional insights
- Variations in experimental design that might be more efficient or robust

Generate 3-5 alternative approaches that test the same core research question but with different methodologies.
"""
        
        messages = [{
            "role": "user",
            "content": f"""
Original experiment:
Title: {experiment.title}
Description: {experiment.description}
Hypothesis: {experiment.hypothesis}
Current parameters: {json.dumps(experiment.parameters, indent=2)}

Generate 3-5 alternative experimental approaches that test the same research question
but with different methodologies, parameter ranges, or evaluation approaches.

Each alternative should:
1. Test the same core hypothesis
2. Use a different methodological approach
3. Be feasible on T4 GPU
4. Provide complementary insights

Return in JSON format:
{{
  "alternatives": [
    {{
      "title": "Alternative approach title",
      "description": "How this approach differs from the original",
      "hypothesis": "Same or refined hypothesis",
      "parameters": {{"param1": "value1"}},
      "priority": 7,
      "estimated_duration_minutes": 120,
      "rationale": "Why this approach might be better"
    }}
  ]
}}
"""
        }]
        
        response = self.call_llm(messages, system_prompt)
        
        try:
            alternatives_data = json.loads(response)
            alternatives = []
            
            for alt_data in alternatives_data.get("alternatives", []):
                alternative = ResearchExperiment(
                    id=str(uuid.uuid4()),
                    title=alt_data["title"],
                    description=alt_data["description"],
                    hypothesis=alt_data["hypothesis"],
                    parameters=alt_data["parameters"],
                    priority=alt_data["priority"],
                    estimated_duration=alt_data["estimated_duration_minutes"]
                )
                
                alternative.metadata = {
                    "original_experiment_id": experiment.id,
                    "alternative_rationale": alt_data.get("rationale", ""),
                    "created_at": datetime.now().isoformat()
                }
                
                alternatives.append(alternative)
            
            return alternatives
            
        except json.JSONDecodeError:
            return [experiment]  # Return original if parsing fails
    
    def optimize_for_t4_gpu(self, experiment: ResearchExperiment) -> ResearchExperiment:
        """Optimize experiment specifically for T4 GPU constraints."""
        
        system_prompt = """
You are an expert in GPU optimization and hardware-aware ML research.
Your task is to optimize research experiments specifically for Tesla T4 GPU constraints.

T4 GPU specifications to consider:
- 16GB VRAM
- 2560 CUDA cores
- Memory bandwidth limitations
- Power consumption constraints
- Training time limitations

Optimization strategies:
- Batch size optimization
- Model size constraints
- Memory-efficient implementations
- Training time optimization
- Mixed precision training
- Gradient accumulation strategies
"""
        
        messages = [{
            "role": "user",
            "content": f"""
Optimize this experiment for T4 GPU constraints:

Title: {experiment.title}
Description: {experiment.description}
Hypothesis: {experiment.hypothesis}
Parameters: {json.dumps(experiment.parameters, indent=2)}
Estimated Duration: {experiment.estimated_duration} minutes

Optimize the experiment to:
1. Fit within T4 GPU memory constraints (16GB VRAM)
2. Maximize training efficiency
3. Minimize training time while maintaining validity
4. Use appropriate batch sizes and model configurations
5. Implement memory-efficient strategies

Return optimized parameters and estimated duration:

{{
  "optimized_experiment": {{
    "title": "T4-optimized experiment title",
    "description": "Description highlighting T4 optimizations",
    "hypothesis": "Same hypothesis",
    "parameters": {{"optimized_param1": "value1"}},
    "estimated_duration_minutes": 90
  }},
  "optimizations_applied": [
    "List of specific optimizations"
  ],
  "resource_requirements": {{
    "estimated_vram_usage": "8GB",
    "estimated_training_time": "90 minutes",
    "batch_size": 32
  }}
}}
"""
        }]
        
        response = self.call_llm(messages, system_prompt)
        
        try:
            optimization_data = json.loads(response)
            optimized_exp = optimization_data["optimized_experiment"]
            
            optimized_experiment = ResearchExperiment(
                id=str(uuid.uuid4()),
                title=optimized_exp["title"],
                description=optimized_exp["description"],
                hypothesis=optimized_exp["hypothesis"],
                parameters=optimized_exp["parameters"],
                priority=experiment.priority,
                estimated_duration=optimized_exp["estimated_duration_minutes"]
            )
            
            optimized_experiment.metadata = {
                "original_experiment_id": experiment.id,
                "optimizations_applied": optimization_data.get("optimizations_applied", []),
                "resource_requirements": optimization_data.get("resource_requirements", {}),
                "optimized_at": datetime.now().isoformat()
            }
            
            return optimized_experiment
            
        except json.JSONDecodeError:
            return experiment  # Return original if parsing fails
    
    def process(self, input_data: Any) -> Any:
        """Process input data for improvement."""
        if isinstance(input_data, dict) and "experiment" in input_data and "review" in input_data:
            return self.improve_experiment(input_data["experiment"], input_data["review"])
        elif isinstance(input_data, ResearchExperiment):
            return self.generate_alternative_approaches(input_data)
        
        return {"error": "Invalid input data type for improvement agent"}
