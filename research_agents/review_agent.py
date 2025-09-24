"""Research Review Agent - evaluates and reviews research experiments."""

import json
from typing import List, Dict, Any, Tuple
from datetime import datetime

from .base_agent import BaseAgent, ResearchExperiment


class ReviewAgent(BaseAgent):
    """Agent responsible for reviewing and evaluating research experiments."""
    
    def __init__(self, openrouter_api_key: str, model: str = "x-ai/grok-4-fast:free"):
        super().__init__("review_agent", openrouter_api_key, model)
        
        # Review criteria weights
        self.review_criteria = {
            "novelty": 0.25,      # How novel/innovative is this research?
            "feasibility": 0.20,   # How feasible is this on T4 GPU?
            "impact": 0.20,       # Potential impact on the field
            "clarity": 0.15,      # How clear is the hypothesis and methodology?
            "efficiency": 0.20    # How efficiently can this be tested?
        }
    
    def review_experiment(self, experiment: ResearchExperiment) -> Dict[str, Any]:
        """Review a research experiment and provide detailed feedback."""
        
        system_prompt = f"""
You are an expert research reviewer specializing in Machine Learning and MoE architectures.
You have extensive experience reviewing research proposals and experimental designs.

Your task is to provide a comprehensive review of research experiments for MoE LLM training on T4 GPUs.

Review criteria and weights:
- Novelty (25%): How novel/innovative is this research? Does it explore new territory?
- Feasibility (20%): How feasible is this experiment on T4 GPU hardware constraints?
- Impact (20%): What's the potential impact on the field if successful?
- Clarity (15%): How clear is the hypothesis, methodology, and expected outcomes?
- Efficiency (20%): How efficiently can this experiment be conducted and evaluated?

For each criterion, provide:
1. A score from 1-10
2. Detailed reasoning for the score
3. Specific suggestions for improvement

Also provide:
- Overall recommendation (approve/revise/reject)
- Priority assessment (1-10)
- Estimated success probability (0-1)
- Key risks and mitigation strategies
- Suggested modifications

Be thorough but constructive in your feedback.
"""
        
        messages = [{
            "role": "user",
            "content": f"""
Please review this research experiment:

**Title:** {experiment.title}

**Description:** {experiment.description}

**Hypothesis:** {experiment.hypothesis}

**Parameters to test:**
{json.dumps(experiment.parameters, indent=2)}

**Estimated duration:** {experiment.estimated_duration} minutes

**Priority:** {experiment.priority}/10

Provide a comprehensive review covering all criteria. Return your review in this JSON format:
{{
  "scores": {{
    "novelty": {{
      "score": 8,
      "reasoning": "Detailed explanation of the score"
    }},
    "feasibility": {{
      "score": 7,
      "reasoning": "Detailed explanation considering T4 GPU constraints"
    }},
    "impact": {{
      "score": 9,
      "reasoning": "Assessment of potential impact"
    }},
    "clarity": {{
      "score": 6,
      "reasoning": "Evaluation of hypothesis and methodology clarity"
    }},
    "efficiency": {{
      "score": 8,
      "reasoning": "Assessment of experimental efficiency"
    }}
  }},
  "overall_score": 7.6,
  "recommendation": "approve",
  "priority_assessment": 8,
  "success_probability": 0.75,
  "key_strengths": [
    "List of main strengths"
  ],
  "areas_for_improvement": [
    "List of areas that need improvement"
  ],
  "suggested_modifications": [
    "Specific suggestions for improvement"
  ],
  "risks": [
    "Potential risks and mitigation strategies"
  ],
  "detailed_feedback": "Comprehensive written feedback"
}}
"""
        }]
        
        response = self.call_llm(messages, system_prompt)
        
        try:
            review_data = json.loads(response)
            
            # Calculate weighted overall score
            weighted_score = 0
            for criterion, weight in self.review_criteria.items():
                if criterion in review_data.get("scores", {}):
                    weighted_score += review_data["scores"][criterion]["score"] * weight
            
            review_data["weighted_score"] = round(weighted_score, 2)
            review_data["reviewed_at"] = datetime.now().isoformat()
            review_data["experiment_id"] = experiment.id
            
            return review_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse review response: {e}")
            return self._create_fallback_review(experiment)
    
    def _create_fallback_review(self, experiment: ResearchExperiment) -> Dict[str, Any]:
        """Create a fallback review if LLM parsing fails."""
        return {
            "scores": {
                "novelty": {"score": 5, "reasoning": "Unable to assess - parsing error"},
                "feasibility": {"score": 5, "reasoning": "Unable to assess - parsing error"},
                "impact": {"score": 5, "reasoning": "Unable to assess - parsing error"},
                "clarity": {"score": 5, "reasoning": "Unable to assess - parsing error"},
                "efficiency": {"score": 5, "reasoning": "Unable to assess - parsing error"}
            },
            "overall_score": 5.0,
            "weighted_score": 5.0,
            "recommendation": "revise",
            "priority_assessment": 5,
            "success_probability": 0.5,
            "key_strengths": ["Experiment has clear parameters"],
            "areas_for_improvement": ["Unable to assess due to parsing error"],
            "suggested_modifications": ["Review the experiment description for clarity"],
            "risks": ["Unknown due to parsing error"],
            "detailed_feedback": "Unable to provide detailed feedback due to parsing error. Please review the experiment description.",
            "reviewed_at": datetime.now().isoformat(),
            "experiment_id": experiment.id
        }
    
    def compare_experiments(self, experiments: List[ResearchExperiment]) -> Dict[str, Any]:
        """Compare multiple experiments and rank them."""
        
        reviews = []
        for experiment in experiments:
            review = self.review_experiment(experiment)
            reviews.append({
                "experiment": experiment,
                "review": review
            })
        
        # Sort by weighted score
        reviews.sort(key=lambda x: x["review"]["weighted_score"], reverse=True)
        
        # Generate comparison analysis
        system_prompt = """
You are an expert research coordinator. Analyze the comparison of multiple research experiments
and provide insights about their relative merits, synergies, and recommended execution order.

Consider:
- Which experiments build on each other
- Resource allocation and scheduling
- Risk diversification
- Expected impact and timeline
"""
        
        comparison_text = "Experiment Comparison:\n\n"
        for i, item in enumerate(reviews):
            exp = item["experiment"]
            review = item["review"]
            comparison_text += f"{i+1}. {exp.title} (Score: {review['weighted_score']})\n"
            comparison_text += f"   Recommendation: {review['recommendation']}\n"
            comparison_text += f"   Priority: {review['priority_assessment']}/10\n\n"
        
        messages = [{
            "role": "user",
            "content": f"""
Analyze this comparison of research experiments:

{comparison_text}

Provide insights on:
1. Recommended execution order
2. Potential synergies between experiments
3. Resource allocation suggestions
4. Risk management strategy

Return in JSON format:
{{
  "execution_order": ["experiment_id_1", "experiment_id_2", ...],
  "synergies": [
    {{"experiments": ["id1", "id2"], "description": "How they complement each other"}}
  ],
  "resource_allocation": {{"high_priority": 60, "medium_priority": 30, "low_priority": 10}},
  "risk_assessment": "Overall risk assessment and mitigation strategy",
  "timeline_recommendation": "Suggested timeline and milestones"
}}
"""
        }]
        
        response = self.call_llm(messages, system_prompt)
        
        try:
            comparison_data = json.loads(response)
            comparison_data["experiment_reviews"] = reviews
            comparison_data["comparison_date"] = datetime.now().isoformat()
            return comparison_data
            
        except json.JSONDecodeError:
            return {
                "execution_order": [exp.id for exp in experiments],
                "synergies": [],
                "resource_allocation": {"high_priority": 60, "medium_priority": 30, "low_priority": 10},
                "risk_assessment": "Unable to assess due to parsing error",
                "timeline_recommendation": "Execute in score order",
                "experiment_reviews": reviews,
                "comparison_date": datetime.now().isoformat()
            }
    
    def batch_review(self, experiments: List[ResearchExperiment]) -> List[Dict[str, Any]]:
        """Review multiple experiments efficiently."""
        reviews = []
        
        for experiment in experiments:
            self.logger.info(f"Reviewing experiment: {experiment.title}")
            review = self.review_experiment(experiment)
            reviews.append({
                "experiment_id": experiment.id,
                "experiment_title": experiment.title,
                "review": review
            })
        
        return reviews
    
    def process(self, input_data: Any) -> Any:
        """Process input data for review."""
        if isinstance(input_data, ResearchExperiment):
            return self.review_experiment(input_data)
        elif isinstance(input_data, list) and len(input_data) > 0:
            if isinstance(input_data[0], ResearchExperiment):
                if len(input_data) == 1:
                    return self.review_experiment(input_data[0])
                else:
                    return self.compare_experiments(input_data)
            else:
                return self.batch_review(input_data)
        
        return {"error": "Invalid input data type for review agent"}
