"""Base agent class for all research agents."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from openai import OpenAI


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ResearchExperiment:
    """Research experiment data structure."""
    id: str
    title: str
    description: str
    hypothesis: str
    parameters: Dict[str, Any]
    priority: int
    estimated_duration: int  # in minutes
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BaseAgent(ABC):
    """Base class for all research agents."""
    
    def __init__(self, name: str, openrouter_api_key: str, model: str = "x-ai/grok-4-fast:free"):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        self.model = model
        
        # Initialize OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        
        # Agent conversation history
        self.conversation_history: List[AgentMessage] = []
        
        # Agent-specific configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load agent-specific configuration."""
        config_path = f"configs/{self.name}_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_config(self):
        """Save agent configuration."""
        os.makedirs("configs", exist_ok=True)
        config_path = f"configs/{self.name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def send_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """Send a message and add to conversation history."""
        message = AgentMessage(
            role="assistant",
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.conversation_history.append(message)
        return message
    
    def receive_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """Receive a message from another agent."""
        message = AgentMessage(
            role="user",
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.conversation_history.append(message)
        return message
    
    def call_llm(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        """Call the LLM with messages."""
        try:
            # Prepare messages for API
            api_messages = []
            
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            for msg in self.conversation_history[-10:]:  # Last 10 messages for context
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current messages
            api_messages.extend(messages)
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            response = completion.choices[0].message.content
            self.logger.info(f"LLM response received: {response[:100]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            return f"Error: {str(e)}"
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "name": self.name,
            "model": self.model,
            "conversation_length": len(self.conversation_history),
            "config": self.config,
            "status": "active"
        }
