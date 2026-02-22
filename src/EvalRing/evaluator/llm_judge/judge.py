"""
LLM Judge backends.

An LLMJudge wraps an LLM client and exposes a simple `generate(prompt) -> str`
interface. The evaluator/metric layer calls the judge to get raw outputs,
which are then parsed via the schema module.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMJudge(ABC):
    """
    Abstract base class for an LLM judge backend.
    
    Subclass this to integrate any LLM provider (OpenAI, Anthropic,
    local models, etc.) as a judge.
    """
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._config = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a text response from the judge LLM.
        
        Args:
            prompt: The fully-rendered judge prompt.
            **kwargs: Provider-specific overrides.
            
        Returns:
            Raw string output from the LLM.
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Return metadata about this judge for logging."""
        return {
            "model_name": self.model_name,
            "type": self.__class__.__name__,
            "config": {k: v for k, v in self._config.items() if k != "api_key"}
        }


class OpenAIJudge(LLMJudge):
    """
    LLM Judge backed by the OpenAI Chat Completions API.
    
    Args:
        model_name: OpenAI model to use (e.g., "gpt-4o", "gpt-5.2").
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        temperature: Sampling temperature. Default 0.0 for deterministic judging.
        max_completion_tokens: Max tokens in judge response. Default 512.
        **kwargs: Additional parameters forwarded to the API call.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self._client = None
    
    def _ensure_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install: pip install openai")
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY or pass api_key.")
            self._client = OpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Call the OpenAI Chat Completions API.
        
        Args:
            prompt: The judge prompt (sent as a user message).
            **kwargs: Overrides for temperature, max_completion_tokens, etc.
            
        Returns:
            The model's response text.
        """
        self._ensure_client()
        
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_completion_tokens", self.max_completion_tokens)
        
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluation judge. Always respond with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content.strip()
    
    def get_model_name(self) -> str:
        return self.model_name
