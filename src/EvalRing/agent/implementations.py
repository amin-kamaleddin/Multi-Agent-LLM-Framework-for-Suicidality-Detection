"""
Concrete implementations of agents.
"""

import os
import time
import random
from typing import List, Optional

from .base import BaseAgent, AgentResponse


class MockAgent(BaseAgent):
    """
    A mock agent for testing and demonstration purposes.
    
    Returns random predictions from a predefined list or a fixed response.
    """
    
    def __init__(self, 
                 name: str = "mock_agent",
                 fixed_response: Optional[str] = None,
                 possible_outputs: Optional[List[str]] = None,
                 delay: float = 0.1,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.fixed_response = fixed_response
        self.possible_outputs = possible_outputs or ["positive", "negative", "neutral"]
        self.delay = delay
        
    def initialize(self, **kwargs) -> None:
        """Initialize the mock agent."""
        self._is_initialized = True
        
    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        """
        Make a mock prediction.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse with mock prediction
        """
        if not self._is_initialized:
            raise RuntimeError("Agent must be initialized before prediction")
            
        start_time = time.time()
        
        # Simulate processing time
        if self.delay > 0:
            time.sleep(self.delay)
            
        # Generate response
        if self.fixed_response is not None:
            output = self.fixed_response
        else:
            output = random.choice(self.possible_outputs)
            
        processing_time = time.time() - start_time
        
        return AgentResponse(
            input_id="",  # Will be set by evaluator or batch processor
            input_text=input_text,
            output=output,
            confidence=random.uniform(0.5, 1.0),
            processing_time=processing_time,
            metadata={"mock": True}
        )


class RuleBasedAgent(BaseAgent):
    """
    A simple rule-based agent that uses keyword matching.
    """
    
    def __init__(self, 
                 name: str = "rule_based_agent",
                 rules: Optional[dict] = None,
                 default_output: str = "unknown",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.rules = rules or {}
        self.default_output = default_output
        
    def initialize(self, **kwargs) -> None:
        """Initialize the rule-based agent."""
        # Convert all keywords to lowercase for case-insensitive matching
        self._processed_rules = {}
        for output, keywords in self.rules.items():
            self._processed_rules[output] = [k.lower() for k in keywords]
            
        self._is_initialized = True
        
    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        """
        Make a prediction based on keyword rules.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse with rule-based prediction
        """
        if not self._is_initialized:
            raise RuntimeError("Agent must be initialized before prediction")
            
        start_time = time.time()
        
        text_lower = input_text.lower()
        output = self.default_output
        confidence = 0.0
        matched_keyword = None
        
        # Simple keyword matching
        for rule_output, keywords in self._processed_rules.items():
            for keyword in keywords:
                if keyword in text_lower:
                    output = rule_output
                    confidence = 1.0
                    matched_keyword = keyword
                    break
            if matched_keyword:
                break
                
        processing_time = time.time() - start_time
        
        return AgentResponse(
            input_id="",
            input_text=input_text,
            output=output,
            confidence=confidence,
            processing_time=processing_time,
            metadata={"matched_keyword": matched_keyword} if matched_keyword else {}
        )


class OpenAIAgent(BaseAgent):
    """
    Generic OpenAI-backed agent with streaming, TTFT, TPS, and token tracking.

    Subclass this and set ``self.system_prompt`` to get a working agent.
    Override ``_parse_output`` if you need custom output cleaning.

    Args:
        name: Agent name for reporting.
        model_name: OpenAI model identifier (e.g., "gpt-4o", "gpt-5.2").
        api_key: API key. Falls back to OPEN_ROUTER_KEY, then OPENAI_API_KEY.
        system_prompt: System message sent to the model.
        temperature: Sampling temperature.  Default 0.0.
        max_completion_tokens: Maximum tokens the model may generate.
        **kwargs: Forwarded to BaseAgent.
    """

    def __init__(
        self,
        name: str = "openai-agent",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_completion_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or (
            "https://openrouter.ai/api/v1" if os.environ.get("OPEN_ROUTER_KEY") else None
        )
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.client = None

    # ── lifecycle ──────────────────────────────────────────

    def initialize(self, **kwargs) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required.  Install: pip install openai"
            )
        if not self.api_key:
            raise ValueError(
                "API key not found. Set OPEN_ROUTER_KEY or OPENAI_API_KEY, "
                "or pass api_key to the constructor."
            )
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)
        self._is_initialized = True

    # ── prediction ────────────────────────────────────────

    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        """
        Call the OpenAI Chat Completions API with streaming.

        Returns an ``AgentResponse`` whose metadata includes:
        ``raw_output``, ``model``, ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``, ``ttft``, ``generation_time``, ``tps``, ``total_time``.
        """
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized.  Call initialize() first.")

        if not self.validate_input(input_text):
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output=None,
                error="Invalid input",
            )

        start_time = time.time()

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._build_user_message(input_text)},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            first_token_time = None
            raw_output = ""
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        raw_output += delta

                if hasattr(chunk, "usage") and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

            end_time = time.time()
            if first_token_time is None:
                first_token_time = end_time

            ttft = first_token_time - start_time
            generation_time = end_time - first_token_time
            total_time = end_time - start_time
            tps = completion_tokens / generation_time if generation_time > 0 else 0.0

            output = self._parse_output(raw_output)

            return AgentResponse(
                input_id="",
                input_text=input_text,
                output=output,
                confidence=1.0,
                processing_time=total_time,
                metadata={
                    "raw_output": raw_output,
                    "model": self.model_name,
                    "base_url": self.base_url,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "ttft": ttft,
                    "generation_time": generation_time,
                    "tps": tps,
                    "total_time": total_time,
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output="Error",
                error=str(e),
                processing_time=processing_time,
            )

    # ── hooks for subclasses ──────────────────────────────

    def _build_user_message(self, input_text: str) -> str:
        """
        Build the user-role message from ``input_text``.

        Override this to add task-specific formatting, e.g.::

            def _build_user_message(self, input_text):
                return f"Text to classify:\\n\\n{input_text}"

        Default returns the input as-is.
        """
        return input_text

    def _parse_output(self, raw_output: str) -> str:
        """
        Post-process the model's raw text into the final output.

        Override for label cleaning, JSON extraction, etc.
        Default strips whitespace.
        """
        return raw_output.strip()
