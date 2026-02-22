from typing import Optional
from EvalRing.agent import OpenAIAgent


class OpenAISuicideDetectionAgent(OpenAIAgent):
    """
    Suicide Risk Classification agent using OpenAI.
    Classifies text into: Ideation, Behavior, Indicator, Attempt.

    Subclasses OpenAIAgent — inherits streaming, TTFT/TPS tracking,
    and token counting.  Only the system prompt and output parsing
    are domain-specific.
    """

    SYSTEM_PROMPT = """Classify the text into exactly ONE suicide risk category.

**Indicator**: Emotional distress, depression, or self-harm for coping, with NO reference to suicide, dying, ending life, or wanting it to be over. Purely about pain, not about death.

**Ideation**: ANY mention of suicide, wanting to die, wishing to not exist, wanting "a way out" (of life), wanting "it to end/be over" (in a life-ending sense), asking about methods, thinking about plans, or expressing that death is preferable. Even vague wishes like "I don't want to wake up" or "how much longer do I have to go on" count. If the text references death/suicide/ending life AT ALL — even passively — classify as Ideation, NOT Indicator.

**Behavior**: The person has taken PHYSICAL preparatory actions toward suicide — acquiring means, physically writing/leaving goodbye notes, going to a location, giving away possessions. Requires evidence of concrete actions already performed, not just thoughts or words about plans.

**Attempt**: A suicide attempt has occurred or is actively in progress — the person has ingested substances, cut with lethal intent, jumped, etc.

Key rules:
- "Want a way out", "want it to be over", "give up on life", "don't want to wake up" → Ideation (NOT Indicator)
- Thinking/talking about writing a will or plans → Ideation (NOT Behavior, since no physical action yet)
- When in doubt between Indicator and Ideation → choose Ideation
- When in doubt between Ideation and Behavior → choose Ideation

Respond with ONLY: Ideation, Behavior, Attempt, or Indicator"""

    VALID_LABELS = ["Ideation", "Behavior", "Indicator", "Attempt"]

    def __init__(
        self,
        name: str = "gpt-5-mini-suicide-detector",
        model_name: str = "gpt-5.2",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            model_name=model_name,
            api_key=api_key,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=temperature,
            max_completion_tokens=10,
            **kwargs,
        )

    def _build_user_message(self, input_text: str) -> str:
        return f"Text to classify:\n\n{input_text}"

    def _parse_output(self, raw_output: str) -> str:
        """Ensure the output strictly matches one of the valid labels."""
        clean = raw_output.strip('\'". \n').lower()
        for label in self.VALID_LABELS:
            if label.lower() in clean:
                return label
        return "Indicator"  # fallback
