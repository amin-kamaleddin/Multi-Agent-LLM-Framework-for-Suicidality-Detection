from typing import Optional
import os
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

**Indicator**: Emotional distress, depression, hopelessness, or self-harm for coping, with NO reference to suicide, dying, ending life, or wanting it to be over. Purely about pain, not about death or non-existence.

**Ideation**: The person expresses THOUGHTS, WISHES, or PASSIVE DESIRES about death or suicide — wanting to die, wishing to not exist, wanting "a way out" of life, asking about methods, thinking about plans. This is THINKING/WISHING, not acting. Includes strong language like "all I can think about is ending my life", "I want to be free of these feelings", "hopefully I can be free" — these are still wishes, not actions.

**Behavior**: The person has already taken CONCRETE PREPARATORY ACTIONS toward a specific plan — acquiring means, writing farewell/goodbye messages addressed to specific named people, giving away possessions, going to a location, setting a specific date ("it has to be today/tonight"). Writing "I miss you [name], I'm sorry" to loved ones = goodbye note = Behavior. Expressing a specific time deadline combined with farewell words = Behavior.

**Attempt**: A suicide attempt is ACTIVELY HAPPENING RIGHT NOW or has just occurred — the person has ingested substances, is cutting with lethal intent, has jumped, etc. Requires evidence of an ONGOING or JUST-COMPLETED physical act, not a wish or a plan.

Key rules:
- "All I can think about is ending my life", "I want to be free", "hopefully be free of these feelings" → Ideation (strong wishing, still just thoughts — NOT Attempt)
- Writing farewell words to specific named people ("I miss you dad, I'm sorry") + a specific date or "today" → Behavior (NOT Ideation)
- Acquiring a weapon, buying pills, writing a note and placing it → Behavior
- Currently taking pills, currently cutting → Attempt
- "Want a way out", "want it to be over", "don't want to wake up" → Ideation (NOT Indicator)
- Thinking or talking about a plan WITHOUT doing anything physical → Ideation (NOT Behavior)
- When in doubt between Indicator and Ideation → choose Ideation
- When in doubt between Ideation and Behavior → choose Ideation
- When in doubt between Ideation and Attempt → choose Ideation (Attempt requires active physical action NOW)

Respond with ONLY one word: Ideation, Behavior, Attempt, or Indicator"""

    VALID_LABELS = ["Ideation", "Behavior", "Indicator", "Attempt"]

    def __init__(
        self,
        name: str = "gpt-5-mini-suicide-detector",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ):
        if model_name is None:
            if os.environ.get("OPEN_ROUTER_KEY"):
                model_name = os.environ.get("OPEN_ROUTER_MODEL", "openai/gpt-4o-mini")
            else:
                model_name = os.environ.get("OPENAI_MODEL", "gpt-5.2")

        super().__init__(
            name=name,
            model_name=model_name,
            api_key=api_key,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=temperature,
            max_completion_tokens=500,  # reasoning models (e.g. gpt-5-mini) consume tokens internally before outputting; 500 covers reasoning budget + label output
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
