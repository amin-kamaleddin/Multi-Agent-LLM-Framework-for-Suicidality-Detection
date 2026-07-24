"""
Convenience wrapper to generate the model list for the suicide detection suite.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.EvalRing.utils.generate_model_list import generate_model_list

if __name__ == "__main__":
    out_path = script_dir / "model_list.json"
    requested_mapping = {
            #"Gemini-3.1 pro": ("google/gemini-3.1-pro-preview", 2025),
            #"Grok-4": ("x-ai/grok-4.1-fast", 2025),
            #"Sonnet-4.6": ("anthropic/claude-sonnet-4.6", 2025),
            #"GLM5": ("z-ai/glm-5", 2025),
            "GPT-5.6 Luna": ("openai/gpt-5.6-luna", 2026),
            #"Gemini-3 Flash": ("google/gemini-3-flash-preview", 2024),
            #"Qwen3.5": ("qwen/qwen3.5-plus-02-15", 2024),
            #"GPT-3.5 turbo": ("openai/gpt-3.5-turbo", 2023),
            #"Llama-3": ("meta-llama/llama-3.3-70b-instruct", 2023),
            #"KIMI": ("moonshotai/kimi-k2.5", None),
            #"Minimax": ("minimax/minimax-m2.5", None),
            #"DeepSeek V3.2": ("deepseek/deepseek-v3.2", 2024),
            #"GPT-5 mini": ("openai/gpt-5-mini", 2024),
        }
    requested_mapping = {
            # "Gemini-3.1 pro": ("google/gemini-3.1-pro-preview", 2025),
            #"Grok-4": ("x-ai/grok-4.1-fast", 2025),
            # "Sonnet-4.6": ("anthropic/claude-sonnet-4.6", 2025),
            #"GLM5": ("z-ai/glm-5", 2025),
            "GPT-5.6 Luna": ("openai/gpt-5.6-luna", 2026),
            #"Gemini-3 Flash": ("google/gemini-3-flash-preview", 2024),
            #"Qwen3.5": ("qwen/qwen3.5-plus-02-15", 2024),
            #"GPT-3.5 turbo": ("openai/gpt-3.5-turbo", 2023),
            #"Llama-3": ("meta-llama/llama-3.3-70b-instruct", 2023),
            #"KIMI": ("moonshotai/kimi-k2.5", None),
            #"Minimax": ("minimax/minimax-m2.5", None),
            #"DeepSeek V3.2": ("deepseek/deepseek-v3.2", 2024),
            #"GPT-5 mini": ("openai/gpt-5-mini", 2024),
        }
    
    requested_mapping = {
        "GPT-5.6 Luna": ("openai/gpt-5.6-luna", 2026),
    }
    generate_model_list(out_path, requested_mapping)
