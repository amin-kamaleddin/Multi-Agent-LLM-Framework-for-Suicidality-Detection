import os
import asyncio
from typing import List, Optional
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

class AIDataGenerator:
    """
    A concurrent data generator that uses an LLM (via OpenRouter or OpenAI) to rewrite 
    text instances from one class label to another, guided by a system prompt 
    (e.g., the class definitions).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        max_concurrent: int = 50,
        system_prompt: str = ""
    ):
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.system_prompt = system_prompt
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in OPEN_ROUTER_KEY environment variable.")
            
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
    async def _generate_single(
        self, 
        text: str, 
        original_label: str, 
        target_label: str,
        retries: int = 3
    ) -> str:
        """
        Rewrites a single piece of text to fit the target_label.
        """
        async with self.semaphore:
            prompt = (
                f"You are an expert data generator and text rewriter.\n"
                f"I will provide you with a text that is currently classified as '{original_label}'.\n"
                f"Your task is to rewrite the text so that it naturally, realistically, and accurately "
                f"fits the exact definition of the '{target_label}' class.\n\n"
                f"Original Text (Label: {original_label}):\n{text}\n\n"
                f"Please provide ONLY the rewritten text for the '{target_label}' class. "
                f"Do not include quotes, conversational filler, or markdown blocks. "
                f"It should read like an authentic user post."
            )
            
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            for attempt in range(retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    content = response.choices[0].message.content
                    if content:
                        return content.strip()
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"Error generating data for text (target: {target_label}): {e}")
                        return ""
                    await asyncio.sleep(2 ** attempt)
            return ""

    async def generate_batch(
        self, 
        texts: List[str], 
        current_labels: List[str], 
        target_labels: List[str]
    ) -> List[str]:
        """
        Concurrently process a batch of text rewrites.
        """
        tasks = [
            self._generate_single(text, orig, tgt)
            for text, orig, tgt in zip(texts, current_labels, target_labels)
        ]
        
        # tqdm.gather provides a nice progress bar for the async tasks
        results = await tqdm.gather(*tasks, desc="Generating data")
        return results
        
    def run_dataframe(
        self, 
        df: pd.DataFrame, 
        text_col: str, 
        label_col: str, 
        target_col: str,
        replace_original: bool = False
    ) -> pd.DataFrame:
        """
        Synchronous wrapper to process a Pandas DataFrame.
        """
        # Ensure we have an active event loop for the async execution
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        texts = df[text_col].tolist()
        current_labels = df[label_col].tolist()
        target_labels = list(target_col) if isinstance(target_col, (list, tuple, pd.Series)) else df[target_col].tolist()
        
        results = loop.run_until_complete(
            self.generate_batch(texts, current_labels, target_labels)
        )
        
        df_out = df.copy()
        if replace_original:
            df_out[text_col] = results
            df_out[label_col] = target_labels
            if isinstance(target_col, str) and target_col in df_out.columns:
                df_out.drop(columns=[target_col], inplace=True)
        else:
            df_out['rewritten_text'] = results
            
        return df_out

    def process_csv(
        self,
        input_csv_path: str,
        text_col: str,
        label_col: str,
        target_col: str,
        output_csv_path: Optional[str] = None
    ) -> str:
        """
        Reads a CSV, generates new texts based on the target class, replaces the original 
        text and label fields, and saves to {original}_generated.csv.
        """
        if not output_csv_path:
            base, ext = os.path.splitext(input_csv_path)
            output_csv_path = f"{base}_generated{ext}"
            
        df = pd.read_csv(input_csv_path)
        
        print(f"Loaded {len(df)} rows from {input_csv_path}")
        df_generated = self.run_dataframe(
            df=df,
            text_col=text_col,
            label_col=label_col,
            target_col=target_col,
            replace_original=True
        )
        
        df_generated.to_csv(output_csv_path, index=False)
        print(f"Saved generated dataset to {output_csv_path}")
        return output_csv_path
