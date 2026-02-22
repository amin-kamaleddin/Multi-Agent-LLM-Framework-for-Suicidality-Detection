"""
Concrete implementations of evaluators.
"""

import time
import json
import concurrent.futures
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple

from .base import BaseEvaluator, EvaluationResult, EvaluationMetrics
from ..agent.base import BaseAgent, AgentResponse
from ..dataset.base import BaseDataset, DataSample


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _process_sample(self, sample: DataSample, agent: BaseAgent, max_retries: int) -> Tuple[DataSample, Optional[AgentResponse], Optional[str]]:
        """Process a single sample with retry logic."""
        last_error = None
        for attempt in range(max_retries):
            try:
                response = agent.predict(sample.input_text)
                response.input_id = sample.id
                return sample, response, None
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s...
                    time.sleep(2 ** attempt)
        return sample, None, f"Failed after {max_retries} attempts. Last error: {last_error}"
        
    def evaluate(self, 
                agent: BaseAgent, 
                dataset: BaseDataset, 
                task_name: str,
                version: str = "1.0",
                max_workers: int = 5,
                max_retries: int = 3,
                **kwargs) -> EvaluationResult:
        """
        Evaluate agent on classification task with parallel execution and retries.
        
        Calculates accuracy, precision, recall, and F1 score.
        """
        self.validate_inputs(agent, dataset)
        
        if not agent._is_initialized:
            agent.initialize()
        
        start_time = time.time()
        
        # Collect predictions
        predictions = []
        ground_truth = []
        per_sample_metrics = []
        execution_failures = []
        
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_sample, sample, agent, max_retries): sample 
                for sample in dataset
            }
            
            for future in concurrent.futures.as_completed(futures):
                sample, response, error = future.result()
                
                if error:
                    execution_failures.append({
                        "sample_id": sample.id,
                        "input_text": sample.input_text,
                        "error": error
                    })
                    predictions.append("Error")
                    ground_truth.append(sample.target_output)
                    per_sample_metrics.append({
                        "sample_id": sample.id,
                        "accuracy": 0.0,
                        "prediction": "Error",
                        "ground_truth": sample.target_output,
                        "error": error
                    })
                else:
                    predictions.append(response.output)
                    ground_truth.append(sample.target_output)
                    
                    is_correct = response.output == sample.target_output
                    sample_metric = {
                        "sample_id": sample.id,
                        "accuracy": 1.0 if is_correct else 0.0,
                        "prediction": response.output,
                        "ground_truth": sample.target_output
                    }
                    if response.metadata:
                        sample_metric.update(response.metadata)
                    per_sample_metrics.append(sample_metric)
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(ground_truth, predictions)
        metrics.per_sample_metrics = per_sample_metrics
        
        # Calculate aggregate streaming metrics
        import statistics
        
        def safe_mean(values):
            return statistics.mean(values) if values else 0.0
            
        def safe_stdev(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0
            
        ttfts = [m["ttft"] for m in per_sample_metrics if "ttft" in m]
        tps_list = [m["tps"] for m in per_sample_metrics if "tps" in m]
        gen_times = [m["generation_time"] for m in per_sample_metrics if "generation_time" in m]
        total_times = [m["total_time"] for m in per_sample_metrics if "total_time" in m]
        
        if ttfts:
            metrics.add_metric("avg_ttft", safe_mean(ttfts))
            metrics.add_metric("std_ttft", safe_stdev(ttfts))
        if tps_list:
            metrics.add_metric("avg_tps", safe_mean(tps_list))
            metrics.add_metric("std_tps", safe_stdev(tps_list))
        if gen_times:
            metrics.add_metric("avg_generation_time", safe_mean(gen_times))
        if total_times:
            metrics.add_metric("avg_total_time", safe_mean(total_times))
        
        # Token usage aggregation
        prompt_toks = [m["prompt_tokens"] for m in per_sample_metrics if "prompt_tokens" in m]
        completion_toks = [m["completion_tokens"] for m in per_sample_metrics if "completion_tokens" in m]
        total_toks = [m["total_tokens"] for m in per_sample_metrics if "total_tokens" in m]
        
        if prompt_toks:
            metrics.add_metric("total_prompt_tokens", sum(prompt_toks))
            metrics.add_metric("total_completion_tokens", sum(completion_toks))
            metrics.add_metric("total_tokens", sum(total_toks))
            metrics.add_metric("avg_prompt_tokens", safe_mean(prompt_toks))
            metrics.add_metric("avg_completion_tokens", safe_mean(completion_toks))
        
        duration = time.time() - start_time
        
        # Record metadata for reproducibility
        eval_metadata = {
            "task_name": task_name,
            "version": version,
            "max_workers": max_workers,
            "max_retries": max_retries,
            "agent_info": agent.get_info(),
            "dataset_info": dataset.get_statistics(),
            "execution_failures": execution_failures,
            "kwargs": kwargs
        }
        full_metadata = {**self._metadata, **eval_metadata}
        
        from datetime import datetime
        result = EvaluationResult(
            agent_name=agent.name,
            dataset_name=dataset.name,
            metrics=metrics,
            duration=duration,
            timestamp=datetime.now(),
            task_name=task_name,
            version=version,
            metadata=full_metadata
        )
        
        return result
        
    def retry_failed_cases(self, meta_file_path: Union[str, Path], agent: BaseAgent, dataset: BaseDataset, **kwargs) -> Optional[EvaluationResult]:
        """
        Read Meta.json, extract execution failures, and re-run evaluation on those specific cases.
        """
        meta_path = Path(meta_file_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
            
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        failed_ids = [str(f["sample_id"]) for f in meta.get("execution_failures", [])]
        if not failed_ids:
            print("No execution failures found in Meta.json to retry.")
            return None
            
        # Create a new dataset with only the failed cases
        retry_dataset = dataset.__class__(name=f"{dataset.name}_retry")
        for sample in dataset:
            if str(sample.id) in failed_ids:
                retry_dataset.add_sample(sample)
                
        print(f"Retrying {len(retry_dataset)} failed cases...")
        
        return self.evaluate(
            agent=agent,
            dataset=retry_dataset,
            task_name=meta.get("task_name", "retry_task"),
            version=meta.get("version", "1.0"),
            max_workers=meta.get("max_workers", 1),
            max_retries=meta.get("max_retries", 3),
            **kwargs
        )
        
    def _calculate_classification_metrics(self, y_true: List[Any], y_pred: List[Any]) -> EvaluationMetrics:
        """Calculate classification metrics."""
        metrics = EvaluationMetrics()
        
        # Accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0
        metrics.add_metric("accuracy", accuracy)
        
        # For multi-class, calculate macro-averaged precision, recall, f1
        classes = set(y_true) | set(y_pred)
        
        if len(classes) > 0:
            precisions = []
            recalls = []
            f1s = []
            
            for cls in classes:
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                
            metrics.add_metric("precision", sum(precisions) / len(precisions))
            metrics.add_metric("recall", sum(recalls) / len(recalls))
            metrics.add_metric("f1_score", sum(f1s) / len(f1s))
            
        return metrics
