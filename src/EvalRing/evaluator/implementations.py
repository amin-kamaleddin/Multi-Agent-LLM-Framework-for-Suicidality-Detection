"""
Concrete implementations of evaluators.
"""

import time
import json
import os
import datetime
import concurrent.futures
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple

from .base import BaseEvaluator, EvaluationResult, EvaluationMetrics
from ..agent import resolve_classification_prediction
from ..agent.base import BaseAgent, AgentResponse
from ..dataset.base import BaseDataset, DataSample


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _process_sample(
        self,
        sample: DataSample,
        agent: BaseAgent,
        max_retries: int,
        error_cb=None,
    ) -> Tuple[DataSample, Optional[AgentResponse], Optional[str]]:
        """Process a single sample with retry logic."""
        last_error = None
        attempt = 0
        rate_limit_attempt = 0
        last_error_type = None
        while True:
            try:
                response = agent.predict(sample.input_text)
                if getattr(response, 'error', None) or response.output == "Error":
                    raise RuntimeError(getattr(response, 'error', None) or "Unknown prediction error")
                    
                resolved = resolve_classification_prediction(response.output)
                if resolved.label is not None:
                    response.output = resolved.label
                if resolved.confidence is not None and response.confidence is None:
                    response.confidence = resolved.confidence
                if resolved.class_scores is not None:
                    response.metadata = response.metadata or {}
                    response.metadata.setdefault("class_scores", resolved.class_scores)
                response.input_id = sample.id
                return sample, response, None
            except Exception as e:
                last_error = str(e)
                last_error_type = e.__class__.__name__
                error_lower = last_error.lower()
                is_rate_limit = "429" in error_lower or "rate limit" in error_lower or "empty output" in error_lower

                if error_cb:
                    try:
                        error_cb({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "sample_id": str(sample.id),
                            "error_type": last_error_type,
                            "error_message": last_error,
                            "is_rate_limit": is_rate_limit,
                            "attempt": attempt + 1,
                            "rate_limit_attempt": rate_limit_attempt + 1 if is_rate_limit else 0,
                            "max_retries": max_retries,
                        })
                    except Exception:
                        pass
                
                if is_rate_limit:
                    if rate_limit_attempt < 30:
                        time.sleep(10)
                        rate_limit_attempt += 1
                        continue
                    else:
                        break
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    attempt += 1
                else:
                    break
        return sample, None, (
            f"Failed after {max_retries} attempts (or 30 rate limits). "
            f"Last error type: {last_error_type or 'UnknownError'}. "
            f"Last error: {last_error}"
        )
        
    def get_cache_info(self,
                       agent: BaseAgent,
                       dataset: BaseDataset,
                       **kwargs) -> dict:
        self.validate_inputs(agent, dataset)
        samples = list(dataset)
        agent_mode = kwargs.get("agent_mode", "")
        base_class = kwargs.get("base_class", "")
        model_name = getattr(agent, "model_name", "")
        
        ignore_errors = kwargs.get("ignore_errors", False)
        
        self.global_cache.preload_legacy_cache(
            model_name=model_name,
            agent_mode=agent_mode,
            base_class=base_class,
            ignore_errors=ignore_errors
        )
        
        uncached = 0
        cached = 0
        
        try:
            from tqdm import tqdm
            samples_iter = tqdm(samples, desc=f"Checking cache for {model_name or 'model'}", leave=False, disable=not kwargs.get("show_progress", False))
        except ImportError:
            samples_iter = samples
            
        for sample in samples_iter:
            cache_key = self.global_cache.generate_key(
                model_name=model_name,
                payload=sample.input_text,
                params={"agent_mode": agent_mode, "base_class": base_class}
            )
            cached_result = self.global_cache.lookup(
                cache_key=cache_key,
                sample_id=str(sample.id),
                model_name=model_name,
                agent_mode=agent_mode,
                base_class=base_class,
                ignore_errors=ignore_errors,
                input_text=sample.input_text
            )
            if cached_result:
                cached += 1
            else:
                uncached += 1
                
        return {"uncached": uncached, "cached": cached, "total": len(samples)}

    def evaluate(self, 
                agent: BaseAgent, 
                dataset: BaseDataset, 
                task_name: str,
                version: str = "1.0",
                max_workers: int = 5,
                max_retries: int = 3,
                show_progress: bool = False,
                seed: int = 42,
                exit_on_first_error: bool = False,
                partial_cb = None,
                error_cb = None,
                **kwargs) -> EvaluationResult:
        """
        Evaluate agent on classification task with parallel execution and retries.
        
        Calculates accuracy, precision, recall, and F1 score.
        """
        self.validate_inputs(agent, dataset)
        
        if not agent._is_initialized:
            agent.initialize()
        
        start_time = time.time()
        
        # Collect predictions (deterministic order: same as dataset iteration order)
        samples = list(dataset)
        
        agent_mode = kwargs.get("agent_mode", "")
        base_class = kwargs.get("base_class", "")
        model_name = getattr(agent, "model_name", "")
        ignore_errors = kwargs.get("ignore_errors", False)
        
        # Preload legacy run caching for runs_only and both modes
        self.global_cache.preload_legacy_cache(
            model_name=model_name,
            agent_mode=agent_mode,
            base_class=base_class,
            ignore_errors=ignore_errors
        )
        
        uncached_samples = []
        cache_hits = 0
        per_sample_by_id = {}

        for sample in samples:
            cache_key = self.global_cache.generate_key(
                model_name=model_name,
                payload=sample.input_text,
                params={"agent_mode": agent_mode, "base_class": base_class}
            )
            cached_result = self.global_cache.lookup(
                cache_key=cache_key,
                sample_id=str(sample.id),
                model_name=model_name,
                agent_mode=agent_mode,
                base_class=base_class,
                ignore_errors=ignore_errors,
                input_text=sample.input_text
            )
            if cached_result:
                is_correct = cached_result.get("prediction") == sample.target_output
                sample_metric = {
                    "sample_id": sample.id,
                    "input_text": sample.input_text,
                    "accuracy": 1.0 if is_correct else 0.0,
                    "prediction": cached_result.get("prediction", ""),
                    "ground_truth": sample.target_output,
                    "from_cache": True,
                    "cache_key": cache_key
                }
                if "prediction_confidence" in cached_result:
                    sample_metric["prediction_confidence"] = cached_result["prediction_confidence"]
                if "class_scores" in cached_result:
                    sample_metric["class_scores"] = cached_result["class_scores"]
                if "ttft" in cached_result:
                    sample_metric["ttft"] = cached_result["ttft"]
                if "total_time" in cached_result:
                    sample_metric["total_time"] = cached_result["total_time"]
                if "generation_time" in cached_result:
                    sample_metric["generation_time"] = cached_result["generation_time"]

                per_sample_by_id[str(sample.id)] = sample_metric
                cache_hits += 1
                if partial_cb:
                    partial_cb(sample_metric)
            else:
                uncached_samples.append(sample)
        
        tasks = uncached_samples
        requested_workers = max(1, int(max_workers))
        cpu_count = os.cpu_count() or 1
        planned_parallelism = min(requested_workers, len(tasks)) if tasks else 0

        print(
            "[Concurrency Plan] "
            f"tasks={len(tasks)}, requested_max_workers={requested_workers}, "
            f"planned_parallelism={planned_parallelism}, local_cpu_count={cpu_count}"
        )
        execution_failures = []
        retry_backoff_failures = 0
        rate_limit_failures = 0
        rate_limit_sleep_events = 0
        
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=requested_workers) as executor:
            effective_executor_workers = getattr(executor, "_max_workers", requested_workers)
            print(
                "[Concurrency Executor] "
                f"executor_max_workers={effective_executor_workers}, submitted_tasks={len(tasks)}"
            )
            futures = {
                executor.submit(self._process_sample, task, agent, max_retries, error_cb): task 
                for task in tasks
            }
            
            def timeout_as_completed(fs, idle_timeout=300.0):
                not_done = set(fs.keys())
                while not_done:
                    done, not_done = concurrent.futures.wait(not_done, timeout=idle_timeout, return_when=concurrent.futures.FIRST_COMPLETED)
                    if not done:
                        import sys
                        print(f"\n[Fatal Timeout] {len(not_done)} tasks hung for over {idle_timeout}s! Killing them.", file=sys.stderr)
                        class MockTimeoutFuture:
                            def __init__(self, task):
                                self._task = task
                            def result(self):
                                return self._task, None, f"Fatal timeout: Task hung and did not complete within {idle_timeout}s"
                        for f in not_done:
                            f.cancel()
                            yield MockTimeoutFuture(fs[f])
                        break
                    for f in done:
                        yield f

            iterator = timeout_as_completed(futures, idle_timeout=300.0)
            if show_progress:
                try:
                    from tqdm import tqdm
                    import sys
                    iterator = tqdm(iterator, total=len(futures), file=sys.stdout, desc="Evaluating")
                except ImportError:
                    pass
            
            completed_count = 0
            for future in iterator:
                sample, response, error = future.result()
                completed_count += 1
                
                if error:
                    print(
                        "[Sample Failure] "
                        f"sample_id={sample.id}, error={error}"
                    )
                    err_lower = str(error).lower()
                    if "rate limit" in err_lower or "30 rate limits" in err_lower or "429" in err_lower or "empty output" in err_lower:
                        rate_limit_failures += 1
                    else:
                        retry_backoff_failures += 1
                    if "30 rate limits" in err_lower or "empty output" in err_lower:
                        rate_limit_sleep_events += 30
                        
                    execution_failures.append({
                        "sample_id": sample.id,
                        "input_text": sample.input_text,
                        "error": error
                    })
                    err_metric = {
                        "sample_id": sample.id,
                        "input_text": sample.input_text,
                        "accuracy": 0.0,
                        "prediction": "Error",
                        "ground_truth": sample.target_output,
                        "error": error
                    }
                    per_sample_by_id[str(sample.id)] = err_metric
                    if partial_cb:
                        partial_cb(err_metric)
                else:
                    assert response is not None
                    
                    is_correct = response.output == sample.target_output
                    sample_metric = {
                        "sample_id": sample.id,
                        "input_text": sample.input_text,
                        "accuracy": 1.0 if is_correct else 0.0,
                        "prediction": response.output,
                        "ground_truth": sample.target_output
                    }
                    if response.confidence is not None:
                        sample_metric["prediction_confidence"] = response.confidence
                    if response.metadata:
                        sample_metric.update(response.metadata)
                    per_sample_by_id[str(sample.id)] = sample_metric
                    
                    # Write back to GlobalCache
                    cache_key = self.global_cache.generate_key(
                        model_name=model_name,
                        payload=sample.input_text,
                        params={"agent_mode": agent_mode, "base_class": base_class}
                    )
                    self.global_cache.set(cache_key, sample_metric)
                    
                    if partial_cb:
                        partial_cb(sample_metric)

        if rate_limit_failures or retry_backoff_failures:
            print(
                "[Concurrency Diagnostics] "
                f"non_rate_limit_failures={retry_backoff_failures}, "
                f"rate_limit_failures={rate_limit_failures}, "
                f"estimated_rate_limit_sleep_events={rate_limit_sleep_events}"
            )

        # Emit ordered per-sample metrics in the same order as the dataset.
        per_sample_metrics = [per_sample_by_id[str(s.id)] for s in samples]
        predictions = [m["prediction"] for m in per_sample_metrics]
        ground_truth = [m["ground_truth"] for m in per_sample_metrics]
        
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
