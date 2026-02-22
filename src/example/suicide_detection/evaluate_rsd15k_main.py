import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Add src directories to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    from EvalRing.dataset import CSVDataset
    from EvalRing.agent import MockAgent
    from EvalRing.evaluator import ClassificationEvaluator
    from llm_agent import OpenAISuicideDetectionAgent
except ImportError as e:
    print(f"Error importing EvalRing: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def run_main_eval():
    print("Running main evaluation framework on RSD_15K dataset (limited to 50 queries)...")
    
    # Paths
    base_dir = current_file.parent
    data_path = base_dir / "data" / "rsd_15k.csv"
    
    if not data_path.exists():
        print(f"Dataset file not found at: {data_path}")
        return

    # 1. Load Dataset
    print(f"Loading dataset from: {data_path}")
    dataset = CSVDataset(name="rsd_15k_sample")
    dataset.load_data(
        source=data_path,
        text_field="text",
        label_field="sentiment",
        id_field="users" # Assuming 'users' column can serve as ID, or use None for auto-generated
    )
    
    print(f"Total samples loaded: {len(dataset._samples)}")
    
    # Limit to 50 queries
    dataset._samples = dataset._samples[:50]
    print(f"Samples limited to: {len(dataset._samples)}")
    
    # 2. Setup Agent
    # We will try to use the full LLM agent if an OpenAI API key is available.
    # Otherwise, we fall back to the MockAgent.
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        print("OpenAI API key found. Using full LLM Agent.")
        agent = OpenAISuicideDetectionAgent(
            api_key=api_key
        )
        agent.initialize()
    else:
        print("No OPENAI_API_KEY found in environment. Falling back to MockAgent.")
        possible_outputs = ["Ideation", "Behavior", "Indicator", "Attempt"]
        agent = MockAgent(
            name="MockSuicideDetectionAgent",
            possible_outputs=possible_outputs
        )
        agent.initialize()
    
    # 3. Setup Evaluator
    evaluator = ClassificationEvaluator()
    
    # 4. Run Evaluation
    print("\nStarting evaluation (Parallel execution enabled)...")
    result = evaluator.evaluate(
        agent=agent,
        dataset=dataset,
        task_name="suicide_risk_classification",
        version="demo_v1",
        max_workers=5,
        max_retries=3,
        seed=42 # Example hardcoded param
    )
    
    # 5. Output Results
    print("\nEvaluation Results:")
    print(f"Agent: {result.agent_name}")
    print(f"Task: {result.task_name}")
    print(f"Duration: {result.duration:.4f} seconds")
    
    print("\nMetrics:")
    for metric_name, value in result.metrics.to_dict().items():
        if metric_name == "metadata": continue # Skip metadata printing if complex
        print(f"- {metric_name}: {value}")
        
    print("\nPer-sample Results:")
    all_cases = []
    failed_cases = []
    
    # Access per_sample_metrics directly from the metrics object
    per_sample_metrics = getattr(result.metrics, "per_sample_metrics", [])
    
    for i, sample_res in enumerate(per_sample_metrics):
         print(f"Sample {sample_res['sample_id']}: Pred={sample_res['prediction']}, GT={sample_res['ground_truth']} -> Acc={sample_res['accuracy']}")
         if sample_res['prediction'] == 'Error':
             print(f"  -> Error details: {sample_res.get('error', 'Unknown error')}")
         
         # Find the original text for the case
         # Since Per_sample_metrics is generated in the same order as dataset._samples,
         # we can directly access the corresponding sample by index.
         original_text = "Text not found"
         if i < len(dataset._samples):
             original_text = dataset._samples[i].input_text
                 
         case_data = {
             "sample_id": sample_res['sample_id'],
             "text": original_text,
             "prediction": sample_res['prediction'],
             "ground_truth": sample_res['ground_truth'],
             "accuracy": sample_res['accuracy'],
             "ttft": sample_res.get('ttft', 'N/A'),
             "tps": sample_res.get('tps', 'N/A'),
             "total_time": sample_res.get('total_time', 'N/A'),
             "generation_time": sample_res.get('generation_time', 'N/A'),
             "prompt_tokens": sample_res.get('prompt_tokens', 'N/A'),
             "completion_tokens": sample_res.get('completion_tokens', 'N/A'),
             "total_tokens": sample_res.get('total_tokens', 'N/A')
         }
         
         all_cases.append(case_data)
         
         # Check if accuracy is not 1.0 (meaning it failed)
         if float(sample_res['accuracy']) < 1.0:
             failed_cases.append(case_data)
             print(f"  -> Added to failed cases log.")
             
    # 6. Log results to a separate directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = current_file.parent / ".EvalRing" / f"run_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving reports to: {eval_dir}")
    
    # Write result.md
    result_md_path = eval_dir / "result.md"
    with open(result_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Results\n\n")
        f.write(f"**Agent:** {result.agent_name}\n")
        f.write(f"**Task:** {result.task_name}\n")
        f.write(f"**Duration:** {result.duration:.4f} seconds\n")
        f.write(f"**Samples:** {len(all_cases)}\n\n")
        
        # Calculate cost estimate (gpt-5.2 pricing estimate: $2/1M input, $8/1M output)
        metrics_dict = result.metrics.to_dict()['metrics']
        total_prompt = metrics_dict.get('total_prompt_tokens', 0)
        total_completion = metrics_dict.get('total_completion_tokens', 0)
        cost_usd = (total_prompt * 2.0 / 1_000_000) + (total_completion * 8.0 / 1_000_000)
        result.metrics.add_metric('total_cost_usd', cost_usd)
        metrics_dict = result.metrics.to_dict()['metrics']
        
        f.write("## Metrics\n")
        
        # Define display order for metrics
        priority_metrics = ["accuracy", "precision", "recall", "f1_score", "avg_ttft", "std_ttft", "avg_tps", "std_tps", "avg_generation_time", "avg_total_time", "total_prompt_tokens", "total_completion_tokens", "total_tokens", "avg_prompt_tokens", "avg_completion_tokens", "total_cost_usd"]
        
        # Write metrics in priority order first
        for key in priority_metrics:
            if key in metrics_dict:
                 f.write(f"- **{key}**: {metrics_dict[key]}\n")
        
        # Write remaining metrics
        for metric_name, value in metrics_dict.items():
            if metric_name not in priority_metrics:
                f.write(f"- **{metric_name}**: {value}\n")
            
    # Write all_cases.txt
    all_cases_path = eval_dir / "all_cases.txt"
    with open(all_cases_path, "w", encoding="utf-8") as f:
        f.write(f"All Cases Log - Agent: {result.agent_name} - Task: {result.task_name}\n")
        f.write("="*80 + "\n\n")
        for case in all_cases:
            status = "PASS" if float(case['accuracy']) == 1.0 else "FAIL"
            f.write(f"Sample ID: {case['sample_id']} [{status}]\n")
            f.write(f"Ground Truth: {case['ground_truth']}\n")
            f.write(f"Prediction:   {case['prediction']}\n")
            
            # Write detailed timing stats if available
            if case['ttft'] != 'N/A':
                 f.write(f"Stats: TTFT={case['ttft']:.4f}s | GenTime={case['generation_time']:.4f}s | TPS={case['tps']:.2f} | Total={case['total_time']:.4f}s\n")
            if case['prompt_tokens'] != 'N/A':
                 f.write(f"Tokens: Prompt={case['prompt_tokens']} | Completion={case['completion_tokens']} | Total={case['total_tokens']}\n")
            
            f.write(f"Text:\n{case['text']}\n")
            f.write("-" * 40 + "\n\n")
            
    # Write failed_cases.txt
    if failed_cases:
        failed_log_path = eval_dir / "failed_cases.txt"
        with open(failed_log_path, "w", encoding="utf-8") as f:
            f.write(f"Failed Cases Log - Agent: {result.agent_name} - Task: {result.task_name}\n")
            f.write("="*80 + "\n\n")
            for case in failed_cases:
                f.write(f"Sample ID: {case['sample_id']}\n")
                f.write(f"Ground Truth: {case['ground_truth']}\n")
                f.write(f"Prediction:   {case['prediction']}\n")
                f.write(f"Text:\n{case['text']}\n")
                f.write("-" * 40 + "\n\n")
        print(f"Logged {len(failed_cases)} failed cases.")
    else:
        print("No failed cases to log.")
        
    # Write Meta.json
    meta_json_path = eval_dir / "Meta.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(result.metadata, f, indent=4, default=str)
        
    print(f"Saved Meta.json with {len(result.metadata.get('execution_failures', []))} execution failures.")
    
    # Optional: Demonstrate retry if there are execution failures
    if result.metadata.get("execution_failures"):
        print("\n--- Retrying Execution Failures ---")
        retry_result = evaluator.retry_failed_cases(
            meta_file_path=meta_json_path,
            agent=agent,
            dataset=dataset
        )
        if retry_result:
            print(f"Retry completed. New Accuracy: {retry_result.metrics.get_metric('accuracy')}")

if __name__ == "__main__":
    run_main_eval()
