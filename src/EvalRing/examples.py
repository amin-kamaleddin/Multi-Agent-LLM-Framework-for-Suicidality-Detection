"""
Example usage of the EvalRing framework.

This module demonstrates how to use the base classes and implementations
for common evaluation tasks.
"""

import json
from pathlib import Path
from .base import BaseDataset, BaseAgent, AgentResponse, DataSample
from .implementations import create_dataset, create_agent, create_evaluator


def create_sample_dataset(filepath: str) -> None:
    """Create a sample JSON dataset for demonstration."""
    sample_data = [
        {
            "id": "sample_1",
            "input": "I love this product! It's amazing and works perfectly.",
            "target": "positive",
            "metadata": {"source": "product_review", "length": 58}
        },
        {
            "id": "sample_2", 
            "input": "This is terrible. I hate it and want my money back.",
            "target": "negative",
            "metadata": {"source": "product_review", "length": 54}
        },
        {
            "id": "sample_3",
            "input": "The item is okay, nothing special but does the job.",
            "target": "neutral",
            "metadata": {"source": "product_review", "length": 51}
        },
        {
            "id": "sample_4",
            "input": "Absolutely fantastic! Best purchase I've made all year.",
            "target": "positive",
            "metadata": {"source": "product_review", "length": 52}
        },
        {
            "id": "sample_5",
            "input": "Poor quality, broke after just one use. Very disappointed.",
            "target": "negative", 
            "metadata": {"source": "product_review", "length": 61}
        }
    ]
    
    with open(filepath, 'w') as f:
        json.dump(sample_data, f, indent=2)


def example_basic_evaluation():
    """Example of basic evaluation using the framework."""
    print("=== Basic Evaluation Example ===")
    
    # Create sample dataset
    dataset_path = "sample_dataset.json"
    create_sample_dataset(dataset_path)
    
    # Load dataset
    dataset = create_dataset(
        "json",
        name="sentiment_dataset",
        description="Sample sentiment analysis dataset"
    )
    dataset.load_data(dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Create a rule-based agent
    agent = create_agent(
        "rule_based",
        name="sentiment_agent",
        positive_keywords=["love", "amazing", "perfectly", "fantastic", "best"],
        negative_keywords=["terrible", "hate", "poor", "disappointed", "broke"],
        default_output="neutral"
    )
    agent.initialize()
    print("Created agent: {}".format(agent))
    
    # Create evaluator
    evaluator = create_evaluator(
        "classification",
        name="sentiment_evaluator",
        output_dir="./results"
    )
    print("Created evaluator: {}".format(evaluator))
    
    # Run evaluation
    result = evaluator.evaluate(
        agent=agent,
        dataset=dataset,
        task_name="sentiment_analysis_example",
        version="1.0"
    )
    
    # Display results
    print("\nEvaluation Results:")
    print("Agent: {}".format(result.agent_name))
    print("Dataset: {}".format(result.dataset_name))
    print("Task: {}".format(result.task_name))
    print("Duration: {:.2f} seconds".format(result.duration))
    print("\nMetrics:")
    for metric_name, metric_value in result.metrics.metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Clean up
    Path(dataset_path).unlink(missing_ok=True)


def example_custom_agent():
    """Example of creating a custom agent by inheriting from BaseAgent."""
    print("\n=== Custom Agent Example ===")
    
    class CustomSentimentAgent(BaseAgent):
        """Custom sentiment analysis agent using simple heuristics."""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.positive_words = {"good", "great", "excellent", "amazing", "love", "wonderful"}
            self.negative_words = {"bad", "terrible", "awful", "hate", "poor", "disappointing"}
            
        def initialize(self, **kwargs) -> None:
            """Initialize the custom agent."""
            self._is_initialized = True
            print(f"Custom agent {self.name} initialized")
            
        def predict(self, input_text: str, **kwargs):
            """Make prediction using custom logic."""
            if not self._is_initialized:
                raise RuntimeError("Agent not initialized")
                
            if not self.validate_input(input_text):
                return AgentResponse(
                    input_id="",
                    input_text=input_text,
                    output=None,
                    error="Invalid input"
                )
            
            words = set(input_text.lower().split())
            positive_score = len(words & self.positive_words)
            negative_score = len(words & self.negative_words)
            
            if positive_score > negative_score:
                output = "positive"
                confidence = min(1.0, positive_score / 5.0)
            elif negative_score > positive_score:
                output = "negative"
                confidence = min(1.0, negative_score / 5.0)
            else:
                output = "neutral"
                confidence = 0.5
                
            return AgentResponse(
                input_id="",
                input_text=input_text,
                output=output,
                confidence=confidence,
                metadata={"positive_score": positive_score, "negative_score": negative_score}
            )
    
    # Create and test custom agent
    custom_agent = CustomSentimentAgent(name="custom_sentiment_agent")
    custom_agent.initialize()
    
    # Test on sample inputs
    test_inputs = [
        "This product is absolutely amazing!",
        "I hate this terrible service",
        "It's okay, nothing special"
    ]
    
    print("Testing custom agent:")
    for i, input_text in enumerate(test_inputs):
        response = custom_agent.predict(input_text)
        response.input_id = f"test_{i}"
        print(f"  Input: {input_text}")
        print(f"  Output: {response.output} (confidence: {response.confidence:.2f})")
        print()


def example_custom_dataset():
    """Example of creating a custom dataset by inheriting from BaseDataset."""
    print("=== Custom Dataset Example ===")
    
    class CustomTextDataset(BaseDataset):
        """Custom dataset that generates synthetic text data."""
        
        def load_data(self, source, num_samples=100, **kwargs):
            """Generate synthetic data."""
            import random
            
            positive_templates = [
                "I love {} because it's {}",
                "{} is absolutely {}",
                "This {} makes me feel {}"
            ]
            
            negative_templates = [
                "I hate {} because it's {}",
                "{} is terribly {}",
                "This {} makes me feel {}"
            ]
            
            positive_words = ["amazing", "wonderful", "fantastic", "great", "excellent"]
            negative_words = ["terrible", "awful", "horrible", "bad", "disappointing"]
            objects = ["the product", "the service", "this experience", "the quality", "the support"]
            
            self._samples = []
            
            for i in range(num_samples):
                if random.random() < 0.5:  # 50% positive
                    template = random.choice(positive_templates)
                    obj = random.choice(objects)
                    adj = random.choice(positive_words)
                    text = template.format(obj, adj)
                    target = "positive"
                else:  # 50% negative
                    template = random.choice(negative_templates)
                    obj = random.choice(objects)
                    adj = random.choice(negative_words)
                    text = template.format(obj, adj)
                    target = "negative"
                
                sample = DataSample(
                    id=f"synthetic_{i}",
                    input_text=text,
                    target_output=target,
                    metadata={"synthetic": True, "template": template}
                )
                self._samples.append(sample)
        
        def validate_data(self) -> bool:
            """Validate synthetic data."""
            return len(self._samples) > 0
    
    # Create custom dataset
    custom_dataset = CustomTextDataset(
        name="synthetic_sentiment",
        description="Synthetically generated sentiment dataset"
    )
    custom_dataset.load_data(None, num_samples=20)
    
    print(f"Created custom dataset with {len(custom_dataset)} samples")
    print("Sample data:")
    for i in range(min(3, len(custom_dataset))):
        sample = custom_dataset[i]
        print(f"  {sample.id}: {sample.input_text[:50]}... -> {sample.target_output}")


def example_batch_evaluation():
    """Example of evaluating multiple agents on the same dataset."""
    print("\n=== Batch Evaluation Example ===")
    
    # Create sample dataset
    dataset_path = "batch_dataset.json"
    create_sample_dataset(dataset_path)
    
    # Load dataset
    dataset = create_dataset("json", name="batch_dataset")
    dataset.load_data(dataset_path)
    
    # Create multiple agents
    agents = [
        create_agent("mock", name="mock_agent_1", possible_outputs=["positive", "negative", "neutral"]),
        create_agent("rule_based", name="rule_agent_1", 
                    positive_keywords=["love", "amazing"], 
                    negative_keywords=["hate", "terrible"]),
        create_agent("rule_based", name="rule_agent_2",
                    positive_keywords=["fantastic", "best"],
                    negative_keywords=["poor", "disappointed"])
    ]
    
    # Initialize agents
    for agent in agents:
        agent.initialize()
    
    # Create evaluator
    evaluator = create_evaluator("classification", output_dir="./batch_results")
    
    # Evaluate all agents
    results = []
    for agent in agents:
        result = evaluator.evaluate(
            agent=agent,
            dataset=dataset,
            task_name="batch_sentiment_analysis",
            version="1.0"
        )
        results.append(result)
    
    # Compare results
    print("\nBatch Evaluation Results:")
    print(f"{'Agent':<20} {'Accuracy':<10} {'F1 Score':<10} {'Duration':<10}")
    print("-" * 55)
    
    for result in results:
        accuracy = result.metrics.get_metric("accuracy", 0)
        f1_score = result.metrics.get_metric("f1_score", 0)
        print(f"{result.agent_name:<20} {accuracy:<10.4f} {f1_score:<10.4f} {result.duration:<10.2f}")
    
    # Clean up
    Path(dataset_path).unlink(missing_ok=True)


if __name__ == "__main__":
    """Run all examples when the module is executed directly."""
    example_basic_evaluation()
    example_custom_agent()
    example_custom_dataset()
    example_batch_evaluation()
    print("\n=== All examples completed successfully! ===")
