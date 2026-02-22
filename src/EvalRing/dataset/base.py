"""
Base classes for datasets in the EvalRing framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from dataclasses import dataclass, field
import json
import pandas as pd
from pathlib import Path


@dataclass
class DataSample:
    """Represents a single data sample for evaluation."""
    id: str
    input_text: str
    target_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "input_text": self.input_text,
            "target_output": self.target_output,
            "metadata": self.metadata
        }


class BaseDataset(ABC):
    """
    Abstract base class for all datasets in EvalRing.
    
    This class provides a standardized interface for loading and accessing
    different types of datasets (JSON, CSV, DataFrame, etc.).
    """
    
    def __init__(self, 
                 name: str,
                 description: Optional[str] = None,
                 version: str = "1.0"):
        self.name = name
        self.description = description or f"Dataset: {name}"
        self.version = version
        self._samples: List[DataSample] = []
        self._metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def load_data(self, source: Union[str, Path, pd.DataFrame], **kwargs) -> None:
        """
        Load data from the specified source.
        
        Args:
            source: Data source (file path, URL, or DataFrame)
            **kwargs: Additional parameters specific to the dataset type
        """
        pass
    
    @abstractmethod
    def validate_data(self) -> bool:
        """
        Validate the loaded data for consistency and correctness.
        
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def add_sample(self, sample: DataSample) -> None:
        """Add a single sample to the dataset."""
        self._samples.append(sample)
    
    def get_sample(self, index: int) -> DataSample:
        """Get a sample by index."""
        return self._samples[index]
    
    def get_sample_by_id(self, sample_id: str) -> Optional[DataSample]:
        """Get a sample by its ID."""
        for sample in self._samples:
            if sample.id == sample_id:
                return sample
        return None
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._samples)
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over samples in the dataset."""
        return iter(self._samples)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[DataSample, List[DataSample]]:
        """Get sample(s) by index or slice."""
        return self._samples[index]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._samples:
            return {"total_samples": 0}
        
        # Basic statistics
        stats = {
            "total_samples": len(self._samples),
            "name": self.name,
            "version": self.version,
            "description": self.description
        }
        
        # Try to infer additional statistics
        try:
            input_lengths = [len(sample.input_text) for sample in self._samples]
            stats.update({
                "avg_input_length": sum(input_lengths) / len(input_lengths),
                "min_input_length": min(input_lengths),
                "max_input_length": max(input_lengths)
            })
        except Exception:
            pass
        
        return stats
    
    def split(self, train_ratio: float = 0.8, seed: Optional[int] = None) -> Tuple['BaseDataset', 'BaseDataset']:
        """
        Split the dataset into train and test sets.
        
        Args:
            train_ratio: Ratio of data to use for training
            seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        indices = list(range(len(self._samples)))
        random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_dataset = self.__class__(name=f"{self.name}_train", version=self.version)
        test_dataset = self.__class__(name=f"{self.name}_test", version=self.version)
        
        for idx in train_indices:
            train_dataset.add_sample(self._samples[idx])
        
        for idx in test_indices:
            test_dataset.add_sample(self._samples[idx])
        
        return train_dataset, test_dataset
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save dataset to file."""
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "metadata": self._metadata,
            "samples": [sample.to_dict() for sample in self._samples]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """Load dataset from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.name = data.get("name", self.name)
        self.description = data.get("description", self.description)
        self.version = data.get("version", self.version)
        self._metadata = data.get("metadata", {})
        
        self._samples = []
        for sample_data in data.get("samples", []):
            sample = DataSample(
                id=sample_data["id"],
                input_text=sample_data["input_text"],
                target_output=sample_data["target_output"],
                metadata=sample_data.get("metadata", {})
            )
            self._samples.append(sample)
