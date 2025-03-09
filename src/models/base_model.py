# src/models/base_model.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class BaseTaskOffloadingModel(ABC):
    """Abstract base class for task offloading models"""
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.trained = False
    
    @abstractmethod
    def train(self, training_data: Dict[str, Any], **kwargs) -> None:
        """Train the model using the provided data"""
        pass
    
    @abstractmethod
    def decide_offloading(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide offloading strategy based on current state"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        raise NotImplementedError("Save functionality not implemented for this model")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk"""
        raise NotImplementedError("Load functionality not implemented for this model")