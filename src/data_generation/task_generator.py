# src/data_generation/task_generator.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Task:
    """Class representing a computational task in vehicular networks"""
    task_id: int
    size_mb: float      # Size in MB
    cpu_cycles: int     # Required CPU cycles
    deadline_ms: int    # Deadline in milliseconds
    data_in_size: float # Input data size
    data_out_size: float # Output data size
    divisible: bool     # Whether task can be divided
    priority: int       # Task priority (1-10)
    dependencies: List[int] = None  # IDs of tasks this one depends on

class TaskGenerator:
    """Generator for computational tasks with different characteristics"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the task generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        
    def generate_small_task(self) -> Task:
        """Generate a small task (0.1-1 MB)"""
        task_id = np.random.randint(10000)
        size = self.rng.uniform(0.1, 1.0)  # MB
        cycles = int(size * self.rng.uniform(100, 200) * 10**6)  # CPU cycles
        deadline = int(self.rng.uniform(100, 500))  # ms
        data_in = size
        data_out = size * self.rng.uniform(0.05, 0.2)  # Output is smaller than input
        divisible = self.rng.choice([True, False], p=[0.3, 0.7])
        priority = self.rng.randint(1, 5)  # Lower priority
        
        return Task(
            task_id=task_id,
            size_mb=size,
            cpu_cycles=cycles,
            deadline_ms=deadline,
            data_in_size=data_in,
            data_out_size=data_out,
            divisible=divisible,
            priority=priority
        )
    
    def generate_medium_task(self) -> Task:
        """Generate a medium task (1-5 MB)"""
        task_id = np.random.randint(10000)
        size = self.rng.uniform(1.0, 5.0)  # MB
        cycles = int(size * self.rng.uniform(150, 250) * 10**6)  # CPU cycles
        deadline = int(self.rng.uniform(300, 1000))  # ms
        data_in = size
        data_out = size * self.rng.uniform(0.1, 0.3)  # Output is smaller than input
        divisible = self.rng.choice([True, False], p=[0.6, 0.4])
        priority = self.rng.randint(3, 8)  # Medium priority
        
        return Task(
            task_id=task_id,
            size_mb=size,
            cpu_cycles=cycles,
            deadline_ms=deadline,
            data_in_size=data_in,
            data_out_size=data_out,
            divisible=divisible,
            priority=priority
        )
    
    def generate_large_task(self) -> Task:
        """Generate a large task (5-20 MB)"""
        task_id = np.random.randint(10000)
        size = self.rng.uniform(5.0, 20.0)  # MB
        cycles = int(size * self.rng.uniform(200, 300) * 10**6)  # CPU cycles
        deadline = int(self.rng.uniform(800, 2000))  # ms
        data_in = size
        data_out = size * self.rng.uniform(0.1, 0.3)  # Output is smaller than input
        divisible = self.rng.choice([True, False], p=[0.8, 0.2])
        priority = self.rng.randint(6, 10)  # Higher priority
        
        return Task(
            task_id=task_id,
            size_mb=size,
            cpu_cycles=cycles,
            deadline_ms=deadline,
            data_in_size=data_in,
            data_out_size=data_out,
            divisible=divisible,
            priority=priority
        )
    
    def generate_mixed_tasks(self, n_small: int, n_medium: int, n_large: int) -> List[Task]:
        """Generate a mixture of small, medium and large tasks
        
        Args:
            n_small: Number of small tasks
            n_medium: Number of medium tasks
            n_large: Number of large tasks
            
        Returns:
            List of Task objects
        """
        tasks = []
        for _ in range(n_small):
            tasks.append(self.generate_small_task())
            
        for _ in range(n_medium):
            tasks.append(self.generate_medium_task())
            
        for _ in range(n_large):
            tasks.append(self.generate_large_task())
            
        return tasks
    
    def generate_dependent_tasks(self, n_tasks: int, max_depth: int = 3, branching_factor: int = 2) -> List[Task]:
        """Generate a set of tasks with dependencies (forming a DAG)
        
        Args:
            n_tasks: Total number of tasks
            max_depth: Maximum depth of the dependency tree
            branching_factor: Maximum number of child tasks per parent
            
        Returns:
            List of Task objects with dependencies
        """
        tasks = []
        task_types = {
            'small': self.generate_small_task,
            'medium': self.generate_medium_task,
            'large': self.generate_large_task
        }
        
        # Create the tasks
        for i in range(n_tasks):
            task_type = self.rng.choice(['small', 'medium', 'large'])
            task = task_types[task_type]()
            task.task_id = i
            task.dependencies = []
            tasks.append(task)
        
        # Add dependencies (ensuring a DAG)
        for i in range(1, n_tasks):
            # Each task can depend on any previous task (to ensure DAG)
            if i > 0:
                n_deps = self.rng.randint(0, min(i, branching_factor))
                if n_deps > 0:
                    deps = self.rng.choice(range(i), size=n_deps, replace=False)
                    tasks[i].dependencies = deps.tolist()
        
        return tasks
    
    def tasks_to_dataframe(self, tasks: List[Task]) -> pd.DataFrame:
        """Convert list of tasks to a DataFrame for easier analysis"""
        data = []
        for task in tasks:
            task_dict = {
                'task_id': task.task_id,
                'size_mb': task.size_mb,
                'cpu_cycles': task.cpu_cycles,
                'deadline_ms': task.deadline_ms,
                'data_in_size': task.data_in_size,
                'data_out_size': task.data_out_size,
                'divisible': task.divisible,
                'priority': task.priority,
                'dependencies': str(task.dependencies) if task.dependencies else "[]"
            }
            data.append(task_dict)
        return pd.DataFrame(data)

    def save_tasks(self, tasks: List[Task], filepath: str) -> None:
        """Save tasks to a CSV file"""
        df = self.tasks_to_dataframe(tasks)
        df.to_csv(filepath, index=False)

# Example usage
if __name__ == "__main__":
    generator = TaskGenerator(seed=42)
    
    # Generate and save mixed tasks
    mixed_tasks = generator.generate_mixed_tasks(n_small=10, n_medium=5, n_large=2)
    generator.save_tasks(mixed_tasks, "data/raw/mixed_tasks.csv")
    
    # Generate and save dependent tasks
    dependent_tasks = generator.generate_dependent_tasks(n_tasks=15)
    generator.save_tasks(dependent_tasks, "data/raw/dependent_tasks.csv")