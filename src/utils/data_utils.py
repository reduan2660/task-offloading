# src/utils/data_utils.py
import os
import pandas as pd
import json
import networkx as nx
from typing import Dict, List, Tuple, Any

def load_scenario_data(scenario_path: str) -> Dict[str, Any]:
    """
    Load all data files for a scenario
    
    Args:
        scenario_path: Path to the scenario directory
        
    Returns:
        Dictionary containing all scenario data
    """
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_path}")
    
    # Load metadata
    metadata_path = os.path.join(scenario_path, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load road network
    network_path = os.path.join(scenario_path, "network_road_network.json")
    with open(network_path, 'r') as f:
        network_data = json.load(f)
    road_network = nx.node_link_graph(network_data)
    
    # Load RSUs
    rsus_path = os.path.join(scenario_path, "network_rsus.csv")
    rsus = pd.read_csv(rsus_path)
    
    # Load vehicles
    vehicles_path = os.path.join(scenario_path, "network_vehicles.csv")
    vehicles = pd.read_csv(vehicles_path)
    
    # Load connections
    connections_path = os.path.join(scenario_path, "network_connections.json")
    with open(connections_path, 'r') as f:
        connections = json.load(f)
    
    # Load tasks
    tasks_path = os.path.join(scenario_path, "tasks.csv")
    tasks = pd.read_csv(tasks_path)
    
    # Load vehicle-task assignments
    vehicle_tasks_path = os.path.join(scenario_path, "vehicle_tasks.json")
    with open(vehicle_tasks_path, 'r') as f:
        vehicle_tasks = json.load(f)
    
    return {
        'metadata': metadata,
        'road_network': road_network,
        'rsus': rsus,
        'vehicles': vehicles,
        'connections': connections,
        'tasks': tasks,
        'vehicle_tasks': vehicle_tasks
    }

def load_multiple_scenarios(scenario_dir: str, pattern: str = 'scenario_*') -> List[Dict[str, Any]]:
    """
    Load multiple scenarios matching a pattern
    
    Args:
        scenario_dir: Directory containing scenario directories
        pattern: Pattern to match scenario directories
        
    Returns:
        List of scenario data dictionaries
    """
    import glob
    
    scenario_paths = glob.glob(os.path.join(scenario_dir, pattern))
    
    scenarios = []
    for path in scenario_paths:
        if os.path.isdir(path):
            try:
                scenario_data = load_scenario_data(path)
                scenarios.append(scenario_data)
            except Exception as e:
                print(f"Error loading scenario {path}: {e}")
    
    return scenarios

def split_scenario_data(scenario_data: Dict[str, Any], 
                      train_ratio: float = 0.7,
                      seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split scenario data into training and testing sets
    
    Args:
        scenario_data: Dictionary containing scenario data
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (training_data, testing_data)
    """
    import numpy as np
    
    # Create copies of data to avoid modifying original
    train_data = {k: v.copy() if hasattr(v, 'copy') else v for k, v in scenario_data.items()}
    test_data = {k: v.copy() if hasattr(v, 'copy') else v for k, v in scenario_data.items()}
    
    # Split vehicle-task assignments
    vehicle_tasks = scenario_data['vehicle_tasks']
    task_ids = []
    
    for v_tasks in vehicle_tasks.values():
        task_ids.extend(v_tasks)
    
    # Shuffle task IDs
    np.random.seed(seed)
    np.random.shuffle(task_ids)
    
    # Split tasks into train and test sets
    train_size = int(len(task_ids) * train_ratio)
    train_task_ids = set(task_ids[:train_size])
    test_task_ids = set(task_ids[train_size:])
    
    # Update vehicle_tasks for training and testing
    train_vehicle_tasks = {}
    test_vehicle_tasks = {}
    
    for vehicle_id, v_tasks in vehicle_tasks.items():
        train_vehicle_tasks[vehicle_id] = [t for t in v_tasks if t in train_task_ids]
        test_vehicle_tasks[vehicle_id] = [t for t in v_tasks if t in test_task_ids]
    
    train_data['vehicle_tasks'] = train_vehicle_tasks
    test_data['vehicle_tasks'] = test_vehicle_tasks
    
    return train_data, test_data