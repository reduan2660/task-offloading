# src/data_generation/scenario_builder.py

import pandas as pd
import numpy as np
import json
import os
import networkx as nx
from typing import Dict, List, Tuple, Optional
from .task_generator import Task, TaskGenerator
from .network_generator import RSU, Vehicle, NetworkGenerator

class ScenarioBuilder:
    """Build complete task offloading scenarios by combining tasks and networks"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the scenario builder
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.task_generator = TaskGenerator(seed=seed)
        self.network_generator = NetworkGenerator(seed=seed)
    
    def load_network(self, filepath_prefix: str) -> Tuple[nx.Graph, List[RSU], List[Vehicle], Dict]:
        """Load a previously saved network
        
        Args:
            filepath_prefix: Prefix for the input filepaths
            
        Returns:
            Tuple of (road_network, rsus, vehicles, connections)
        """
        # Load road network
        road_network = nx.read_graphml(f"{filepath_prefix}_road_network.graphml")
        
        # Load RSUs
        rsu_df = pd.read_csv(f"{filepath_prefix}_rsus.csv")
        rsus = []
        for _, row in rsu_df.iterrows():
            rsu = RSU(
                rsu_id=row['rsu_id'],
                location=(row['location_x'], row['location_y']),
                coverage_radius=row['coverage_radius'],
                processing_power=row['processing_power'],
                bandwidth=row['bandwidth'],
                memory=row['memory'],
                queue_length=row['queue_length']
            )
            rsus.append(rsu)
        
        # Load vehicles
        vehicle_df = pd.read_csv(f"{filepath_prefix}_vehicles.csv")
        vehicles = []
        for _, row in vehicle_df.iterrows():
            # Parse path from string
            path_str = row['path'].replace('[', '').replace(']', '').split('),')
            path = []
            for p in path_str:
                coords = p.replace('(', '').replace(')', '').split(',')
                if len(coords) >= 2:
                    path.append((float(coords[0]), float(coords[1])))
            
            vehicle = Vehicle(
                vehicle_id=row['vehicle_id'],
                initial_location=(row['initial_location_x'], row['initial_location_y']),
                path=path,
                speed=row['speed'],
                computing_power=row['computing_power'],
                current_location=(row['current_location_x'], row['current_location_y']) 
                    if not pd.isna(row['current_location_x']) else None
            )
            vehicles.append(vehicle)
        
        # Load connections
        with open(f"{filepath_prefix}_connections.json", 'r') as f:
            connections = json.load(f)
            # Convert string keys back to integers
            connections = {int(k): v for k, v in connections.items()}
        
        return road_network, rsus, vehicles, connections
    
    def load_tasks(self, filepath: str) -> List[Task]:
        """Load tasks from a CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            List of Task objects
        """
        df = pd.read_csv(filepath)
        tasks = []
        
        for _, row in df.iterrows():
            # Parse dependencies from string
            deps_str = row['dependencies']
            if deps_str == '[]' or pd.isna(deps_str):
                dependencies = None
            else:
                dependencies = [int(x) for x in deps_str.strip('[]').split(',') if x.strip()]
            
            task = Task(
                task_id=row['task_id'],
                size_mb=row['size_mb'],
                cpu_cycles=row['cpu_cycles'],
                deadline_ms=row['deadline_ms'],
                data_in_size=row['data_in_size'],
                data_out_size=row['data_out_size'],
                divisible=bool(row['divisible']),
                priority=row['priority'],
                dependencies=dependencies
            )
            tasks.append(task)
        
        return tasks
    
    def assign_tasks_to_vehicles(self, tasks: List[Task], vehicles: List[Vehicle], 
                              task_distribution: Optional[str] = 'random') -> Dict[int, List[int]]:
        """Assign tasks to vehicles
        
        Args:
            tasks: List of Task objects
            vehicles: List of Vehicle objects
            task_distribution: How to distribute tasks ('random', 'proportional', 'capability')
            
        Returns:
            Dictionary mapping vehicle IDs to lists of task IDs
        """
        vehicle_tasks = {v.vehicle_id: [] for v in vehicles}
        
        if task_distribution == 'random':
            # Randomly assign tasks to vehicles
            for task in tasks:
                vehicle_id = self.rng.choice([v.vehicle_id for v in vehicles])
                vehicle_tasks[vehicle_id].append(task.task_id)
                
        elif task_distribution == 'proportional':
            # Assign more tasks to vehicles with higher computational power
            weights = np.array([v.computing_power for v in vehicles])
            weights = weights / np.sum(weights)
            
            for task in tasks:
                vehicle_id = self.rng.choice([v.vehicle_id for v in vehicles], p=weights)
                vehicle_tasks[vehicle_id].append(task.task_id)
                
        elif task_distribution == 'capability':
            # Assign larger tasks to vehicles with higher computational power
            sorted_vehicles = sorted(vehicles, key=lambda v: v.computing_power)
            sorted_tasks = sorted(tasks, key=lambda t: t.cpu_cycles)
            
            # Divide tasks into segments based on vehicle count
            n_vehicles = len(vehicles)
            segments = np.array_split(sorted_tasks, n_vehicles)
            
            # Assign task segments to vehicles (smallest to largest)
            for i, vehicle in enumerate(sorted_vehicles):
                for task in segments[i]:
                    vehicle_tasks[vehicle.vehicle_id].append(task.task_id)
        
        return vehicle_tasks
    
    def build_scenario(self, n_vehicles: int, n_rsus: int, 
                     n_small_tasks: int, n_medium_tasks: int, n_large_tasks: int,
                     area_size: float = 1000.0, 
                     task_distribution: str = 'random',
                     output_dir: str = 'data/processed/') -> str:
        """Build a complete task offloading scenario
        
        Args:
            n_vehicles: Number of vehicles
            n_rsus: Number of RSUs
            n_small_tasks: Number of small tasks
            n_medium_tasks: Number of medium tasks
            n_large_tasks: Number of large tasks
            area_size: Size of the area in meters
            task_distribution: How to distribute tasks among vehicles
            output_dir: Directory to save scenario files
            
        Returns:
            Scenario ID string
        """
        # Create scenario ID
        scenario_id = f"scenario_{n_vehicles}v_{n_rsus}r_{n_small_tasks+n_medium_tasks+n_large_tasks}t"
        scenario_path = os.path.join(output_dir, scenario_id)
        os.makedirs(scenario_path, exist_ok=True)
        
        # Generate road network
        road_network, positions = self.network_generator.generate_random_road_network(
            n_nodes=max(20, n_rsus * 3), area_size=area_size)
        
        # Place RSUs
        rsus = self.network_generator.place_rsus(road_network, n_rsus=n_rsus)
        
        # Generate vehicles with paths
        vehicles = self.network_generator.generate_vehicle_paths(
            road_network, n_vehicles=n_vehicles)
        
        # Generate tasks
        tasks = self.task_generator.generate_mixed_tasks(
            n_small=n_small_tasks, n_medium=n_medium_tasks, n_large=n_large_tasks)
        
        # Get vehicle-RSU connections
        connections = self.network_generator.get_vehicle_rsu_connections(vehicles, rsus)
        
        # Assign tasks to vehicles
        vehicle_tasks = self.assign_tasks_to_vehicles(tasks, vehicles, task_distribution)
        
        # Save all components
        self.network_generator.save_network(
            road_network, rsus, vehicles, connections, 
            os.path.join(scenario_path, "network"))
        
        self.task_generator.save_tasks(
            tasks, os.path.join(scenario_path, "tasks.csv"))
        
        # Save vehicle-task assignments
        with open(os.path.join(scenario_path, "vehicle_tasks.json"), 'w') as f:
            json.dump(vehicle_tasks, f)
        
        # Create scenario metadata
        metadata = {
            "scenario_id": scenario_id,
            "n_vehicles": n_vehicles,
            "n_rsus": n_rsus,
            "n_tasks": len(tasks),
            "task_distribution": task_distribution,
            "area_size": area_size,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(scenario_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        return scenario_id
    
    def generate_multi_scenario_dataset(self, 
                                      scenario_configs: List[Dict],
                                      output_dir: str = 'data/processed/') -> List[str]:
        """Generate multiple scenarios based on configurations
        
        Args:
            scenario_configs: List of scenario configuration dictionaries
            output_dir: Directory to save scenario files
            
        Returns:
            List of scenario IDs
        """
        scenario_ids = []
        
        for config in scenario_configs:
            scenario_id = self.build_scenario(
                n_vehicles=config.get('n_vehicles', 10),
                n_rsus=config.get('n_rsus', 5),
                n_small_tasks=config.get('n_small_tasks', 10),
                n_medium_tasks=config.get('n_medium_tasks', 5),
                n_large_tasks=config.get('n_large_tasks', 2),
                area_size=config.get('area_size', 1000.0),
                task_distribution=config.get('task_distribution', 'random'),
                output_dir=output_dir
            )
            scenario_ids.append(scenario_id)
        
        return scenario_ids
    
    def time_evolve_scenario(self, scenario_id: str, time_steps: int, 
                           dt: float = 1.0, output_dir: str = 'data/processed/') -> List[str]:
        """Evolve a scenario over time to simulate vehicle movement
        
        Args:
            scenario_id: ID of the scenario to evolve
            time_steps: Number of time steps to evolve
            dt: Time step duration in seconds
            output_dir: Directory to save evolved scenario files
            
        Returns:
            List of evolved scenario IDs
        """
        scenario_path = os.path.join(output_dir, scenario_id)
        
        # Load original scenario
        road_network, rsus, vehicles, connections = self.load_network(
            os.path.join(scenario_path, "network"))
        
        evolved_scenario_ids = []
        
        for t in range(1, time_steps + 1):
            # Create evolved scenario ID
            evolved_id = f"{scenario_id}_t{t}"
            evolved_path = os.path.join(output_dir, evolved_id)
            os.makedirs(evolved_path, exist_ok=True)
            
            # Update vehicle positions
            for vehicle in vehicles:
                # Simple linear interpolation between waypoints
                current_waypoint_idx = 0
                distance_traveled = vehicle.speed * dt * t
                
                path = vehicle.path
                total_distance = 0
                
                # Find current position based on distance traveled
                for i in range(len(path) - 1):
                    segment_length = np.sqrt(
                        (path[i+1][0] - path[i][0])**2 + 
                        (path[i+1][1] - path[i][1])**2
                    )
                    
                    if total_distance + segment_length >= distance_traveled:
                        # Interpolate position within this segment
                        fraction = (distance_traveled - total_distance) / segment_length
                        x = path[i][0] + fraction * (path[i+1][0] - path[i][0])
                        y = path[i][1] + fraction * (path[i+1][1] - path[i][1])
                        vehicle.current_location = (x, y)
                        break
                    
                    total_distance += segment_length
                    current_waypoint_idx += 1
                
                # If vehicle reached end of path, keep it at the last waypoint
                if vehicle.current_location is None:
                    vehicle.current_location = path[-1]
            
            # Update connections based on new positions
            connections = self.network_generator.get_vehicle_rsu_connections(vehicles, rsus)
            
            # Save evolved network
            self.network_generator.save_network(
                road_network, rsus, vehicles, connections, 
                os.path.join(evolved_path, "network"))
            
            # Copy tasks and vehicle-task assignments from original scenario
            import shutil
            shutil.copy(
                os.path.join(scenario_path, "tasks.csv"),
                os.path.join(evolved_path, "tasks.csv")
            )
            shutil.copy(
                os.path.join(scenario_path, "vehicle_tasks.json"),
                os.path.join(evolved_path, "vehicle_tasks.json")
            )
            
            # Create evolved scenario metadata
            with open(os.path.join(scenario_path, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            metadata.update({
                "scenario_id": evolved_id,
                "parent_scenario": scenario_id,
                "time_step": t,
                "dt": dt,
                "timestamp": pd.Timestamp.now().isoformat()
            })
            
            with open(os.path.join(evolved_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f)
            
            evolved_scenario_ids.append(evolved_id)
        
        return evolved_scenario_ids

# Example usage
if __name__ == "__main__":
    builder = ScenarioBuilder(seed=42)
    
    # Build a single scenario
    scenario_id = builder.build_scenario(
        n_vehicles=15,
        n_rsus=8,
        n_small_tasks=20,
        n_medium_tasks=10,
        n_large_tasks=5,
        task_distribution='proportional'
    )
    print(f"Created scenario: {scenario_id}")
    
    # Generate a time-evolved scenario
    evolved_ids = builder.time_evolve_scenario(scenario_id, time_steps=10)
    print(f"Created {len(evolved_ids)} evolved scenarios")
    
    # Generate multiple scenarios with different configurations
    configs = [
        {
            'n_vehicles': 10,
            'n_rsus': 5,
            'n_small_tasks': 15,
            'n_medium_tasks': 8,
            'n_large_tasks': 3,
            'task_distribution': 'random'
        },
        {
            'n_vehicles': 20,
            'n_rsus': 10,
            'n_small_tasks': 30,
            'n_medium_tasks': 15,
            'n_large_tasks': 5,
            'task_distribution': 'capability'
        },
        {
            'n_vehicles': 5,
            'n_rsus': 3,
            'n_small_tasks': 8,
            'n_medium_tasks': 4,
            'n_large_tasks': 2,
            'task_distribution': 'proportional'
        }
    ]
    
    scenario_ids = builder.generate_multi_scenario_dataset(configs)
    print(f"Created {len(scenario_ids)} scenarios")