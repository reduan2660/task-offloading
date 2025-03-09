# scripts/generate_data.py

import os
import argparse
import yaml
from src.data_generation.task_generator import TaskGenerator
from src.data_generation.network_generator import NetworkGenerator
from src.data_generation.scenario_builder import ScenarioBuilder

def main(config_path, output_dir, seed=None):
    """Generate task offloading data based on configuration"""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize generators
    task_gen = TaskGenerator(seed=seed)
    network_gen = NetworkGenerator(seed=seed)
    scenario_builder = ScenarioBuilder(seed=seed)
    
    # Generate standalone tasks
    print("Generating standalone tasks...")
    for task_config in config.get('tasks', []):
        tasks = task_gen.generate_mixed_tasks(
            n_small=task_config.get('n_small', 10),
            n_medium=task_config.get('n_medium', 5),
            n_large=task_config.get('n_large', 2)
        )
        task_gen.save_tasks(tasks, os.path.join(output_dir, 'raw', task_config.get('filename', 'tasks.csv')))
    
    # Generate dependent tasks
    print("Generating dependent tasks...")
    for dep_task_config in config.get('dependent_tasks', []):
        tasks = task_gen.generate_dependent_tasks(
            n_tasks=dep_task_config.get('n_tasks', 15),
            max_depth=dep_task_config.get('max_depth', 3),
            branching_factor=dep_task_config.get('branching_factor', 2)
        )
        task_gen.save_tasks(tasks, os.path.join(output_dir, 'raw', dep_task_config.get('filename', 'dependent_tasks.csv')))
    
    # Generate networks
    print("Generating networks...")
    for network_config in config.get('networks', []):
        if network_config.get('type', 'grid') == 'grid':
            road_network, positions = network_gen.generate_grid_road_network(
                rows=network_config.get('rows', 5),
                cols=network_config.get('cols', 5),
                cell_size=network_config.get('cell_size', 200.0)
            )
        else:
            road_network, positions = network_gen.generate_random_road_network(
                n_nodes=network_config.get('n_nodes', 25),
                area_size=network_config.get('area_size', 1000.0),
                connectivity=network_config.get('connectivity', 0.3)
            )
        
        rsus = network_gen.place_rsus(
            road_network, 
            n_rsus=network_config.get('n_rsus', 8)
        )
        
        vehicles = network_gen.generate_vehicle_paths(
            road_network, 
            n_vehicles=network_config.get('n_vehicles', 15)
        )
        
        connections = network_gen.get_vehicle_rsu_connections(vehicles, rsus)
        
        network_gen.save_network(
            road_network, rsus, vehicles, connections, 
            os.path.join(output_dir, 'raw', network_config.get('filename_prefix', 'network'))
        )
    
    # Generate complete scenarios
    print("Generating scenarios...")
    scenario_ids = scenario_builder.generate_multi_scenario_dataset(
        config.get('scenarios', []),
        output_dir=os.path.join(output_dir, 'processed')
    )
    
    print(f"Generated {len(scenario_ids)} scenarios")
    
    # Generate time-evolved scenarios
    print("Generating time-evolved scenarios...")
    for evolution_config in config.get('time_evolution', []):
        base_scenario = evolution_config.get('base_scenario')
        if not base_scenario and scenario_ids:
            base_scenario = scenario_ids[0]
        
        if base_scenario:
            evolved_ids = scenario_builder.time_evolve_scenario(
                base_scenario,
                time_steps=evolution_config.get('time_steps', 10),
                dt=evolution_config.get('dt', 1.0),
                output_dir=os.path.join(output_dir, 'processed')
            )
            print(f"Generated {len(evolved_ids)} time-evolved scenarios")
    
    print("Data generation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate task offloading data')
    parser.add_argument('--config', type=str, default='config/simulation_params.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for generated data')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args.config, args.output_dir, args.seed)