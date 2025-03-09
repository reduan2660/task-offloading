# scripts/evaluate_models.py
import os
import argparse
import json
import pandas as pd
import numpy as np
from src.utils.data_utils import load_scenario_data, split_scenario_data
from src.models.paper1_model import LyapunovBasedTaskOffloadingModel
import time

def evaluate_model_on_scenario(model, scenario_data, scenario_id):
    """
    Evaluate a model on a single scenario
    
    Args:
        model: Trained model to evaluate
        scenario_data: Dictionary containing scenario data
        scenario_id: ID of the scenario
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Start timing
    start_time = time.time()
    
    # Evaluate model
    metrics = model.evaluate(scenario_data)
    
    # End timing
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # Add scenario ID and evaluation time to metrics
    metrics['scenario_id'] = scenario_id
    metrics['model_name'] = model.name
    metrics['evaluation_time'] = evaluation_time
    
    return metrics

def main(scenarios_dir, output_dir, v_param=0.5):
    """
    Train and evaluate the Lyapunov-based model on scenarios
    
    Args:
        scenarios_dir: Directory containing scenario data
        output_dir: Directory to save results
        v_param: Lyapunov control parameter (trade-off between energy and queue stability)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scenario directories
    scenario_paths = [os.path.join(scenarios_dir, d) for d in os.listdir(scenarios_dir)
                     if os.path.isdir(os.path.join(scenarios_dir, d))]
    
    # Initialize results storage
    all_metrics = []
    
    # Process each scenario
    for scenario_path in scenario_paths:
        scenario_id = os.path.basename(scenario_path)
        print(f"Processing scenario: {scenario_id}")
        
        try:
            # Load scenario data
            scenario_data = load_scenario_data(scenario_path)
            
            # Split into training and testing data
            train_data, test_data = split_scenario_data(scenario_data)
            
            # Initialize model
            model = LyapunovBasedTaskOffloadingModel(V_param=v_param)
            
            # Train model
            print(f"Training model on scenario: {scenario_id}")
            model.train(train_data)
            
            # Evaluate model
            print(f"Evaluating model on scenario: {scenario_id}")
            metrics = evaluate_model_on_scenario(model, test_data, scenario_id)
            
            # Store metrics
            all_metrics.append(metrics)
            
            print(f"Results for scenario {scenario_id}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"Error processing scenario {scenario_id}: {e}")
    
    # Save all metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = os.path.join(output_dir, "paper1_model_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"Evaluation complete. Results saved to {metrics_file}")
    
    # Calculate and display average metrics
    if all_metrics:
        print("\nAverage metrics across all scenarios:")
        for key in ['success_rate', 'avg_energy', 'avg_delay', 'evaluation_time']:
            avg_value = metrics_df[key].mean()
            print(f"  Average {key}: {avg_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Lyapunov-based task offloading model')
    parser.add_argument('--scenarios-dir', type=str, default='data/processed',
                        help='Directory containing processed scenario data')
    parser.add_argument('--output-dir', type=str, default='data/results',
                        help='Directory to save evaluation results')
    parser.add_argument('--v-param', type=float, default=0.5,
                        help='Lyapunov control parameter (trade-off between energy and queue stability)')
    
    args = parser.parse_args()
    main(args.scenarios_dir, args.output_dir, args.v_param)