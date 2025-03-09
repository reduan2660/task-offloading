# scripts/run_simulation.py
import os
import subprocess
import argparse
import time

def main(scenarios_dir, output_dir, v_param=0.5):
    """
    Run the full simulation pipeline
    
    Args:
        scenarios_dir: Directory containing scenario data
        output_dir: Directory to save results
        v_param: Lyapunov control parameter
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Run evaluation
    print("Evaluating Lyapunov-based model...")
    subprocess.run([
        "python", "scripts/evaluate_models.py",
        "--scenarios-dir", scenarios_dir,
        "--output-dir", output_dir,
        "--v-param", str(v_param)
    ])
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total simulation time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run task offloading simulation')
    parser.add_argument('--scenarios-dir', type=str, default='data/processed',
                        help='Directory containing processed scenario data')
    parser.add_argument('--output-dir', type=str, default='data/results',
                        help='Directory to save simulation results')
    parser.add_argument('--v-param', type=float, default=0.5,
                        help='Lyapunov control parameter')
    
    args = parser.parse_args()
    main(args.scenarios_dir, args.output_dir, args.v_param)