# src/models/paper1_model.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import math
from ..utils.data_utils import load_scenario_data
from .base_model import BaseTaskOffloadingModel

class LyapunovBasedTaskOffloadingModel(BaseTaskOffloadingModel):
    """
    Implementation of the Lyapunov-based task offloading model
    Based on the paper: "Task Offloading and Resource Allocation in Vehicular Networks: 
    A Lyapunov-Based Deep Reinforcement Learning Approach"
    """
    
    def __init__(self, V_param: float = 0.5, name: str = "LyapunovModel"):
        """
        Initialize the Lyapunov-based task offloading model
        
        Args:
            V_param: Lyapunov control parameter (trade-off between energy and queue stability)
            name: Model name
        """
        super().__init__(name=name)
        self.V_param = V_param
        self.queue_lengths = {}  # RSU queue lengths
        self.vehicle_queues = {} # Vehicle local queue lengths
        self.rsu_processing_rates = {}  # Processing rates for each RSU
        self.rsu_bandwidth = {}  # Bandwidth for each RSU
        
    def train(self, training_data: Dict[str, Any], **kwargs) -> None:
        """
        Train the model using the provided data.
        For this Lyapunov-based model, training involves setting the model parameters
        based on the network characteristics.
        
        Args:
            training_data: Dictionary containing network and task information
        """
        if 'rsus' not in training_data or 'vehicles' not in training_data:
            raise ValueError("Training data must contain RSU and vehicle information")
        
        rsus = training_data['rsus']
        vehicles = training_data['vehicles']
        
        # Initialize RSU processing rates and bandwidths
        for _, rsu in rsus.iterrows():
            self.rsu_processing_rates[rsu['rsu_id']] = rsu['processing_power']
            self.rsu_bandwidth[rsu['rsu_id']] = rsu['bandwidth']
            self.queue_lengths[rsu['rsu_id']] = 0
        
        # Initialize vehicle queues
        for _, vehicle in vehicles.iterrows():
            self.vehicle_queues[vehicle['vehicle_id']] = 0
        
        self.trained = True
    
    def _calculate_transmission_delay(self, data_size: float, bandwidth: float, 
                                     distance: float) -> float:
        """
        Calculate transmission delay based on Shannon's formula
        
        Args:
            data_size: Size of data to transmit in MB
            bandwidth: Channel bandwidth in Mbps
            distance: Distance between vehicle and RSU in meters
            
        Returns:
            Transmission delay in milliseconds
        """
        # Simple path loss model
        path_loss = 128.1 + 37.6 * math.log10(distance / 1000)
        signal_power = 23 - path_loss  # dBm
        noise_power = -174 + 10 * math.log10(bandwidth * 1e6)  # dBm
        snr = 10 ** ((signal_power - noise_power) / 10)
        
        # Shannon capacity
        capacity = bandwidth * math.log2(1 + snr)  # Mbps
        
        # Convert data size from MB to Mb
        data_size_mb = data_size * 8
        
        # Calculate delay in ms
        delay = (data_size_mb / capacity) * 1000
        
        return delay
    
    def _calculate_processing_delay(self, task_size: float, cpu_cycles: int, 
                                   processing_power: float) -> float:
        """
        Calculate processing delay
        
        Args:
            task_size: Size of task in MB
            cpu_cycles: Required CPU cycles per bit
            processing_power: Processing power in GIPS
            
        Returns:
            Processing delay in milliseconds
        """
        # Convert task size from MB to bits
        task_size_bits = task_size * 8 * 1024 * 1024
        
        # Total required cycles
        total_cycles = task_size_bits * cpu_cycles
        
        # Processing delay in ms
        delay = (total_cycles / (processing_power * 1e9)) * 1000
        
        return delay
    
    def _calculate_energy_consumption(self, task_size: float, cpu_cycles: int, 
                                     is_local: bool, transmit_power: float = 0.1,
                                     transmission_time: float = 0) -> float:
        """
        Calculate energy consumption
        
        Args:
            task_size: Size of task in MB
            cpu_cycles: Required CPU cycles per bit
            is_local: Whether task is executed locally
            transmit_power: Transmission power in Watts
            transmission_time: Transmission time in ms
            
        Returns:
            Energy consumption in Joules
        """
        if is_local:
            # Local execution energy model: E = k * f^2 * C
            # where k is energy coefficient, f is CPU frequency, C is cycles
            k = 1e-26  # Energy coefficient
            task_size_bits = task_size * 8 * 1024 * 1024
            total_cycles = task_size_bits * cpu_cycles
            
            # Assuming CPU frequency is 1 GHz for local execution
            f = 1e9
            energy = k * (f ** 2) * total_cycles
        else:
            # Transmission energy: P * t
            energy = transmit_power * (transmission_time / 1000)  # Convert ms to s
        
        return energy
    
    def _calculate_lyapunov_drift(self, queue_length: float) -> float:
        """
        Calculate Lyapunov drift
        
        Args:
            queue_length: Current queue length
            
        Returns:
            Lyapunov drift
        """
        return 0.5 * (queue_length ** 2)
    
    def decide_offloading(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide task offloading strategy based on current state
        
        Args:
            state: Current state including task, vehicle, and RSU information
            
        Returns:
            Decision dictionary including offloading decision and resource allocation
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before making decisions")
        
        # Extract state information
        task = state['task']
        vehicle = state['vehicle']
        vehicle_location = (vehicle['current_location_x'], vehicle['current_location_y'])
        available_rsus = state['available_rsus']
        connections = state['connections']
        
        task_id = task['task_id']
        task_size = task['size_mb']
        cpu_cycles = task['cpu_cycles']
        deadline = task['deadline_ms']
        
        vehicle_id = vehicle['vehicle_id']
        vehicle_queue = self.vehicle_queues.get(vehicle_id, 0)
        vehicle_processing_power = vehicle['computing_power']
        
        # Calculate local processing delay and energy
        local_processing_delay = self._calculate_processing_delay(
            task_size, cpu_cycles, vehicle_processing_power)
        local_energy = self._calculate_energy_consumption(
            task_size, cpu_cycles, is_local=True)
        
        # Initialize best strategy
        best_strategy = {
            'offload': False,
            'rsu_id': None,
            'expected_delay': local_processing_delay,
            'expected_energy': local_energy,
            'resource_allocation': 1.0  # Use 100% of local resources
        }
        
        # Calculate Lyapunov-based cost for local execution
        local_cost = self.V_param * local_energy + vehicle_queue * task_size
        best_cost = local_cost
        
        # Check each available RSU for potential offloading
        for _, rsu in available_rsus.iterrows():
            rsu_id = rsu['rsu_id']
            
            # Check if vehicle is connected to this RSU
            if vehicle_id not in connections or rsu_id not in connections[str(vehicle_id)]:
                continue
            
            rsu_location = (rsu['location_x'], rsu['location_y'])
            distance = math.sqrt(
                (vehicle_location[0] - rsu_location[0])**2 + 
                (vehicle_location[1] - rsu_location[1])**2)
            
            # Calculate transmission delay
            transmission_delay = self._calculate_transmission_delay(
                task_size, rsu['bandwidth'], distance)
            
            # Calculate processing delay at RSU
            rsu_processing_delay = self._calculate_processing_delay(
                task_size, cpu_cycles, rsu['processing_power'])
            
            # Total delay for offloading
            offload_delay = transmission_delay + rsu_processing_delay
            
            # Calculate energy consumption for transmission
            offload_energy = self._calculate_energy_consumption(
                task_size, cpu_cycles, is_local=False, 
                transmission_time=transmission_delay)
            
            # Get current RSU queue length
            rsu_queue = self.queue_lengths.get(rsu_id, 0)
            
            # Calculate Lyapunov-based cost for offloading
            offload_cost = self.V_param * offload_energy - vehicle_queue * task_size + rsu_queue * task_size
            
            # Check if offloading to this RSU is better
            if offload_cost < best_cost and offload_delay <= deadline:
                best_cost = offload_cost
                best_strategy = {
                    'offload': True,
                    'rsu_id': rsu_id,
                    'expected_delay': offload_delay,
                    'expected_energy': offload_energy,
                    'resource_allocation': 0.8  # Use 80% of RSU resources (simplified)
                }
        
        # Update queue lengths based on decision
        if best_strategy['offload']:
            # Decrease vehicle queue
            self.vehicle_queues[vehicle_id] = max(0, vehicle_queue - task_size)
            # Increase RSU queue
            rsu_id = best_strategy['rsu_id']
            self.queue_lengths[rsu_id] = self.queue_lengths.get(rsu_id, 0) + task_size
        else:
            # Increase vehicle queue (task stays local)
            self.vehicle_queues[vehicle_id] = vehicle_queue + task_size
        
        return best_strategy
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Dictionary containing test scenarios
            
        Returns:
            Performance metrics dictionary
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        total_energy = 0
        total_delay = 0
        successful_tasks = 0
        failed_tasks = 0
        
        tasks = test_data['tasks']
        vehicles = test_data['vehicles']
        rsus = test_data['rsus']
        connections = test_data['connections']
        vehicle_tasks = test_data['vehicle_tasks']
        
        # Reset queue lengths for evaluation
        self.queue_lengths = {rsu_id: 0 for rsu_id in self.rsu_processing_rates.keys()}
        self.vehicle_queues = {int(v_id): 0 for v_id in vehicle_tasks.keys()}
        
        # Process each task
        for vehicle_id, task_ids in vehicle_tasks.items():
            vehicle_id = int(vehicle_id)
            vehicle_data = vehicles[vehicles['vehicle_id'] == vehicle_id].iloc[0]
            
            for task_id in task_ids:
                task_data = tasks[tasks['task_id'] == int(task_id)].iloc[0]
                
                # Get available RSUs for this vehicle
                available_rsus_ids = connections.get(str(vehicle_id), [])
                available_rsus = rsus[rsus['rsu_id'].isin(available_rsus_ids)]
                
                # Decide offloading strategy
                state = {
                    'task': task_data,
                    'vehicle': vehicle_data,
                    'available_rsus': available_rsus,
                    'connections': connections
                }
                
                decision = self.decide_offloading(state)
                
                # Update metrics
                if decision['expected_delay'] <= task_data['deadline_ms']:
                    successful_tasks += 1
                else:
                    failed_tasks += 1
                
                total_energy += decision['expected_energy']
                total_delay += decision['expected_delay']
        
        # Calculate metrics
        total_tasks = successful_tasks + failed_tasks
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        avg_energy = total_energy / total_tasks if total_tasks > 0 else 0
        avg_delay = total_delay / total_tasks if total_tasks > 0 else 0
        
        return {
            'success_rate': success_rate,
            'avg_energy': avg_energy,
            'avg_delay': avg_delay,
            'total_energy': total_energy,
            'total_delay': total_delay,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks
        }