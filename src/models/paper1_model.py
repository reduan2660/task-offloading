# src/models/paper1_model.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import math
from ..utils.data_utils import load_scenario_data
from .base_model import BaseTaskOffloadingModel

class LyapunovBasedTaskOffloadingModel(BaseTaskOffloadingModel):
    """
    Enhanced implementation of the Lyapunov-based task offloading model
    Based on the paper: "Task Offloading and Resource Allocation in Vehicular Networks: 
    A Lyapunov-Based Deep Reinforcement Learning Approach"
    """
    
    def __init__(self, V_param: float = 0.5, weight_delay: float = 0.7, 
                 weight_energy: float = 0.3, name: str = "LyapunovModel"):
        # Add constant for deadline safety factor
        global deadline_safety_factor
        deadline_safety_factor = 0.8  # Will keep delays at 80% of deadline for safety
        """
        Initialize the enhanced Lyapunov-based task offloading model
        
        Args:
            V_param: Lyapunov control parameter (trade-off between energy and queue stability)
            weight_delay: Weight factor for delay in the decision function
            weight_energy: Weight factor for energy in the decision function
            name: Model name
        """
        super().__init__(name=name)
        self.V_param = V_param
        self.weight_delay = weight_delay
        self.weight_energy = weight_energy
        self.queue_lengths = {}  # RSU queue lengths
        self.vehicle_queues = {} # Vehicle local queue lengths
        self.rsu_processing_rates = {}  # Processing rates for each RSU
        self.rsu_bandwidth = {}  # Bandwidth for each RSU
        
        # Historical performance tracking
        self.historical_delays = {}  # Track delay history by RSU
        self.historical_queue_lengths = {} # Historical queue length tracking
        self.rsu_reliability = {}  # Track reliability by RSU
        
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
            self.historical_delays[rsu['rsu_id']] = []
            self.historical_queue_lengths[rsu['rsu_id']] = []
            self.rsu_reliability[rsu['rsu_id']] = 1.0  # Initialize with perfect reliability
        
        # Initialize vehicle queues with capacity considerations
        for _, vehicle in vehicles.iterrows():
            self.vehicle_queues[vehicle['vehicle_id']] = {
                'current': 0,
                'capacity': vehicle.get('queue_capacity', 100),  # Default capacity if not provided
                'processing_power': vehicle['computing_power']
            }
        
        # Adaptive V parameter based on network density
        if 'adapt_V' in kwargs and kwargs['adapt_V']:
            num_vehicles = len(vehicles)
            num_rsus = len(rsus)
            network_density = num_vehicles / max(num_rsus, 1)
            
            # Adjust V parameter based on network density
            if network_density > 10:  # High density
                self.V_param = min(1.2, self.V_param * 1.5)
            elif network_density < 3:  # Low density
                self.V_param = max(0.3, self.V_param * 0.8)
        
        self.trained = True
    
    def _calculate_transmission_delay(self, data_size: float, bandwidth: float, 
                                     distance: float, vehicle_speed: float = 0) -> float:
        """
        Calculate transmission delay based on Shannon's formula with mobility considerations
        
        Args:
            data_size: Size of data to transmit in MB
            bandwidth: Channel bandwidth in Mbps
            distance: Distance between vehicle and RSU in meters
            vehicle_speed: Speed of the vehicle in m/s
            
        Returns:
            Transmission delay in milliseconds
        """
        # Safety checks to prevent extreme values
        distance = min(1000, max(10, distance))  # Limit distance range to 10-1000m
        bandwidth = max(1, bandwidth)  # Ensure minimum bandwidth of 1 Mbps
        
        # Simplified path loss model with capped values
        mobility_factor = 1 + 0.05 * min(1, vehicle_speed / 30)  # Reduced mobility impact
        path_loss = min(100, 40 + 20 * math.log10(distance / 100))  # Simplified and capped path loss
        
        # Consider interference based on distance (with smaller effect)
        interference_factor = max(0.7, 1 - (distance / 2000))
        effective_bandwidth = bandwidth * interference_factor
        
        # Calculate SNR with reasonable bounds
        signal_power = max(-70, 20 - path_loss)  # dBm (bounded)
        noise_power = -100  # Simplified fixed noise floor (dBm)
        snr = max(1, 10 ** ((signal_power - noise_power) / 10))  # Ensure minimum SNR
        
        # Shannon capacity with realistic upper bound
        capacity = min(bandwidth, effective_bandwidth * math.log2(1 + snr))  # Mbps (capped)
        
        # Convert data size from MB to Mb
        data_size_mb = data_size * 8
        
        # Calculate base delay with safety floor
        base_delay = (data_size_mb / max(0.5, capacity)) * 1000
        
        # Simplified retransmission model
        retransmission_factor = 1 + min(0.5, distance / 2000)
        
        # Final bounded delay
        delay = min(deadline_safety_factor * 200, base_delay * retransmission_factor)
        
        return delay
    
    def _calculate_processing_delay(self, task_size: float, cpu_cycles: int, 
                                   processing_power: float, queue_length: float = 0,
                                   priority: int = 3) -> float:
        """
        Calculate processing delay with queue awareness
        
        Args:
            task_size: Size of task in MB
            cpu_cycles: Required CPU cycles per bit
            processing_power: Processing power in GIPS
            queue_length: Current queue length
            priority: Task priority (1-5, where 1 is highest)
            
        Returns:
            Processing delay in milliseconds
        """
        # Ensure reasonable values and prevent overflows
        task_size = min(10, task_size)  # Cap task size at 10MB for calculation
        cpu_cycles = min(1000, cpu_cycles)  # Cap CPU cycles per bit
        processing_power = max(0.1, processing_power)  # Ensure minimum processing power
        
        # Convert task size from MB to bits (with overflow protection)
        task_size_bits = task_size * 8 * 1024 * 1024
        
        # Total required cycles (with safety bounds)
        total_cycles = min(1e12, task_size_bits * cpu_cycles)  # Cap at reasonable value
        
        # Simplified queue factor (smaller impact)
        queue_factor = 1 + min(1, 0.1 * queue_length)
        
        # Priority influence (higher priority reduces delay)
        priority_factor = max(0.5, min(1.5, 0.8 + (priority / 10)))
        
        # Processing delay in ms with reasonable bounds
        base_delay = (total_cycles / (processing_power * 1e9)) * 1000
        base_delay = min(deadline_safety_factor * 200, base_delay)  # Cap at reasonable value
        
        delay = base_delay * queue_factor * priority_factor
        
        # Final safety bound
        return min(deadline_safety_factor * 300, delay)
    
    def _calculate_energy_consumption(self, task_size: float, cpu_cycles: int, 
                                     is_local: bool, transmit_power: float = 0.1,
                                     transmission_time: float = 0, 
                                     processing_power: float = 1.0,
                                     mobility: float = 0) -> float:
        """
        Calculate energy consumption with enhanced models
        
        Args:
            task_size: Size of task in MB
            cpu_cycles: Required CPU cycles per bit
            is_local: Whether task is executed locally
            transmit_power: Transmission power in Watts
            transmission_time: Transmission time in ms
            processing_power: Processing power in GHz
            mobility: Vehicle mobility factor (0-1)
            
        Returns:
            Energy consumption in Joules
        """
        # Energy calculation safety
        task_size = min(10, task_size)  # Cap task size at 10MB
        cpu_cycles = min(1000, cpu_cycles)  # Cap CPU cycles
        transmission_time = min(5000, transmission_time)  # Cap transmission time at 5s
        processing_power = max(0.1, min(5.0, processing_power))  # Reasonable processing power range
        
        if is_local:
            # Simplified local execution energy model
            k = 1e-11  # Modified energy coefficient for reasonable values
            task_size_bits = task_size * 8 * 1024 * 1024
            
            # Calculate total cycles with safety bounds
            total_cycles = min(1e10, task_size_bits * cpu_cycles)
            
            # Simplified energy calculation
            energy = k * (processing_power ** 2) * total_cycles
            
            # Apply safety bounds
            energy = min(10.0, max(0.001, energy))
            
        else:
            # Simplified transmission energy model
            base_energy = transmit_power * (transmission_time / 1000)  # Convert ms to s
            
            # Simpler mobility factor
            mobility_factor = 1 + 0.2 * mobility
            
            # Reduced preparation energy
            preparation_energy = 0.01 * task_size
            
            # Calculate total energy with safety bounds
            energy = min(5.0, (base_energy * mobility_factor) + preparation_energy)
        
        return max(0.001, energy)  # Ensure positive energy value
    
    def _calculate_lyapunov_drift(self, queue_length: float, threshold: float = 10.0) -> float:
        """
        Calculate enhanced Lyapunov drift with threshold awareness
        
        Args:
            queue_length: Current queue length
            threshold: Queue stability threshold
            
        Returns:
            Lyapunov drift
        """
        # Quadratic function with threshold awareness
        if queue_length <= threshold:
            return 0.5 * (queue_length ** 2)
        else:
            # Heavier penalty for exceeding threshold
            return 0.5 * (threshold ** 2) + 2 * threshold * (queue_length - threshold)
    
    def _calculate_reliability_score(self, rsu_id: int, task_deadline: float) -> float:
        """
        Calculate reliability score for an RSU based on historical performance
        
        Args:
            rsu_id: RSU identifier
            task_deadline: Deadline of the current task
            
        Returns:
            Reliability score (0-1)
        """
        if not self.historical_delays.get(rsu_id, []):
            return 1.0  # Default perfect reliability if no history
        
        # Get recent delay history (last 20 tasks)
        recent_delays = self.historical_delays[rsu_id][-20:]
        
        # Calculate deadline miss ratio
        deadline_misses = sum(1 for delay in recent_delays if delay > task_deadline)
        miss_ratio = deadline_misses / len(recent_delays) if recent_delays else 0
        
        # Exponential reliability function
        reliability = math.exp(-3 * miss_ratio)
        
        return reliability
    
    def _predict_future_queue_length(self, rsu_id: int, time_window: int = 5) -> float:
        """
        Predict future queue length based on historical trends
        
        Args:
            rsu_id: RSU identifier
            time_window: Prediction window (number of recent observations)
            
        Returns:
            Predicted future queue length
        """
        if rsu_id not in self.historical_queue_lengths or len(self.historical_queue_lengths[rsu_id]) < 2:
            return self.queue_lengths.get(rsu_id, 0)
        
        # Get recent queue length history
        recent_queue = self.historical_queue_lengths[rsu_id][-time_window:]
        
        # Simple linear trend prediction
        if len(recent_queue) >= 2:
            trend = (recent_queue[-1] - recent_queue[0]) / len(recent_queue)
            prediction = max(0, self.queue_lengths[rsu_id] + trend * 2)  # Predict 2 steps ahead
            return prediction
        
        return self.queue_lengths.get(rsu_id, 0)
    
    def _get_adaptive_resource_allocation(self, task_priority: int, queue_length: float, 
                                        rsu_id: int = None) -> float:
        """
        Determine adaptive resource allocation based on task priority and queue state
        
        Args:
            task_priority: Priority of the task (1-5, where 1 is highest)
            queue_length: Current queue length
            rsu_id: RSU identifier (None for local execution)
            
        Returns:
            Resource allocation ratio (0-1)
        """
        # Base allocation by priority (higher priority gets more resources)
        base_allocation = max(0.5, 1 - ((task_priority - 1) / 10))
        
        if rsu_id is not None:
            # RSU allocation considers current queue length
            queue_factor = 1 / (1 + 0.1 * queue_length)
            allocation = min(1.0, base_allocation * queue_factor * 1.2)  # Boost RSU allocation
        else:
            # Local allocation is more conservative to save energy
            allocation = base_allocation * 0.9
        
        return min(1.0, max(0.3, allocation))  # Ensure allocation is between 30% and 100%
    
    def decide_offloading(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide task offloading strategy based on current state with enhanced decision logic
        
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
        priority = task.get('priority', 3)  # Default priority is medium (3)
        
        vehicle_id = vehicle['vehicle_id']
        vehicle_speed = vehicle.get('speed', 0)
        vehicle_queue = self.vehicle_queues.get(vehicle_id, {}).get('current', 0)
        vehicle_processing_power = vehicle['computing_power']
        
        # Safety checks for input values
        deadline = max(100, deadline)  # Ensure reasonable deadline
        task_size = min(10, task_size)  # Cap task size for calculations
        cpu_cycles = min(1000, cpu_cycles)  # Cap CPU cycles

        # Simplified deadline urgency
        deadline_urgency = max(0.1, min(1.0, 10 / deadline))
        
        # Calculate local processing delay and energy with safety bounds
        local_processing_delay = self._calculate_processing_delay(
            task_size, cpu_cycles, vehicle_processing_power, vehicle_queue, priority)
        local_energy = self._calculate_energy_consumption(
            task_size, cpu_cycles, is_local=True, processing_power=vehicle_processing_power)
        
        # Calculate resource allocation for local execution
        local_allocation = self._get_adaptive_resource_allocation(priority, vehicle_queue)
        
        # Initialize best strategy with local execution
        best_strategy = {
            'offload': False,
            'rsu_id': None,
            'expected_delay': local_processing_delay,
            'expected_energy': local_energy,
            'resource_allocation': local_allocation,
            'reliability': 1.0  # Local processing has perfect reliability
        }
        
        # Simplified Lyapunov-based cost for local execution
        vehicle_queue_drift = min(5, self._calculate_lyapunov_drift(vehicle_queue))
        local_cost = (self.V_param * local_energy * self.weight_energy + 
                     local_processing_delay * self.weight_delay + 
                     vehicle_queue_drift)
        
        # Simpler priority factor
        priority_factor = 1 + (0.1 * (6 - priority))
        local_cost /= priority_factor
        
        best_cost = local_cost
        found_viable_offload = False
        
        # Check each available RSU for potential offloading
        for _, rsu in available_rsus.iterrows():
            rsu_id = rsu['rsu_id']
            
            # Check if vehicle is connected to this RSU
            if vehicle_id not in connections or rsu_id not in connections.get(str(vehicle_id), []):
                continue
            
            rsu_location = (rsu['location_x'], rsu['location_y'])
            distance = math.sqrt(
                (vehicle_location[0] - rsu_location[0])**2 + 
                (vehicle_location[1] - rsu_location[1])**2)
            
            # Calculate transmission delay with safety bounds
            transmission_delay = self._calculate_transmission_delay(
                task_size, rsu['bandwidth'], distance, vehicle_speed)
            
            # Get current RSU queue length (with safety cap)
            rsu_queue = min(10, self.queue_lengths.get(rsu_id, 0))
            
            # Calculate processing delay at RSU with safety bounds
            rsu_processing_delay = self._calculate_processing_delay(
                task_size, cpu_cycles, rsu['processing_power'], rsu_queue, priority)
            
            # Total delay for offloading
            offload_delay = transmission_delay + rsu_processing_delay
            
            # Calculate energy consumption for transmission
            mobility_factor = min(1, vehicle_speed / 30)
            offload_energy = self._calculate_energy_consumption(
                task_size, cpu_cycles, is_local=False, 
                transmission_time=transmission_delay,
                mobility=mobility_factor)
            
            # Simplified reliability calculation
            reliability = max(0.7, self._calculate_reliability_score(rsu_id, deadline))
            
            # Calculate resource allocation for RSU execution
            rsu_allocation = self._get_adaptive_resource_allocation(priority, rsu_queue, rsu_id)
            
            # Simplified cost function
            queue_drift = min(5, self._calculate_lyapunov_drift(rsu_queue))
            offload_cost = (
                self.V_param * offload_energy * self.weight_energy + 
                offload_delay * self.weight_delay +
                queue_drift
            )
            
            # Simplified reliability and priority adjustments
            offload_cost *= (2 - reliability)
            offload_cost /= priority_factor
            
            # Debug information
            # print(f"RSU {rsu_id}: delay={offload_delay}, deadline={deadline}, cost={offload_cost}")
            
            # Check if offloading meets deadline and improves cost
            if offload_delay <= deadline and offload_cost < best_cost:
                best_cost = offload_cost
                best_strategy = {
                    'offload': True,
                    'rsu_id': rsu_id,
                    'expected_delay': offload_delay,
                    'expected_energy': offload_energy,
                    'resource_allocation': rsu_allocation,
                    'reliability': reliability
                }
                found_viable_offload = True
        
        # Force local execution if it can meet deadline and no viable offloading found
        if not found_viable_offload and local_processing_delay <= deadline:
            best_strategy = {
                'offload': False,
                'rsu_id': None,
                'expected_delay': local_processing_delay,
                'expected_energy': local_energy,
                'resource_allocation': local_allocation,
                'reliability': 1.0
            }
        
        # Update queue lengths and historical data based on decision
        if best_strategy['offload']:
            # Decrease vehicle queue
            if vehicle_id in self.vehicle_queues:
                self.vehicle_queues[vehicle_id]['current'] = max(0, vehicle_queue - task_size)
            
            # Increase RSU queue
            rsu_id = best_strategy['rsu_id']
            self.queue_lengths[rsu_id] = self.queue_lengths.get(rsu_id, 0) + task_size
            
            # Update historical data
            self.historical_delays[rsu_id].append(best_strategy['expected_delay'])
            self.historical_queue_lengths[rsu_id].append(self.queue_lengths[rsu_id])
            
            # Limit history length
            if len(self.historical_delays[rsu_id]) > 10:
                self.historical_delays[rsu_id] = self.historical_delays[rsu_id][-10:]
            if len(self.historical_queue_lengths[rsu_id]) > 10:
                self.historical_queue_lengths[rsu_id] = self.historical_queue_lengths[rsu_id][-10:]
                
        else:
            # Increase vehicle queue (task stays local)
            if vehicle_id in self.vehicle_queues:
                current = self.vehicle_queues[vehicle_id]['current']
                capacity = self.vehicle_queues[vehicle_id]['capacity']
                # Ensure we don't exceed capacity
                self.vehicle_queues[vehicle_id]['current'] = min(capacity, current + task_size)
        
        return best_strategy
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model performance on test data with enhanced metrics
        
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
        offloaded_tasks = 0
        local_tasks = 0
        high_priority_success = 0
        high_priority_total = 0
        
        # Track delay satisfaction ratio
        delay_satisfaction = []
        
        # Track energy efficiency
        energy_per_bit = []
        
        # Track load balancing
        rsu_load = {rsu_id: 0 for rsu_id in self.rsu_processing_rates.keys()}
        
        tasks = test_data['tasks']
        vehicles = test_data['vehicles']
        rsus = test_data['rsus']
        connections = test_data['connections']
        vehicle_tasks = test_data['vehicle_tasks']
        
        # Reset queue lengths for evaluation
        self.queue_lengths = {rsu_id: 0 for rsu_id in self.rsu_processing_rates.keys()}
        self.vehicle_queues = {
            int(v_id): {'current': 0, 'capacity': 100, 'processing_power': 1.0} 
            for v_id in vehicle_tasks.keys()
        }
        
        # Update vehicle processing power
        for _, vehicle in vehicles.iterrows():
            if vehicle['vehicle_id'] in self.vehicle_queues:
                self.vehicle_queues[vehicle['vehicle_id']]['processing_power'] = vehicle['computing_power']
        
        # Process each task
        for vehicle_id, task_ids in vehicle_tasks.items():
            vehicle_id = int(vehicle_id)
            vehicle_data = vehicles[vehicles['vehicle_id'] == vehicle_id].iloc[0]
            
            for task_id in task_ids:
                task_data = tasks[tasks['task_id'] == int(task_id)].iloc[0]
                
                # Check if high priority task
                priority = task_data.get('priority', 3)
                is_high_priority = priority <= 2
                
                if is_high_priority:
                    high_priority_total += 1
                
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
                
                # Track offloading decision
                if decision['offload']:
                    offloaded_tasks += 1
                    rsu_load[decision['rsu_id']] += task_data['size_mb']
                else:
                    local_tasks += 1
                
                # Update metrics
                delay_ratio = decision['expected_delay'] / task_data['deadline_ms']
                delay_satisfaction.append(min(1, 1 / delay_ratio))
                
                energy_per_bit.append(decision['expected_energy'] / (task_data['size_mb'] * 8 * 1024 * 1024))
                
                if decision['expected_delay'] <= task_data['deadline_ms']:
                    successful_tasks += 1
                    if is_high_priority:
                        high_priority_success += 1
                else:
                    failed_tasks += 1
                
                total_energy += decision['expected_energy']
                total_delay += decision['expected_delay']
        
        # Calculate metrics
        total_tasks = successful_tasks + failed_tasks
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        avg_energy = total_energy / total_tasks if total_tasks > 0 else 0
        avg_delay = total_delay / total_tasks if total_tasks > 0 else 0
        offload_rate = offloaded_tasks / total_tasks if total_tasks > 0 else 0
        
        # High priority task success rate
        high_priority_success_rate = (high_priority_success / high_priority_total 
                                      if high_priority_total > 0 else 1.0)
        
        # Average delay satisfaction
        avg_delay_satisfaction = sum(delay_satisfaction) / len(delay_satisfaction) if delay_satisfaction else 0
        
        # Average energy efficiency
        avg_energy_efficiency = sum(energy_per_bit) / len(energy_per_bit) if energy_per_bit else 0
        
        # Load balancing metric (Jain's fairness index)
        if sum(rsu_load.values()) > 0:
            square_sum = sum(x ** 2 for x in rsu_load.values())
            sum_squared = sum(rsu_load.values()) ** 2
            num_rsus = len(rsu_load)
            load_balance = sum_squared / (num_rsus * square_sum) if square_sum > 0 else 1
        else:
            load_balance = 1  # Perfect balance if no tasks offloaded
        
        # Combined metric (weighted average of success rate, energy, delay, and load balance)
        combined_metric = (0.4 * success_rate + 
                          0.2 * (1 - avg_energy / max(avg_energy, 1e-6)) + 
                          0.2 * (1 - avg_delay / max(avg_delay, 1e-6)) + 
                          0.1 * load_balance +
                          0.1 * high_priority_success_rate)
        
        return {
            'success_rate': success_rate,
            'avg_energy': avg_energy,
            'avg_delay': avg_delay,
            'total_energy': total_energy,
            'total_delay': total_delay,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'offload_rate': offload_rate,
            'high_priority_success_rate': high_priority_success_rate,
            'avg_delay_satisfaction': avg_delay_satisfaction,
            'energy_efficiency': avg_energy_efficiency,
            'load_balance': load_balance,
            'combined_metric': combined_metric
        }