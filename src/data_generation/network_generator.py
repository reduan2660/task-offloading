# src/data_generation/network_generator.py

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class RSU:
    """Roadside Unit class"""
    rsu_id: int
    location: Tuple[float, float]  # (x, y) coordinates
    coverage_radius: float         # Coverage radius in meters
    processing_power: float        # Processing capability in GIPS (Giga Instructions Per Second)
    bandwidth: float               # Communication bandwidth in Mbps
    memory: float                  # Available memory in MB
    queue_length: int = 0          # Current queue length
    
    def to_dict(self) -> Dict:
        """Convert RSU object to dictionary"""
        return {
            'rsu_id': self.rsu_id,
            'location_x': self.location[0],
            'location_y': self.location[1],
            'coverage_radius': self.coverage_radius,
            'processing_power': self.processing_power,
            'bandwidth': self.bandwidth,
            'memory': self.memory,
            'queue_length': self.queue_length
        }
    
@dataclass
class Vehicle:
    """Vehicle class"""
    vehicle_id: int
    initial_location: Tuple[float, float]  # Starting (x, y) coordinates
    path: List[Tuple[float, float]]        # List of waypoints
    speed: float                           # Speed in m/s
    computing_power: float                 # Local computing power in GIPS
    current_location: Tuple[float, float] = None  # Current location
    
    def to_dict(self) -> Dict:
        """Convert Vehicle object to dictionary"""
        return {
            'vehicle_id': self.vehicle_id,
            'initial_location_x': self.initial_location[0],
            'initial_location_y': self.initial_location[1],
            'path': str(self.path),
            'speed': self.speed,
            'computing_power': self.computing_power,
            'current_location_x': self.current_location[0] if self.current_location else None,
            'current_location_y': self.current_location[1] if self.current_location else None
        }

class NetworkGenerator:
    """Generate realistic RSU-vehicle networks"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the network generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        
    def generate_grid_road_network(self, rows: int, cols: int, cell_size: float = 200.0) -> nx.Graph:
        """Generate a grid road network
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            cell_size: Size of each grid cell in meters
            
        Returns:
            NetworkX graph representing the road network
        """
        G = nx.grid_2d_graph(rows, cols)
        
        # Convert to a more standard graph representation
        mapping = {(i, j): f"node_{i}_{j}" for i, j in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        
        # Add node positions and edge weights
        pos = {}
        for node in G.nodes():
            i, j = map(int, node.split('_')[1:])
            pos[node] = (i * cell_size, j * cell_size)
            G.nodes[node]['pos'] = (i * cell_size, j * cell_size)
        
        # Add edge weights (distances)
        for u, v in G.edges():
            u_pos = G.nodes[u]['pos']
            v_pos = G.nodes[v]['pos']
            dist = np.sqrt((u_pos[0] - v_pos[0])**2 + (u_pos[1] - v_pos[1])**2)
            G[u][v]['weight'] = dist
            G[u][v]['distance'] = dist
        
        return G, pos
    
    def generate_random_road_network(self, n_nodes: int, area_size: float, 
                                    connectivity: float = 0.3) -> Tuple[nx.Graph, Dict]:
        """Generate a random road network
        
        Args:
            n_nodes: Number of nodes (intersections)
            area_size: Size of the area in meters
            connectivity: Probability of edge creation (higher = more connected)
            
        Returns:
            NetworkX graph representing the road network and positions dictionary
        """
        # Generate random node positions
        pos = {i: (self.rng.uniform(0, area_size), self.rng.uniform(0, area_size)) 
               for i in range(n_nodes)}
        
        # Create empty graph
        G = nx.Graph()
        
        # Add nodes with positions
        for i in range(n_nodes):
            G.add_node(i, pos=pos[i])
        
        # Add edges with some randomness to ensure connectivity
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if self.rng.random() < connectivity:
                    dist = np.sqrt((pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2)
                    G.add_edge(i, j, weight=dist, distance=dist)
        
        # Ensure the graph is connected
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(1, len(components)):
                node1 = list(components[0])[0]
                node2 = list(components[i])[0]
                dist = np.sqrt((pos[node1][0] - pos[node2][0])**2 + 
                              (pos[node1][1] - pos[node2][1])**2)
                G.add_edge(node1, node2, weight=dist, distance=dist)
        
        return G, pos
    
    def place_rsus(self, road_network: nx.Graph, n_rsus: int, 
                  coverage_radius_range: Tuple[float, float] = (100.0, 300.0),
                  proc_power_range: Tuple[float, float] = (10.0, 50.0),
                  bandwidth_range: Tuple[float, float] = (10.0, 100.0),
                  memory_range: Tuple[float, float] = (1024.0, 4096.0)) -> List[RSU]:
        """Place RSUs in the road network
        
        Args:
            road_network: NetworkX graph representing the road network
            n_rsus: Number of RSUs to place
            coverage_radius_range: Range of coverage radius (min, max) in meters
            proc_power_range: Range of processing power (min, max) in GIPS
            bandwidth_range: Range of bandwidth (min, max) in Mbps
            memory_range: Range of memory (min, max) in MB
            
        Returns:
            List of RSU objects
        """
        nodes = list(road_network.nodes())
        selected_nodes = self.rng.choice(nodes, size=min(n_rsus, len(nodes)), replace=False)
        
        rsus = []
        for i, node in enumerate(selected_nodes):
            pos = road_network.nodes[node]['pos']
            coverage = self.rng.uniform(*coverage_radius_range)
            proc_power = self.rng.uniform(*proc_power_range)
            bandwidth = self.rng.uniform(*bandwidth_range)
            memory = self.rng.uniform(*memory_range)
            
            rsu = RSU(
                rsu_id=i,
                location=pos,
                coverage_radius=coverage,
                processing_power=proc_power,
                bandwidth=bandwidth,
                memory=memory
            )
            rsus.append(rsu)
        
        return rsus
    
    def generate_vehicle_paths(self, road_network: nx.Graph, n_vehicles: int, 
                              min_path_length: int = 3, max_path_length: int = 10,
                              speed_range: Tuple[float, float] = (10.0, 30.0),
                              computing_power_range: Tuple[float, float] = (1.0, 5.0)) -> List[Vehicle]:
        """Generate vehicles with paths in the road network
        
        Args:
            road_network: NetworkX graph representing the road network
            n_vehicles: Number of vehicles to generate
            min_path_length: Minimum path length in nodes
            max_path_length: Maximum path length in nodes
            speed_range: Range of vehicle speeds (min, max) in m/s
            computing_power_range: Range of computing power (min, max) in GIPS
            
        Returns:
            List of Vehicle objects
        """
        vehicles = []
        nodes = list(road_network.nodes())
        
        for i in range(n_vehicles):
            # Pick random start and end nodes
            start_node = self.rng.choice(nodes)
            
            # Generate a path of random length
            path_length = self.rng.randint(min_path_length, max_path_length + 1)
            
            # Use random walks to generate paths
            path = [start_node]
            current = start_node
            
            for _ in range(path_length - 1):
                neighbors = list(road_network.neighbors(current))
                if not neighbors:
                    break
                next_node = self.rng.choice(neighbors)
                path.append(next_node)
                current = next_node
            
            # Extract node positions to form the path
            path_coords = [road_network.nodes[node]['pos'] for node in path]
            
            # Assign random speed and computing power
            speed = self.rng.uniform(*speed_range)
            computing_power = self.rng.uniform(*computing_power_range)
            
            vehicle = Vehicle(
                vehicle_id=i,
                initial_location=path_coords[0],
                path=path_coords,
                speed=speed,
                computing_power=computing_power,
                current_location=path_coords[0]
            )
            vehicles.append(vehicle)
        
        return vehicles
    
    def get_vehicle_rsu_connections(self, vehicles: List[Vehicle], rsus: List[RSU]) -> Dict:
        """Determine which RSUs each vehicle can connect to based on distance
        
        Args:
            vehicles: List of Vehicle objects
            rsus: List of RSU objects
            
        Returns:
            Dictionary mapping vehicle IDs to lists of accessible RSU IDs
        """
        connections = {}
        
        for vehicle in vehicles:
            vehicle_rsus = []
            v_loc = vehicle.current_location
            
            for rsu in rsus:
                rsu_loc = rsu.location
                dist = np.sqrt((v_loc[0] - rsu_loc[0])**2 + (v_loc[1] - rsu_loc[1])**2)
                
                if dist <= rsu.coverage_radius:
                    vehicle_rsus.append(rsu.rsu_id)
            
            connections[vehicle.vehicle_id] = vehicle_rsus
        
        return connections
    
    def save_network(self, road_network: nx.Graph, rsus: List[RSU], 
                    vehicles: List[Vehicle], connections: Dict, 
                    filepath_prefix: str) -> None:
        """Save the network components to files
        
        Args:
            road_network: NetworkX graph representing the road network
            rsus: List of RSU objects
            vehicles: List of Vehicle objects
            connections: Dictionary of vehicle-RSU connections
            filepath_prefix: Prefix for the output filepaths
        """
        # Save road network
        nx.write_graphml(road_network, f"{filepath_prefix}_road_network.graphml")
        
        # Save RSUs
        rsu_df = pd.DataFrame([rsu.to_dict() for rsu in rsus])
        rsu_df.to_csv(f"{filepath_prefix}_rsus.csv", index=False)
        
        # Save vehicles
        vehicle_df = pd.DataFrame([vehicle.to_dict() for vehicle in vehicles])
        vehicle_df.to_csv(f"{filepath_prefix}_vehicles.csv", index=False)
        
        # Save connections
        with open(f"{filepath_prefix}_connections.json", 'w') as f:
            json.dump(connections, f)

# Example usage
if __name__ == "__main__":
    generator = NetworkGenerator(seed=42)
    
    # Create road network (grid layout)
    road_network, positions = generator.generate_grid_road_network(rows=5, cols=5)
    
    # Place RSUs
    rsus = generator.place_rsus(road_network, n_rsus=10)
    
    # Generate vehicles with paths
    vehicles = generator.generate_vehicle_paths(road_network, n_vehicles=20)
    
    # Get vehicle-RSU connections
    connections = generator.get_vehicle_rsu_connections(vehicles, rsus)
    
    # Save the network
    generator.save_network(road_network, rsus, vehicles, connections, 
                          "data/raw/grid_network")
    
    # Create a random road network
    random_network, positions = generator.generate_random_road_network(n_nodes=25, area_size=1000)
    
    # Place RSUs
    rsus = generator.place_rsus(random_network, n_rsus=8)
    
    # Generate vehicles with paths
    vehicles = generator.generate_vehicle_paths(random_network, n_vehicles=15)
    
    # Get vehicle-RSU connections
    connections = generator.get_vehicle_rsu_connections(vehicles, rsus)
    
    # Save the network
    generator.save_network(random_network, rsus, vehicles, connections,
                         "data/raw/random_network")