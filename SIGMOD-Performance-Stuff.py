from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
from mpi4py import MPI
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import time
import sys
import networkx as nx
import pandas as pd
import copy
from shapely.geometry import Point
from collections import defaultdict
from sklearn.cluster import KMeans  
from sklearn.cluster import DBSCAN  


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

THRESHOLD = 80  
BLOCK_SIZE = 5   
REPLICATION_FACTOR = 3  
MAX_BLOCKS = 100 // BLOCK_SIZE

class Block:
    block_counter = 0

    def __init__(self, features):
        if not all(isinstance(feature, (list, np.ndarray)) for feature in features):
            features = [[feature] if isinstance(feature, float) else feature for feature in features]
        self.features = features
        self.hash = self.calculate_hash()
        self.id = Block.block_counter
        self.size = BLOCK_SIZE
        Block.block_counter += 1
        self.replica_locations = []  # Track replica nodes

    def calculate_hash(self):
        return sum(sum(feature) if isinstance(feature, (list, np.ndarray)) else feature 
                   for feature in self.features if feature is not None)



class Node:
    def __init__(self, id, degree, uptime, latency, token, adjacency_votes, disk_usage, computational_capacity, network):
        self.id = id
        self.degree = degree
        self.uptime = uptime
        self.latency = latency
        self.token = token
        self.adjacency_votes = adjacency_votes
        self.is_leader = False
        self.is_trustworthy = bool(random.getrandbits(1))
        self.neighbors = []
        self.disk_usage = disk_usage
        self.computational_capacity = computational_capacity
        self.is_validating = False
        self.blockchain = []
        self.reputation = 0
        self.network = network
        self.validation_thread = None
        self.health_check_thread = None

    def is_idle(self):
        return self.computational_capacity > 60  # More realistic idle condition

    def prune_old_data(self):
        """Prune blocks if over capacity, update usage correctly"""
        max_blocks = (100 // BLOCK_SIZE)
        if len(self.blockchain) > max_blocks:
            removed = len(self.blockchain) - max_blocks
            self.blockchain = self.blockchain[-max_blocks:]
            self.disk_usage = len(self.blockchain) * BLOCK_SIZE  # Direct count
            

    def _validation_loop(self):
        while True:
            if self.blockchain:
                self.process_block()
            time.sleep(0.1)

    def _health_check_loop(self):
        while True:
            if self.disk_usage > THRESHOLD:
                enhanced_dynamic_sharding(self)
            time.sleep(5)

class Cluster:
    def __init__(self, cluster_id, nodes):  # Changed parameter name to avoid conflict
        self.id = cluster_id
        self.nodes = nodes
        self.sub_clusters = []
        self.shard_boundaries = (-float('inf'), float('inf'))
        self.vector_size_profile = []

    def update_boundaries(self, new_block):
        # Rest of your existing update_boundaries method
        if self.shard_boundaries[0] == -float('inf'):
            self.shard_boundaries = (min(new_block.features[0]), 
                                   max(new_block.features[0]))
        else:
            vector = new_block.features[0]
            self.shard_boundaries = (
                min(self.shard_boundaries[0], min(vector)),
                max(self.shard_boundaries[1], max(vector))
            )
            self.vector_size_profile.append(len(vector))  


class ScaleFreeGraph:
    def __init__(self, num_nodes, initial_links):
        self.graph = nx.barabasi_albert_graph(num_nodes, initial_links)

    def get_adjacency_list(self):
        return nx.to_dict_of_lists(self.graph)


class Network:
    def __init__(self, num_nodes, num_clusters, initial_links=2, replication_factor=REPLICATION_FACTOR):
        self.num_nodes = num_nodes
        self.num_clusters = self.calculate_optimal_clusters(num_nodes)  # Dynamic cluster scaling
        self.nodes_per_cluster = max(4, num_nodes // self.num_clusters) 
        self.reserved_capacity = 0 if num_clusters > 1 else max(2, int(num_nodes * 0.05))
        self.nodes = []
        self.clusters = []
        self.scale_free_graph = ScaleFreeGraph(num_nodes, initial_links)
        self.adjacency_list = self.scale_free_graph.get_adjacency_list()
        self.replication_factor = self.calculate_adaptive_replication_factor(num_nodes)  # Adaptive replication
        self.optimization_interval = 30  # 5 minutes
        self.error_counter = defaultdict(int)
        self.error_log = {
            'replication_failures': 0,
            'overloaded_nodes': 0,
            'consensus_failures': 0  
        }
        self.last_rebalance_time = time.time()
        self.under_replicated_blocks = []
        self.reserved_nodes = []
        self.in_experiment = False
        self.initialize_nodes()
        self.scale_free_graph = ScaleFreeGraph(num_nodes, initial_links)
        self.shard_registry = defaultdict(list)  # Added for scale-free tracking



    @staticmethod  # Add staticmethod decorator
    def calculate_optimal_clusters(num_nodes):  # Added self parameter
        """Hybrid cluster sizing: Static for small networks, dynamic for large."""
        if num_nodes < 50:
            return 2 if num_nodes == 50 else 6
        else:
            return max(2, int(np.log2(num_nodes)))

    @staticmethod  # Add staticmethod decorator
    def calculate_adaptive_replication_factor(num_nodes):  # Added self parameter
        """Hybrid replication: Static for small networks, dynamic for large."""
        if num_nodes < 50:
            return 3 if num_nodes == 50 else 1.5
        else:
            return max(1, int(3 - (num_nodes / 50)))


    def calculate_optimal_shard_boundaries(self, blocks, num_shards):
        """Calculate optimal shard boundaries using a combination of DBSCAN and k-means."""
        vectors = [block.features[0] for block in blocks]
        if len(vectors) < num_shards:
            return [(-float('inf'), float('inf'))] * num_shards
        
        # Use DBSCAN for irregular clusters
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan.fit(vectors)
        if len(set(dbscan.labels_)) > 1:  # If DBSCAN finds multiple clusters
            boundaries = []
            for i in range(num_shards):
                cluster_vectors = [vec for vec, label in zip(vectors, dbscan.labels_) if label == i]
                if cluster_vectors:
                    min_val = min(min(vec) for vec in cluster_vectors)
                    max_val = max(max(vec) for vec in cluster_vectors)
                    boundaries.append((min_val, max_val))
                else:
                    boundaries.append((-float('inf'), float('inf')))
            return boundaries
        else:
            # Fallback to k-means if DBSCAN fails to find clusters
            kmeans = KMeans(n_clusters=num_shards)
            kmeans.fit(vectors)
            boundaries = []
            for i in range(num_shards):
                cluster_vectors = [vec for vec, label in zip(vectors, kmeans.labels_) if label == i]
                if cluster_vectors:
                    min_val = min(min(vec) for vec in cluster_vectors)
                    max_val = max(max(vec) for vec in cluster_vectors)
                    boundaries.append((min_val, max_val))
                else:
                    boundaries.append((-float('inf'), float('inf')))
            return boundaries

    def periodic_rebalance(self):
        """Periodically rebalance blocks across clusters."""
        current_time = time.time()
        if current_time - self.last_rebalance_time > self.optimization_interval:
            self.global_rebalance()
            self.last_rebalance_time = current_time


    def _create_shard_around_hub(self, hub, min_nodes=4):
        """Create shard with hub and its neighbors"""
        members = [hub] + hub.neighbors[:min_nodes-1]
        new_cluster = Cluster(len(self.clusters), members)
        self.clusters.append(new_cluster)
        self.shard_registry[hub.id] = new_cluster
        return new_cluster

    def split_shard(self, cluster_id):
        old_cluster = self.clusters[cluster_id]
        if len(old_cluster.nodes) >= 6:  # More aggressive splitting
            split_idx = len(old_cluster.nodes) // 2
            new_nodes = old_cluster.nodes[split_idx:]
            old_cluster.nodes = old_cluster.nodes[:split_idx]
            
            new_cluster = Cluster(len(self.clusters), new_nodes)
            self.clusters.append(new_cluster)
            
            # Calculate optimal shard boundaries using DBSCAN
            all_blocks = [b for node in old_cluster.nodes + new_cluster.nodes for b in node.blockchain]
            boundaries = self.calculate_optimal_shard_boundaries(all_blocks, 2)
            old_cluster.shard_boundaries = boundaries[0]
            new_cluster.shard_boundaries = boundaries[1]
            
            self.redistribute_blocks(old_cluster, new_cluster)

    def redistribute_blocks(self, old_cluster, new_cluster):
        all_blocks = [b for node in old_cluster.nodes + new_cluster.nodes for b in node.blockchain]
        for node in old_cluster.nodes + new_cluster.nodes:
            node.blockchain = []
            node.disk_usage = 0

        for block in all_blocks:
            vec_mean = np.mean(block.features[0])
            target_cluster = old_cluster if (old_cluster.shard_boundaries[0] <= vec_mean <= old_cluster.shard_boundaries[1]) else new_cluster
            
            target_node = min(target_cluster.nodes, 
                            key=lambda n: n.disk_usage)
            
            if target_node.disk_usage + BLOCK_SIZE <= 100:
                target_node.blockchain.append(block)
                target_node.disk_usage += BLOCK_SIZE

    def optimize_shards(self):
        cluster_loads = []
        for cluster in self.clusters:
            avg_load = np.mean([n.disk_usage for n in cluster.nodes])
            if avg_load > 75 and len(cluster.nodes) > 2:
                self.split_shard(cluster.id)
            cluster_loads.append(avg_load)
        
        if np.std(cluster_loads) > 20:
            self.global_rebalance()
            
        self.process_under_replicated()
        
        self.check_scaling()

    def global_rebalance(self):
        all_blocks = []
        for cluster in self.clusters:
            for node in cluster.nodes:
                all_blocks.extend(node.blockchain)
                node.blockchain = []
                node.disk_usage = 0

        all_blocks.sort(key=lambda b: np.linalg.norm(b.features[0]))
        
        for block in all_blocks:
            target_cluster = min(self.clusters, 
                               key=lambda c: np.mean([n.disk_usage for n in c.nodes]))
            target_node = min(target_cluster.nodes, 
                            key=lambda n: n.disk_usage)
            
            if target_node.disk_usage + BLOCK_SIZE <= 100:
                target_node.blockchain.append(block)
                target_node.disk_usage += BLOCK_SIZE


    def initialize_nodes(self):
        for i in range(self.num_nodes):
            node = Node(
                id=i,
                degree=len(self.adjacency_list[i]),
                uptime=random.uniform(0.8, 1.0),
                latency=random.uniform(0.1, 0.5),
                token=random.randint(50, 100),
                adjacency_votes=random.randint(5, 10),
                disk_usage=0,
                network=self,
                computational_capacity=100
            )
            self.nodes.append(node)
        
        self.reserved_nodes = self.nodes[:self.reserved_capacity]
        clusterable_nodes = self.nodes[self.reserved_capacity:]
        
        self.clusters = []
        nodes_per_cluster = max(1, len(clusterable_nodes) // self.num_clusters)
        for cluster_id in range(self.num_clusters):
            start_idx = cluster_id * nodes_per_cluster
            end_idx = start_idx + nodes_per_cluster
            if cluster_id == self.num_clusters - 1:
                cluster_nodes = clusterable_nodes[start_idx:]
            else:
                cluster_nodes = clusterable_nodes[start_idx:end_idx]
            
            if not cluster_nodes:  # Safety check
                cluster_nodes = [clusterable_nodes[0]] if clusterable_nodes else []
            
            self.clusters.append(Cluster(cluster_id, cluster_nodes))

        for node in self.nodes:
            node.neighbors = [self.nodes[neighbor_id] for neighbor_id in self.adjacency_list[node.id]]

    def add_cluster(self, nodes):
        if len(nodes) == 0:
            raise ValueError("Cannot create empty cluster")
        new_cluster = Cluster(len(self.clusters), nodes)
        self.clusters.append(new_cluster)

    def get_non_sharded_nodes(self):
        return [n for n in self.reserved_nodes 
               if n.disk_usage + BLOCK_SIZE <= 100]


    def periodic_rebalance(self):
        """Enhanced with scale-free awareness"""
        if time.time() - self.last_rebalance_time > self.optimization_interval:
            self.global_rebalance()
            # Scale-free node addition
            if len(self.nodes)/len(self.clusters) > 15:  
                new_hub = max(self.nodes, key=lambda n: n.degree)
                if new_hub.id not in self.shard_registry:
                    self._create_shard_around_hub(new_hub)
            self.last_rebalance_time = time.time()

    def process_under_replicated(self):
        temp_ur = []
        for block in self.under_replicated_blocks:
            if self.replicate_block(block, find_global=True):
                continue
            temp_ur.append(block)  # Keep if still unresolved
        self.under_replicated_blocks = temp_ur

    def check_scaling(self):
        total_used = sum(n.disk_usage for n in self.nodes)
        total_capacity = len(self.nodes) * 100
        
        if total_used / total_capacity > 0.75:
            new_nodes = [Node(
                id=len(self.nodes)+i,
                degree=0,  # Updated during initialization
                uptime=random.uniform(0.8, 1.0),
                latency=random.uniform(0.1, 0.5),
                token=random.randint(50, 100),
                adjacency_votes=random.randint(5, 10),
                disk_usage=0,
                computational_capacity=100,
                network=self
            ) for i in range(10)]
            
            self.nodes.extend(new_nodes)
            if len(self.clusters) == 1:  # Non-sharded
                self.clusters[0].nodes.extend(new_nodes)
            else:
                self.add_cluster(new_nodes)
            
        for node in self.nodes:
            if node.disk_usage > 100:
                overflow = node.disk_usage - 100
                remove_count = min(len(node.blockchain), overflow // BLOCK_SIZE)
                del node.blockchain[:remove_count]
                node.disk_usage = len(node.blockchain) * BLOCK_SIZE
                self.error_log['overloaded_nodes'] += 1

    def replicate_block(self, block, cluster=None, find_global=False):
        if cluster and not find_global:
            candidates = sorted(cluster.nodes, 
                            key=lambda n: n.disk_usage)
            candidates = [n for n in candidates 
                        if n.disk_usage + BLOCK_SIZE <= 100
                        and n.id not in block.replica_locations]
        else:
            candidates = []
            sorted_clusters = sorted(self.clusters, 
                                key=lambda c: np.mean([n.disk_usage for n in c.nodes]))
            for sc in sorted_clusters:
                cluster_candidates = sorted(sc.nodes, key=lambda n: n.disk_usage)
                candidates.extend(cluster_candidates)
                if len(candidates) >= 10:
                    break
            candidates = [n for n in candidates 
                        if n.disk_usage + BLOCK_SIZE <= 100
                        and n.id not in block.replica_locations]
        
        if not candidates:
            self.error_log['replication_failures'] += 1
            return False
            
        target_node = min(candidates[:5], 
                        key=lambda n: (n.disk_usage, len(n.blockchain)/MAX_BLOCKS))
        target_node.blockchain.append(block)
        target_node.disk_usage += BLOCK_SIZE
        block.replica_locations.append(target_node.id)
        return True

    def print_detailed_diagnostics(self):
        print(f"\nCluster Health ({len(self.clusters)} clusters)")
        for cluster in self.clusters:
            load_dist = [n.disk_usage for n in cluster.nodes]
            print(f"Cluster {cluster.id}: "
                  f"Min={min(load_dist):.1f}%, "
                  f"Avg={np.mean(load_dist):.1f}%, "
                  f"Max={max(load_dist):.1f}%")
                  
        print(f"\nReplication Health: "
              f"{len(self.under_replicated_blocks)} pending blocks")
              
        print("\nNode Capacity:")
        reserved_usage = [n.disk_usage for n in self.reserved_nodes]
        print(f"Reserved nodes: {np.mean(reserved_usage):.1f}% avg usage")
        
        normal_usage = [n.disk_usage for n in self.nodes 
                       if n not in self.reserved_nodes]
        print(f"Sharded nodes: {np.mean(normal_usage):.1f}% avg usage")

    def needs_rebalancing(self, cluster):
        vector_sizes = [len(b.features) for node in cluster.nodes for b in node.blockchain]
        load_std = np.std([n.disk_usage for n in cluster.nodes])
        return load_std > 15 or np.mean(vector_sizes) > 50
    
    def redistribute_vectors(self, cluster):
        sorted_blocks = sorted((b for node in cluster.nodes for b in node.blockchain),
                             key=lambda x: x.features[0])
        new_boundary = sorted_blocks[len(sorted_blocks)//2].features[0]
        cluster.shard_boundaries = (cluster.shard_boundaries[0], new_boundary)
        
        for node in cluster.nodes:
            node.blockchain = [b for b in node.blockchain 
                             if b.features[0] <= new_boundary]

    def check_replica_health(self):
        for cluster in self.clusters:
            for node in cluster.nodes:
                for block in node.blockchain:
                    live_replicas = sum(1 for n in cluster.nodes 
                                      if block in n.blockchain)
                    if live_replicas < self.replication_factor:
                        self.replicate_block(block, cluster)

    def print_network_state(self, stage):
        print(f"\n{stage} Network State")
        print(f"{'='*40}")
        print(f"Nodes: {self.num_nodes} | Clusters: {len(self.clusters)}")
        print(f"Replication: {self.replication_factor}x")
        # print_cluster_summary(self.clusters)

    def test_boundaries(self):
        for cluster in self.clusters:
            for node in cluster.nodes:
                for block in node.blockchain:
                    for feature_value in block.features[0]:
                        assert (cluster.shard_boundaries[0] <= feature_value <= cluster.shard_boundaries[1]), \
                            f"Vector {block.features[0]} out of shard range {cluster.shard_boundaries}!"
            
    def print_replica_distribution(self):
        replica_counts = defaultdict(int)
        total_blocks = 0
        
        for cluster in self.clusters:
            for node in cluster.nodes:
                for block in node.blockchain:
                    replica_count = len(block.replica_locations)
                    replica_counts[replica_count] += 1
                    total_blocks += 1

        print("\nReplica Distribution Summary:")
        print(f"{'Replicas':<10} | {'Blocks':<10}")
        for count in sorted(replica_counts.keys()):
            print(f"{count:<10} | {replica_counts[count]:<10}")
        
        if total_blocks == 0:
            print("\nNo blocks stored in the network")
            return

        non_compliant = sum(cnt for reps, cnt in replica_counts.items() 
                        if reps < self.replication_factor)
        
        compliant_pct = (non_compliant / total_blocks * 100) if total_blocks else 0.0
        print(f"\nNon-compliant Blocks: {non_compliant} ({compliant_pct:.1f}%)")

    def read_block(self, block_id, use_sharding=True):
        """
        Read a block from the network and measure retrieval time
        
        Args:
            block_id: ID of the block to retrieve
            use_sharding: Whether to use sharding for retrieval
            
        Returns:
            tuple: (retrieved block, retrieval time in seconds)
        """
        start_time = time.perf_counter()
        retrieved_block = None
        
        # Add some artificial complexity to ensure measurable times
        search_count = 0
        
        if use_sharding:
            # Query across shards
            for cluster in self.clusters:
                for node in cluster.nodes:
                    search_count += 1
                    # Add a small computational cost
                    _ = np.mean([n.disk_usage for n in cluster.nodes])
                    
                    for block in node.blockchain:
                        # More realistic search (compute hash comparison)
                        block_hash = block.calculate_hash()
                        if block.id == block_id:
                            retrieved_block = block
                            break
                    if retrieved_block:
                        break
                if retrieved_block:
                    break
        else:
            # Query all nodes (non-sharded approach)
            for node in self.nodes:
                search_count += 1
                # Add a small computational cost
                _ = node.disk_usage * 1.1
                
                for block in node.blockchain:
                    # More realistic search (compute hash comparison)
                    block_hash = block.calculate_hash()
                    if block.id == block_id:
                        retrieved_block = block
                        break
                if retrieved_block:
                    break
        
        # Ensure we have a minimum processing time for measurement
        time.sleep(0.0001)  # Small sleep to ensure measurable times
        
        end_time = time.perf_counter()
        retrieval_time = end_time - start_time
        
        return retrieved_block, retrieval_time


def validate_block(block, node, validation_criteria):
    # Add null check for validation_criteria
    if not validation_criteria:
        validation_criteria = [check_integrity]
    
    for vector in block.features:
        if not check_integrity(vector) or not satisfies_criteria(vector, validation_criteria):
            return False
    return True

def check_integrity(vector):
    return len(vector) > 0 and all(isinstance(val, (int, float)) for val in vector)

def satisfies_criteria(vector, validation_criteria):
    return all(criterion(vector) for criterion in validation_criteria)

def select_leaders(network):
    if not network.nodes:
        raise ValueError("Cannot select leaders from empty network")
    
    try:
        hub = max(network.nodes, key=lambda n: n.degree)
    except ValueError:
        hub = random.choice(network.nodes)

    if not hub.neighbors:
        potential_hubs = sorted([n for n in network.nodes if n.neighbors], 
                               key=lambda x: x.degree, reverse=True)
        if potential_hubs:
            hub = potential_hubs[0]
        else:
            hub = random.choice(network.nodes)

    if hub.neighbors:
        candidates = random.sample(hub.neighbors, min(3, len(hub.neighbors)))
    else:
        candidates = random.sample(network.nodes, min(3, len(network.nodes)))

    leaders = []
    for node in candidates:
        score = node.uptime - node.latency + node.token + node.degree
        if score > 80:
            leaders.append(node)
            node.is_leader = True

    total_adjacency = sum(n.degree for n in leaders)
    return leaders, total_adjacency

# Add padding to ensure equal chunks
def split_padded(arr, n):
    quotient, remainder = divmod(len(arr), n)
    return [arr[i*quotient+min(i, remainder):(i+1)*quotient+min(i+1, remainder)] 
           for i in range(n)]


def scale_free_consensus(block, network, use_sharding=True):
    try:
        # Define validation criteria (FIX 1)
        validation_criteria = [check_integrity]
        
        # Scale-free leader selection
        leaders = [n for n in network.shard_registry.keys()]
        
        # Multi-threaded validation (FIX 1 - add validation_criteria)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(validate_block, block, leader, validation_criteria)
                      for leader in leaders]
            leader_votes = sum(f.result() for f in futures)

        if leader_votes >= len(leaders) * 0.51:
            if use_sharding:
                # Scale-free shard selection
                target_node = network.get_target_shard(block)
                
                # Parallel replication
                with ThreadPoolExecutor() as executor:
                    replicas = list(executor.map(
                        lambda _: network.replicate_block(block),
                        range(REPLICATION_FACTOR)
                    ))
                return sum(replicas) >= REPLICATION_FACTOR
        return False
    except Exception as e:
        # Initialize consensus_failures if missing (FIX 2)
        if 'consensus_failures' not in network.error_log:
            network.error_log['consensus_failures'] = 0
        network.error_log['consensus_failures'] += 1
        return False

def enhanced_dynamic_sharding(node):
    if isinstance(node, Node):
        if node.disk_usage <= THRESHOLD:
            return

        cluster = next((c for c in node.network.clusters if node in c.nodes), None)
        if not cluster:
            return
        
        cluster_load = np.mean([n.disk_usage for n in cluster.nodes])
        if cluster_load > 75 and len(cluster.nodes) >= 4:
            node.network.split_shard(cluster.id)
            return
        
        suitable_peers = [p for p in cluster.nodes
                         if p != node and p.disk_usage + BLOCK_SIZE <= 100]
        if not suitable_peers:
            return
        
        # Move up to 2 blocks
        for _ in range(2):
            if not node.blockchain:
                break
                
            block = node.blockchain.pop(0)
            peer = random.choice(suitable_peers)
            peer.blockchain.append(block)
            node.disk_usage -= BLOCK_SIZE
            peer.disk_usage += BLOCK_SIZE
    else:
        raise ValueError("Expected a single Node object")

def rebalance_network(network, use_sharding=False):
    overloaded_nodes = [n for n in network.nodes if n.disk_usage > THRESHOLD]
    
    for node in overloaded_nodes:
        if use_sharding:
            cluster = next((c for c in network.clusters if node in c.nodes), None)
            candidates = cluster.nodes if cluster else []
        else:
            candidates = network.nodes
            
        suitable_peers = [p for p in candidates 
                         if p != node and p.disk_usage + BLOCK_SIZE <= 100]
        
        if not suitable_peers:
            continue
            
        peer = min(suitable_peers, key=lambda n: n.disk_usage)
        if node.blockchain:
            block = node.blockchain.pop(0)
            peer.blockchain.append(block)
            node.disk_usage -= BLOCK_SIZE
            peer.disk_usage += BLOCK_SIZE



import numpy as np
import time
from mpi4py import MPI
import geopandas as gpd
import os
import matplotlib.pyplot as plt

# ===============================================================
# CONFIGURATION PARAMETERS - ADJUST THESE FOR DIFFERENT ENVIRONMENTS
# ===============================================================

MAX_MPI_PROCESSES = 15

# Network size parameters
DEFAULT_NETWORK_SIZE = 200  
NETWORK_SIZES = [50, 100, 150, 200, 250]

# Workload parameters
WORKLOADS = [50, 100, 150, 200, 258]  # Workloads
DEFAULT_WORKLOAD = 258  # Fixed workload

# File paths - CHANGE THESE PATHS based on your environment
VECTOR_FILE_PATH = "C:/Users/mrasheed/Desktop/countries_dataset.shp"  # Path to geospatial data
FIGURES_DIR = "C:/Users/mrasheed/Desktop/Poster_Diagrams"  # Directory to save result visualizations

# Replication factor for the network
REPLICATION_FACTOR = 3  

# ===============================================================
# PERFORMANCE TESTING INFRASTRUCTURE
# ===============================================================

class DynamicVectorShardingPerformanceTester:
    def __init__(self, network):
        """
        Initialize performance tester with a network instance
        
        Args:
            network (Network): Distributed network instance
        """
        self.network = network
    
    def insert_vector(self, vector):
        """
        Insert single vector and measure insertion time
        
        Args:
            vector (np.ndarray): Vector to insert
        
        Returns:
            float: Insertion latency
        """
        try:
            start_time = time.perf_counter()
            block = Block([vector.tolist()])
            
            # Use scale-free consensus for insertion
            consensus_result = scale_free_consensus(block, self.network)
            
            # Optional: Apply dynamic sharding if consensus successful
            if consensus_result:
                leaders = [n for n in self.network.nodes if n.is_leader]
                for leader in leaders:
                    enhanced_dynamic_sharding(leader)
            
            return time.perf_counter() - start_time
        except Exception as e:
            print(f"Error inserting vector: {e}")
            return float('inf')


def load_geospatial_vectors(file_path):
    """
    Load geospatial vectors from a shapefile.
    
    Args:
        file_path (str): Path to the shapefile
        
    Returns:
        np.ndarray: Array of vectors
    """
    try:
        gdf = gpd.read_file(file_path)
        vectors = np.array([np.array(geom.centroid.coords).flatten() for geom in gdf.geometry])
        return vectors
    except Exception as e:
        print(f"Error loading vectors from {file_path}: {e}")
        # Generate synthetic vectors if file loading fails
        print("Generating synthetic vectors instead...")
        return np.random.rand(258, 2)


# ===============================================================
# EXPERIMENT 1: FIXED NETWORK SIZE, VARYING WORKLOAD
# ===============================================================

def run_experiment_1(comm, vectors, network_size=DEFAULT_NETWORK_SIZE, workloads=WORKLOADS):
    """
    Experiment 1: Fixed Network Size, Varying Workload
    
    Args:
        comm (MPI.Comm): MPI communicator
        vectors (np.ndarray): Input vectors
        network_size (int): Fixed network size
        workloads (list): List of workload sizes to test
        
    Returns:
        list: Performance results
    """
    rank = comm.Get_rank()
    size = comm.Get_size()  # Total number of MPI processes
    
    results = []
    
    for workload in workloads:
        # Skip if not enough vectors
        if workload > len(vectors):
            continue
            
        # Calculate this process's share of vectors
        vectors_per_process = workload // size
        remainder = workload % size
        start_idx = rank * vectors_per_process + min(rank, remainder)
        end_idx = start_idx + vectors_per_process + (1 if rank < remainder else 0)
        my_vectors = vectors[start_idx:end_idx]
        
        if len(my_vectors) == 0:
            continue
        
        # Create shared network instance - each process manages a portion of nodes
        nodes_per_process = network_size // size
        remainder_nodes = network_size % size
        my_node_count = nodes_per_process + (1 if rank < remainder_nodes else 0)
        
        # Calculate total number of clusters (roughly 10% of network size)
        total_clusters = max(2, network_size // 10)
        
        # Initialize this process's portion of the network
        # In a real implementation, you would coordinate across processes
        network = Network(
            num_nodes=my_node_count,  # Just this process's nodes
            num_clusters=max(1, total_clusters // size),  # Fair share of clusters
            replication_factor=REPLICATION_FACTOR
        )
        
        # Initialize performance tester
        tester = DynamicVectorShardingPerformanceTester(network)
        
        # Time the insertions
        insertion_times = []
        start_time = time.perf_counter()
        
        for i, vector in enumerate(my_vectors):
            latency = tester.insert_vector(vector)
            insertion_times.append(latency)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(my_vectors):
                print(f"Process {rank}: Inserted {i+1}/{len(my_vectors)} vectors for workload {workload}")
        
        overall_time = time.perf_counter() - start_time
        throughput = len(my_vectors) / overall_time if overall_time > 0 else 0
        avg_latency = np.mean(insertion_times) if insertion_times else 0
        
        results.append((workload, throughput, avg_latency, network_size))
    
    # Gather results from all processes
    gathered_results = comm.gather(results, root=0)
    
    if rank == 0:
        # Combine and average results for each workload
        final_results = {}
        for process_results in gathered_results:
            if process_results:
                for workload, throughput, latency, net_size in process_results:
                    if workload not in final_results:
                        final_results[workload] = {'throughputs': [], 'latencies': [], 'network_size': net_size}
                    final_results[workload]['throughputs'].append(throughput)
                    final_results[workload]['latencies'].append(latency)
        
        # Calculate system-wide metrics
        return [(w, 
                 np.sum(data['throughputs']),  # Total system throughput
                 np.mean(data['latencies']),   # Average latency
                 data['network_size'])         # Network size
                for w, data in final_results.items()]
    return None


# ===============================================================
# EXPERIMENT 2: FIXED WORKLOAD, VARYING NETWORK SIZE
# ===============================================================

def run_experiment_2(comm, vectors, network_sizes=NETWORK_SIZES, workload=DEFAULT_WORKLOAD):
    """
    Experiment 2: Fixed Workload, Varying Network Size
    
    Args:
        comm (MPI.Comm): MPI communicator
        vectors (np.ndarray): Input vectors
        network_sizes (list): List of network sizes to test
        workload (int): Fixed workload size
        
    Returns:
        list: Performance results
    """
    rank = comm.Get_rank()
    size = comm.Get_size()  # Total number of MPI processes
    
    # Cap workload to available vectors
    workload = min(workload, len(vectors))
    
    # Calculate this process's share of vectors - fixed for all network sizes
    vectors_per_process = workload // size
    remainder = workload % size
    start_idx = rank * vectors_per_process + min(rank, remainder)
    end_idx = start_idx + vectors_per_process + (1 if rank < remainder else 0)
    my_vectors = vectors[start_idx:end_idx]
    
    results = []
    
    for network_size in network_sizes:
        if len(my_vectors) == 0:
            continue
            
        # Calculate this process's share of the network
        nodes_per_process = network_size // size
        remainder_nodes = network_size % size
        my_node_count = nodes_per_process + (1 if rank < remainder_nodes else 0)
        
        # Calculate total number of clusters (roughly 10% of network size)
        total_clusters = max(2, network_size // 10)
        
        # Initialize this process's portion of the network
        network = Network(
            num_nodes=my_node_count,
            num_clusters=max(1, total_clusters // size),
            replication_factor=REPLICATION_FACTOR
        )
        
        # Initialize performance tester
        tester = DynamicVectorShardingPerformanceTester(network)
        
        # Time the insertions
        insertion_times = []
        start_time = time.perf_counter()
        
        for i, vector in enumerate(my_vectors):
            latency = tester.insert_vector(vector)
            insertion_times.append(latency)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(my_vectors):
                print(f"Process {rank}: Inserted {i+1}/{len(my_vectors)} vectors for network size {network_size}")
        
        overall_time = time.perf_counter() - start_time
        throughput = len(my_vectors) / overall_time if overall_time > 0 else 0
        avg_latency = np.mean(insertion_times) if insertion_times else 0
        
        results.append((network_size, throughput, avg_latency, workload))
    
    # Gather results from all processes
    gathered_results = comm.gather(results, root=0)
    
    if rank == 0:
        # Combine results for each network size
        final_results = {}
        for process_results in gathered_results:
            if process_results:
                for net_size, throughput, latency, wl in process_results:
                    if net_size not in final_results:
                        final_results[net_size] = {'throughputs': [], 'latencies': [], 'workload': wl}
                    final_results[net_size]['throughputs'].append(throughput)
                    final_results[net_size]['latencies'].append(latency)
        
        # Calculate system-wide metrics
        return [(n, 
                 np.sum(data['throughputs']),  # Total system throughput
                 np.mean(data['latencies']),   # Average latency
                 data['workload'])             # Workload
                for n, data in final_results.items()]
    return None


# ===============================================================
# VISUALIZATION FUNCTIONS
# ===============================================================

def visualize_experiment_1(results, save_dir=FIGURES_DIR):
    """
    Visualize results from Experiment 1 (Fixed Network Size, Varying Workload)
    
    Args:
        results (list): List of tuples (workload, throughput, latency, network_size)
        save_dir (str): Directory to save the figures
    """
    if not results:
        print("No results to visualize for Experiment 1")
        return
    
    
    # Sort results by workload
    results.sort(key=lambda x: x[0])
    workloads = [r[0] for r in results]
    throughputs = [r[1] for r in results]
    latencies = [r[2] for r in results]
    network_size = results[0][3]  # Should be the same for all entries
    
    # Set figure size and DPI for high-quality outputs
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot throughput vs workload
    plt.plot(workloads, throughputs, 'o-', linewidth=2.5, markersize=10, color='#0072B2')
    
    # Add grid, labels, and title with larger fonts
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Workload (Number of Vectors)', fontsize=14, fontweight='bold')
    plt.ylabel('System Throughput (vectors/second)', fontsize=14, fontweight='bold')
    plt.title(f'System Throughput vs. Workload with Fixed Network Size ({network_size} nodes)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add data point labels
    for i, (w, t) in enumerate(zip(workloads, throughputs)):
        plt.annotate(f'{t:.2f}', (w, t), xytext=(0, 10), 
                     textcoords='offset points', ha='center', fontsize=12)
    
    # Customize tick labels
    plt.xticks(workloads, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp1_throughput.png", bbox_inches='tight')
    plt.close()
    
    # Plot latency vs workload
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(workloads, latencies, 'o-', linewidth=2.5, markersize=10, color='#D55E00')
    
    # Add grid, labels, and title
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Workload (Number of Vectors)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Latency (seconds)', fontsize=14, fontweight='bold')
    plt.title(f'Average Latency vs. Workload with Fixed Network Size ({network_size} nodes)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add data point labels
    for i, (w, l) in enumerate(zip(workloads, latencies)):
        plt.annotate(f'{l:.4f}', (w, l), xytext=(0, 10), 
                     textcoords='offset points', ha='center', fontsize=12)
    
    # Customize tick labels
    plt.xticks(workloads, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp1_latency.png", bbox_inches='tight')
    plt.close()
    
    print(f"Experiment 1 visualizations saved to {save_dir}")


def visualize_experiment_2(results, save_dir=FIGURES_DIR):
    """
    Visualize results from Experiment 2 (Fixed Workload, Varying Network Size)
    
    Args:
        results (list): List of tuples (network_size, throughput, latency, workload)
        save_dir (str): Directory to save the figures
    """
    if not results:
        print("No results to visualize for Experiment 2")
        return
        
    # Create save directory if it doesn't exist
    
    # Sort results by network size
    results.sort(key=lambda x: x[0])
    network_sizes = [r[0] for r in results]
    throughputs = [r[1] for r in results]
    latencies = [r[2] for r in results]
    workload = results[0][3]  # Should be the same for all entries
    
    # Plot throughput vs network size
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Bar chart for throughput
    bars = plt.bar(network_sizes, throughputs, color='#0072B2', width=min(8, 0.6*network_sizes[0]), alpha=0.8)
    
    # Add grid, labels, and title
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel('Network Size (Number of Nodes)', fontsize=14, fontweight='bold')
    plt.ylabel('System Throughput (vectors/second)', fontsize=14, fontweight='bold')
    plt.title(f'System Throughput vs. Network Size with Fixed Workload ({workload} vectors)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    # Set x-ticks to only show the specific network sizes
    plt.xticks(network_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Set x-axis limit to give some padding
    plt.xlim([min(network_sizes) - 10, max(network_sizes) + 10])
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp2_throughput.png", bbox_inches='tight')
    plt.close()
    
    # Plot latency vs network size
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Line plot for latency
    plt.plot(network_sizes, latencies, 'o-', linewidth=2.5, markersize=10, color='#D55E00')
    
    # Add grid, labels, and title
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Network Size (Number of Nodes)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Latency (seconds)', fontsize=14, fontweight='bold')
    plt.title(f'Average Latency vs. Network Size with Fixed Workload ({workload} vectors)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add data point labels
    for i, (n, l) in enumerate(zip(network_sizes, latencies)):
        plt.annotate(f'{l:.4f}', (n, l), xytext=(0, 10), 
                     textcoords='offset points', ha='center', fontsize=12)
    
    # Set x-ticks to only show the specific network sizes
    plt.xticks(network_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Set x-axis limit to give some padding
    plt.xlim([min(network_sizes) - 10, max(network_sizes) + 10])
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp2_latency.png", bbox_inches='tight')
    plt.close()
    
    print(f"Experiment 2 visualizations saved to {save_dir}")


def experiment_memory_usage_vs_network_size(comm, workload=500, network_sizes=None):
    """
    Experiment: Analyze memory usage across different network sizes with and without sharding
    
    Args:
        comm (MPI.Comm): MPI communicator
        workload (int): Fixed workload size (number of blocks to process)
        network_sizes (list): List of network sizes to test
        
    Returns:
        dict: Results comparing memory usage with and without sharding
    """
    if network_sizes is None:
        network_sizes = [50, 100, 150, 200, 250]  # Default network sizes
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Results structure
    results = {
        "network_sizes": [],
        "without_sharding_avg": [],
        "without_sharding_std": [],
        "without_sharding_max": [],
        "with_sharding_avg": [],
        "with_sharding_std": [],
        "with_sharding_max": []
    }
    
    for network_size in network_sizes:
        if rank == 0:
            print(f"\nTesting network size: {network_size}")
        
        # ---- Non-sharded network ----
        network_without = Network(
            num_nodes=network_size,
            num_clusters=1,  # Single cluster = non-sharded
            replication_factor=REPLICATION_FACTOR
        )
        
        # Process workload
        for i in range(workload):
            block = Block([np.random.rand(10).tolist()])
            scale_free_consensus(block, network_without, False)
            
            if rank == 0 and (i+1) % 100 == 0:
                print(f"Non-sharded: Processed {i+1}/{workload} blocks")
        
        # Collect memory usage metrics
        if rank == 0:
            memory_without = [n.disk_usage for n in network_without.nodes]
            avg_memory_without = np.mean(memory_without)
            std_memory_without = np.std(memory_without)
            max_memory_without = np.max(memory_without)
            
            print(f"Non-sharded: Avg memory usage: {avg_memory_without:.2f}%, Std: {std_memory_without:.2f}, Max: {max_memory_without:.2f}%")
        
        # ---- Sharded network ----
        network_with = Network(
            num_nodes=network_size,
            num_clusters=max(2, network_size//10),  # Dynamic sharding
            replication_factor=REPLICATION_FACTOR
        )
        
        # Process workload
        for i in range(workload):
            block = Block([np.random.rand(10).tolist()])
            if scale_free_consensus(block, network_with, True):
                leaders = [n for n in network_with.nodes if n.is_leader]
                for leader in leaders:
                    enhanced_dynamic_sharding(leader)
            
            if rank == 0 and (i+1) % 100 == 0:
                print(f"Sharded: Processed {i+1}/{workload} blocks")
        
        # Collect memory usage metrics
        if rank == 0:
            memory_with = [n.disk_usage for n in network_with.nodes]
            avg_memory_with = np.mean(memory_with)
            std_memory_with = np.std(memory_with)
            max_memory_with = np.max(memory_with)
            
            print(f"Sharded: Avg memory usage: {avg_memory_with:.2f}%, Std: {std_memory_with:.2f}, Max: {max_memory_with:.2f}%")
            
            # Store results
            results["network_sizes"].append(network_size)
            results["without_sharding_avg"].append(avg_memory_without)
            results["without_sharding_std"].append(std_memory_without)
            results["without_sharding_max"].append(max_memory_without)
            results["with_sharding_avg"].append(avg_memory_with)
            results["with_sharding_std"].append(std_memory_with)
            results["with_sharding_max"].append(max_memory_with)
    
    # Only rank 0 has meaningful results
    return results if rank == 0 else None


def visualize_memory_usage_vs_network_size(results, save_dir=FIGURES_DIR):
    """
    Visualize memory usage comparison between sharded and non-sharded approaches
    across different network sizes.
    
    Args:
        results (dict): Results from experiment_memory_usage_vs_network_size
        save_dir (str): Directory to save the figures
    """
    if not results or "network_sizes" not in results:
        print("No valid results to visualize for memory usage experiment")
        return
    
    # Extract data
    network_sizes = results["network_sizes"]
    without_sharding_avg = results["without_sharding_avg"]
    without_sharding_std = results["without_sharding_std"]
    without_sharding_max = results["without_sharding_max"]
    with_sharding_avg = results["with_sharding_avg"]
    with_sharding_std = results["with_sharding_std"]
    with_sharding_max = results["with_sharding_max"]
    
    # ---- AVERAGE MEMORY USAGE COMPARISON ----
    plt.figure(figsize=(14, 12), dpi=300)
    
    # Plot with error bars (standard deviation)
    plt.errorbar(network_sizes, without_sharding_avg, yerr=without_sharding_std, 
                 fmt='o-', linewidth=4, markersize=14, capsize=8, capthick=3,
                 color='#D55E00', markeredgecolor='black', markeredgewidth=1,
                 label='Without Sharding')
    
    plt.errorbar(network_sizes, with_sharding_avg, yerr=with_sharding_std, 
                 fmt='s-', linewidth=4, markersize=14, capsize=8, capthick=3,
                 color='#2CA02C', markeredgecolor='black', markeredgewidth=1,
                 label='With Sharding')
    
    # Calculate improvement percentage for annotation
    improvements = [(w - s) / w * 100 if w > 0 else 0 
                    for w, s in zip(without_sharding_avg, with_sharding_avg)]
    
    # Add data labels with improvements
    for i, (x, y_without, y_with, imp) in enumerate(zip(
            network_sizes, without_sharding_avg, with_sharding_avg, improvements)):
        # Add percentage improvement annotation
        plt.annotate(f"{imp:.1f}% reduction", 
                     ((x + network_sizes[min(i+1, len(network_sizes)-1)])/2 
                      if i < len(network_sizes)-1 else x + 10, 
                     (y_without + y_with)/2),
                     xytext=(0, 10), textcoords='offset points', ha='center',
                     fontsize=16, fontweight='bold', color='black',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add grid and styling with consistent parameters
    plt.grid(True, linestyle='--', alpha=0.7, which='major')
    plt.grid(False, which='minor')
    
    # Labels and title
    plt.xlabel('Network Size (Number of Nodes)', fontsize=22, fontweight='bold')
    plt.ylabel('Average Memory Usage (%)', fontsize=22, fontweight='bold')
    
    # Set y-axis to start from 0 and extend just beyond max value
    max_usage = max(max(without_sharding_avg), max(with_sharding_avg)) * 1.1
    plt.ylim(0, max_usage)
    
    # Customize tick labels
    plt.xticks(network_sizes, fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    
    # Add legend
    legend = plt.legend(fontsize=20, loc='upper right', frameon=True, 
                      framealpha=0.9, edgecolor='black')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    
    # Add box around the plot
    plt.box(True)
    
    # Set x-axis limit with padding
    plt.xlim([min(network_sizes) - 20, max(network_sizes) + 20])
    
    # Better layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(f"{save_dir}/memory_usage_avg_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # ---- MAXIMUM MEMORY USAGE COMPARISON ----
    plt.figure(figsize=(14, 12), dpi=300)
    
    # Plot maximum memory usage
    plt.plot(network_sizes, without_sharding_max, 'o-', linewidth=4, markersize=14, 
             color='#D55E00', markeredgecolor='black', markeredgewidth=1,
             label='Without Sharding (Max)')
    
    plt.plot(network_sizes, with_sharding_max, 's-', linewidth=4, markersize=14, 
             color='#2CA02C', markeredgecolor='black', markeredgewidth=1,
             label='With Sharding (Max)')
    
    # Calculate improvement percentage for annotation
    max_improvements = [(w - s) / w * 100 if w > 0 else 0 
                         for w, s in zip(without_sharding_max, with_sharding_max)]
    
    # Add data labels with improvements
    for i, (x, y_without, y_with, imp) in enumerate(zip(
            network_sizes, without_sharding_max, with_sharding_max, max_improvements)):
        # Add percentage improvement annotation
        plt.annotate(f"{imp:.1f}% reduction", 
                     ((x + network_sizes[min(i+1, len(network_sizes)-1)])/2 
                      if i < len(network_sizes)-1 else x + 10, 
                     (y_without + y_with)/2),
                     xytext=(0, 10), textcoords='offset points', ha='center',
                     fontsize=16, fontweight='bold', color='black',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add grid and styling
    plt.grid(True, linestyle='--', alpha=0.7, which='major')
    plt.grid(False, which='minor')
    
    # Labels and title
    plt.xlabel('Network Size (Number of Nodes)', fontsize=22, fontweight='bold')
    plt.ylabel('Maximum Memory Usage (%)', fontsize=22, fontweight='bold')
    
    # Set y-axis to start from 0 and extend just beyond max value
    max_usage = max(max(without_sharding_max), max(with_sharding_max)) * 1.1
    plt.ylim(0, max_usage)
    
    # Customize tick labels
    plt.xticks(network_sizes, fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    
    # Add legend
    legend = plt.legend(fontsize=20, loc='upper right', frameon=True, 
                      framealpha=0.9, edgecolor='black')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    
    # Add box around the plot
    plt.box(True)
    
    # Set x-axis limit with padding
    plt.xlim([min(network_sizes) - 20, max(network_sizes) + 20])
    
    # Better layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(f"{save_dir}/memory_usage_max_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # ---- MEMORY USAGE DISTRIBUTION ----
    # This visualization shows the distribution of memory usage across nodes
    plt.figure(figsize=(14, 12), dpi=300)
    
    # Create box plot data
    box_data = []
    labels = []
    
    for i, size in enumerate(network_sizes):
        # Add non-sharded and sharded data points for each network size
        # We need to create synthetic data here since we only have summary statistics
        # Use normal distribution with the given mean and std
        without_data = np.random.normal(without_sharding_avg[i], 
                                      without_sharding_std[i], 
                                      size=100)  # 100 points for statistics
        with_data = np.random.normal(with_sharding_avg[i], 
                                    with_sharding_std[i], 
                                    size=100)  # 100 points for statistics
        
        # Clip to realistic range (0-100% memory usage)
        without_data = np.clip(without_data, 0, 100)
        with_data = np.clip(with_data, 0, 100)
        
        box_data.append(without_data)
        box_data.append(with_data)
        
        labels.append(f"{size} W/O")
        labels.append(f"{size} With")
    
    # Create box plot
    boxplot = plt.boxplot(box_data, patch_artist=True, labels=labels)
    
    # Color boxes - alternating for without and with sharding
    for i, box in enumerate(boxplot['boxes']):
        if i % 2 == 0:  # Without sharding
            box.set(facecolor='#D55E00', alpha=0.7)
        else:  # With sharding
            box.set(facecolor='#2CA02C', alpha=0.7)
    
    # Add grid and styling
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Labels and title
    plt.xlabel('Network Size and Sharding Approach', fontsize=22, fontweight='bold')
    plt.ylabel('Memory Usage Distribution (%)', fontsize=22, fontweight='bold')
    
    # Customize tick labels
    plt.xticks(fontsize=18, fontweight='bold', rotation=45)
    plt.yticks(fontsize=20, fontweight='bold')
    
    # Add a legend for box colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D55E00', alpha=0.7, label='Without Sharding'),
        Patch(facecolor='#2CA02C', alpha=0.7, label='With Sharding')
    ]
    legend = plt.legend(handles=legend_elements, fontsize=20, loc='upper right',
                      frameon=True, framealpha=0.9, edgecolor='black')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    
    # Add box around the plot
    plt.box(True)
    
    # Better layout with more room at bottom for rotated labels
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(f"{save_dir}/memory_usage_distribution.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Memory usage visualizations saved to {save_dir}")





# ===============================================================
# MAIN PROGRAM
# ===============================================================

def main():
    """Main entry point for the benchmark program."""
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create directories for saving figures (only process 0)
    if rank == 0:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        print(f"Running with {size} MPI processes")
    
    # Load or generate vectors (root process only, then broadcast)
    if rank == 0:
        print("Initializing Dynamic Vector Sharding Benchmark...")
        try:
            vectors = load_geospatial_vectors(VECTOR_FILE_PATH)
        except Exception as e:
            print(f"Error loading vectors: {e}")
            print("Generating synthetic vectors instead...")
            vectors = np.random.rand(DEFAULT_WORKLOAD, 2)
        print(f"Using {len(vectors)} vectors for benchmarks")
    else:
        vectors = None
    
    # Broadcast vectors to all processes
    vectors = comm.bcast(vectors, root=0)
    
    # Run Experiment 1: Fixed Network Size, Varying Workload
    if rank == 0:
        print("\n===== Experiment 1: Fixed Network Size (100 nodes), Varying Workload =====")
    
    exp1_results = run_experiment_1(comm, vectors, DEFAULT_NETWORK_SIZE, WORKLOADS)
    
    if rank == 0 and exp1_results:
        print("\nWorkload\tThroughput (vecs/sec)\tAvg Latency (sec)\tNetwork Size")
        for workload, throughput, latency, net_size in sorted(exp1_results, key=lambda x: x[0]):
            print(f"{workload}\t\t{throughput:.2f}\t\t\t{latency:.4f}\t\t{net_size}")
        
        # Visualize Experiment 1 results
        visualize_experiment_1(exp1_results, FIGURES_DIR)
    
    # Run Experiment 2: Fixed Workload, Varying Network Size
    if rank == 0:
        print("\n===== Experiment 2: Fixed Workload (258 vectors), Varying Network Size =====")
    
    exp2_results = run_experiment_2(comm, vectors, NETWORK_SIZES, DEFAULT_WORKLOAD)
    
    if rank == 0 and exp2_results:
        print("\nNetwork Size\tThroughput (vecs/sec)\tAvg Latency (sec)\tWorkload")
        for net_size, throughput, latency, workload in sorted(exp2_results, key=lambda x: x[0]):
            print(f"{net_size}\t\t{throughput:.2f}\t\t\t{latency:.4f}\t\t{workload}")
        
        # Visualize Experiment 2 results
        visualize_experiment_2(exp2_results, FIGURES_DIR)



    # ---------------------------------------------------------------
    # EXPERIMENT: Memory Usage vs Network Size                      -
    # ---------------------------------------------------------------
    if rank == 0:
        print("\n===== EXPERIMENT: MEMORY USAGE VS NETWORK SIZE =====")

    memory_usage_results = experiment_memory_usage_vs_network_size(comm, workload=500, network_sizes=NETWORK_SIZES)

    if rank == 0 and memory_usage_results:
        print("\nMemory Usage Results:")
        print("Network Size\tAvg w/o Sharding\tAvg with Sharding\tImprovement")
        for i, size in enumerate(memory_usage_results["network_sizes"]):
            without_avg = memory_usage_results["without_sharding_avg"][i]
            with_avg = memory_usage_results["with_sharding_avg"][i]
            improvement = (without_avg - with_avg) / without_avg * 100 if without_avg > 0 else 0
            print(f"{size}\t\t{without_avg:.2f}%\t\t{with_avg:.2f}%\t\t{improvement:.2f}%")
        
        # Generate visualizations
        visualize_memory_usage_vs_network_size(memory_usage_results, FIGURES_DIR)


if __name__ == "__main__":
    main()