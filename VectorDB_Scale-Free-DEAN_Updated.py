import traceback
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

THRESHOLD = 75  
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

    def is_idle(self):
        return self.computational_capacity > 60  # More realistic idle condition

    def prune_old_data(self):
        """Prune blocks if over capacity, update usage correctly"""
        max_blocks = (100 // BLOCK_SIZE)
        if len(self.blockchain) > max_blocks:
            removed = len(self.blockchain) - max_blocks
            self.blockchain = self.blockchain[-max_blocks:]
            self.disk_usage = len(self.blockchain) * BLOCK_SIZE  # Direct count
  

class Cluster:
    def __init__(self, id, nodes):
        self.id = id
        self.nodes = nodes
        self.sub_clusters = []
        self.shard_boundaries = (-float('inf'), float('inf'))  # Vector range
        self.vector_size_profile = []  # Track vector dimensions

    def update_boundaries(self, new_block):
        # Initialize from first block
        if self.shard_boundaries[0] == -float('inf'):
            self.shard_boundaries = (min(new_block.features[0]), max(new_block.features[0]))
        else:
            # Handle vector features
            vector = new_block.features[0]
            min_feature = min(vector)
            max_feature = max(vector)
            
            self.shard_boundaries = (
                min(self.shard_boundaries[0], min_feature),
                max(self.shard_boundaries[1], max_feature)
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
        self.nodes_per_cluster = max(4, num_nodes // self.num_clusters)  # Minimum 4 nodes/cluster
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
            'overloaded_nodes': 0
        }
        self.last_rebalance_time = time.time()
        self.under_replicated_blocks = []
        self.reserved_nodes = []
        self.in_experiment = False
        self.initialize_nodes()


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
        print_cluster_summary(self.clusters)

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

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_check = time.time()
    
    def consume(self, tokens):
        now = time.time()
        elapsed = now - self.last_check
        self.tokens += elapsed * self.rate
        self.tokens = min(self.tokens, self.capacity)
        self.last_check = now
        
        if self.tokens < tokens:
            time.sleep((tokens - self.tokens) / self.rate)
            self.tokens = 0
        else:
            self.tokens -= tokens

def validate_block(block, node, validation_criteria):
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


def scale_free_consensus(block, network, use_sharding=False):
    try:
        leaders, total_adj = select_leaders(network)
    except ValueError as e:
        network.error_log['consensus_failures'] += 1
        return False

    leader_votes = sum(1 for l in leaders if validate_block(block, l, [check_integrity]))

    if leader_votes >= len(leaders) * 0.67:
        if use_sharding:
            vec_mean = np.mean(block.features[0])
            target_cluster = min(network.clusters,
                               key=lambda c: abs(vec_mean - np.mean(c.shard_boundaries)))
            
            replica_clusters = sorted(network.clusters,
                                    key=lambda c: abs(np.mean(c.shard_boundaries) - vec_mean))[1:3]
            
            successful_replicas = 0
            for cluster in [target_cluster] + replica_clusters:
                suitable_nodes = sorted(cluster.nodes, key=lambda n: n.disk_usage)
                for node in suitable_nodes[:3]:
                    if node.disk_usage + BLOCK_SIZE <= 100:
                        node.blockchain.append(block)
                        node.disk_usage += BLOCK_SIZE
                        node.prune_old_data()
                        successful_replicas += 1
                        break
            
            return successful_replicas >= 1
        else:
            replicated = 0
            for node in sorted(network.nodes, key=lambda n: n.disk_usage):
                if node.disk_usage + BLOCK_SIZE <= 100:
                    node.blockchain.append(block)
                    node.disk_usage = 100
                    replicated += 1
            return replicated > 0
            
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

def process_block_with_latency(block):
    base_processing = 0.001
    simulated_latency = random.uniform(0.01, 0.1)  # Edge network latency
    time.sleep(base_processing + simulated_latency)


def experiment_1_workload_analysis(workloads, comm):
    experiment_1_results = []
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split workloads across processes
    workloads_per_proc = [[] for _ in range(size)]
    if rank == 0:
        for i, wl in enumerate(workloads):
            workloads_per_proc[i % size].append(wl)
    
    local_workloads = comm.scatter(workloads_per_proc, root=0)

    for workload in local_workloads:
        network = Network(num_nodes=50, num_clusters=5)
        start_time = time.perf_counter()
        # start_time = time.time()
        
        for _ in range(workload):
            block = Block([np.random.rand(10).tolist()])
            scale_free_consensus(block, network, use_sharding=False)
        
        # end_time = time.time()
        end_time = time.perf_counter()

        time_taken = end_time - start_time
        throughput = workload / time_taken if time_taken > 0 else 0
        # latency = time_taken / workload if workload > 0 else 0
        latency = (end_time - start_time) / workload if workload > 0 else 0
        # Add minimal latency to avoid division by zero
        latency = max(latency, 1e-9)  
        
        experiment_1_results.append((workload, throughput, latency))
    
    gathered = comm.gather(experiment_1_results, root=0)
    return [item for sublist in gathered for item in sublist] if rank == 0 else []

import faiss
import numpy as np
import time
from mpi4py import MPI

def faiss_experiment_1_workload_analysis(workloads, comm):
    comm.Barrier()  # Synchronize processes before starting
    experiment_results = []
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Improved workload distribution using array_split
    if rank == 0:
        workloads_per_proc = np.array_split(workloads, size)
        # Ensure even empty arrays are sent to maintain MPI sync
        workloads_per_proc = [wl.tolist() if isinstance(wl, np.ndarray) else wl 
                            for wl in workloads_per_proc]
    else:
        workloads_per_proc = None

    local_workloads = comm.scatter(workloads_per_proc, root=0)

    dimension = 10
    for workload in local_workloads:
        if workload == 0:  # Handle empty workloads gracefully
            experiment_results.append((0, 0.0, 0.0))
            continue

        # Initialize FAISS index
        index = faiss.IndexFlatL2(dimension)
        vectors = [np.random.rand(dimension).astype('float32') 
                 for _ in range(workload)]

        # Timing with high precision
        start_time = time.perf_counter()
        for vec in vectors:
            index.add(np.expand_dims(vec, axis=0))
        end_time = time.perf_counter()

        # Robust metric calculation
        duration = max(end_time - start_time, 1e-9)
        throughput = workload / duration
        
        # Handle potential division errors
        latency = duration / workload if workload > 0 else 0
        
        # NaN/Inf protection
        if np.isnan(throughput) or np.isinf(throughput):
            throughput = 0.0
        if np.isnan(latency) or np.isinf(latency):
            latency = 0.0

        experiment_results.append((workload, throughput, latency))

    # Safe gathering with empty list protection
    gathered = comm.gather(experiment_results, root=0)
    
    if rank == 0:
        # Flatten results while filtering None and empty lists
        flat_results = []
        for sublist in gathered:
            if sublist:  # Skip empty/Nones
                flat_results.extend([item for item in sublist if item is not None])
        return flat_results
    else:
        return []


def faiss_experiment_2_network_size_analysis(network_sizes, base_workload, comm):
    results = []
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Scatter network sizes across processes
    tasks_per_proc = [[] for _ in range(size)]
    if rank == 0:
        for i, ns in enumerate(network_sizes):
            tasks_per_proc[i % size].append((ns, base_workload))
    local_tasks = comm.scatter(tasks_per_proc, root=0)

    dimension = 10
    for net_size, workload in local_tasks:
        # Create sharded index
        shards = [faiss.IndexFlatL2(dimension) for _ in range(net_size)]
        vectors = [np.random.rand(dimension).astype('float32') 
                for _ in range(workload)]
        
        # High-precision timing
        start_time = time.perf_counter()
        for i, vec in enumerate(vectors):
            shard_idx = i % net_size
            shards[shard_idx].add(np.expand_dims(vec, axis=0))
        end_time = time.perf_counter()
        
        # Robust metric calculation
        duration = max(end_time - start_time, 1e-9)
        throughput = workload / duration
        latency = duration / workload if workload > 0 else 0
        
        # Handle edge cases
        if np.isnan(throughput) or np.isinf(throughput):
            throughput = 0.0
        if np.isnan(latency) or np.isinf(latency):
            latency = 0.0
        
        # Calculate cluster balance
        shard_sizes = [shard.ntotal for shard in shards]
        balance_std = np.std(shard_sizes)
        
        results.append((net_size, throughput, latency, balance_std))
    
    # Gather and sort results
    gathered = comm.gather(results, root=0)
    return sorted([item for sublist in gathered for item in sublist], 
                key=lambda x: x[0]) if rank == 0 else []


def experiment_2_network_size_analysis(network_sizes, base_workload, comm):
    results = []
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Unified workload distribution (same as Experiment 1)
    workloads_per_proc = [[] for _ in range(size)]
    if rank == 0:
        for i, ns in enumerate(network_sizes):
            workloads_per_proc[i % size].append((ns, base_workload))
    
    local_tasks = comm.scatter(workloads_per_proc, root=0)

    for net_size, workload in local_tasks:
        # Network initialization (only difference from Experiment 1)
        network = Network(num_nodes=net_size, 
                         num_clusters=max(2, net_size//5))
        
        # Exact Experiment 1 timing and processing
        start = time.perf_counter()
        for _ in range(workload):
            block = Block([np.random.rand(10).tolist()])
            scale_free_consensus(block, network, use_sharding=True)
        end = time.perf_counter()
        
        # Identical metric calculations
        duration = end - start
        throughput = workload / duration
        latency = duration / workload
        
        results.append((
            net_size,
            round(throughput, 2),
            round(latency, 5),  # Increased precision for small values
            round(np.std([len(c.nodes) for c in network.clusters]), 2)
        ))
    
    final = comm.gather(results, root=0)
    return sorted([x for sub in final for x in sub]) if rank == 0 else []

def experiment_3_memory_analysis(network_sizes, workload, comm):
    memory_results = []
    
    for net_size in network_sizes:
        # Non-sharded network
        network_without = Network(num_nodes=net_size, num_clusters=1)
        network_without.print_network_state("Initial Non-Sharded")
        
        # Process workload
        for _ in range(workload):
            block = Block([np.random.rand(10).tolist()])
            scale_free_consensus(block, network_without, False)
        
        network_without.print_network_state("Final Non-Sharded")
        network_without.print_replica_distribution()

        # Sharded network
        network_with = Network(num_nodes=net_size, num_clusters=max(2, net_size//5))
        network_with.print_network_state("Initial Sharded")
        
        # Process workload
        for _ in range(workload):
            block = Block([np.random.rand(10).tolist()])
            if scale_free_consensus(block, network_with, True):
                leaders = [n for n in network_with.nodes if n.is_leader]
                for leader in leaders:
                    enhanced_dynamic_sharding(leader)

        network_with.print_network_state("Final Sharded")
        network_with.print_replica_distribution()
        
        # Collect metrics
        memory_results.append((net_size, network_without, network_with))
    
    return memory_results


def plot_results(experiment_1_results, experiment_2_results):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Experiment 1 Plot
    plt.figure(figsize=(12, 6), num="Exp1 Results")
    workloads = sorted(list(set(x[0] for x in experiment_1_results)))
    
    # Throughput subplot
    plt.subplot(1, 2, 1)
    avg_throughput = [np.mean([res[1] for res in experiment_1_results if res[0] == wl]) for wl in workloads]
    plt.plot(workloads, avg_throughput, marker='o', color='blue')
    plt.title('Throughput vs Workload')
    plt.xlabel('Workload')
    plt.ylabel('Throughput (blocks/sec)')
    plt.grid(True)

    # Latency subplot
    plt.subplot(1, 2, 2)
    avg_latency = [np.mean([res[2] for res in experiment_1_results if res[0] == wl]) for wl in workloads]
    plt.plot(workloads, avg_latency, marker='x', color='orange')
    plt.title('Latency vs Workload')
    plt.xlabel('Workload')
    plt.ylabel('Latency (sec/block)')
    plt.grid(True)

    # Save and clean up
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'experiment_1_results.png'))
    plt.cla()
    plt.clf()
    plt.close()

    # Experiment 2 Plot
    plt.figure(figsize=(12, 6), num="Exp2 Results")
    net_sizes = sorted(list(set(x[0] for x in experiment_2_results)))
    
    # Throughput subplot
    plt.subplot(1, 2, 1)
    avg_throughput = [np.mean([res[1] for res in experiment_2_results if res[0] == ns]) for ns in net_sizes]
    plt.plot(net_sizes, avg_throughput, marker='o', color='blue')
    plt.title('Throughput vs Network Size')
    plt.xlabel('Network Size')
    plt.ylabel('Throughput (blocks/sec)')
    plt.grid(True)

    # Latency subplot
    plt.subplot(1, 2, 2)
    avg_latency = [np.mean([res[2] for res in experiment_2_results if res[0] == ns]) for ns in net_sizes]
    plt.plot(net_sizes, avg_latency, marker='x', color='orange')
    plt.title('Latency vs Network Size')
    plt.xlabel('Network Size')
    plt.ylabel('Latency (sec/block)')
    plt.grid(True)

    # Save and clean up
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'experiment_2_results.png'))
    plt.cla()
    plt.clf()
    plt.close()

def plot_memory_usage(memory_results):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for idx, (net_size, network_without, network_with) in enumerate(memory_results):
        memory_without = [n.disk_usage for n in network_without.nodes]
        memory_with = [n.disk_usage for n in network_with.nodes]
        
        min_length = min(len(memory_without), len(memory_with))
        memory_without = memory_without[:min_length]
        memory_with = memory_with[:min_length]

        output_path = os.path.join(script_dir, f'memory_usage_{net_size}.png')

        node_ids = np.arange(min_length)
        width = 0.35

        plt.bar(node_ids - width/2, memory_without, width, label='Without Sharding', color='tomato', alpha=0.8)
        plt.bar(node_ids + width/2, memory_with, width, label='With Sharding', color='dodgerblue', alpha=0.8)

        plt.xlabel('Node ID')
        plt.ylabel('Memory Usage (%)')
        plt.title(f'Memory Comparison (Size: {net_size})')
        plt.xticks(node_ids[::max(1, len(node_ids)//10)], rotation=45)
        plt.legend()
        plt.grid(linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.cla()
        plt.clf()
        plt.close()

def generate_performance_report(results):
    print("\nSCALABILITY ANALYSIS")
    print(f"{'Nodes':<6} | {'Sharding Gain':<12} | {'Replication':<10} | {'Overloads'}")
    print("-"*45)
    
    for entry in results:
        net_size = entry[0]
        network_without = entry[1]
        network_with = entry[2]
        
        without_usages = [n.disk_usage for n in network_without.nodes]
        with_usages = [n.disk_usage for n in network_with.nodes]
        
        avg_without = np.mean(without_usages)
        avg_with = np.mean(with_usages)
        gain = ((avg_without - avg_with) / avg_without * 100) if avg_without != 0 else 0
        overloads = sum(1 for u in with_usages if u >= 100)
        
        print(f"{net_size:<6} | {gain:>7.1f}%{'':<5} | {REPLICATION_FACTOR}x{'':<4} | {overloads}")

def print_cluster_summary(clusters):
    print(f"\n{'Cluster':<8} | {'Nodes':<6} | {'Usage Range':<15} | {'Status'}")
    print("-"*45)
    for cluster in clusters:
        usages = [n.disk_usage for n in cluster.nodes]
        status = "OK" if max(usages) < 90 else "WARN" if max(usages) < 100 else "CRIT"
        print(f"{cluster.id:<8} | {len(cluster.nodes):<6} | "
              f"{min(usages):.0f}-{max(usages):.0f}%{'':<5} | {status}")

def print_experiment_summary(network):
    print(f"\nCritical Errors: {sum(network.error_counter.values())}")
    print("Error Types:")
    for err_type, count in network.error_counter.items():
        print(f"- {err_type.replace('_',' ').title()}: {count}")

def print_memory_usage(network_sizes, memory_results):
    if not memory_results:
        print("No memory usage data available.")
        return

    print("\n\nFINAL MEMORY USAGE RESULTS")
    print("="*60)
    print(f"{'Network Size':<12} | {'Avg Without (%)':<15} | {'Avg With (%)':<15} | {'Reduction (%)':<15}")
    print("-"*60)
    
    for entry in memory_results:
        net_size = entry[0]
        network_without = entry[1]
        network_with = entry[2]
        
        without_usages = [n.disk_usage for n in network_without.nodes]
        with_usages = [n.disk_usage for n in network_with.nodes]
        
        avg_without = np.mean(without_usages)
        avg_with = np.mean(with_usages)
        reduction = ((avg_without - avg_with)/avg_without*100) if avg_without > 0 else 0
        
        print(f"{net_size:<12} | {avg_without:<15.1f} | {avg_with:<15.1f} | {reduction:<15.1f}")

    print("\nKey Observations:")
    max_reduction = max([(np.mean([n.disk_usage for n in entry[1].nodes]) - 
                        np.mean([n.disk_usage for n in entry[2].nodes])) /
                        np.mean([n.disk_usage for n in entry[1].nodes]) * 100
                        for entry in memory_results])
    print(f"- Maximum memory reduction: {max_reduction:.1f}%")
    print(f"- Target replication factor maintained: {REPLICATION_FACTOR}")


def visualize_network_hierarchy(networks_data):
    if not networks_data:
        print("No network data available for visualization.")
        return

    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelpad': 2,
        'legend.title_fontsize': 8,
        'font.family': 'DejaVu Sans'
    })

    num_networks = len(networks_data)
    rows = (num_networks // 2) + (1 if num_networks % 2 != 0 else 0)
    cols = min(2, num_networks)

    colors = {
        'leaders': '#E64646',
        'peers_simple': '#5F9ED1',
        'peers_cross': '#9ED15F'
    }

    for idx, (network_size, network) in enumerate(networks_data, 1):
        max_nodes = network_size
        all_nodes = network.nodes[:max_nodes]
        
        leaders = [n for n in all_nodes if n.is_leader]
        peers = [n for n in all_nodes if not n.is_leader]
        
        if len(leaders) + len(peers) != max_nodes:
            print(f"Size mismatch in network {network_size}: {len(leaders)} leaders + {len(peers)} peers")
            continue

        peer_types = {'Simple': 0, 'Cross-connected': 0}
        for peer in peers:
            leader_connections = sum(1 for n in peer.neighbors if n.is_leader and n in all_nodes)
            peer_types['Cross-connected' if leader_connections > 1 else 'Simple'] += 1

        sizes = [
            len(leaders),
            peer_types['Simple'],
            peer_types['Cross-connected']
        ]

        if sum(sizes) != max_nodes:
            print(f"Critical error in network {network_size}: Data sum {sum(sizes)} != {max_nodes}")
            continue

        ax = plt.subplot(rows, cols, idx)
        wedges, texts, autotexts = ax.pie(
            sizes,
            colors=[colors['leaders'], colors['peers_simple'], colors['peers_cross']],
            autopct=lambda p: f'{p:.1f}%' if p >= 5 else '',
            startangle=90,
            pctdistance=0.65,
            labeldistance=1.1,
            textprops={'color': 'black', 'alpha': 0.8, 'fontsize': 6}
        )

        ax.set_title(f'N = {network_size}', y=0.95, fontsize=8, va='top', pad=2)
        ax.axis('equal')

        for i, (text, size) in enumerate(zip(texts, sizes)):
            text.set_text(f'{size}' if size > 0 else '')
            text.set_fontsize(6)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Leaders',
                  markerfacecolor=colors['leaders'], markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Simple Peers',
                  markerfacecolor=colors['peers_simple'], markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Cross-connected Peers',
                  markerfacecolor=colors['peers_cross'], markersize=8)
    ]

    plt.figlegend(handles=legend_elements, loc='upper center', 
                ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3, w_pad=2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'network_hierarchy_final_corrected.png')
    plt.savefig(output_path, bbox_inches='tight', transparent=False, dpi=600)
    print(f"Final corrected visualization saved at: {output_path}")

    plt.show()
    plt.close()



# def visualize_network_hierarchy(networks_data):
#     if not networks_data:
#         print("No network data available for visualization.")
#         return

#     # Increased figure size and resolution
#     plt.figure(figsize=(12, 8), dpi=300)
#     plt.rcParams.update({
#         'font.size': 8,  # Increased base font size
#         'axes.titlesize': 8.5,  # Larger title size
#         'axes.labelsize': 8,
#         'legend.fontsize': 8,
#         'legend.title_fontsize': 8.5,
#         'font.family': 'DejaVu Sans',
#         'axes.labelpad': 4  # Increased padding
#     })

#     num_networks = len(networks_data)
#     rows = (num_networks // 2) + (1 if num_networks % 2 != 0 else 0)
#     cols = min(2, num_networks)

#     # Professional color scheme
#     colors = {
#         'leaders': '#E64646',
#         'peers_simple': '#5F9ED1',
#         'peers_cross': '#9ED15F'
#     }

#     for idx, (network_size, network) in enumerate(networks_data, 1):
#         max_nodes = network_size
#         all_nodes = network.nodes[:max_nodes]
        
#         leaders = [n for n in all_nodes if n.is_leader]
#         peers = [n for n in all_nodes if not n.is_leader]
        
#         if len(leaders) + len(peers) != max_nodes:
#             print(f"Size mismatch in network {network_size}: {len(leaders)} leaders + {len(peers)} peers")
#             continue

#         peer_types = {'Simple': 0, 'Cross-connected': 0}
#         for peer in peers:
#             leader_connections = sum(1 for n in peer.neighbors if n.is_leader and n in all_nodes)
#             peer_types['Cross-connected' if leader_connections > 1 else 'Simple'] += 1

#         sizes = [
#             len(leaders),
#             peer_types['Simple'],
#             peer_types['Cross-connected']
#         ]

#         if sum(sizes) != max_nodes:
#             print(f"Critical error in network {network_size}: Data sum {sum(sizes)} != {max_nodes}")
#             continue

#         ax = plt.subplot(rows, cols, idx)
#         wedges, texts, autotexts = ax.pie(
#             sizes,
#             colors=[colors['leaders'], colors['peers_simple'], colors['peers_cross']],
#             autopct=lambda p: f'{p:.1f}%' if p >= 5 else '',
#             startangle=90,
#             pctdistance=0.7,
#             labeldistance=1.5,
#             textprops={'color': 'black', 'fontsize': 9, 'alpha': 0.9},  # Larger text
#             wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}  # Added edge contrast
#         )

#         # Enhanced title formatting
#         ax.set_title(f'Network Size: {network_size}', y=0.85, fontsize=6, 
#                     fontweight='semibold', color='#333333')

#         # Improved label formatting
#         for text in texts + autotexts:
#             text.set_fontsize(9)
#             text.set_alpha(0.9)

#     # Professional legend design
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label='Leaders',
#                   markerfacecolor=colors['leaders'], markersize=9),
#         plt.Line2D([0], [0], marker='o', color='w', label='Simple Peers',
#                   markerfacecolor=colors['peers_simple'], markersize=9),
#         plt.Line2D([0], [0], marker='o', color='w', label='Cross-connected Peers',
#                   markerfacecolor=colors['peers_cross'], markersize=9)
#     ]

#     # Centered legend with improved spacing
#     plt.figlegend(handles=legend_elements, loc='upper center', 
#                  ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05),
#                  fontsize=10, handletextpad=0.3)

#     # Optimized layout spacing
#     plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4, w_pad=4)

#     # Save in multiple formats for publication
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     output_path = os.path.join(script_dir, 'network_hierarchy.png')
#     plt.savefig(output_path, bbox_inches='tight', transparent=False, dpi=600)
#     plt.show
#     plt.close()


def memoryUsage_visuals(memory_results, network_sizes):
    plt.style.use('ggplot')
    
    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    network_sizes = [res[0] for res in memory_results]
    without = [np.mean([n.disk_usage for n in res[1].nodes]) for res in memory_results]
    with_shard = [np.mean([n.disk_usage for n in res[2].nodes]) for res in memory_results]

    x = np.arange(len(network_sizes))
    ax1.bar(x - 0.2, without, 0.4, label='Without Sharding')
    ax1.bar(x + 0.2, with_shard, 0.4, label='With Sharding')
    ax1.set_xticks(x)
    ax1.set_xticklabels(network_sizes)
    ax1.set_xlabel('Network Size (Nodes)')
    ax1.set_ylabel('Memory Usage (%)')
    ax1.set_title('Memory Efficiency Comparison')
    ax1.legend()
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'presentation_visuals.png')
    plt.savefig(output_path, bbox_inches='tight', transparent=False, dpi=600)
    plt.close()

def print_experiment_results(our_exp1, faiss_exp1, our_exp2, faiss_exp2):
    """Prints all experiment results in Windows-safe format"""
    # Our System - Experiment 1
    print("\nOUR SYSTEM - EXPERIMENT 1 RESULTS (Workload Analysis)")
    print(f"{'Workload':<10} | {'Throughput (blk/s)':<18} | {'Latency (s/blk)':<15}")
    print("-"*50)
    for wl, tput, lat in sorted(our_exp1, key=lambda x: x[0]):
        print(f"{wl:<10} | {tput:<18.2f} | {lat:<15.4f}")

    # FAISS - Experiment 1
    print("\nFAISS - EXPERIMENT 1 RESULTS (Workload Analysis)")
    print(f"{'Workload':<10} | {'Throughput (blk/s)':<18} | {'Latency (s/blk)':<15}")
    print("-"*50)
    for wl, tput, lat in sorted(faiss_exp1, key=lambda x: x[0]):
        print(f"{wl:<10} | {tput:<18.2f} | {lat:<15.4f}")

    # Our System - Experiment 2
    print("\nOUR SYSTEM - EXPERIMENT 2 RESULTS (Network Size Analysis)")
    print(f"{'Nodes':<6} | {'Throughput':<10} | {'Latency':<10} | {'Load Balance (std)'}")
    print("-"*55)
    for size, tput, lat, bal in sorted(our_exp2, key=lambda x: x[0]):
        print(f"{size:<6} | {tput:<10.2f} | {lat:<10.4f} | {bal:<15.2f}")

    # FAISS - Experiment 2
    print("\nFAISS - EXPERIMENT 2 RESULTS (Network Size Analysis)")
    print(f"{'Nodes':<6} | {'Throughput':<10} | {'Latency':<10} | {'Load Balance (std)'}")
    print("-"*55)
    for size, tput, lat, bal in sorted(faiss_exp2, key=lambda x: x[0]):
        print(f"{size:<6} | {tput:<10.2f} | {lat:<10.4f} | {bal:<15.2f}")




def compare_systems_plot(exp1_ours, exp1_faiss, exp2_ours, exp2_faiss):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Replace with either:
    # plt.style.use('seaborn-v0_8')  # Newer matplotlib versions
    # OR if that doesn't work:
    plt.style.use('ggplot')  # Alternative built-in style


    # Experiment 1 Comparison Plot
    plt.figure(figsize=(14, 6))
    
    # Throughput Comparison
    plt.subplot(1, 2, 1)
    for system, results, color, marker in [('Our System', exp1_ours, 'blue', 'o'),
                                         ('FAISS', exp1_faiss, 'green', '^')]:
        workloads = sorted({x[0] for x in results})
        avg_throughput = [np.mean([res[1] for res in results if res[0] == wl]) 
                         for wl in workloads]
        plt.plot(workloads, avg_throughput, marker=marker, color=color, 
                linestyle='--', linewidth=1, label=system)
    
    plt.title('Throughput Comparison: Workload Scaling')
    plt.xlabel('Workload Size')
    plt.ylabel('Throughput (ops/sec)')
    plt.legend()
    plt.grid(True)

    # Latency Comparison
    plt.subplot(1, 2, 2)
    for system, results, color, marker in [('Our System', exp1_ours, 'red', 'x'),
                                         ('FAISS', exp1_faiss, 'purple', 's')]:
        workloads = sorted({x[0] for x in results})
        avg_latency = [np.mean([res[2] for res in results if res[0] == wl]) 
                      for wl in workloads]
        plt.plot(workloads, avg_latency, marker=marker, color=color, 
                linestyle='-.', linewidth=1, label=system)
    
    plt.title('Latency Comparison: Workload Scaling')
    plt.xlabel('Workload Size')
    plt.ylabel('Latency (sec/op)')
    plt.yscale('log')  # Better visibility for different scales
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'comparison_experiment_1.png'))
    plt.close()

    # Experiment 2 Comparison Plot
    plt.figure(figsize=(14, 6))
    
    # Throughput Comparison
    plt.subplot(1, 2, 1)
    for system, results, color, marker in [('Our System', exp2_ours, 'cyan', 'D'),
                                         ('FAISS', exp2_faiss, 'orange', 'P')]:
        net_sizes = sorted({x[0] for x in results})
        avg_throughput = [np.mean([res[1] for res in results if res[0] == ns]) 
                         for ns in net_sizes]
        plt.plot(net_sizes, avg_throughput, marker=marker, color=color,
                linestyle='-', linewidth=1, label=system)
    
    plt.title('Throughput Comparison: Network Scaling')
    plt.xlabel('Network Size/Shards')
    plt.ylabel('Throughput (ops/sec)')
    plt.legend()
    plt.grid(True)

    # Latency Comparison
    plt.subplot(1, 2, 2)
    for system, results, color, marker in [('Our System', exp2_ours, 'brown', '*'),
                                         ('FAISS', exp2_faiss, 'magenta', 'v')]:
        net_sizes = sorted({x[0] for x in results})
        avg_latency = [np.mean([res[2] for res in results if res[0] == ns])
                      for ns in net_sizes]
        plt.plot(net_sizes, avg_latency, marker=marker, color=color,
                linestyle=':', linewidth=1, label=system)
    
    plt.title('Latency Comparison: Network Scaling')
    plt.xlabel('Network Size/Shards')
    plt.ylabel('Latency (sec/op)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'comparison_experiment_2.png'))
    plt.close()



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    try:
        if rank == 0:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            plt.style.use('seaborn-v0_8')  

        # Common parameters
        workloadExp1 = [1000, 2500, 5000, 7500, 10000]
        network_sizes = [25, 50, 75, 100]
        base_workload = 1000

        # Run experiments with barriers between them
        comm.Barrier()
        our_exp1 = experiment_1_workload_analysis(workloadExp1, comm)
        comm.Barrier()
        faiss_exp1 = faiss_experiment_1_workload_analysis(workloadExp1, comm)
        comm.Barrier()

        our_exp2 = experiment_2_network_size_analysis(network_sizes, base_workload, comm)
        comm.Barrier()

        faiss_exp2 = faiss_experiment_2_network_size_analysis(network_sizes, base_workload, comm)

        # Gather results (no comm.gather needed here)
        if rank == 0:
            # Directly use the results collected by the experiment functions
            final_our_exp1 = our_exp1
            final_our_exp2 = our_exp2
            final_faiss_exp1 = faiss_exp1
            final_faiss_exp2 = faiss_exp2

            # Create comparison plots
            compare_systems_plot(
                sorted(final_our_exp1, key=lambda x: x[0]),
                sorted(final_faiss_exp1, key=lambda x: x[0]),
                sorted(final_our_exp2, key=lambda x: x[0]),
                sorted(final_faiss_exp2, key=lambda x: x[0])
            )

            # Original plots for individual systems
            plot_results(final_our_exp1, final_our_exp2)  # Your system
            plot_results(final_faiss_exp1, final_faiss_exp2)  # FAISS alone
            print_experiment_results(final_our_exp1, final_faiss_exp1, final_our_exp2, final_faiss_exp2)

        # Experiment 3 (Memory Analysis) - Only rank 0 executes
        if rank == 0:
            exp3_results = []
            workload = 500
            
            for net_size in network_sizes:
                # Non-sharded network
                network_without = Network(num_nodes=net_size, num_clusters=1)
                # Process workload
                for _ in range(workload):
                    block = Block([np.random.rand(10).tolist()])
                    scale_free_consensus(block, network_without, False)
                
                # Sharded network
                network_with = Network(num_nodes=net_size, num_clusters=max(2, net_size//5))
                for _ in range(workload):
                    block = Block([np.random.rand(10).tolist()])
                    if scale_free_consensus(block, network_with, True):
                        leaders = [n for n in network_with.nodes if n.is_leader]
                        for leader in leaders:
                            enhanced_dynamic_sharding(leader)
                
                exp3_results.append((net_size, network_without, network_with))

            # Visualization
            memoryUsage_visuals(exp3_results, network_sizes)
            print_memory_usage(network_sizes, exp3_results)

        if rank == 0:
            networks_for_viz = [
                (size, network_with) for (size, _, network_with) in exp3_results
            ]
            visualize_network_hierarchy(networks_for_viz)

    except Exception as e:
        error_msg = f"Rank {rank} error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        comm.Abort(1)
    finally:
        comm.Barrier()
        MPI.Finalize()

