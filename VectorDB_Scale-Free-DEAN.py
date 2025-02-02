import traceback
import geopandas as gpd
from mpi4py import MPI
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from shapely.geometry import Point
import sys
import networkx as nx
import pandas as pd

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


THRESHOLD = 80  # Memory usage threshold for dynamic sharding

class Block:
    block_counter = 0 

    def __init__(self, features):
        if not all(isinstance(feature, (list, np.ndarray)) for feature in features):
            features = [[feature] if isinstance(feature, float) else feature for feature in features]
        self.features = features
        self.hash = self.calculate_hash()
        self.id = Block.block_counter 
        Block.block_counter += 1

    def calculate_hash(self):
        return sum(sum(feature) if isinstance(feature, (list, np.ndarray)) else feature for feature in self.features if feature is not None)

class Node:
    def __init__(self, id, degree, uptime, latency, token, adjacency_votes, disk_usage, computational_capacity):
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

    def is_idle(self):
        return random.choice([True, False])

class Cluster:
    def __init__(self, id, nodes):
        self.id = id
        self.nodes = nodes
        self.sub_clusters = []

    def assign_sub_clusters(self, sub_clusters):
        if not isinstance(sub_clusters, list):
            raise TypeError("Sub-clusters must be a list.")
        self.sub_clusters = sub_clusters

    def is_idle(self):
        return all(node.is_idle() for node in self.nodes)

class ScaleFreeGraph:
    def __init__(self, num_nodes, initial_links):
        self.graph = nx.barabasi_albert_graph(num_nodes, initial_links)

    def get_adjacency_list(self):
        return nx.to_dict_of_lists(self.graph)

class Network:
    def __init__(self, num_nodes, num_clusters, initial_links=2):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.nodes_per_cluster = num_nodes // num_clusters
        self.nodes = []
        self.clusters = []
        self.sub_clusters = []
        self.scale_free_graph = ScaleFreeGraph(num_nodes, initial_links)
        self.adjacency_list = self.scale_free_graph.get_adjacency_list()

    def initialize_nodes(self):
        for i in range(self.num_nodes):
            node = Node(
                id=i,
                degree=len(self.adjacency_list[i]),
                uptime=random.random(),
                latency=random.uniform(0.1, 1.0),
                token=random.randint(1, 100),
                adjacency_votes=random.randint(1, 10),
                disk_usage=random.randint(50, 100),
                computational_capacity=random.randint(30, 100),
            )
            self.nodes.append(node)

        # Assign Node objects to neighbors
        for node in self.nodes:
            node.neighbors = [self.nodes[neighbor_id] for neighbor_id in self.adjacency_list[node.id]]

        for cluster_id in range(self.num_clusters):
            start_idx = cluster_id * self.nodes_per_cluster
            end_idx = start_idx + self.nodes_per_cluster
            cluster_nodes = self.nodes[start_idx:end_idx]
            cluster = Cluster(id=cluster_id, nodes=cluster_nodes)
            self.clusters.append(cluster)

        remaining_nodes = len(self.nodes) % self.num_clusters
        for i in range(remaining_nodes):
            self.clusters[i % self.num_clusters].nodes.append(self.nodes[-(i + 1)])

    def create_sub_clusters(self, num_sub_clusters):
        for cluster in self.clusters:
            sub_cluster_size = len(cluster.nodes) // num_sub_clusters
            cluster.sub_clusters = []
            for sub_cluster_id in range(num_sub_clusters):
                start_idx = sub_cluster_id * sub_cluster_size
                end_idx = start_idx + sub_cluster_size
                sub_cluster_nodes = cluster.nodes[start_idx:end_idx]
                cluster.sub_clusters.append(sub_cluster_nodes)

            remaining_nodes = len(cluster.nodes) % num_sub_clusters
            for i in range(remaining_nodes):
                cluster.sub_clusters[i % num_sub_clusters].append(cluster.nodes[-(i + 1)])

    def get_nodes_for_rank(self, rank, size):
        nodes_per_rank = len(self.nodes) // size
        start_idx = rank * nodes_per_rank
        end_idx = start_idx + nodes_per_rank

        if rank == size - 1:
            end_idx = len(self.nodes)

        return self.nodes[start_idx:end_idx]

# Algorithm 1: Block Validation Protocol
def validate_block(block, node, validation_criteria):

    for vector in block.features:
        if not check_integrity(vector):
            return False
        if not satisfies_criteria(vector, validation_criteria):
            return False
    return True

def check_integrity(vector):

    return len(vector) > 0 and all(isinstance(val, (int, float)) for val in vector)

def satisfies_criteria(vector, validation_criteria):
    """
    Checks if the vector satisfies the given validation criteria.
    """
    return all(criterion(vector) for criterion in validation_criteria)

# Algorithm 2: Leader Selection in Scale-Free Topology
def select_leaders(network, leader_threshold=80):

    leaders = []
    leader_peer_mapping = {}
    cross_connected_peers = {}

    # Select leaders based on the score threshold
    for node in network.nodes:
        score = calculate_score(node)
        if score > leader_threshold:
            node.is_leader = True
            leaders.append(node)
            leader_peer_mapping[node.id] = [neighbor.id for neighbor in node.neighbors]

            # Track cross-connected peers
            for neighbor in node.neighbors:
                if neighbor.id in cross_connected_peers:
                    cross_connected_peers[neighbor.id].append(node.id)
                else:
                    cross_connected_peers[neighbor.id] = [node.id]

    return leaders, leader_peer_mapping, cross_connected_peers

# def select_leaders(network, leader_threshold):

#     leaders = []
#     leader_peer_mapping = {}

#     # Calculate scores for all nodes
#     for node in network.nodes:
#         score = calculate_score(node)  
#         if score > leader_threshold: 
#             node.is_leader = True
#             leaders.append(node)
#             leader_peer_mapping[node.id] = [neighbor.id for neighbor in node.neighbors]

#     return leaders, leader_peer_mapping




def select_best_leader(candidates):

    best_leader = max(candidates, key=lambda x: calculate_score(x))
    return best_leader

def calculate_score(node):

    return node.uptime - node.latency + node.token + node.degree

def random_sample(hub, K):

    neighbor_nodes = random.sample(hub.neighbors, K)  # Directly sample Node objects
    return neighbor_nodes

# Algorithm 3: Scale-Free Consensus (Leader-Only)
def scale_free_consensus(block, network):
    """
    Implements scale-free consensus where only leaders participate.
    """
    leaders, _ = select_leaders(network)
    leader_votes = 0

    for leader in leaders:
        if validate_block(block, leader, [check_integrity]):
            leader_votes += 1

    # Consensus is reached if a majority of leaders agree
    if leader_votes > len(leaders) / 2:
        append_block(block, network)
        return True
    return False

def append_block(block, network):
    """
    Appends a block to the blockchain of all nodes in the network.
    """
    for node in network.nodes:
        node.blockchain.append(block)
    print(f"Block with hash {block.hash} appended to the blockchain by rank {rank}")

def dynamic_sharding(leader):
    """
    Adjusts shard boundaries dynamically based on memory usage.
    """
    shard_load = leader.disk_usage  
    if shard_load > THRESHOLD:
        print(f"Dynamic sharding triggered for leader {leader.id} (disk_usage: {shard_load})")
        new_boundary = adjust_shard_boundary(leader, shard_load)
        leader.shard_boundary = new_boundary
        load_balancing(leader)
    else:
        print(f"No dynamic sharding needed for leader {leader.id} (disk_usage: {shard_load})")


def adjust_shard_boundary(leader, shard_load):
    """
    Adjusts the shard boundary based on the current memory load.
    """
    current_boundary = leader.shard_boundary if hasattr(leader, 'shard_boundary') else 100  # Default boundary
    overload_factor = (shard_load - THRESHOLD) / THRESHOLD
    adjustment_factor = 1 - overload_factor
    new_boundary = current_boundary * adjustment_factor
    return new_boundary

def load_balancing(leader):
    """
    Balances the load by offloading data to other nodes.
    """
    for node in leader.neighbors:
        if node.disk_usage < THRESHOLD:
            # Offload data to the neighbor
            data_to_offload = leader.blockchain.pop()
            node.blockchain.append(data_to_offload)
            leader.disk_usage -= 10  # Simulate memory reduction
            node.disk_usage += 10  # Simulate memory increase
            print(f"Data offloaded from leader {leader.id} to node {node.id}")
            break

# Generate Synthetic Datasets
def generate_synthetic_datasets(data_sizes):

    datasets = []
    for num_records in data_sizes:
        synthetic_data = {
            'id': list(range(num_records)),
            'vector_data': [np.random.rand(10).tolist() for _ in range(num_records)],
            'geometry': [
                Point(np.random.uniform(-180, 180), np.random.uniform(-90, 90))
                for _ in range(num_records)
            ]
        }
        gdf = gpd.GeoDataFrame(synthetic_data, crs="EPSG:4326")
        datasets.append((gdf, num_records))
    return datasets

# Simulate Block Processing
def process_block(block):
    """
    Simulates block processing.
    """
    time.sleep(0.001)

# Experiment 1: Workload Analysis
def experiment_1_workload_analysis(workloads, comm):
    experiment_1_results = []

    if rank == 0:
        datasets = generate_synthetic_datasets(workloads)
    else:
        datasets = None

    datasets = comm.bcast(datasets, root=0)

    for idx, (gdf, workload) in enumerate(datasets):
        local_workload = workload // size
        start_time = time.time()

        for _ in range(local_workload):
            process_block({'features': [np.random.rand(10).tolist()]})

        end_time = time.time()
        throughput = workload / (end_time - start_time) if end_time > start_time else 0
        latency = (end_time - start_time) / workload if workload > 0 else 0
        experiment_1_results.append((workload, throughput, latency))

    return experiment_1_results

# Updated Experiment 2: Network Size Analysis
def experiment_2_network_size_analysis(network_sizes, workload, comm):
    experiment_2_results = []

    for net_size in network_sizes:
        num_clusters = 5  
        nodes_per_cluster = net_size // num_clusters  # Nodes per cluster

        network = Network(num_nodes=net_size, num_clusters=num_clusters, initial_links=2)
        network.initialize_nodes()

        local_workload = workload // size
        start_time = time.time()

        for _ in range(local_workload):
            process_block({'features': [np.random.rand(10).tolist()]})

        end_time = time.time()
        throughput = workload / (end_time - start_time) if end_time > start_time else 0
        latency = (end_time - start_time) / workload if workload > 0 else 0

        experiment_2_results.append((net_size, throughput, latency))

    return experiment_2_results

# Experiment 3: Memory Usage Comparison
def experiment_memory_usage(network_sizes, workload, comm):
    """
    Compares memory usage with and without dynamic sharding.
    """
    memory_results = []

    for net_size in network_sizes:
        
        network = Network(num_nodes=net_size, num_clusters=5, initial_links=2)
        network.initialize_nodes()

        # Case 1: Without Dynamic Sharding
        start_time = time.time()
        for _ in range(workload):
            block = Block([np.random.rand(10).tolist()])
            scale_free_consensus(block, network)
        end_time = time.time()
        memory_without_sharding = [node.disk_usage for node in network.nodes]

        # Case 2: With Dynamic Sharding
        for node in network.nodes:
            if node.is_leader:
                dynamic_sharding(node)
        start_time_sharding = time.time()
        for _ in range(workload):
            block = Block([np.random.rand(10).tolist()])
            scale_free_consensus(block, network)
        end_time_sharding = time.time()
        memory_with_sharding = [node.disk_usage for node in network.nodes]

        # Append results
        memory_results.append((net_size, memory_without_sharding, memory_with_sharding))

    return memory_results

# def experiment_memory_usage(network_sizes, workload, comm):
#     """
#     Compares memory usage with and without dynamic sharding.
#     """
#     memory_results = []

#     for net_size in network_sizes:
#         # Initialize network
#         network = Network(num_nodes=net_size, num_clusters=5, initial_links=2)
#         network.initialize_nodes()

#         # Case 1: Without Dynamic Sharding
#         start_time = time.time()
#         for _ in range(workload):
#             block = Block([np.random.rand(10).tolist()])
#             scale_free_consensus(block, network)
#         end_time = time.time()
#         memory_without_sharding = [node.disk_usage for node in network.nodes]

#         # Case 2: With Dynamic Sharding
#         for node in network.nodes:
#             if node.is_leader:
#                 dynamic_sharding(node)
#         start_time_sharding = time.time()
#         for _ in range(workload):
#             block = Block([np.random.rand(10).tolist()])
#             scale_free_consensus(block, network)
#         end_time_sharding = time.time()
#         memory_with_sharding = [node.disk_usage for node in network.nodes]

#         
#         memory_results.append((net_size, memory_without_sharding, memory_with_sharding))

#     return memory_results

# Plot Results
def plot_results(experiment_1_results, experiment_2_results, memory_results):
    # Plot Experiment 1: Throughput and Latency
    workloads = list(set(x[0] for x in experiment_1_results))
    workloads.sort()
    avg_throughput1 = [np.mean([res[1] for res in experiment_1_results if res[0] == wl]) for wl in workloads]
    avg_latency1 = [np.mean([res[2] for res in experiment_1_results if res[0] == wl]) for wl in workloads]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(workloads, avg_throughput1, marker='o', label='Throughput', color='blue')
    plt.title('Experiment 1: Throughput vs Workload')
    plt.xlabel('Workload')
    plt.ylabel('Throughput (blocks/sec)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(workloads, avg_latency1, marker='x', label='Latency', color='orange')
    plt.title('Experiment 1: Latency vs Workload')
    plt.xlabel('Workload')
    plt.ylabel('Latency (sec/block)')
    plt.grid(True)
    plt.legend()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # plt.savefig(script_dir, 'experiment_1_results.png')
    output_path = os.path.join(script_dir, 'experiment_1_results.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


    # Plot Experiment 2: Throughput and Latency
    net_sizes = list(set(x[0] for x in experiment_2_results))
    net_sizes.sort()
    avg_throughput2 = [np.mean([res[1] for res in experiment_2_results if res[0] == ns]) for ns in net_sizes]
    avg_latency2 = [np.mean([res[2] for res in experiment_2_results if res[0] == ns]) for ns in net_sizes]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(net_sizes, avg_throughput2, marker='o', label='Throughput', color='blue')
    plt.title('Experiment 2: Throughput vs Network Size')
    plt.xlabel('Network Size')
    plt.ylabel('Throughput (blocks/sec)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(net_sizes, avg_latency2, marker='x', label='Latency', color='orange')
    plt.title('Experiment 2: Latency vs Network Size')
    plt.xlabel('Network Size')
    plt.ylabel('Latency (sec/block)')
    plt.grid(True)
    plt.legend()
    # plt.savefig(script_dir, 'experiment_2_results.png')
    output_path = os.path.join(script_dir, 'experiment_2_results.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Plot Memory Usage Comparison
    net_sizes = [res[0] for res in memory_results]
    memory_without = [np.mean(res[1]) for res in memory_results]
    memory_with = [np.mean(res[2]) for res in memory_results]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

    # Plot Without Dynamic Sharding
    axes[0].plot(net_sizes, memory_without, marker='o', color='red')
    axes[0].set_title('Memory Usage Without Dynamic Sharding')
    axes[0].set_xlabel('Network Size')
    axes[0].set_ylabel('Average Memory Usage (%)')
    axes[0].grid(True)

    # Plot With Dynamic Sharding
    axes[1].plot(net_sizes, memory_with, marker='x', color='green')
    axes[1].set_title('Memory Usage With Dynamic Sharding')
    axes[1].set_xlabel('Network Size')
    axes[1].set_ylabel('Average Memory Usage (%)')
    axes[1].grid(True)

    plt.tight_layout()
    output_path = os.path.join(script_dir, 'memory_usage_comparison.png')
    plt.savefig(output_path)
    plt.close()


    
import matplotlib.pyplot as plt
import networkx as nx

def visualize_leaders_and_peers(network):
    """
    Visualizes the leaders, their connected peer nodes, and highlights cross connected peers.
    """
    leaders, leader_peer_mapping, cross_connected_peers = select_leaders(network)

    G = nx.Graph()

    # Add all nodes to the graph
    for node in network.nodes:
        G.add_node(node.id, is_leader=node.is_leader)

    # Add edges to the graph
    for node in network.nodes:
        for neighbor in node.neighbors:
            G.add_edge(node.id, neighbor.id)

    # (leaders in red, peers in blue, cross-connected peers in green)
    node_colors = []
    for node in G.nodes():
        if network.nodes[node].is_leader:
            node_colors.append('red')  
        elif node in cross_connected_peers and len(cross_connected_peers[node]) > 1:
            node_colors.append('green')  
        else:
            node_colors.append('blue') 

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')

    
    plt.scatter([], [], color='red', label='Leaders')
    plt.scatter([], [], color='blue', label='Peers')
    plt.scatter([], [], color='green', label='Cross-Connected Peers')
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title="Node Type")

    plt.title("Leaders and Their Connected Peer Nodes")
    plt.axis('off')  
    plt.show()

    print("Leader-Peer Mapping:")
    for leader_id, peers in leader_peer_mapping.items():
        print(f"Leader {leader_id} is connected to peers: {peers}")

    print("\nCross-Connected Peers:")
    for peer_id, leaders_connected in cross_connected_peers.items():
        if len(leaders_connected) > 1:
            print(f"Peer {peer_id} is connected to leaders: {leaders_connected}")

# def visualize_leaders_and_peers(network, leader_threshold):
#     """
#     Visualizes the leaders and their connected peer nodes in the network.
#     """
#     leaders, leader_peer_mapping = select_leaders(network, leader_threshold)

#     # Create a graph for visualization
#     G = nx.Graph()

#     # Add all nodes to the graph
#     for node in network.nodes:
#         G.add_node(node.id, is_leader=node.is_leader)

#     # Add edges to the graph
#     for node in network.nodes:
#         for neighbor in node.neighbors:
#             G.add_edge(node.id, neighbor.id)

#     # Define node colors (leaders in red, peers in blue)
#     node_colors = []
#     for node in G.nodes():
#         if network.nodes[node].is_leader:
#             node_colors.append('red')  # Leaders
#         else:
#             node_colors.append('blue')  # Peers

#     # Draw the graph
#     pos = nx.spring_layout(G)  # Layout for better visualization
#     plt.figure(figsize=(10, 8))
#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
#     nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
#     nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')

#     # Add a legend
#     plt.scatter([], [], color='red', label='Leaders')
#     plt.scatter([], [], color='blue', label='Peers')
#     plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title="Node Type")

#     plt.title("Leaders and Their Connected Peer Nodes")
#     plt.axis('off')  # Hide axes
#     plt.show()

#     # Print leader-peer mapping
#     print("Leader-Peer Mapping:")
#     for leader_id, peers in leader_peer_mapping.items():
#         print(f"Leader {leader_id} is connected to peers: {peers}")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        if rank == 0:
            print(f"MPI initialized. Total processes: {size}", flush=True)

        network = Network(num_nodes=50, num_clusters=5, initial_links=2)
        network.initialize_nodes()



        # Select leaders and visualize
        visualize_leaders_and_peers(network)

        # # Select leaders
        # leaders, leader_peer_mapping = select_leaders(network, leader_threshold=80)

        # # Print results
        # print(f"Number of leaders: {len(leaders)}")
        # for leader_id, peers in leader_peer_mapping.items():
        #     print(f"Leader {leader_id} is connected to peers: {peers}")

        
        if rank == 0:
            visualize_leaders_and_peers(network, leader_threshold=80)

        # Rest of experiments...
        workloads = [2500, 5000, 7500, 10000]
        experiment_1_results = experiment_1_workload_analysis(workloads, comm)

        network_sizes = [20, 40, 60, 80, 100]
        workload = 10000
        experiment_2_results = experiment_2_network_size_analysis(network_sizes, workload, comm)

        memory_results = experiment_memory_usage(network_sizes, workload, comm)

        # Gather and output results
        all_exp1 = comm.gather(experiment_1_results, root=0)
        all_exp2 = comm.gather(experiment_2_results, root=0)
        all_memory = comm.gather(memory_results, root=0)

        if rank == 0:
            flat_exp1 = [item for sublist in all_exp1 for item in sublist]
            flat_exp2 = [item for sublist in all_exp2 for item in sublist]
            flat_memory = [item for sublist in all_memory for item in sublist]

            # Visualize results
            plot_results(flat_exp1, flat_exp2, flat_memory)

    except Exception as e:
        print(f"Rank {rank}: Error - {e}", flush=True)
        traceback.print_exc()
    finally:
        comm.Barrier()
        MPI.Finalize()

if __name__ == "__main__":
    main()