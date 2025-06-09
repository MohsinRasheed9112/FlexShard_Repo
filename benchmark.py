import json
import numpy as np
import time
from web3 import Web3
from mpi4py import MPI

def load_geospatial_vectors(file_path):
    """Load geospatial vectors from a file."""
    import geopandas as gpd
    gdf = gpd.read_file(file_path)
    vectors = [np.array(geom.centroid.coords).flatten() for geom in gdf.geometry]
    return vectors

def initialize_contract():
    """Initialize Web3 and contract in each process"""
    try:
        web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        if not web3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")
        
        with open("/home/vboxuser/my-hardhat-project/artifacts/contracts/Binecone.sol/Binecone.json") as f:
            contract_data = json.load(f)
        
        contract_address = Web3.to_checksum_address("0xB7f8BC63BbcaD18155201308C8f3540b07f84F5e")
        contract = web3.eth.contract(address=contract_address, abi=contract_data['abi'])
        account = web3.eth.accounts[0]
        
        return contract, account, web3
    except Exception as e:
        print(f"Contract initialization error: {e}")
        raise

class BineconePerformanceTester:
    def __init__(self, contract, account, web3):
        self.contract = contract
        self.account = account
        self.web3 = web3
        self.tx_params = {
            'from': self.account,
            'gas': 300000,
            'gasPrice': Web3.to_wei('20', 'gwei')
        }
    
    def insert_vector(self, vector):
        try:
            vector_to_insert = [int(abs(x * 1000)) for x in vector]
            start_time = time.perf_counter()
            tx_hash = self.contract.functions.insertVector(vector_to_insert).transact(self.tx_params)
            self.web3.eth.wait_for_transaction_receipt(tx_hash)
            return time.perf_counter() - start_time
        except Exception as e:
            print(f"Error inserting vector: {e}")
            return float('inf')

def run_experiment_1(comm, vectors):
    """Experiment 1: Fixed 10 MPI processes, varying workloads"""
    rank = comm.Get_rank()
    contract, account, web3 = initialize_contract()
    tester = BineconePerformanceTester(contract, account, web3)
    
    workloads = [50, 100, 150, 200, 258]
    fixed_processes = 18
    
    results = []
    
    if rank < fixed_processes: 
        for workload in workloads:
            # Calculate each process's share
            vectors_per_process = workload // fixed_processes
            remainder = workload % fixed_processes
            start_idx = rank * vectors_per_process + min(rank, remainder)
            end_idx = start_idx + vectors_per_process + (1 if rank < remainder else 0)
            my_vectors = vectors[start_idx:end_idx]
            
            # Time the insertions
            insertion_times = []
            start_time = time.perf_counter()
            for i, vector in enumerate(my_vectors):
                latency = tester.insert_vector(vector)
                insertion_times.append(latency)
                if (i + 1) % 10 == 0:  # Progress every 10 vectors
                    print(f"Process {rank}: Inserted {i+1}/{len(my_vectors)} vectors for workload {workload}")
            
            overall_time = time.perf_counter() - start_time
            throughput = len(my_vectors) / overall_time if overall_time > 0 else 0
            avg_latency = np.mean(insertion_times) if insertion_times else 0
            
            results.append((workload, throughput, avg_latency))
    
    # Gather results from all active processes
    gathered_results = comm.gather(results, root=0)
    
    if rank == 0:
        # Combine and average results for each workload
        final_results = {}
        for process_results in gathered_results:
            if process_results:  # Skip inactive processes
                for workload, throughput, latency in process_results:
                    if workload not in final_results:
                        final_results[workload] = {'throughputs': [], 'latencies': []}
                    final_results[workload]['throughputs'].append(throughput)
                    final_results[workload]['latencies'].append(latency)
        
        # Calculate averages
        return [(w, 
                 np.mean(data['throughputs']), 
                 np.mean(data['latencies'])) 
                for w, data in final_results.items()]
    return None

def run_experiment_2(comm, vectors):
    """Experiment 2: Fixed 258 vectors, varying MPI processes"""
    rank = comm.Get_rank()
    contract, account, web3 = initialize_contract()
    tester = BineconePerformanceTester(contract, account, web3)
    
    total_workload = 258
    process_counts = [1, 3, 6, 9, 12]
    
    results = []
    
    for num_procs in process_counts:
        if rank < num_procs:  # Only active processes participate
            # Calculate each process's share
            vectors_per_process = total_workload // num_procs
            remainder = total_workload % num_procs
            start_idx = rank * vectors_per_process + min(rank, remainder)
            end_idx = start_idx + vectors_per_process + (1 if rank < remainder else 0)
            my_vectors = vectors[start_idx:end_idx]
            
            # Time the insertions
            insertion_times = []
            start_time = time.perf_counter()
            for i, vector in enumerate(my_vectors):
                latency = tester.insert_vector(vector)
                insertion_times.append(latency)
                if (i + 1) % 10 == 0:  # Progress every 10 vectors
                    print(f"Process {rank}: Inserted {i+1}/{len(my_vectors)} vectors for {num_procs}-process test")
            
            overall_time = time.perf_counter() - start_time
            throughput = len(my_vectors) / overall_time if overall_time > 0 else 0
            avg_latency = np.mean(insertion_times) if insertion_times else 0
            
            results.append((num_procs, throughput, avg_latency))
    
    # Gather results from all processes
    gathered_results = comm.gather(results, root=0)
    
    if rank == 0:
        # Combine results
        final_results = {}
        for process_results in gathered_results:
            if process_results:  # Skip inactive processes
                for num_procs, throughput, latency in process_results:
                    if num_procs not in final_results:
                        final_results[num_procs] = {'throughputs': [], 'latencies': []}
                    final_results[num_procs]['throughputs'].append(throughput)
                    final_results[num_procs]['latencies'].append(latency)
        
        # Calculate total throughput (sum across all processes)
        return [(p, 
                 np.sum(data['throughputs']),  # Total system throughput
                 np.mean(data['latencies']))  # Average latency
                for p, data in final_results.items()]
    return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("Initializing benchmark...")
        vectors = load_geospatial_vectors("/home/vboxuser/Downloads/countries_dataset.shp")
        print(f"Loaded {len(vectors)} vectors")
    else:
        vectors = None
    
    # Broadcast vectors to all processes
    vectors = comm.bcast(vectors, root=0)
    
    if rank == 0:
        print("\n=== Experiment 1: Fixed Resources (10 MPI procs), Varying Workload ===")
        exp1_results = run_experiment_1(comm, vectors)
        if exp1_results:
            print("\nWorkload\tThroughput (vecs/sec)\tAvg Latency (sec)")
            for workload, throughput, latency in sorted(exp1_results, key=lambda x: x[0]):
                print(f"{workload}\t\t{throughput:.2f}\t\t\t{latency:.4f}")
        
        print("\n=== Experiment 2: Fixed Workload (258 vectors), Varying MPI Processes ===")
        exp2_results = run_experiment_2(comm, vectors)
        if exp2_results:
            print("\nProcesses\tTotal Throughput\tAvg Latency")
            for procs, throughput, latency in sorted(exp2_results, key=lambda x: x[0]):
                print(f"{procs}\t\t{throughput:.2f}\t\t\t{latency:.4f}")
    else:
        run_experiment_1(comm, vectors)
        run_experiment_2(comm, vectors)


if __name__ == "__main__":
    main()







#code for faiss to calculate the performance



import os
import time
import numpy as np
import faiss
from mpi4py import MPI
import geopandas as gpd

def load_geospatial_vectors(file_path):
    """Load geospatial vectors from a file, with fallback to random generation."""
    try:
        if file_path and os.path.exists(file_path):
            gdf = gpd.read_file(file_path)
            vectors = np.array([np.array(geom.centroid.coords).flatten() for geom in gdf.geometry])
            return vectors

    except Exception as e:
        print(f"Error loading geospatial file: {e}")
        print("Generating random vectors for testing...")
        return [np.random.rand(2) for _ in range(258)]


class FAISSPerformanceTester:
    def __init__(self, dimension):
        """
        Initialize FAISS index with specific configuration
        
        Args:
            dimension (int): Dimensionality of input vectors
        """
        # Using flat index for simple insertion performance test
        self.index = faiss.IndexFlatL2(dimension)
    
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
            self.index.add(vector.reshape(1, -1))
            return time.perf_counter() - start_time
        except Exception as e:
            print(f"Error inserting vector: {e}")
            return float('inf')
    
    def insert_batch(self, vectors):
        """
        Insert batch of vectors and measure insertion time
        
        Args:
            vectors (np.ndarray): Batch of vectors to insert
        
        Returns:
            float: Batch insertion latency
        """
        try:
            start_time = time.perf_counter()
            self.index.add(vectors)
            return time.perf_counter() - start_time
        except Exception as e:
            print(f"Error inserting batch: {e}")
            return float('inf')

def run_experiment_1(comm, vectors):
    """
    Experiment 1: Fixed Resources (5 MPI processes), Varying Workload
    
    Args:
        comm (MPI.Comm): MPI communicator
        vectors (np.ndarray): Input vectors
    
    Returns:
        list: Performance results
    """
    rank = comm.Get_rank()
    dimension = vectors.shape[1]
    
    workloads = [50, 100, 150, 200, 258]
    fixed_processes = 18
    
    results = []
    
    if rank < fixed_processes:
        for workload in workloads:
            # Calculate each process's share
            vectors_per_process = workload // fixed_processes
            remainder = workload % fixed_processes
            start_idx = rank * vectors_per_process + min(rank, remainder)
            end_idx = start_idx + vectors_per_process + (1 if rank < remainder else 0)
            my_vectors = vectors[start_idx:end_idx]
            
            # Initialize FAISS index
            tester = FAISSPerformanceTester(dimension)
            
            # Time the insertions
            insertion_times = []
            start_time = time.perf_counter()
            
            for i, vector in enumerate(my_vectors):
                latency = tester.insert_vector(vector)
                insertion_times.append(latency)
                
                if (i + 1) % 10 == 0:
                    print(f"Process {rank}: Inserted {i+1}/{len(my_vectors)} vectors for workload {workload}")
            
            overall_time = time.perf_counter() - start_time
            throughput = len(my_vectors) / overall_time if overall_time > 0 else 0
            avg_latency = np.mean(insertion_times) if insertion_times else 0
            
            results.append((workload, throughput, avg_latency))
    
    # Gather results from all active processes
    gathered_results = comm.gather(results, root=0)
    
    if rank == 0:
        # Combine and average results for each workload
        final_results = {}
        for process_results in gathered_results:
            if process_results:
                for workload, throughput, latency in process_results:
                    if workload not in final_results:
                        final_results[workload] = {'throughputs': [], 'latencies': []}
                    final_results[workload]['throughputs'].append(throughput)
                    final_results[workload]['latencies'].append(latency)
        
        # Calculate averages
        return [(w, 
                 np.mean(data['throughputs']), 
                 np.mean(data['latencies'])) 
                for w, data in final_results.items()]
    return None

def run_experiment_2(comm, vectors):
    """
    Experiment 2: Fixed Workload (258 vectors), Varying MPI Processes
    
    Args:
        comm (MPI.Comm): MPI communicator
        vectors (np.ndarray): Input vectors
    
    Returns:
        list: Performance results
    """
    rank = comm.Get_rank()
    dimension = vectors.shape[1]
    
    total_workload = 258
    process_counts = [1, 3, 6, 9, 12]
    
    results = []
    
    for num_procs in process_counts:
        if rank < num_procs:
            # Calculate each process's share
            vectors_per_process = total_workload // num_procs
            remainder = total_workload % num_procs
            start_idx = rank * vectors_per_process + min(rank, remainder)
            end_idx = start_idx + vectors_per_process + (1 if rank < remainder else 0)
            my_vectors = vectors[start_idx:end_idx]
            
            # Initialize FAISS index
            tester = FAISSPerformanceTester(dimension)
            
            # Time the insertions
            insertion_times = []
            start_time = time.perf_counter()
            
            for i, vector in enumerate(my_vectors):
                latency = tester.insert_vector(vector)
                insertion_times.append(latency)
                
                if (i + 1) % 10 == 0:
                    print(f"Process {rank}: Inserted {i+1}/{len(my_vectors)} vectors for {num_procs}-process test")
            
            overall_time = time.perf_counter() - start_time
            throughput = len(my_vectors) / overall_time if overall_time > 0 else 0
            avg_latency = np.mean(insertion_times) if insertion_times else 0
            
            results.append((num_procs, throughput, avg_latency))
    
    # Gather results from all processes
    gathered_results = comm.gather(results, root=0)
    
    if rank == 0:
        # Combine results
        final_results = {}
        for process_results in gathered_results:
            if process_results:
                for num_procs, throughput, latency in process_results:
                    if num_procs not in final_results:
                        final_results[num_procs] = {'throughputs': [], 'latencies': []}
                    final_results[num_procs]['throughputs'].append(throughput)
                    final_results[num_procs]['latencies'].append(latency)
        
        # Calculate total throughput and average latency
        return [(p, 
                 np.sum(data['throughputs']),  # Total system throughput
                 np.mean(data['latencies']))   # Average latency
                for p, data in final_results.items()]
    return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Set file path for geospatial vectors
    vector_file_path = "C:/Users/mrasheed/Desktop/Vector-Dataset/countries_dataset.shp"
    
    if rank == 0:
        print("Initializing FAISS Benchmark...")
        vectors = load_geospatial_vectors(vector_file_path)
        print(f"Loaded {len(vectors)} vectors")
    else:
        vectors = None
    
    # Broadcast vectors to all processes
    vectors = comm.bcast(vectors, root=0)
    
    if rank == 0:
        print("\n=== Experiment 1: Fixed Resources (5 MPI procs), Varying Workload ===")
        exp1_results = run_experiment_1(comm, vectors)
        if exp1_results:
            print("\nWorkload\tThroughput (vecs/sec)\tAvg Latency (sec)")
            for workload, throughput, latency in sorted(exp1_results, key=lambda x: x[0]):
                print(f"{workload}\t\t{throughput:.2f}\t\t\t{latency:.4f}")
        
        print("\n=== Experiment 2: Fixed Workload (258 vectors), Varying MPI Processes ===")
        exp2_results = run_experiment_2(comm, vectors)
        if exp2_results:
            print("\nProcesses\tTotal Throughput\tAvg Latency")
            for procs, throughput, latency in sorted(exp2_results, key=lambda x: x[0]):
                print(f"{procs}\t\t{throughput:.2f}\t\t\t{latency:.4f}")
    else:
        run_experiment_1(comm, vectors)
        run_experiment_2(comm, vectors)

if __name__ == "__main__":
    main()


