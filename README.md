# FlexShard: A High-Performance Blockchain Sharding Framework for Reliable Vector Data Management at the Edge

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

FlexShard is a novel blockchain-integrated vector data management system designed specifically for the dynamic, decentralized nature of edge computing environments. The system addresses critical challenges in managing high-dimensional vector data across resource-constrained edge devices while ensuring data integrity, provenance, and fault tolerance.

## Key Features

- **Dynamic Vector Sharding**: Hierarchical, clustering-aware sharding that organizes vectors based on semantic similarity
- **Scale-Free Network Topology**: Balanced network structure with adaptive leader selection
- **Lightweight Consensus**: Energy-efficient two-phase consensus protocol activating only 51% adjacent quorum
- **Hybrid Architecture**: On-chain metadata for provenance, off-chain vector storage for efficiency
- **Edge-Optimized**: Designed for resource-constrained environments with microsecond-level latencies
- **Fault Tolerance & Self-Healing**: Supports re-sharding and topology reformation under dynamic edge node participation.

## System Architecture

The system is organized into two sharding layers:

1. **Semantic Vector Shards**: Vectors are grouped using semantic-aware clustering algorithms.
2. **Scale-Free Role-Based Partitioning**: Nodes are organized into a scale-free network with adaptive leader/peer roles based on their capacity and graph centrality.

The architecture supports five core protocols:

- `Balanced Scale-Free Graph Construction`
- `Find Reliable Leader`
- `Dynamic Sharding`
- `Block-Validation`
- `Resource Load-Balance`

Each protocol is implemented with low communication overhead and designed for scalability across edge devices.


## Performance Highlights

- **57.8% higher throughput** than FAISS at low workloads
- **38× improvement** over blockchain-based alternatives (Binecone)
- **Microsecond-level latencies** (11-25 microseconds)
- **22.5% lower memory usage** compared to non-sharded approaches
- **Linear scalability** up to 200+ nodes

## Installation

```bash
git clone https://github.com/MohsinRasheed9112/FlexShard_Repo.git
cd FlexShard_Repo
pip install -r requirements.txt

## Prerequisites

- Python 3.7.0 or higher
- Linux/Unix environment (tested on Red Hat Enterprise Linux 8)
- MPI implementation (MPICH2 recommended)
- At least 4GB RAM per node
- Network connectivity between nodes

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt

## Support

For questions, contributions, or discussions:

- Email: [mrasheed@augusta.edu](mailto:mrasheed@augusta.edu)
- GitHub: https://github.com/MohsinRasheed9112/FlexShard_Repo
