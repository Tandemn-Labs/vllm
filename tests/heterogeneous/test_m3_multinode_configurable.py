#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configurable multi-node test for Milestone 3:
Cross-node heterogeneous communication.

This test supports ANY topology configuration via environment variables
or a config file.

Example 1: 2 nodes (4xA10G + 1xA10G = 5 GPUs total)
    export HETERO_TP_SIZES="4,1"
    export HETERO_PP_SIZE="2"
    
    # Node 0 (4xA10G):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=<NODE_0_IP> --master_port=29500 \
        test_m3_multinode_configurable.py
    
    # Node 1 (1xA10G):
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
        --master_addr=<NODE_0_IP> --master_port=29500 \
        test_m3_multinode_configurable.py

Example 2: 3 nodes (4xA10G + 2xT4 + 1xA100 = 7 GPUs total)
    export HETERO_TP_SIZES="4,2,1"
    export HETERO_PP_SIZE="3"
    
Example 3: 4 nodes (2xA10G + 2xA10G + 2xT4 + 2xV100 = 8 GPUs)
    export HETERO_TP_SIZES="2,2,2,2"
    export HETERO_PP_SIZE="4"

Example 4: Single node with mixed TP (8 GPUs on one machine)
    export HETERO_TP_SIZES="4,1,2,1"
    export HETERO_PP_SIZE="4"
    
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
        test_m3_multinode_configurable.py

You can also use a config file:
    export HETERO_CONFIG_FILE="hetero_config.json"
    
Where hetero_config.json contains:
{
    "per_stage_tp_sizes": [4, 1],
    "pipeline_parallel_size": 2,
    "test_sizes_mb": [1, 10, 100],
    "run_bandwidth_test": true,
    "run_latency_test": true,
    "run_stress_test": false
}
"""

import json
import os
import socket
import time

import torch
import torch.distributed as dist

from vllm.distributed.heterogeneous_parallel import (get_current_stage_info,
                                                     is_heterogeneous_mode)
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel,
                                             ensure_model_parallel_initialized,
                                             get_pp_group,
                                             init_distributed_environment)


class HeterogeneousTestConfig:
    """Configuration for heterogeneous testing."""

    def __init__(self):
        """Initialize configuration from environment or file."""

        # Try loading from config file first
        config_file = os.environ.get("HETERO_CONFIG_FILE")
        if config_file and os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
            self.per_stage_tp_sizes = config["per_stage_tp_sizes"]
            self.pipeline_parallel_size = config["pipeline_parallel_size"]
            self.test_sizes_mb = config.get("test_sizes_mb", [1, 10, 100])
            self.run_bandwidth_test = config.get("run_bandwidth_test", True)
            self.run_latency_test = config.get("run_latency_test", True)
            self.run_stress_test = config.get("run_stress_test", False)
            self.iterations = config.get("iterations", 10)
        else:
            # Load from environment variables
            tp_sizes_str = os.environ.get("HETERO_TP_SIZES", "4,1,2,1")
            self.per_stage_tp_sizes = [int(x) for x in tp_sizes_str.split(",")]

            pp_size_str = os.environ.get("HETERO_PP_SIZE")
            if pp_size_str:
                self.pipeline_parallel_size = int(pp_size_str)
            else:
                self.pipeline_parallel_size = len(self.per_stage_tp_sizes)

            # Test configuration
            test_sizes_str = os.environ.get("TEST_SIZES_MB", "1,10,100")
            self.test_sizes_mb = [int(x) for x in test_sizes_str.split(",")]

            self.run_bandwidth_test = os.environ.get("RUN_BANDWIDTH",
                                                     "1") == "1"
            self.run_latency_test = os.environ.get("RUN_LATENCY", "1") == "1"
            self.run_stress_test = os.environ.get("RUN_STRESS", "0") == "1"
            self.iterations = int(os.environ.get("TEST_ITERATIONS", "10"))

        # Validate configuration
        assert len(self.per_stage_tp_sizes) == self.pipeline_parallel_size, \
            f"""TP sizes length {len(self.per_stage_tp_sizes)} !=
            PP size {self.pipeline_parallel_size}"""

        self.world_size = sum(self.per_stage_tp_sizes)

    def get_node_to_stage_mapping(self) -> dict[int, list[int]]:
        """Map node rank to stages it handles."""
        mapping: dict[int, list[int]] = {}
        cumulative_gpus = 0

        for stage_idx, tp_size in enumerate(self.per_stage_tp_sizes):
            # Determine which node(s) this stage spans

            # This is a simplified mapping - adjust based on actual deployment
            # For now, assume stages map to nodes sequentially
            node_rank = stage_idx if stage_idx < int(
                os.environ.get("NNODES", 1)) else 0

            if node_rank not in mapping:
                mapping[node_rank] = []
            mapping[node_rank].append(stage_idx)

            cumulative_gpus += tp_size

        return mapping

    def print_config(self):
        """Print the current configuration."""
        print("\n" + "=" * 70)
        print("HETEROGENEOUS TEST CONFIGURATION")
        print("=" * 70)
        print(f"Pipeline Stages: {self.pipeline_parallel_size}")
        print(f"Per-Stage TP Sizes: {self.per_stage_tp_sizes}")
        print(f"Total GPUs Required: {self.world_size}")
        print(f"Test Sizes (MB): {self.test_sizes_mb}")
        print(f"Test Iterations: {self.iterations}")
        print("Tests Enabled:")
        print(f"  - Bandwidth: {self.run_bandwidth_test}")
        print(f"  - Latency: {self.run_latency_test}")
        print(f"  - Stress: {self.run_stress_test}")
        print("=" * 70)


def get_node_info() -> dict:
    """Get information about current node."""
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "127.0.0.1"

    node_rank = int(os.environ.get("NODE_RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Detect GPU type
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_count = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(
            local_rank).total_memory / (1024**3)
    else:
        gpu_name = "CPU"
        gpu_count = 0
        gpu_memory_gb = 0

    return {
        'hostname': hostname,
        'ip': ip,
        'node_rank': node_rank,
        'local_rank': local_rank,
        'gpu_name': gpu_name,
        'gpu_count': gpu_count,
        'gpu_memory_gb': gpu_memory_gb,
    }


def test_basic_communication(config: HeterogeneousTestConfig):
    """Test basic communication pattern for the configured topology."""

    info = get_current_stage_info()
    node_info = get_node_info()
    pp_group = get_pp_group()
    rank = dist.get_rank()

    print(f"\n[Rank {rank}] Node {node_info['node_rank']}: "
          f"{node_info['hostname']}")
    print(f"[Rank {rank}] GPU: {node_info['gpu_name']} "
          f"({node_info['gpu_memory_gb']:.1f}GB)")
    print(f"[Rank {rank}] Stage {info['stage']}/"
          f"{config.pipeline_parallel_size-1}, TP={info['tp_size']}")

    # Forward pass simulation
    if info['stage'] < config.pipeline_parallel_size - 1:  # Not last stage
        # Create tensor based on TP size
        tensor_size = 1024 * info['tp_size']
        tensor = torch.randn(10, tensor_size, device='cuda')

        tensor_dict = {
            'activations': tensor,
            'stage': info['stage'],
            'node': node_info['node_rank'],
            'gpu_type': node_info['gpu_name'],
        }

        print(f"[Rank {rank}] Stage {info['stage']} → "
              f"Stage {info['stage']+1}")
        pp_group.send_tensor_dict_heterogeneous(tensor_dict)
        print(f"✅ [Rank {rank}] Sent from stage {info['stage']}")

    if info['stage'] > 0:  # Not first stage
        print(f"[Rank {rank}] Waiting to receive at stage {info['stage']}")
        received = pp_group.recv_tensor_dict_heterogeneous()

        prev_stage = received.get('stage', -1)
        prev_node = received.get('node', -1)
        prev_gpu = received.get('gpu_type', 'unknown')

        print(f"✅ [Rank {rank}] Stage {info['stage']} received "
              f"from stage {prev_stage}")
        print(f"   Source: Node {prev_node} ({prev_gpu})")
        print(f"   Shape: {received['activations'].shape}")


def test_bandwidth(config: HeterogeneousTestConfig):
    """Test bandwidth for the configured topology."""

    if not config.run_bandwidth_test:
        return

    info = get_current_stage_info()
    pp_group = get_pp_group()
    rank = dist.get_rank()

    print(f"\n[Rank {rank}] Testing bandwidth for stage {info['stage']}")

    results = []

    for size_mb in config.test_sizes_mb:
        elements = (size_mb * 1024 * 1024) // 4  # float32

        # Adjust size based on TP to keep total communication constant
        elements_per_rank = elements // max(1, info['tp_size'])

        # Only test between consecutive stages
        if info['stage'] == 0:
            tensor = torch.randn(elements_per_rank, device='cuda')
            tensor_dict = {'data': tensor}

            torch.cuda.synchronize()
            start_time = time.time()

            pp_group.send_tensor_dict_heterogeneous(tensor_dict)

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            # Account for gathering if TP > 1
            actual_size_mb = size_mb if info['tp_size'] == 1 else size_mb
            bandwidth = actual_size_mb / elapsed

            results.append({
                'size_mb': size_mb,
                'time_ms': elapsed * 1000,
                'bandwidth_mbps': bandwidth,
            })

            print(f"  Sent {size_mb}MB: {elapsed*1000:.2f}ms "
                  f"({bandwidth:.2f} MB/s)")

        elif info['stage'] == 1:
            received = pp_group.recv_tensor_dict_heterogeneous()
            # Receiver doesn't measure in this simple test
            print(f"  Received {received['data'].shape}")
            pass

    return results


def test_latency(config: HeterogeneousTestConfig):
    """Test latency for small messages."""

    if not config.run_latency_test:
        return

    info = get_current_stage_info()
    pp_group = get_pp_group()
    rank = dist.get_rank()

    print(f"\n[Rank {rank}] Testing latency for stage {info['stage']}")

    latencies = []

    for i in range(config.iterations):
        if info['stage'] == 0:
            # Small tensor for latency test
            tensor = torch.randn(10, device='cuda')
            tensor_dict = {
                'data': tensor,
                'timestamp': time.time(),
                'iteration': i,
            }

            torch.cuda.synchronize()
            pp_group.send_tensor_dict_heterogeneous(tensor_dict)

        elif info['stage'] == 1:
            torch.cuda.synchronize()
            received = pp_group.recv_tensor_dict_heterogeneous()
            torch.cuda.synchronize()

            if 'timestamp' in received:
                e2e_latency = (time.time() - received['timestamp']) * 1000
                latencies.append(e2e_latency)

    if latencies and info['stage'] == 1:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print(f"  Latency over {len(latencies)} iterations:")
        print(f"    Avg: {avg_latency:.2f}ms")
        print(f"    Min: {min_latency:.2f}ms")
        print(f"    Max: {max_latency:.2f}ms")


def test_stress(config: HeterogeneousTestConfig):
    """Stress test with continuous communication."""

    if not config.run_stress_test:
        return

    info = get_current_stage_info()
    pp_group = get_pp_group()
    rank = dist.get_rank()

    print(f"\n[Rank {rank}] Running stress test for stage {info['stage']}")

    # Continuous send/recv for 10 seconds
    start_time = time.time()
    count = 0

    while time.time() - start_time < 10:
        if info['stage'] < config.pipeline_parallel_size - 1:
            tensor = torch.randn(1000, 1000, device='cuda')
            tensor_dict = {'data': tensor, 'seq': count}
            pp_group.send_tensor_dict_heterogeneous(tensor_dict)

        if info['stage'] > 0:
            received = pp_group.recv_tensor_dict_heterogeneous()
            assert 'data' in received

        count += 1

    print(f"✅ [Rank {rank}] Stress test: {count} iterations in 10s")


def verify_topology(config: HeterogeneousTestConfig):
    """Verify the actual topology matches configuration."""

    node_info = get_node_info()
    info = get_current_stage_info()
    rank = dist.get_rank()

    # Gather info from all ranks
    all_info = [None] * dist.get_world_size()
    dist.all_gather_object(
        all_info,
        {
            'rank': rank,
            'node_rank': node_info['node_rank'],
            'stage': info['stage'],
            'tp_size': info['tp_size'],
            'tp_rank': info['tp_rank'],
            'gpu_name': node_info['gpu_name'],
            'gpu_memory': node_info['gpu_memory_gb'],
            'hostname': node_info['hostname'],
        },
    )

    if rank == 0:
        print("\n" + "=" * 70)
        print("TOPOLOGY VERIFICATION")
        print("=" * 70)

        # Group by stage
        stages: dict[int, list[dict]] = {}
        for rank_info in all_info:
            if rank_info is None:
                continue
            stage = rank_info['stage']
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(rank_info)

        # Verify each stage
        for stage_idx in sorted(stages.keys()):
            stage_ranks = stages[stage_idx]
            expected_tp = config.per_stage_tp_sizes[stage_idx]
            actual_tp = len(stage_ranks)

            print(f"\nStage {stage_idx}:")
            print(f"  Expected TP: {expected_tp}, Actual: {actual_tp}")

            if expected_tp != actual_tp:
                print("  ⚠️ WARNING: TP size mismatch!")

            # Group by node within stage
            nodes_in_stage: dict[int, list[dict]] = {}
            for r in stage_ranks:
                node = r['node_rank']
                if node not in nodes_in_stage:
                    nodes_in_stage[node] = []
                nodes_in_stage[node].append(r)

            for node in sorted(nodes_in_stage.keys()):
                node_ranks = nodes_in_stage[node]
                print(f"    Node {node} ({node_ranks[0]['hostname']}):")
                print(f"      GPUs: {len(node_ranks)} x "
                      f"{node_ranks[0]['gpu_name']} "
                      f"({node_ranks[0]['gpu_memory']:.1f}GB)")
                print(f"      Ranks: {[r['rank'] for r in node_ranks]}")
                print(f"      TP ranks: {[r['tp_rank'] for r in node_ranks]}")

        # Summary
        print("\n" + "-" * 70)
        print("Summary:")
        print(f"  Total ranks: {len(all_info)}")
        print(f"  Pipeline stages: {len(stages)}")
        # node_ranks_set = {r['node_rank'] for r in all_info if r is not None}
        # print(f"  Nodes involved: {len(node_ranks_set)}")

        # Check if topology matches config
        all_match = True
        for stage_idx, expected_tp in enumerate(config.per_stage_tp_sizes):
            actual_tp = len(stages.get(stage_idx, []))
            if actual_tp != expected_tp:
                all_match = False
                break

        if all_match:
            print("  ✅ Topology matches configuration!")
        else:
            print("  ❌ Topology does NOT match configuration!")

        print("=" * 70)


def main():
    """Main test function."""

    # Load configuration
    config = HeterogeneousTestConfig()

    # Get environment info
    world_size = int(os.environ.get("WORLD_SIZE", config.world_size))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    nnodes = int(os.environ.get("NNODES", 1))

    if rank == 0:
        config.print_config()
        print("\nRuntime Info:")
        print(f"  World Size: {world_size}")
        print(f"  Nodes: {nnodes}")

    # Validate world size
    if world_size != config.world_size:
        raise RuntimeError(
            f"World size mismatch! Config expects {config.world_size} GPUs "
            f"but got {world_size}. Check your torchrun parameters.")

    # Initialize distributed
    print(f"[Rank {rank}] Initializing (Node {node_rank}, Local {local_rank})")

    # Set CUDA device for this rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )

    # Initialize heterogeneous model parallel
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=max(config.per_stage_tp_sizes),
        pipeline_model_parallel_size=config.pipeline_parallel_size,
        per_stage_tp_sizes=config.per_stage_tp_sizes,
    )

    assert is_heterogeneous_mode(), "Heterogeneous mode not enabled"

    dist.barrier()

    # Verify topology
    verify_topology(config)
    dist.barrier()

    # Run tests
    tests = [
        ("Basic Communication", lambda: test_basic_communication(config)),
        ("Bandwidth Test", lambda: test_bandwidth(config)),
        ("Latency Test", lambda: test_latency(config)),
        ("Stress Test", lambda: test_stress(config)),
    ]

    for test_name, test_func in tests:
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            print('=' * 50)

        dist.barrier()

        try:
            test_func()
            dist.barrier()

            if rank == 0:
                print(f"\n✅ {test_name} PASSED")

        except Exception as e:
            print(f"\n❌ [Rank {rank}] {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED!")
        print("=" * 70)

    # Clean up
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
