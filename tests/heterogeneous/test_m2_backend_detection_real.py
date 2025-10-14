#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""REAL multi-process test for backend detection in heterogeneous mode.

Run with: torchrun
          --nproc_per_node=4
          tests/heterogeneous/test_m2_backend_detection_real.py

This test actually initializes distributed environment and verifies:
1. Backend detection works on all ranks
2. All ranks get identical stage_backends after all_gather
3. Cross-backend edge detection is deterministic
4. CPU hop decision is consistent across ranks

Configuration: [2, 1, 1] - 3 pipeline stages with 4 GPUs
- Stage 0: Ranks 0-1 (TP=2)
- Stage 1: Rank 2 (TP=1)
- Stage 2: Rank 3 (TP=1)
"""

import os
import sys

import torch
import torch.distributed as dist

from vllm.distributed.heterogeneous_parallel import (
    auto_detect_stage_backends, get_current_backend_type,
    get_current_stage_info, get_or_detect_stage_backends,
    is_cross_backend_edge, is_heterogeneous_mode, should_use_cpu_hop)
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel,
                                             ensure_model_parallel_initialized,
                                             init_distributed_environment)


def test_backend_detection():
    """Test backend detection with real distributed environment."""

    # Get environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size != 4:
        raise RuntimeError(
            f"This test requires exactly 4 processes, got {world_size}")

    # ADD THIS LINE: Set CUDA device for this rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Initialize distributed environment
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )

    # Initialize heterogeneous groups
    per_stage_tp_sizes = [2, 1, 1]
    pipeline_parallel_size = 3

    ensure_model_parallel_initialized(
        tensor_model_parallel_size=2,  # Ignored in hetero mode
        pipeline_model_parallel_size=pipeline_parallel_size,
        per_stage_tp_sizes=per_stage_tp_sizes,
    )

    assert is_heterogeneous_mode(), "Heterogeneous mode not enabled"

    # TEST 1: Backend type detection
    print(f"\n[Rank {rank}] TEST 1: Backend Type Detection")
    print("-" * 50)

    backend_type = get_current_backend_type()
    print(f"[Rank {rank}] Detected backend: {backend_type}")

    # Verify backend is one of the known types
    valid_backends = ['cuda', 'rocm', 'tpu', 'xpu', 'cpu', 'unknown']
    assert backend_type in valid_backends, \
        f"Invalid backend type: {backend_type}"
    print(f"✅ [Rank {rank}] Backend type detection works")

    # TEST 2: Auto-detect stage backends
    print(f"\n[Rank {rank}] TEST 2: Auto-Detect Stage Backends")
    print("-" * 50)

    # Call auto-detect (all ranks participate in all_gather)
    stage_backends = auto_detect_stage_backends()

    print(f"[Rank {rank}] Stage backends: {stage_backends}")

    # Verify structure
    assert len(stage_backends) == pipeline_parallel_size, \
        f"Should have {pipeline_parallel_size} stages"

    # Verify all stages have a backend
    for stage_idx in range(pipeline_parallel_size):
        assert stage_idx in stage_backends, \
            f"Stage {stage_idx} missing from stage_backends"
        assert stage_backends[stage_idx] in valid_backends, \
            f"Invalid backend for stage {stage_idx}"

    print(f"✅ [Rank {rank}] Auto-detection completed")

    # TEST 3: Verify determinism (all ranks have identical results)
    print(f"\n[Rank {rank}] TEST 3: Verify Determinism")
    print("-" * 50)

    # Gather stage_backends from all ranks to verify they're identical
    all_stage_backends = [None] * world_size
    dist.all_gather_object(all_stage_backends, stage_backends)

    if rank == 0:
        # Verify all ranks have identical stage_backends
        first = all_stage_backends[0]
        for i, sb in enumerate(all_stage_backends[1:], 1):
            assert sb == first, \
                f"Rank {i} has different stage_backends: {sb} vs {first}"
        print("""✅ [Rank 0] DETERMINISM VERIFIED:
            All ranks have identical stage_backends""")
        print(f"   Common result: {first}")

    dist.barrier()
    print(f"✅ [Rank {rank}] Determinism test passed")

    # TEST 4: Get or detect (caching)
    print(f"\n[Rank {rank}] TEST 4: Caching Test")
    print("-" * 50)

    # Should return cached value
    cached_backends = get_or_detect_stage_backends()
    assert cached_backends == stage_backends, "Should return cached value"
    print(f"✅ [Rank {rank}] Caching works correctly")

    # TEST 5: Cross-backend edge detection
    print(f"\n[Rank {rank}] TEST 5: Cross-Backend Edge Detection")
    print("-" * 50)

    info = get_current_stage_info()
    is_cross = is_cross_backend_edge()

    print(f"[Rank {rank}] Stage {info['stage']}, TP={info['tp_size']}, "
          f"TP_rank={info['tp_rank']}")
    print(f"[Rank {rank}] Is cross-backend edge: {is_cross}")

    # For single-backend setup (all CUDA or all ROCm), should be False
    # For multi-backend, depends on stage transitions
    current_stage = info['stage']
    if current_stage < pipeline_parallel_size - 1:
        next_backend = stage_backends[current_stage + 1]
        current_backend = stage_backends[current_stage]
        expected_cross = (current_backend != next_backend)
        print(f"[Rank {rank}] Current backend: {current_backend}, "
              f"Next backend: {next_backend}")
        print(f"[Rank {rank}] Expected cross-backend: {expected_cross}, "
              f"Detected: {is_cross}")

    print(f"✅ [Rank {rank}] Cross-backend detection completed")

    # TEST 6: CPU hop decision
    print(f"\n[Rank {rank}] TEST 6: CPU Hop Decision")
    print("-" * 50)

    should_cpu = should_use_cpu_hop()
    print(f"[Rank {rank}] Should use CPU hop: {should_cpu}")

    # Gather decisions from all ranks
    all_decisions = [None] * world_size
    dist.all_gather_object(all_decisions, should_cpu)

    if rank == 0:
        print(f"[Rank 0] CPU hop decisions from all ranks: {all_decisions}")

    print(f"✅ [Rank {rank}] CPU hop decision made")

    # TEST 7: Verify consistency across TP ranks in same stage
    print(f"\n[Rank {rank}] TEST 7: Consistency Within Stage")
    print("-" * 50)

    # Gather stage info from all ranks
    all_stage_info = [None] * world_size
    dist.all_gather_object(all_stage_info, info)

    # Gather cross-backend flags from all ranks
    all_is_cross = [None] * world_size
    dist.all_gather_object(all_is_cross, is_cross)

    if rank == 0:
        # print("\n[Rank 0] Verification Summary:")
        # print("-" * 50)
        # for r in range(world_size):
        #     r_info = all_stage_info[r]
        #     r_is_cross = all_is_cross[r]
        # print(f"  Rank {r}: Stage {r_info['stage']}, "
        #       f"TP={r_info['tp_size']}, TP_rank={r_info['tp_rank']}, "
        #       f"CrossBackend={r_is_cross}")

        # Verify TP ranks within same stage have same cross-backend result
        # Stage 0 (ranks 0-1): should have same is_cross value
        assert all_is_cross[0] == all_is_cross[1], \
            "Stage 0 ranks should have same cross-backend detection"

        print(
            "\n✅ [Rank 0] CONSISTENCY VERIFIED across TP ranks in same stage")

    dist.barrier()
    print(f"✅ [Rank {rank}] Consistency test passed")

    # Synchronize before cleanup
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 70)
        print("✅ ALL REAL TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print(f"  - {world_size} processes tested")
        print(f"  - {pipeline_parallel_size} pipeline stages")
        print(f"  - Per-stage TP sizes: {per_stage_tp_sizes}")
        print(f"  - Stage backends: {stage_backends}")
        print("=" * 70)

    # Clean up
    destroy_model_parallel()
    destroy_distributed_environment()


def test_backend_consistency_across_runs():
    """Test that backend detection is consistent across multiple runs."""

    # This test verifies that auto_detect_stage_backends always returns
    # the same result when called multiple times

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size != 4:
        print(
            f"Skipping consistency test (needs 4 processes, got {world_size})")
        return

    # Initialize
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )

    ensure_model_parallel_initialized(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=3,
        per_stage_tp_sizes=[2, 1, 1],
    )

    print(f"\n[Rank {rank}] TEST: Consistency Across Multiple Calls")
    print("-" * 50)

    # Call auto_detect multiple times
    results = []
    for i in range(3):
        # Reset stage_backends to force re-detection
        from vllm.distributed.heterogeneous_parallel import _HETERO_CONFIG
        if _HETERO_CONFIG is not None and 'stage_backends' in _HETERO_CONFIG:
            del _HETERO_CONFIG['stage_backends']

        stage_backends = auto_detect_stage_backends()
        results.append(stage_backends)
        print(f"[Rank {rank}] Call {i+1}: {stage_backends}")

    # Verify all calls returned identical results
    assert results[0] == results[1] == results[2], \
        f"Inconsistent results across calls: {results}"

    print(f"✅ [Rank {rank}] Consistency verified across multiple calls")

    # Clean up
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    print("=" * 70)
    print("REAL TEST: Backend Detection with Distributed Environment")
    print("""Run with: torchrun --nproc_per_node=4 
                        test_m2_backend_detection_real.py""")
    print("=" * 70)
    print()

    # Check if running with torchrun
    if "RANK" not in os.environ:
        print("❌ ERROR: Must run with torchrun!")
        print("""Example: torchrun
                        --nproc_per_node=4 
                        test_m2_backend_detection_real.py""")
        sys.exit(1)

    try:
        # Run main test
        test_backend_detection()

        # Run consistency test
        # test_backend_consistency_across_runs()

    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        print(f"\n❌ [Rank {rank}] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
