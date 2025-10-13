#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test script for Milestone 2: Parallel State & Group Creation
Run with: torchrun --nproc_per_node=4 test_m2_parallel_groups.py

Configuration: [2, 1, 1] - 3 pipeline stages with 4 total GPUs
- Stage 0: Ranks 0-1 (TP=2)
- Stage 1: Rank 2 (TP=1)  
- Stage 2: Rank 3 (TP=1)
"""

import os

import torch
import torch.distributed as dist

from vllm.distributed.heterogeneous_parallel import (get_current_stage_info,
                                                     get_next_stage_tp_size,
                                                     get_prev_stage_tp_size,
                                                     is_heterogeneous_mode)
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel,
                                             ensure_model_parallel_initialized,
                                             get_pp_group, get_tp_group,
                                             init_distributed_environment)


def test_group_creation():
    """Test heterogeneous group creation with [2, 1, 1] configuration."""

    # Initialize distributed environment
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size != 4:
        raise RuntimeError(
            f"This test requires exactly 4 GPUs, got {world_size}")

    # Initialize PyTorch distributed
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )

    # Initialize heterogeneous model parallel groups
    per_stage_tp_sizes = [2, 1, 1]
    pipeline_parallel_size = 3

    ensure_model_parallel_initialized(
        tensor_model_parallel_size=2,  # Ignored in hetero mode
        pipeline_model_parallel_size=pipeline_parallel_size,
        per_stage_tp_sizes=per_stage_tp_sizes,
    )

    # Verify heterogeneous mode is active
    assert is_heterogeneous_mode(), "Heterogeneous mode not enabled"

    # Get current stage info
    info = get_current_stage_info()
    rank = dist.get_rank()

    print(f"Rank {rank}: Stage {info['stage']}, TP size {info['tp_size']}, "
          f"TP rank {info['tp_rank']}")

    # Verify stage assignment based on rank
    if rank < 2:
        assert info['stage'] == 0, f"Rank {rank} should be in stage 0"
        assert info['tp_size'] == 2, "Stage 0 should have TP=2"
        assert info[
            'tp_rank'] == rank, f"Rank {rank} should have tp_rank={rank}"
    elif rank == 2:
        assert info['stage'] == 1, "Rank 2 should be in stage 1"
        assert info['tp_size'] == 1, "Stage 1 should have TP=1"
        assert info['tp_rank'] == 0, "Rank 2 should have tp_rank=0"
    elif rank == 3:
        assert info['stage'] == 2, "Rank 3 should be in stage 2"
        assert info['tp_size'] == 1, "Stage 2 should have TP=1"
        assert info['tp_rank'] == 0, "Rank 3 should have tp_rank=0"

    print(f"✅ Rank {rank}: Stage assignment verified")

    # Test next/prev stage TP size queries
    next_tp = get_next_stage_tp_size()
    prev_tp = get_prev_stage_tp_size()

    if info['stage'] == 0:
        assert next_tp == 1, f"Stage 0 next should be TP=1, got {next_tp}"
        assert prev_tp is None, "Stage 0 should have no prev"
    elif info['stage'] == 1:
        assert next_tp == 1, f"Stage 1 next should be TP=1, got {next_tp}"
        assert prev_tp == 2, f"Stage 1 prev should be TP=2, got {prev_tp}"
    elif info['stage'] == 2:
        assert next_tp is None, "Stage 2 should have no next"
        assert prev_tp == 1, f"Stage 2 prev should be TP=1, got {prev_tp}"

    print(f"✅ Rank {rank}: Next/prev TP sizes verified")

    # Test TP group
    tp_group = get_tp_group()
    assert tp_group is not None, "TP group should be initialized"
    assert tp_group.world_size == info['tp_size'], \
        f"TP group size mismatch: {tp_group.world_size} vs {info['tp_size']}"

    print(f"✅ Rank {rank}: TP group verified (size={tp_group.world_size})")

    # Test PP group
    pp_group = get_pp_group()
    assert pp_group is not None, "PP group should be initialized"
    # For heterogeneous mode, verify correct PP group assignment
    if rank == 1:
        # Rank 1 should be in dummy single-rank PP group
        assert pp_group.world_size == 1, \
            f"""Rank 1 should be in dummy PP group (size=1),
             got {pp_group.world_size}"""
        assert pp_group.ranks == [1], \
            f"Rank 1 PP group should be [1], got {pp_group.ranks}"
        print(
            f"✅ Rank {rank}: Dummy PP group verified (ranks={pp_group.ranks})")
    else:
        # Ranks 0, 2, 3 should be in main PP group
        assert pp_group.world_size == pipeline_parallel_size, \
            f"""Main PP group size mismatch: 
            {pp_group.world_size} vs {pipeline_parallel_size}"""
        assert set(pp_group.ranks) == {0, 2, 3}, \
            f"Main PP group should be [0, 2, 3], got {pp_group.ranks}"
        print(
            f"✅ Rank {rank}: Main PP group verified (ranks={pp_group.ranks})")

    # Synchronize all ranks
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED for Milestone 2!")
        print("=" * 50)

    # Clean up
    destroy_model_parallel()
    destroy_distributed_environment()


def test_pp_connectivity():
    """Test PP group connectivity between stages."""

    # This test verifies the PP group structure
    # For [2, 1, 1] configuration, expected PP groups:
    # Main PP group: [0, 2, 3] (TP rank 0 from each stage)
    # Dummy PP group: [1] (single rank group for TP rank 1 in stage 0)

    pp_group = get_pp_group()
    rank = dist.get_rank()

    # Verify PP group structure
    if rank == 1:
        # Rank 1 should be in a dummy single-rank PP group
        assert len(
            pp_group.ranks) == 1, "Rank 1 should be in single-rank PP group"
        assert pp_group.ranks[
            0] == 1, "Rank 1 PP group should only contain itself"
        print(f"✅ Rank {rank} (dummy PP group): {pp_group.ranks}")
    else:
        # Ranks 0, 2, 3 should be in the main PP group
        assert len(pp_group.ranks
                   ) == 3, "Main PP group should have 3 ranks (one per stage)"
        assert set(pp_group.ranks) == {
            0, 2, 3
        }, f"Main PP group should be [0, 2, 3], got {pp_group.ranks}"
        print(f"✅ Rank {rank} (main PP group): {pp_group.ranks}")

    print(f"✅ Rank {rank}: PP connectivity verified")


if __name__ == "__main__":
    test_group_creation()
    test_pp_connectivity()
