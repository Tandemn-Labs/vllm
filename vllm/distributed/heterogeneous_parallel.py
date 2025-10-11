# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Heterogeneous parallel state management for mixed GPU deployments.

This module provides global state management and query functions for
heterogeneous tensor parallelism across pipeline stages with different TP sizes.
"""

from typing import Optional

# Global state for heterogeneous configuration
_HETERO_CONFIG: Optional[dict] = None


def set_heterogeneous_config(config: dict) -> None:
    """Set the heterogeneous configuration globally.
    
    Args:
        config: Dictionary containing heterogeneous configuration:
            - 'per_stage_tp_sizes': List of TP sizes per stage
            - 'rank_to_stage_info': Mapping from rank to stage information
            - 'pipeline_parallel_size': Number of pipeline stages
    """
    global _HETERO_CONFIG
    _HETERO_CONFIG = config


def get_heterogeneous_config() -> Optional[dict]:
    """Get the heterogeneous configuration.
    
    Returns:
        Dictionary containing heterogeneous configuration, or None if not set.
    """
    return _HETERO_CONFIG


def is_heterogeneous_mode() -> bool:
    """Check if heterogeneous mode is enabled.
    
    Returns:
        True if heterogeneous configuration is set, False otherwise.
    """
    return _HETERO_CONFIG is not None


def get_current_stage_info() -> dict:
    """Get current process's stage information.
    
    Returns:
        Dictionary containing:
            - 'stage': Pipeline stage index
            - 'tp_size': TP size for this stage
            - 'tp_rank': TP rank within the stage
            - 'pp_rank': PP rank (same as stage)
    
    Raises:
        ValueError: If heterogeneous mode is not enabled or rank not found.
    """
    if not is_heterogeneous_mode():
        raise ValueError("Heterogeneous mode is not enabled")

    assert _HETERO_CONFIG is not None  # For mypy
    import torch.distributed as dist
    rank = dist.get_rank()

    if rank not in _HETERO_CONFIG['rank_to_stage_info']:
        raise ValueError(
            f"Rank {rank} not found in heterogeneous configuration")

    info = _HETERO_CONFIG['rank_to_stage_info'][rank]
    # Add pp_rank for convenience
    info['pp_rank'] = info['stage']
    return info


def get_next_stage_tp_size() -> Optional[int]:
    """Get TP size of the next pipeline stage.
    
    Returns:
        TP size of next stage, or None if this is the last stage.
    """
    if not is_heterogeneous_mode():
        return None

    assert _HETERO_CONFIG is not None  # For mypy
    info = get_current_stage_info()
    next_stage = info['stage'] + 1

    per_stage_tp_sizes = _HETERO_CONFIG['per_stage_tp_sizes']
    if next_stage >= len(per_stage_tp_sizes):
        return None  # Last stage

    return per_stage_tp_sizes[next_stage]


def get_prev_stage_tp_size() -> Optional[int]:
    """Get TP size of the previous pipeline stage.
    
    Returns:
        TP size of previous stage, or None if this is the first stage.
    """
    if not is_heterogeneous_mode():
        return None

    assert _HETERO_CONFIG is not None  # For mypy
    info = get_current_stage_info()
    prev_stage = info['stage'] - 1

    if prev_stage < 0:
        return None  # First stage

    per_stage_tp_sizes = _HETERO_CONFIG['per_stage_tp_sizes']
    return per_stage_tp_sizes[prev_stage]


def get_heterogeneous_pp_groups() -> dict[str, list]:
    """Get the PP group configuration for heterogeneous setup.
    
    Returns:
        Dictionary with 'main' PP group and 'dummy' groups.
        
    Note:
        SIMPLIFIED APPROACH: Only TP rank 0 from each stage participates
        in the main PP group. Other ranks get dummy single-rank groups.
    """
    if not is_heterogeneous_mode():
        return {}

    assert _HETERO_CONFIG is not None  # For mypy
    per_stage_tp_sizes = _HETERO_CONFIG['per_stage_tp_sizes']
    pp_groups: dict[str, list] = {}

    # Build the main PP group with only TP rank 0 from each stage
    main_pp_ranks = []
    gpu_offset = 0
    for stage_idx, tp_size in enumerate(per_stage_tp_sizes):
        # Always take the first rank (TP rank 0) from each stage
        main_pp_ranks.append(gpu_offset)
        gpu_offset += tp_size

    pp_groups['main'] = main_pp_ranks  # e.g., [0, 4, 5, 7]

    # Create list of dummy single-rank groups
    dummy_groups: list[list[int]] = []
    gpu_offset = 0
    for stage_idx, tp_size in enumerate(per_stage_tp_sizes):
        if tp_size > 1:
            # For stages with TP > 1, create dummy groups for TP ranks 1,2,3...
            for tp_rank in range(1, tp_size):
                dummy_rank = gpu_offset + tp_rank
                dummy_groups.append([dummy_rank])
        gpu_offset += tp_size

    pp_groups['dummy'] = dummy_groups  # e.g., [[1], [2], [3], [6]]

    return pp_groups


def is_pp_primary_rank() -> bool:
    """Check if current rank is the primary PP communicator for its stage.
    
    In heterogeneous mode with simplified PP groups, only TP rank 0 from each
    stage participates in actual PP communication. Other ranks have dummy
    single-rank PP groups.
    
    Returns:
        True if this rank is TP rank 0 in its stage, False otherwise.
        Returns True if not in heterogeneous mode (for compatibility).
    """
    if not is_heterogeneous_mode():
        return True  # In uniform mode, all ranks participate

    info = get_current_stage_info()
    return info['tp_rank'] == 0


def reset_heterogeneous_config() -> None:
    """Reset the heterogeneous configuration.
    
    This should be called when destroying the model parallel groups.
    """
    global _HETERO_CONFIG
    _HETERO_CONFIG = None
