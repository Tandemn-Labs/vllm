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


def get_next_stage_pp_rank_0() -> Optional[int]:
    """Get the global rank of the next pipeline stage's TP rank 0.
    
    Returns:
        Global rank of next stage's TP rank 0,
         or None if this is the last stage.
        
    Note:
        In heterogeneous mode, only TP rank 0 from each stage participates
        in the main PP group, so this returns the rank that would receive
        data in a pipeline transfer.
    """
    if not is_heterogeneous_mode():
        return None

    # Get current stage info
    current_stage_info = get_current_stage_info()
    current_stage = current_stage_info['stage']

    # Get PP groups configuration
    pp_groups = get_heterogeneous_pp_groups()
    pp_groups_main = pp_groups['main']  # e.g., [0, 4, 5, 7]

    # Find next stage index
    next_stage_idx = current_stage + 1

    # Check if next stage exists
    if next_stage_idx < len(pp_groups_main):
        return pp_groups_main[next_stage_idx]

    # This is the last stage
    return None


def get_prev_stage_pp_rank_0() -> Optional[int]:
    """Get the global rank of the previous pipeline stage's TP rank 0.
    
    Returns:
        Global rank of previous stage's TP rank 0,
         or None if this is the first stage.
        
    Note:
        In heterogeneous mode, only TP rank 0 from each stage participates
        in the main PP group, so this returns the rank that would send
        data in a pipeline transfer.
    """
    if not is_heterogeneous_mode():
        return None

    # Get current stage info
    current_stage_info = get_current_stage_info()
    current_stage = current_stage_info['stage']

    # Get PP groups configuration
    pp_groups = get_heterogeneous_pp_groups()
    pp_groups_main = pp_groups['main']  # e.g., [0, 4, 5, 7]

    # Find previous stage index
    prev_stage_idx = current_stage - 1

    # Check if previous stage exists
    if prev_stage_idx >= 0:
        return pp_groups_main[prev_stage_idx]

    # This is the first stage
    return None


def get_stage_pp_rank_0(stage_idx: int) -> Optional[int]:
    """Get the global rank of a specific pipeline stage's TP rank 0.
    
    Args:
        stage_idx: Pipeline stage index (0-based).
        
    Returns:
        Global rank of the stage's TP rank 0, or None if stage doesn't exist.
        
    Note:
        This is a general helper that can get any stage's primary PP rank.
    """
    if not is_heterogeneous_mode():
        return None

    # Get PP groups configuration
    pp_groups = get_heterogeneous_pp_groups()
    pp_groups_main = pp_groups['main']  # e.g., [0, 4, 5, 7]

    # Check if stage exists
    if 0 <= stage_idx < len(pp_groups_main):
        return pp_groups_main[stage_idx]

    return None


def reset_heterogeneous_config() -> None:
    """Reset the heterogeneous configuration.
    
    This should be called when destroying the model parallel groups.
    """
    global _HETERO_CONFIG
    _HETERO_CONFIG = None


def get_current_backend_type() -> str:
    """Detect the current hardware backend type.
    
    Returns:
        String identifier: 'cuda', 'rocm', 'tpu', 'xpu', or 'cpu'
        
    Note:
        This uses vLLM's platform detection system to determine the
        underlying hardware runtime.
    """
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        # NVIDIA GPUs with CUDA runtime
        return 'cuda'
    elif current_platform.is_rocm():
        # AMD GPUs with ROCm runtime
        return 'rocm'
    elif current_platform.is_tpu():
        # Google TPUs with XLA runtime
        return 'tpu'
    elif current_platform.is_xpu():
        # Intel XPUs
        return 'xpu'
    elif current_platform.is_cpu():
        # CPU-only
        return 'cpu'
    else:
        # Fallback for unrecognized platforms
        return 'unknown'


# def is_cross_backend_edge() -> bool:
#     """Check if current PP edge crosses hardware backend boundaries.
#     In heterogenous mode, we assume that the current node is homogenous.

#     We check here if the rank is sending to or
#     receiving from a different hardware backend (e.g., CUDA → ROCm).
#     When true, communication should use CPU Backend.
#     """
#     # check if heterogeneous mode is enabled
#     if not is_heterogeneous_mode():
#         return False
#     # double check
#     assert _HETERO_CONFIG is not None

#     # Get current stage info
#     info = get_current_stage_info()

#     # TODO (hetarth): ideally we need to check the backend of the
#     # next and previous stage
#     # Check if stage backends are available for explicit backend comparison
#     try:
#         # naively assume cross-backend edge when TP sizes differ
#         next_tp = get_next_stage_tp_size()
#         prev_tp = get_prev_stage_tp_size()
#         # current_tp = info['tp_size']

#         stage_backends = get_or_detect_stage_backends()
#         current_stage = info['stage']
#         current_backend = stage_backends.get(current_stage)

#         if next_tp is not None:
#             # Check next stage backend
#             next_stage = current_stage + 1
#             if next_stage in stage_backends:
#                 next_backend = stage_backends[next_stage]
#                 if current_backend != next_backend:
#                     return True

#         if prev_tp is not None:
#             # Check previous stage backend
#             prev_stage = current_stage - 1
#             if prev_stage in stage_backends:
#                 prev_backend = stage_backends[prev_stage]
#                 if current_backend != prev_backend:
#                     return True
#     except Exception:
#         pass

#     # if next_tp is not None and next_tp != current_tp:
#     #     return True
#     # if prev_tp is not None and prev_tp != current_tp:
#     #     return True

#     return False

# def should_use_cpu_hop() -> bool:
#     """Determine if CPU hop should be used for current PP communication.

#     CPU hop is required when:
#     1. Crossing backend boundaries (CUDA ↔ ROCm ↔ TPU)
#     2. Different TP sizes between stages (gather/broadcast required)

#     Returns:
#         True if CPU hop should be used, False if direct device-to-device is OK
#     """
#     if not is_heterogeneous_mode():
#         return False

#     # Always use CPU hop for cross-backend edges
#     return bool(is_cross_backend_edge())


def get_stage_backends() -> dict[int, str]:
    """Get stage backend info.
    
    Returns:
        Dictionary mapping stage index to backend type
        
    This is a convenience function that returns 
    existing stage_backends if already set
    """
    if not is_heterogeneous_mode():
        raise ValueError("Heterogeneous mode must be enabled")

    assert _HETERO_CONFIG is not None

    # Check if already populated
    if 'stage_backends' in _HETERO_CONFIG and _HETERO_CONFIG['stage_backends']:
        return _HETERO_CONFIG['stage_backends']
    else:
        raise ValueError("Dict of stage backends were not set")


def auto_detect_stage_backends() -> dict:
    """
    This should automatically detect the backend
     for all stages on all workers.
    NOTE: Always call this function after the
     heterogenous config is set,
    however BEFORE any cross-backend comms are done.
    This function detects the backend for
     all stages using get_current_backend_type
    and then gathers this from all ranks. 
    This just adds everything to the heterogenous config
     as a hashmap.
    """
    # mandatory checks
    if not is_heterogeneous_mode():
        return {}
    # mandatory checks
    assert _HETERO_CONFIG is not None
    import torch.distributed as dist

    # Step 1: get current rank's backend
    rank = dist.get_rank()
    current_backend = get_current_backend_type()
    # Step 2: Gather this info from all ranks
    all_backends = [None] * dist.get_world_size()
    # do the all gather using current_backend
    dist.all_gather_object(all_backends, current_backend)
    # the way this works is that =
    # Each rank has detected its OWN backend:
    # Rank 0 (T4):     current_backend = 'cuda'
    # Rank 1 (T4):     current_backend = 'cuda'
    # Rank 2 (T4):     current_backend = 'cuda'
    # Rank 3 (T4):     current_backend = 'cuda'
    # Rank 4 (MI300x): current_backend = 'rocm'
    # ... etc

    # # Each rank has its own OUTPUT list (initially filled with None):
    # Rank 0: all_backends = [None, None, None, None, None, None, None, None]
    # Rank 1: all_backends = [None, None, None, None, None, None, None, None]
    # ...etc (all ranks have empty lists)
    # all-gather fills the info up for all ranks

    # Step 3: Map stages to the backends
    per_stage_tp_sizes = _HETERO_CONFIG['per_stage_tp_sizes']
    stage_backends = {}

    gpu_offset = 0
    for stage_idx, tp_size in enumerate(per_stage_tp_sizes):
        # Collect all backends for this stage
        stage_rank_backends: list[str] = []
        for tp_rank in range(tp_size):
            global_rank = gpu_offset + tp_rank
            backend = all_backends[global_rank]
            if backend is not None:
                stage_rank_backends.append(backend)

        # Verify all ranks in the stage have the
        #  same backend (homogeneity within node)
        unique_backends = set(stage_rank_backends)
        if len(unique_backends) != 1:
            raise ValueError(
                f"Stage {stage_idx} has mixed backends: {unique_backends}. "
                f"This violates the assumption of homogeneity within a node. ")

        # Use the backend from TP rank 0 as representative for the stage
        stage_backends[stage_idx] = stage_rank_backends[0]
        gpu_offset += tp_size

    # Step 4: Set the stage backends in the heterogenous config
    _HETERO_CONFIG['stage_backends'] = stage_backends

    # Step 5: Log the detected configuration (only from rank 0)
    if rank == 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Auto-detected stage backends:")
        for stage_idx, backend in stage_backends.items():
            tp_size = per_stage_tp_sizes[stage_idx]
            logger.info("  Stage %s (TP=%s): %s", stage_idx, tp_size, backend)

    return stage_backends
