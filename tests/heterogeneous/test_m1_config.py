# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test configuration parsing and validation for heterogeneous TP+PP.

This test validates Milestone 1: Configuration Layer changes for supporting
heterogeneous tensor parallel sizes across pipeline parallel stages.
"""

import pytest

from vllm.config.parallel import ParallelConfig


def test_config_parsing():
    """Test that per-stage TP sizes are parsed correctly."""
    config = ParallelConfig(
        pipeline_parallel_size=4,
        tensor_parallel_size=1,  # Not used when per_stage_tp_sizes is set
        per_stage_tp_sizes=[4, 1, 2, 1],
    )

    # Verify heterogeneous mode is detected
    assert config.is_heterogeneous(), \
        "is_heterogeneous() should return True"

    # Verify correct TP size returned for each stage
    assert config.get_tp_size_for_stage(0) == 4, \
        "Stage 0 should have TP=4"
    assert config.get_tp_size_for_stage(1) == 1, \
        "Stage 1 should have TP=1"
    assert config.get_tp_size_for_stage(2) == 2, \
        "Stage 2 should have TP=2"
    assert config.get_tp_size_for_stage(3) == 1, \
        "Stage 3 should have TP=1"

    # Verify world_size is sum of all TP sizes
    assert config.world_size == 8, \
        "World size should be 4+1+2+1=8"


def test_config_uniform_mode():
    """Test that uniform TP mode still works (backward compatibility)."""
    config = ParallelConfig(
        pipeline_parallel_size=2,
        tensor_parallel_size=4,
        per_stage_tp_sizes=None,  # Use uniform TP
    )

    # Verify heterogeneous mode is NOT detected
    assert not config.is_heterogeneous(), \
        "is_heterogeneous() should return False"

    # Verify all stages return same TP size
    assert config.get_tp_size_for_stage(0) == 4, \
        "Stage 0 should have TP=4"
    assert config.get_tp_size_for_stage(1) == 4, \
        "Stage 1 should have TP=4"

    # Verify world_size follows uniform calculation
    assert config.world_size == 8, \
        "World size should be 2*4=8"


def test_config_validation_wrong_length():
    """Test that wrong per_stage_tp_sizes length raises ValueError."""
    with pytest.raises(ValueError, match="per_stage_tp_sizes length"):
        ParallelConfig(
            pipeline_parallel_size=4,
            tensor_parallel_size=1,
            per_stage_tp_sizes=[4, 1, 2],  # Only 3 values for 4 stages
        )


def test_config_validation_negative_value():
    """Test that negative TP size raises ValueError."""
    with pytest.raises(ValueError, match="must be positive integers"):
        ParallelConfig(
            pipeline_parallel_size=4,
            tensor_parallel_size=1,
            per_stage_tp_sizes=[4, -1, 2, 1],  # Negative value
        )


def test_config_validation_zero_value():
    """Test that zero TP size raises ValueError."""
    with pytest.raises(ValueError, match="must be positive integers"):
        ParallelConfig(
            pipeline_parallel_size=4,
            tensor_parallel_size=1,
            per_stage_tp_sizes=[4, 0, 2, 1],  # Zero value
        )


def test_config_hash_includes_heterogeneous():
    """Test that compute_hash includes per_stage_tp_sizes."""
    config1 = ParallelConfig(
        pipeline_parallel_size=4,
        tensor_parallel_size=2,
        per_stage_tp_sizes=[4, 1, 2, 1],
    )

    config2 = ParallelConfig(
        pipeline_parallel_size=4,
        tensor_parallel_size=2,
        per_stage_tp_sizes=[2, 2, 2, 2],
    )

    # Different per_stage_tp_sizes should produce different hashes
    assert config1.compute_hash() != config2.compute_hash(), \
        "Different heterogeneous configs should have different hashes"


def test_config_single_gpu_stages():
    """Test configuration with multiple single-GPU stages."""
    config = ParallelConfig(
        pipeline_parallel_size=5,
        tensor_parallel_size=1,
        per_stage_tp_sizes=[4, 1, 1, 2, 1],
    )

    assert config.is_heterogeneous()
    assert config.world_size == 9  # 4+1+1+2+1
    assert config.get_tp_size_for_stage(1) == 1
    assert config.get_tp_size_for_stage(2) == 1
    assert config.get_tp_size_for_stage(4) == 1


def test_config_all_same_tp_with_explicit_list():
    """Test that explicit list with all same values works."""
    config = ParallelConfig(
        pipeline_parallel_size=3,
        tensor_parallel_size=1,
        per_stage_tp_sizes=[2, 2, 2],
    )

    # Even though all are same, it's still heterogeneous mode
    assert config.is_heterogeneous()
    assert config.world_size == 6
    assert all(config.get_tp_size_for_stage(i) == 2 for i in range(3))


def test_config_heterogeneous_without_tensor_parallel_size():
    """Test that heterogeneous mode works when 
    tensor_parallel_size is not explicitly set."""
    config = ParallelConfig(
        pipeline_parallel_size=4,
        # tensor_parallel_size not specified, should default to 1
        per_stage_tp_sizes=[4, 1, 2, 1],
    )

    # Should still be considered heterogeneous
    assert config.is_heterogeneous()
    assert config.world_size == 8  # 4+1+2+1
    assert config.get_tp_size_for_stage(0) == 4
    assert config.get_tp_size_for_stage(1) == 1
    assert config.get_tp_size_for_stage(2) == 2
    assert config.get_tp_size_for_stage(3) == 1

    # tensor_parallel_size should default to 1 but per_stage_tp_sizes takes
    # precedence
    assert config.tensor_parallel_size == 1


if __name__ == "__main__":
    # Run tests
    print("Running Milestone 1 Configuration Tests...")
    pytest.main([__file__, "-v"])
