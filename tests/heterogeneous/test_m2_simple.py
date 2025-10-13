#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Just a configuration test for Heterogenous TP+PP using a dummy config
   for per_stage_tp_sizes=[4, 1, 2, 1]
"""

import unittest
from unittest.mock import patch

from vllm.config import ParallelConfig
from vllm.distributed.heterogeneous_parallel import (
    get_current_stage_info, get_heterogeneous_config,
    get_heterogeneous_pp_groups, get_next_stage_tp_size,
    get_prev_stage_tp_size, is_heterogeneous_mode, is_pp_primary_rank,
    reset_heterogeneous_config, set_heterogeneous_config)


class TestHeterogeneousParallel(unittest.TestCase):
    """Test heterogeneous parallel state management."""

    def setUp(self):
        """Set up test configuration."""
        self.test_config = {
            'per_stage_tp_sizes': [4, 1, 2, 1],
            'rank_to_stage_info': {
                0: {
                    'stage': 0,
                    'tp_size': 4,
                    'tp_rank': 0
                },
                1: {
                    'stage': 0,
                    'tp_size': 4,
                    'tp_rank': 1
                },
                2: {
                    'stage': 0,
                    'tp_size': 4,
                    'tp_rank': 2
                },
                3: {
                    'stage': 0,
                    'tp_size': 4,
                    'tp_rank': 3
                },
                4: {
                    'stage': 1,
                    'tp_size': 1,
                    'tp_rank': 0
                },
                5: {
                    'stage': 2,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                6: {
                    'stage': 2,
                    'tp_size': 2,
                    'tp_rank': 1
                },
                7: {
                    'stage': 3,
                    'tp_size': 1,
                    'tp_rank': 0
                },
            },
            'pipeline_parallel_size': 4,
        }

    def tearDown(self):
        """Clean up after test."""
        reset_heterogeneous_config()

    def test_config_management(self):
        """Test setting and getting heterogeneous configuration."""
        # Initially not in heterogeneous mode
        self.assertFalse(is_heterogeneous_mode())
        self.assertIsNone(get_heterogeneous_config())

        # Set configuration
        set_heterogeneous_config(self.test_config)

        # Now in heterogeneous mode
        self.assertTrue(is_heterogeneous_mode())
        self.assertEqual(get_heterogeneous_config(), self.test_config)

        # Reset configuration
        reset_heterogeneous_config()
        self.assertFalse(is_heterogeneous_mode())
        self.assertIsNone(get_heterogeneous_config())

        print("✅ Config management test passed")

    @patch('torch.distributed.get_rank')
    def test_stage_info(self, mock_get_rank):
        """Test getting current stage information."""
        set_heterogeneous_config(self.test_config)

        # Test rank 0 (stage 0, TP=4)
        mock_get_rank.return_value = 0
        info = get_current_stage_info()
        self.assertEqual(info['stage'], 0)
        self.assertEqual(info['tp_size'], 4)
        self.assertEqual(info['tp_rank'], 0)
        self.assertEqual(info['pp_rank'], 0)

        # Test rank 4 (stage 1, TP=1)
        mock_get_rank.return_value = 4
        info = get_current_stage_info()
        self.assertEqual(info['stage'], 1)
        self.assertEqual(info['tp_size'], 1)
        self.assertEqual(info['tp_rank'], 0)

        # Test rank 6 (stage 2, TP=2)
        mock_get_rank.return_value = 6
        info = get_current_stage_info()
        self.assertEqual(info['stage'], 2)
        self.assertEqual(info['tp_size'], 2)
        self.assertEqual(info['tp_rank'], 1)

        print("✅ Stage info test passed")

    @patch('torch.distributed.get_rank')
    def test_next_prev_tp_size(self, mock_get_rank):
        """Test getting next/prev stage TP sizes."""
        set_heterogeneous_config(self.test_config)

        # Stage 0 -> next=1, prev=None
        mock_get_rank.return_value = 0
        self.assertEqual(get_next_stage_tp_size(), 1)
        self.assertIsNone(get_prev_stage_tp_size())

        # Stage 1 -> next=2, prev=4
        mock_get_rank.return_value = 4
        self.assertEqual(get_next_stage_tp_size(), 2)
        self.assertEqual(get_prev_stage_tp_size(), 4)

        # Stage 2 -> next=1, prev=1
        mock_get_rank.return_value = 5
        self.assertEqual(get_next_stage_tp_size(), 1)
        self.assertEqual(get_prev_stage_tp_size(), 1)

        # Stage 3 -> next=None, prev=2
        mock_get_rank.return_value = 7
        self.assertIsNone(get_next_stage_tp_size())
        self.assertEqual(get_prev_stage_tp_size(), 2)

        print("✅ Next/prev TP size test passed")

    def test_pp_groups(self):
        """Test PP group configuration with simplified approach."""
        set_heterogeneous_config(self.test_config)

        pp_groups = get_heterogeneous_pp_groups()

        # Should have 'main' and 'dummy' groups
        self.assertIn('main', pp_groups)
        self.assertIn('dummy', pp_groups)

        # Expected configuration for [4,1,2,1]:
        # Main PP group: [0, 4, 5, 7] - only TP rank 0 from each stage
        # Dummy groups: [[1], [2], [3], [6]] - other ranks get single-rank
        # groups

        self.assertEqual(pp_groups['main'], [0, 4, 5, 7])
        self.assertEqual(pp_groups['dummy'], [[1], [2], [3], [6]])

        print("✅ PP groups test passed (simplified approach)")

    @patch('torch.distributed.get_rank')
    def test_pp_primary_rank(self, mock_get_rank):
        """Test PP primary rank identification."""
        set_heterogeneous_config(self.test_config)

        # Test rank 0 (stage 0, TP rank 0) - should be primary
        mock_get_rank.return_value = 0
        self.assertTrue(is_pp_primary_rank())

        # Test rank 1 (stage 0, TP rank 1) - should NOT be primary
        mock_get_rank.return_value = 1
        self.assertFalse(is_pp_primary_rank())

        # Test rank 4 (stage 1, TP rank 0) - should be primary (TP=1)
        mock_get_rank.return_value = 4
        self.assertTrue(is_pp_primary_rank())

        # Test rank 5 (stage 2, TP rank 0) - should be primary
        mock_get_rank.return_value = 5
        self.assertTrue(is_pp_primary_rank())

        # Test rank 6 (stage 2, TP rank 1) - should NOT be primary
        mock_get_rank.return_value = 6
        self.assertFalse(is_pp_primary_rank())

        # Test rank 7 (stage 3, TP rank 0) - should be primary (TP=1)
        mock_get_rank.return_value = 7
        self.assertTrue(is_pp_primary_rank())

        print("✅ PP primary rank test passed")


class TestParallelConfig(unittest.TestCase):
    """Test ParallelConfig heterogeneous support."""

    def test_config_validation(self):
        """Test heterogeneous config validation."""
        # Valid configuration
        config = ParallelConfig(
            pipeline_parallel_size=4,
            per_stage_tp_sizes=[4, 1, 2, 1],
        )

        self.assertTrue(config.is_heterogeneous())
        self.assertEqual(config.get_tp_size_for_stage(0), 4)
        self.assertEqual(config.get_tp_size_for_stage(1), 1)
        self.assertEqual(config.get_tp_size_for_stage(2), 2)
        self.assertEqual(config.get_tp_size_for_stage(3), 1)
        self.assertEqual(config.world_size, 8)  # Sum of TP sizes

        print("✅ ParallelConfig validation test passed")

    def test_invalid_config(self):
        """Test invalid heterogeneous configurations."""
        # Wrong length
        with self.assertRaises(ValueError) as cm:
            _ = ParallelConfig(
                pipeline_parallel_size=4,
                per_stage_tp_sizes=[4, 1, 2],  # Missing one
            )
        self.assertIn("length", str(cm.exception).lower())

        # Negative value
        with self.assertRaises(ValueError) as cm:
            _ = ParallelConfig(
                pipeline_parallel_size=2,
                per_stage_tp_sizes=[4, -1],
            )
        self.assertIn("positive", str(cm.exception).lower())

        print("✅ Invalid config test passed")

    def test_backward_compatibility(self):
        """Test that uniform mode still works."""
        config = ParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
            # per_stage_tp_sizes not set
        )

        self.assertFalse(config.is_heterogeneous())
        self.assertEqual(config.get_tp_size_for_stage(0), 2)
        self.assertEqual(config.get_tp_size_for_stage(1), 2)
        self.assertEqual(config.world_size, 8)  # 2 * 4

        print("✅ Backward compatibility test passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Milestone 2: Parallel State & Group Creation")
    print("=" * 60)

    # Run tests
    unittest.main(verbosity=2)
