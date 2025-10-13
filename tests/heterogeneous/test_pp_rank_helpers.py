#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the PP rank helper functions in heterogeneous_parallel.py"""

import unittest
from unittest.mock import patch

from vllm.distributed.heterogeneous_parallel import (
    get_next_stage_pp_rank_0, get_prev_stage_pp_rank_0, get_stage_pp_rank_0,
    reset_heterogeneous_config, set_heterogeneous_config)


class TestPPRankHelpers(unittest.TestCase):
    """Test PP rank helper functions."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'per_stage_tp_sizes': [4, 1, 2,
                                   1],  # Stage 0: TP=4, Stage 1: TP=1, etc.
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
        set_heterogeneous_config(self.config)

    def tearDown(self):
        """Clean up after each test."""
        reset_heterogeneous_config()

    @patch('torch.distributed.get_rank')
    def test_get_next_stage_pp_rank_0(self, mock_get_rank):
        """Test getting next stage's PP rank 0."""
        # Test from stage 0 -> stage 1
        mock_get_rank.return_value = 0  # Rank 0 is in stage 0
        next_rank = get_next_stage_pp_rank_0()
        self.assertEqual(next_rank, 4,
                         "Stage 0 -> Stage 1 should return rank 4")

        # Test from stage 1 -> stage 2
        mock_get_rank.return_value = 4  # Rank 4 is in stage 1
        next_rank = get_next_stage_pp_rank_0()
        self.assertEqual(next_rank, 5,
                         "Stage 1 -> Stage 2 should return rank 5")

        # Test from stage 2 -> stage 3
        mock_get_rank.return_value = 5  # Rank 5 is in stage 2
        next_rank = get_next_stage_pp_rank_0()
        self.assertEqual(next_rank, 7,
                         "Stage 2 -> Stage 3 should return rank 7")

        # Test from last stage
        mock_get_rank.return_value = 7  # Rank 7 is in stage 3 (last)
        next_rank = get_next_stage_pp_rank_0()
        self.assertIsNone(next_rank, "Last stage should return None")

    @patch('torch.distributed.get_rank')
    def test_get_prev_stage_pp_rank_0(self, mock_get_rank):
        """Test getting previous stage's PP rank 0."""
        # Test from first stage
        mock_get_rank.return_value = 0  # Rank 0 is in stage 0 (first)
        prev_rank = get_prev_stage_pp_rank_0()
        self.assertIsNone(prev_rank, "First stage should return None")

        # Test from stage 1 -> stage 0
        mock_get_rank.return_value = 4  # Rank 4 is in stage 1
        prev_rank = get_prev_stage_pp_rank_0()
        self.assertEqual(prev_rank, 0,
                         "Stage 1 -> Stage 0 should return rank 0")

        # Test from stage 2 -> stage 1
        mock_get_rank.return_value = 5  # Rank 5 is in stage 2
        prev_rank = get_prev_stage_pp_rank_0()
        self.assertEqual(prev_rank, 4,
                         "Stage 2 -> Stage 1 should return rank 4")

        # Test from stage 3 -> stage 2
        mock_get_rank.return_value = 7  # Rank 7 is in stage 3
        prev_rank = get_prev_stage_pp_rank_0()
        self.assertEqual(prev_rank, 5,
                         "Stage 3 -> Stage 2 should return rank 5")

    def test_get_stage_pp_rank_0(self):
        """Test getting any stage's PP rank 0."""
        # Test valid stages
        self.assertEqual(get_stage_pp_rank_0(0), 0,
                         "Stage 0 PP rank 0 should be 0")
        self.assertEqual(get_stage_pp_rank_0(1), 4,
                         "Stage 1 PP rank 0 should be 4")
        self.assertEqual(get_stage_pp_rank_0(2), 5,
                         "Stage 2 PP rank 0 should be 5")
        self.assertEqual(get_stage_pp_rank_0(3), 7,
                         "Stage 3 PP rank 0 should be 7")

        # Test invalid stages
        self.assertIsNone(get_stage_pp_rank_0(-1),
                          "Invalid stage -1 should return None")
        self.assertIsNone(get_stage_pp_rank_0(4),
                          "Invalid stage 4 should return None")
        self.assertIsNone(get_stage_pp_rank_0(100),
                          "Invalid stage 100 should return None")

    @patch('torch.distributed.get_rank')
    def test_non_tp_rank_0(self, mock_get_rank):
        """Test that non-TP-rank-0 processes work correctly."""
        # Test from a non-TP-rank-0 process (rank 2 is TP rank 2 in stage 0)
        mock_get_rank.return_value = 2

        # Should still return correct next stage rank
        next_rank = get_next_stage_pp_rank_0()
        self.assertEqual(next_rank, 4,
                         "Non-TP-rank-0 should still get correct next stage")

        # Test from rank 6 (TP rank 1 in stage 2)
        mock_get_rank.return_value = 6
        next_rank = get_next_stage_pp_rank_0()
        self.assertEqual(next_rank, 7,
                         "Non-TP-rank-0 in stage 2 should get rank 7")

        prev_rank = get_prev_stage_pp_rank_0()
        self.assertEqual(prev_rank, 4,
                         "Non-TP-rank-0 in stage 2 should get rank 4 as prev")

    def test_no_heterogeneous_mode(self):
        """Test functions return None when not in heterogeneous mode."""
        reset_heterogeneous_config()

        self.assertIsNone(get_next_stage_pp_rank_0())
        self.assertIsNone(get_prev_stage_pp_rank_0())
        self.assertIsNone(get_stage_pp_rank_0(0))
        self.assertIsNone(get_stage_pp_rank_0(1))


if __name__ == '__main__':
    print("=" * 70)
    print("Testing PP Rank Helper Functions")
    print("=" * 70)
    unittest.main(verbosity=2)
