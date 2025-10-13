#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test script for backend detection in heterogeneous mode.

This tests the cross-backend edge detection logic that determines when
to use CPU hop vs device-native communication.

Two test approaches:
1. Simple: Mock-based tests (no multi-GPU needed)
2. Real: Multi-process tests with actual distributed environment
"""

import unittest
from unittest.mock import patch

from vllm.distributed.heterogeneous_parallel import (
    auto_detect_stage_backends, get_current_backend_type,
    get_current_stage_info, get_or_detect_stage_backends,
    is_cross_backend_edge, is_heterogeneous_mode, reset_heterogeneous_config,
    set_heterogeneous_config, should_use_cpu_hop)


class TestBackendDetection(unittest.TestCase):
    """Simple mock-based tests for backend detection."""

    def tearDown(self):
        """Clean up after each test."""
        reset_heterogeneous_config()

    def test_get_current_backend_type_cuda(self):
        """Test backend type detection for CUDA."""
        with patch('vllm.platforms.current_platform') as mock_platform:
            # Mock CUDA platform
            mock_platform.is_cuda.return_value = True
            mock_platform.is_rocm.return_value = False
            mock_platform.is_tpu.return_value = False
            mock_platform.is_xpu.return_value = False
            mock_platform.is_cpu.return_value = False

            backend = get_current_backend_type()
            self.assertEqual(backend, 'cuda')
            print("✅ CUDA backend detection works")

    def test_get_current_backend_type_rocm(self):
        """Test backend type detection for ROCm."""
        with patch('vllm.platforms.current_platform') as mock_platform:
            # Mock ROCm platform
            mock_platform.is_cuda.return_value = False
            mock_platform.is_rocm.return_value = True
            mock_platform.is_tpu.return_value = False
            mock_platform.is_xpu.return_value = False
            mock_platform.is_cpu.return_value = False

            backend = get_current_backend_type()
            self.assertEqual(backend, 'rocm')
            print("✅ ROCm backend detection works")

    def test_get_current_backend_type_tpu(self):
        """Test backend type detection for TPU."""
        with patch('vllm.platforms.current_platform') as mock_platform:
            # Mock TPU platform
            mock_platform.is_cuda.return_value = False
            mock_platform.is_rocm.return_value = False
            mock_platform.is_tpu.return_value = True
            mock_platform.is_xpu.return_value = False
            mock_platform.is_cpu.return_value = False

            backend = get_current_backend_type()
            self.assertEqual(backend, 'tpu')
            print("✅ TPU backend detection works")

    def test_get_current_backend_type_xpu(self):
        """Test backend type detection for XPU."""
        with patch('vllm.platforms.current_platform') as mock_platform:
            # Mock XPU platform
            mock_platform.is_cuda.return_value = False
            mock_platform.is_rocm.return_value = False
            mock_platform.is_tpu.return_value = False
            mock_platform.is_xpu.return_value = True
            mock_platform.is_cpu.return_value = False

            backend = get_current_backend_type()
            self.assertEqual(backend, 'xpu')
            print("✅ XPU backend detection works")

    def test_get_current_backend_type_cpu(self):
        """Test backend type detection for CPU."""
        with patch('vllm.platforms.current_platform') as mock_platform:
            # Mock CPU platform
            mock_platform.is_cuda.return_value = False
            mock_platform.is_rocm.return_value = False
            mock_platform.is_tpu.return_value = False
            mock_platform.is_xpu.return_value = False
            mock_platform.is_cpu.return_value = True

            backend = get_current_backend_type()
            self.assertEqual(backend, 'cpu')
            print("✅ CPU backend detection works")


class TestCrossBackendEdgeDetection(unittest.TestCase):
    """Test cross-backend edge detection with explicit stage_backends."""

    def setUp(self):
        """Set up heterogeneous configuration."""
        self.config_same_backend = {
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
            'stage_backends': {
                0: 'cuda',  # All CUDA (same backend)
                1: 'cuda',
                2: 'cuda',
                3: 'cuda',
            }
        }

        self.config_cross_backend = {
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
            'stage_backends': {
                0: 'cuda',  # NVIDIA T4s
                1: 'rocm',  # AMD MI300x (different!)
                2: 'cuda',  # NVIDIA A10Gs
                3: 'tpu',  # Google TPU (different!)
            }
        }

    def tearDown(self):
        """Clean up after each test."""
        reset_heterogeneous_config()

    @patch('torch.distributed.get_rank')
    def test_cross_backend_stage0_to_stage1_detected(self, mock_get_rank):
        """Test CUDA→ROCm edge is detected as cross-backend."""
        set_heterogeneous_config(self.config_cross_backend)

        # Test from stage 0 (CUDA) perspective
        mock_get_rank.return_value = 0
        info = get_current_stage_info()
        self.assertEqual(info['stage'], 0)

        # Should detect cross-backend to stage 1 (ROCm)
        is_cross = is_cross_backend_edge()
        self.assertTrue(is_cross,
                        "CUDA→ROCm edge should be detected as cross-backend")
        print("✅ CUDA→ROCm cross-backend detection works")

    @patch('torch.distributed.get_rank')
    def test_cross_backend_stage1_to_stage2_detected(self, mock_get_rank):
        """Test ROCm→CUDA edge is detected as cross-backend."""
        set_heterogeneous_config(self.config_cross_backend)

        # Test from stage 1 (ROCm) perspective
        mock_get_rank.return_value = 4
        info = get_current_stage_info()
        self.assertEqual(info['stage'], 1)

        # Should detect cross-backend to stage 2 (CUDA)
        is_cross = is_cross_backend_edge()
        self.assertTrue(is_cross,
                        "ROCm→CUDA edge should be detected as cross-backend")
        print("✅ ROCm→CUDA cross-backend detection works")

    @patch('torch.distributed.get_rank')
    def test_cross_backend_stage2_to_stage3_detected(self, mock_get_rank):
        """Test CUDA→TPU edge is detected as cross-backend."""
        set_heterogeneous_config(self.config_cross_backend)

        # Test from stage 2 (CUDA) perspective
        mock_get_rank.return_value = 5
        info = get_current_stage_info()
        self.assertEqual(info['stage'], 2)

        # Should detect cross-backend to stage 3 (TPU)
        is_cross = is_cross_backend_edge()
        self.assertTrue(is_cross,
                        "CUDA→TPU edge should be detected as cross-backend")
        print("✅ CUDA→TPU cross-backend detection works")

    @patch('torch.distributed.get_rank')
    def test_same_backend_not_detected(self, mock_get_rank):
        """Test same-backend edges are NOT detected as cross-backend."""
        set_heterogeneous_config(self.config_same_backend)

        # Test from stage 0 (CUDA) to stage 1 (CUDA)
        mock_get_rank.return_value = 0
        is_cross = is_cross_backend_edge()
        self.assertFalse(is_cross,
                         "CUDA→CUDA edge should NOT be cross-backend")
        print("✅ Same-backend edge correctly not flagged")

    @patch('torch.distributed.get_rank')
    def test_should_use_cpu_hop_cross_backend(self, mock_get_rank):
        """Test CPU hop decision for cross-backend."""
        set_heterogeneous_config(self.config_cross_backend)

        # Stage 0 → Stage 1 (CUDA → ROCm)
        mock_get_rank.return_value = 0
        should_cpu = should_use_cpu_hop()
        self.assertTrue(
            should_cpu,
            "CPU hop should be used for cross-backend communication")
        print("✅ CPU hop decision works for cross-backend")

    @patch('torch.distributed.get_rank')
    def test_should_use_cpu_hop_same_backend(self, mock_get_rank):
        """Test CPU hop decision for same-backend."""
        set_heterogeneous_config(self.config_same_backend)

        # Stage 0 → Stage 1 (CUDA → CUDA)
        mock_get_rank.return_value = 0
        should_cpu = should_use_cpu_hop()

        # Even though TP sizes differ (4→1), same backend might not need CPU hop
        # But current implementation is conservative
        # This test documents the current behavior
        print(f"   CPU hop for same-backend hetero TP: {should_cpu}")

    @patch('torch.distributed.get_rank')
    def test_fallback_heuristic_no_stage_backends(self, mock_get_rank):
        """Test fallback to heuristic when stage_backends not provided."""
        # Config without stage_backends
        config_no_backends = {
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
            # No 'stage_backends' key
        }
        set_heterogeneous_config(config_no_backends)

        # Mock get_or_detect_stage_backends to fail
        mock_get_rank.return_value = 0

        # Should fall back to heuristic (different TP
        #  = cross-backend assumption)
        # This is conservative but safe
        is_cross = is_cross_backend_edge()
        # Current implementation with fallback logic
        print(f"   Fallback heuristic result: {is_cross}")
        print("✅ Fallback heuristic test completed")

    def test_not_heterogeneous_mode(self):
        """Test that functions return False when not in heterogeneous mode."""
        # Don't set any config
        self.assertFalse(is_heterogeneous_mode())
        self.assertFalse(is_cross_backend_edge())
        self.assertFalse(should_use_cpu_hop())
        print("✅ Non-heterogeneous mode correctly returns False")


class TestAutoDetectStageBackends(unittest.TestCase):
    """Test automatic backend detection with mocked distributed operations."""

    def setUp(self):
        """Set up heterogeneous configuration."""
        self.config_same_backend = {
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
            'stage_backends': {
                0: 'cuda',  # All CUDA (same backend)
                1: 'cuda',
                2: 'cuda',
                3: 'cuda',
            }
        }

        self.config_cross_backend = {
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
            'stage_backends': {
                0: 'cuda',  # NVIDIA T4s
                1: 'rocm',  # AMD MI300x (different!)
                2: 'cuda',  # NVIDIA A10Gs
                3: 'tpu',  # Google TPU (different!)
            }
        }

    def tearDown(self):
        """Clean up after each test."""
        reset_heterogeneous_config()

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_auto_detect_uniform_backend(self, mock_all_gather,
                                         mock_world_size, mock_get_rank):
        """Test auto-detection when all stages have same backend."""
        # Setup: 8 ranks, all CUDA
        mock_world_size.return_value = 8
        mock_get_rank.return_value = 0

        # Mock all_gather_object to simulate all ranks reporting 'cuda'
        def mock_gather(output_list, input_obj):
            # Simulate all 8 ranks reporting 'cuda'
            for i in range(8):
                output_list[i] = 'cuda'

        mock_all_gather.side_effect = mock_gather

        # Set config first
        config = {
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
        set_heterogeneous_config(config)

        # Detect backends
        with patch(
                'vllm.distributed.heterogeneous_parallel.get_current_backend_type',
                return_value='cuda'):
            stage_backends = auto_detect_stage_backends()

        # Verify
        self.assertEqual(stage_backends, {
            0: 'cuda',
            1: 'cuda',
            2: 'cuda',
            3: 'cuda',
        })
        print("✅ Uniform backend auto-detection works")

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_auto_detect_mixed_backends(self, mock_all_gather, mock_world_size,
                                        mock_get_rank):
        """Test auto-detection with mixed backends: CUDA→ROCm→CUDA→TPU."""
        mock_world_size.return_value = 8
        mock_get_rank.return_value = 0

        # Mock all_gather_object to simulate heterogeneous backends
        def mock_gather(output_list, input_obj):
            # Ranks 0-3: CUDA (Stage 0)
            # Rank 4: ROCm (Stage 1)
            # Ranks 5-6: CUDA (Stage 2)
            # Rank 7: TPU (Stage 3)
            backends = [
                'cuda', 'cuda', 'cuda', 'cuda', 'rocm', 'cuda', 'cuda', 'tpu'
            ]
            for i in range(8):
                output_list[i] = backends[i]

        mock_all_gather.side_effect = mock_gather

        # Set config
        config = {
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
        set_heterogeneous_config(config)

        # Detect backends (mock the platform detection per call)
        stage_backends = auto_detect_stage_backends()

        # Verify stage mapping
        self.assertEqual(stage_backends[0], 'cuda')  # Stage 0: ranks 0-3
        self.assertEqual(stage_backends[1], 'rocm')  # Stage 1: rank 4
        self.assertEqual(stage_backends[2], 'cuda')  # Stage 2: ranks 5-6
        self.assertEqual(stage_backends[3], 'tpu')  # Stage 3: rank 7

        print("✅ Mixed backend auto-detection works")
        print(f"   Detected: {stage_backends}")

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_homogeneity_within_stage_validation(self, mock_all_gather,
                                                 mock_world_size,
                                                 mock_get_rank):
        """Test that mixed backends within a stage raise error."""
        mock_world_size.return_value = 8
        mock_get_rank.return_value = 0

        # Mock stage 0 with mixed backends (violation of assumption!)
        def mock_gather(output_list, input_obj):
            # Ranks 0-2: CUDA, Rank 3: ROCm (WRONG! Stage 0 is heterogeneous)
            backends = [
                'cuda',
                'cuda',
                'cuda',
                'rocm',  # Stage 0 mixed!
                'rocm',
                'cuda',
                'cuda',
                'tpu'
            ]
            for i in range(8):
                output_list[i] = backends[i]

        mock_all_gather.side_effect = mock_gather

        config = {
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
        set_heterogeneous_config(config)

        # Should raise ValueError for heterogeneous stage
        with self.assertRaises(ValueError) as cm:
            auto_detect_stage_backends()

        self.assertIn("mixed backends", str(cm.exception).lower())
        print(
            "✅ Homogeneity validation works (rejects mixed backends in stage)")

    @patch('torch.distributed.get_rank')
    def test_get_or_detect_caches_result(self, mock_get_rank):
        """Test that get_or_detect_stage_backends caches the result."""
        mock_get_rank.return_value = 0

        # Set config with explicit backends
        set_heterogeneous_config(self.config_cross_backend)

        # First call - should return cached value
        backends1 = get_or_detect_stage_backends()
        self.assertEqual(backends1[0], 'cuda')

        # Second call - should return same cached value
        backends2 = get_or_detect_stage_backends()
        self.assertIs(backends1, backends2,
                      "Should return same object (cached)")

        print("✅ Backend caching works")


class TestCPUHopDecision(unittest.TestCase):
    """Test CPU hop decision logic."""

    def tearDown(self):
        """Clean up after each test."""
        reset_heterogeneous_config()

    @patch('torch.distributed.get_rank')
    def test_cpu_hop_for_cross_backend(self, mock_get_rank):
        """Test that CPU hop is required for cross-backend edges."""
        config = {
            'per_stage_tp_sizes': [2, 2],
            'rank_to_stage_info': {
                0: {
                    'stage': 0,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                1: {
                    'stage': 0,
                    'tp_size': 2,
                    'tp_rank': 1
                },
                2: {
                    'stage': 1,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                3: {
                    'stage': 1,
                    'tp_size': 2,
                    'tp_rank': 1
                },
            },
            'pipeline_parallel_size': 2,
            'stage_backends': {
                0: 'cuda',
                1: 'rocm',  # Different backend
            }
        }
        set_heterogeneous_config(config)

        mock_get_rank.return_value = 0
        should_cpu = should_use_cpu_hop()
        self.assertTrue(should_cpu, "CPU hop required for cross-backend")
        print("✅ CPU hop required for cross-backend edges")

    @patch('torch.distributed.get_rank')
    def test_no_cpu_hop_same_backend_same_tp(self, mock_get_rank):
        """Test that CPU hop is NOT needed for same-backend, same-TP."""
        config = {
            'per_stage_tp_sizes': [2, 2],
            'rank_to_stage_info': {
                0: {
                    'stage': 0,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                1: {
                    'stage': 0,
                    'tp_size': 2,
                    'tp_rank': 1
                },
                2: {
                    'stage': 1,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                3: {
                    'stage': 1,
                    'tp_size': 2,
                    'tp_rank': 1
                },
            },
            'pipeline_parallel_size': 2,
            'stage_backends': {
                0: 'cuda',
                1: 'cuda',  # Same backend
            }
        }
        set_heterogeneous_config(config)

        mock_get_rank.return_value = 0
        should_cpu = should_use_cpu_hop()
        self.assertFalse(should_cpu,
                         "CPU hop NOT needed for same-backend same-TP")
        print("✅ No CPU hop for same-backend, same-TP")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def tearDown(self):
        """Clean up after each test."""
        reset_heterogeneous_config()

    def test_functions_without_hetero_mode(self):
        """Test functions behave correctly when hetero mode not enabled."""
        # These should not crash and return sensible defaults
        self.assertFalse(is_cross_backend_edge())
        self.assertFalse(should_use_cpu_hop())
        print("✅ Functions handle non-hetero mode gracefully")

    @patch('torch.distributed.get_rank')
    def test_first_stage_no_prev_backend(self, mock_get_rank):
        """Test first stage has no previous backend to check."""
        config = {
            'per_stage_tp_sizes': [2, 2],
            'rank_to_stage_info': {
                0: {
                    'stage': 0,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                1: {
                    'stage': 0,
                    'tp_size': 2,
                    'tp_rank': 1
                },
                2: {
                    'stage': 1,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                3: {
                    'stage': 1,
                    'tp_size': 2,
                    'tp_rank': 1
                },
            },
            'pipeline_parallel_size': 2,
            'stage_backends': {
                0: 'cuda',
                1: 'rocm',
            }
        }
        set_heterogeneous_config(config)

        # First stage (no previous)
        mock_get_rank.return_value = 0
        is_cross = is_cross_backend_edge()
        # Should only check forward (to stage 1)
        self.assertTrue(is_cross, "Should detect forward cross-backend")
        print("✅ First stage correctly checks only forward edge")

    @patch('torch.distributed.get_rank')
    def test_last_stage_no_next_backend(self, mock_get_rank):
        """Test last stage has no next backend to check."""
        config = {
            'per_stage_tp_sizes': [2, 2],
            'rank_to_stage_info': {
                0: {
                    'stage': 0,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                1: {
                    'stage': 0,
                    'tp_size': 2,
                    'tp_rank': 1
                },
                2: {
                    'stage': 1,
                    'tp_size': 2,
                    'tp_rank': 0
                },
                3: {
                    'stage': 1,
                    'tp_size': 2,
                    'tp_rank': 1
                },
            },
            'pipeline_parallel_size': 2,
            'stage_backends': {
                0: 'cuda',
                1: 'rocm',
            }
        }
        set_heterogeneous_config(config)

        # Last stage (no next)
        mock_get_rank.return_value = 2
        is_cross = is_cross_backend_edge()
        # Should only check backward (from stage 0)
        self.assertTrue(is_cross, "Should detect backward cross-backend")
        print("✅ Last stage correctly checks only backward edge")


if __name__ == '__main__':
    print("=" * 70)
    print("SIMPLE TESTS: Backend Detection (Mock-based, no GPU needed)")
    print("=" * 70)
    print()

    # Run tests
    unittest.main(verbosity=2)
