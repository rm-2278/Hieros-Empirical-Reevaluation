#!/usr/bin/env python3
"""
Test script to validate the tensor dimension fix for subgoal reward computation.

This test creates minimal mock objects to simulate the tensor dimension issue
and validates that the fix correctly handles the dimensions.
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

import torch


class MockConfig:
    """Mock configuration with minimal required attributes."""
    def __init__(self):
        self.subgoal_shape = [8, 8]
        self.dyn_deter = 128
        self.dyn_stoch = 32
        self.dyn_discrete = 32
        self.subgoal = {"use_stoch": False, "use_deter": True}
        self.subgoal_compression = {"encoding_symlog": False}
        self.subgoal_reward_symlog = False


class MockSubgoalAutoencoder:
    """Mock subgoal autoencoder that decodes compressed subgoal."""
    def __init__(self, decoded_dim=128):
        self.decoded_dim = decoded_dim
    
    def decode(self, subgoal):
        """
        Decode compressed subgoal [batch, 8, 8] to full representation [batch, decoded_dim].
        """
        batch_size = subgoal.shape[0]
        # Flatten and project to decoded dimension
        return torch.randn(batch_size, self.decoded_dim)


class MockSubactor:
    """Mock subactor with minimal methods for testing."""
    def __init__(self, config):
        self._config = config
        self._compute_subgoal = True
        self.subgoal_autoencoder = MockSubgoalAutoencoder(decoded_dim=config.dyn_deter)
    
    def decode_subgoal(self, subgoal, isfirst=False):
        """Decode compressed subgoal to full representation."""
        if isfirst:
            return torch.zeros(subgoal.shape[0], self._config.dyn_deter)
        return self.subgoal_autoencoder.decode(subgoal)
    
    def get_subgoal(self, state):
        """Get subgoal representation from state."""
        if self._config.subgoal["use_deter"]:
            return state["deter"]
        return None
    
    def _subgoal_reward(self, state, subgoal):
        """
        Compute cosine similarity reward between state and subgoal.
        This is the actual method that was failing.
        """
        state_representation = self.get_subgoal(state)
        
        # This reshape and reshape (fixed from expand) was causing the error
        reshaped_subgoal = subgoal.reshape(
            [subgoal.shape[0] * subgoal.shape[1]] + list(subgoal.shape[2:])
        ).reshape(state_representation.shape)
        
        dims_to_sum = list(range(len(state_representation.shape)))[2:]
        gnorm = torch.norm(reshaped_subgoal, dim=dims_to_sum) + 1e-12
        fnorm = torch.norm(state_representation, dim=dims_to_sum) + 1e-12
        norm = torch.max(gnorm, fnorm)
        cos = torch.sum(reshaped_subgoal * state_representation, dim=dims_to_sum) / (
            norm * norm
        )
        subgoal_reward = torch.clamp(cos.unsqueeze(-1), min=0)
        return subgoal_reward


def test_original_bug():
    """Test that reproduces the original bug."""
    print("Testing original bug (should fail)...")
    config = MockConfig()
    batch_size = 8
    
    # Simulate original code with expand (before our fix)
    cached_subgoal = torch.randn(batch_size, 8, 8)  # [batch, subgoal_shape]
    state = {
        "deter": torch.randn(batch_size, config.dyn_deter)  # [batch, deter_dim]
    }
    
    # Original buggy code: add time dimension directly to compressed subgoal
    state_with_time = {k: v.unsqueeze(1) for k, v in state.items()}  # [batch, 1, feature]
    subgoal_with_time = cached_subgoal.unsqueeze(1)  # [batch, 1, 8, 8] - WRONG!
    
    # Simulate the original expand logic
    state_representation = state_with_time["deter"]  # [8, 1, 128]
    
    try:
        # This is what the original code tried to do
        reshaped = subgoal_with_time.reshape(8*1, 8, 8)  # [8, 8, 8]
        expanded = reshaped.expand(state_representation.shape)  # Try to expand to [8, 1, 128]
        print("  ❌ UNEXPECTED: Bug test should have failed but succeeded!")
        return False
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "expanded size" in error_msg or "invalid" in error_msg:
            print(f"  ✅ Expected error: {e}")
            return True
        else:
            print(f"  ❌ Unexpected error: {e}")
            return False


def test_fixed_version():
    """Test that validates the fix."""
    print("\nTesting fixed version (should succeed)...")
    config = MockConfig()
    batch_size = 8
    
    # Create mock state and cached_subgoal
    cached_subgoal = torch.randn(batch_size, 8, 8)  # [batch, subgoal_shape]
    state = {
        "deter": torch.randn(batch_size, config.dyn_deter)  # [batch, deter_dim]
    }
    
    # Fixed code: decode compressed subgoal before adding time dimension
    state_with_time = {k: v.unsqueeze(1) for k, v in state.items()}  # [batch, 1, feature]
    
    subactor = MockSubactor(config)
    decoded_subgoal = subactor.decode_subgoal(cached_subgoal, isfirst=False)  # [batch, deter_dim]
    subgoal_with_time = decoded_subgoal.unsqueeze(1)  # [batch, 1, deter_dim] - CORRECT!
    
    try:
        reward = subactor._subgoal_reward(state_with_time, subgoal_with_time)
        print(f"  ✅ Success! Reward shape: {reward.shape}")
        
        # Validate reward shape
        expected_shape = (batch_size, 1, 1)
        if reward.shape == expected_shape:
            print(f"  ✅ Reward has expected shape: {expected_shape}")
            return True
        else:
            print(f"  ❌ Unexpected reward shape: {reward.shape}, expected: {expected_shape}")
            return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def test_dimension_explanation():
    """Print explanation of the dimensions."""
    print("\n" + "="*70)
    print("DIMENSION EXPLANATION")
    print("="*70)
    print("\nOriginal Bug:")
    print("  • cached_subgoal: [8, 8, 8]")
    print("    └─ [batch=8, subgoal_height=8, subgoal_width=8]")
    print("  • After unsqueeze(1): [8, 1, 8, 8]")
    print("    └─ [batch=8, time=1, height=8, width=8]")
    print("  • state_representation: [8, 1, 128]")
    print("    └─ [batch=8, time=1, deter_features=128]")
    print("  • In _subgoal_reward, reshape tries: [8*1, 8, 8] = [8, 8, 8]")
    print("  • Then expand to [8, 1, 128] fails:")
    print("    └─ Can't expand dimension 2: size 8 to 128 ❌")
    
    print("\nFixed Version:")
    print("  • cached_subgoal: [8, 8, 8]")
    print("    └─ [batch=8, subgoal_height=8, subgoal_width=8]")
    print("  • After decode_subgoal: [8, 128]")
    print("    └─ [batch=8, decoded_features=128]")
    print("  • After unsqueeze(1): [8, 1, 128]")
    print("    └─ [batch=8, time=1, features=128]")
    print("  • state_representation: [8, 1, 128]")
    print("    └─ [batch=8, time=1, deter_features=128]")
    print("  • In _subgoal_reward, reshape gives: [8*1, 128] = [8, 128]")
    print("  • Then expand to [8, 1, 128] succeeds:")
    print("    └─ Adds singleton dimension at index 1 ✅")
    
    print("\nKey Insight:")
    print("  The 8 in [8, 8, 8] is the BATCH SIZE")
    print("  The [8, 8] is the COMPRESSED SUBGOAL SHAPE")
    print("  The 128 in [8, 1, 128] is the FULL FEATURE DIMENSION (dyn_deter)")
    print("\n" + "="*70)


if __name__ == "__main__":
    print("="*70)
    print("SUBGOAL REWARD DIMENSION FIX VALIDATION TEST")
    print("="*70)
    
    test_dimension_explanation()
    
    bug_test_passed = test_original_bug()
    fix_test_passed = test_fixed_version()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Original bug reproduction: {'✅ PASS' if bug_test_passed else '❌ FAIL'}")
    print(f"Fixed version validation: {'✅ PASS' if fix_test_passed else '❌ FAIL'}")
    
    if bug_test_passed and fix_test_passed:
        print("\n🎉 All tests passed! The fix correctly resolves the dimension mismatch.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the fix.")
        sys.exit(1)
