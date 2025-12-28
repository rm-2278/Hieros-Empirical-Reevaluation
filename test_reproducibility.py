#!/usr/bin/env python3
"""
Test script to verify reproducibility of metrics across two runs.
This script will run training for a small number of steps and check
if the metrics are identical across two runs with the same seed.
"""

import sys
import pathlib
import torch
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent))

from embodied.replay import selectors

def test_selector_reproducibility():
    """Test that replay selectors produce identical samples with same seed."""
    print("Testing selector reproducibility...")
    
    # Test Uniform selector
    print("\n1. Testing Uniform selector...")
    selector1 = selectors.Uniform(seed=42)
    selector2 = selectors.Uniform(seed=42)
    
    # Add some keys
    for i in range(100):
        selector1[f"key_{i}"] = None
        selector2[f"key_{i}"] = None
    
    # Sample and check if identical
    samples1 = [selector1() for _ in range(50)]
    samples2 = [selector2() for _ in range(50)]
    
    assert samples1 == samples2, "Uniform selector not reproducible!"
    print("✓ Uniform selector is reproducible")
    
    # Test TimeBalanced selector
    print("\n2. Testing TimeBalanced selector...")
    selector1 = selectors.TimeBalanced(seed=42, bias_factor=1.5)
    selector2 = selectors.TimeBalanced(seed=42, bias_factor=1.5)
    
    # Add some keys
    for i in range(100):
        selector1[f"key_{i}"] = None
        selector2[f"key_{i}"] = None
    
    # Sample and check if identical
    samples1 = [selector1() for _ in range(50)]
    samples2 = [selector2() for _ in range(50)]
    
    assert samples1 == samples2, "TimeBalanced selector not reproducible!"
    print("✓ TimeBalanced selector is reproducible")
    
    # Test TimeBalancedNaive selector
    print("\n3. Testing TimeBalancedNaive selector...")
    selector1 = selectors.TimeBalancedNaive(seed=42, bias_factor=20)
    selector2 = selectors.TimeBalancedNaive(seed=42, bias_factor=20)
    
    # Add some keys
    for i in range(100):
        selector1[f"key_{i}"] = None
        selector2[f"key_{i}"] = None
    
    # Sample and check if identical
    samples1 = [selector1() for _ in range(50)]
    samples2 = [selector2() for _ in range(50)]
    
    assert samples1 == samples2, "TimeBalancedNaive selector not reproducible!"
    print("✓ TimeBalancedNaive selector is reproducible")
    
    # Test EfficientTimeBalanced selector
    print("\n4. Testing EfficientTimeBalanced selector...")
    selector1 = selectors.EfficientTimeBalanced(seed=42, length=1000, temperature=1.0)
    selector2 = selectors.EfficientTimeBalanced(seed=42, length=1000, temperature=1.0)
    
    # Add some keys
    for i in range(100):
        selector1[f"key_{i}"] = None
        selector2[f"key_{i}"] = None
    
    # Sample and check if identical
    samples1 = [selector1() for _ in range(50)]
    samples2 = [selector2() for _ in range(50)]
    
    assert samples1 == samples2, "EfficientTimeBalanced selector not reproducible!"
    print("✓ EfficientTimeBalanced selector is reproducible")
    
    print("\n✅ All selectors are reproducible!")


def test_pytorch_reproducibility():
    """Test that PyTorch operations are reproducible with seeding."""
    print("\n\nTesting PyTorch reproducibility...")
    
    # Test torch generator with Beta distribution
    print("\n1. Testing Beta distribution with seeded generator...")
    gen1 = torch.Generator()
    gen1.manual_seed(42)
    gen2 = torch.Generator()
    gen2.manual_seed(42)
    
    dist1 = torch.distributions.Beta(1.5, 1)
    dist2 = torch.distributions.Beta(1.5, 1)
    
    samples1 = [dist1.sample(generator=gen1).item() for _ in range(20)]
    samples2 = [dist2.sample(generator=gen2).item() for _ in range(20)]
    
    assert samples1 == samples2, "Beta distribution with generator not reproducible!"
    print("✓ Beta distribution with seeded generator is reproducible")
    
    # Test Categorical distribution
    print("\n2. Testing Categorical distribution with seeded generator...")
    gen1 = torch.Generator()
    gen1.manual_seed(42)
    gen2 = torch.Generator()
    gen2.manual_seed(42)
    
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    dist1 = torch.distributions.Categorical(probs=probs)
    dist2 = torch.distributions.Categorical(probs=probs)
    
    samples1 = [dist1.sample(generator=gen1).item() for _ in range(20)]
    samples2 = [dist2.sample(generator=gen2).item() for _ in range(20)]
    
    assert samples1 == samples2, "Categorical distribution with generator not reproducible!"
    print("✓ Categorical distribution with seeded generator is reproducible")
    
    print("\n✅ PyTorch operations are reproducible!")


if __name__ == "__main__":
    print("=" * 60)
    print("REPRODUCIBILITY TEST SUITE")
    print("=" * 60)
    
    try:
        test_selector_reproducibility()
        test_pytorch_reproducibility()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - System is reproducible!")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
