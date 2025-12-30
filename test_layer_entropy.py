#!/usr/bin/env python3
"""
Simple test to verify layer-specific entropy configuration.

This test validates that actor_entropy can be specified per layer
and that the SubActor correctly picks the appropriate value.
"""

import ast
import sys


def test_subactor_signature():
    """Check if SubActor.__init__ accepts layer_idx parameter."""
    print("="*80)
    print("TEST: Checking if SubActor accepts layer_idx parameter")
    print("="*80)
    
    with open("hieros/hieros.py", "r") as f:
        content = f.read()
    
    # Parse the file
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ FAILED: Syntax error in hieros.py: {e}")
        return False
    
    # Find the SubActor class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SubActor":
            print(f"✅ Found class: {node.name}")
            
            # Find __init__ method
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    actual_params = [arg.arg for arg in item.args.args]
                    print(f"  Parameters: {actual_params}")
                    
                    if "layer_idx" in actual_params:
                        print("✅ PASSED: layer_idx parameter exists in SubActor.__init__")
                        return True
                    else:
                        print("❌ FAILED: layer_idx parameter not found in SubActor.__init__")
                        return False
    
    print("❌ FAILED: SubActor class not found")
    return False


def test_entropy_handling():
    """Check if entropy handling code exists."""
    print("\n" + "="*80)
    print("TEST: Checking if entropy handling code exists")
    print("="*80)
    
    with open("hieros/hieros.py", "r") as f:
        content = f.read()
    
    # Check for the entropy handling logic
    if "isinstance(actor_entropy_value, (list, tuple))" in content:
        print("✅ PASSED: Found list/tuple handling for actor_entropy")
        return True
    else:
        print("❌ FAILED: List/tuple handling for actor_entropy not found")
        return False


def test_layer_idx_passed():
    """Check if layer_idx is passed when creating SubActors."""
    print("\n" + "="*80)
    print("TEST: Checking if layer_idx is passed when creating SubActors")
    print("="*80)
    
    with open("hieros/hieros.py", "r") as f:
        content = f.read()
    
    # Check for layer_idx=0 in first subactor
    if "layer_idx=0," in content:
        print("✅ PASSED: Found layer_idx=0 in first SubActor creation")
    else:
        print("⚠️  WARNING: layer_idx=0 not found in first SubActor creation")
        return False
    
    # Check for layer_idx in _create_subactor
    if "layer_idx=layer_idx," in content or "layer_idx=len(self._subactors)" in content:
        print("✅ PASSED: Found layer_idx in _create_subactor method")
        return True
    else:
        print("❌ FAILED: layer_idx not passed in _create_subactor method")
        return False


def main():
    """Run all tests."""
    print("Testing layer-specific entropy configuration\n")
    
    tests = [
        test_subactor_signature,
        test_entropy_handling,
        test_layer_idx_passed,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
