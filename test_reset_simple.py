"""
Simple test to verify reset action is present in policy output.
This test verifies the code structure without requiring full dependencies.
"""
import sys
import pathlib
import ast

def check_reset_in_policy_output():
    """Check if reset is included in policy_output in _policy method."""
    print("=" * 60)
    print("TEST 1: Check reset in policy_output")
    print("=" * 60)
    
    hieros_file = pathlib.Path(__file__).parent / 'hieros' / 'hieros.py'
    content = hieros_file.read_text()
    
    # Check for reset in policy_output for training mode
    if 'policy_output = {"action": action, "log_entropy": logprob, "reset": reset}' in content:
        print("✓ PASS: Reset found in policy_output (training mode)")
        training_pass = True
    else:
        print("✗ FAIL: Reset NOT found in policy_output (training mode)")
        training_pass = False
    
    # Check for reset in policy_output for eval mode
    if 'policy_output = {"action": action, "reset": reset}' in content:
        print("✓ PASS: Reset found in policy_output (eval mode)")
        eval_pass = True
    else:
        print("✗ FAIL: Reset NOT found in policy_output (eval mode)")
        eval_pass = False
    
    return training_pass and eval_pass


def check_reset_logic():
    """Check if reset uses agent decision OR is_last."""
    print("\n" + "=" * 60)
    print("TEST 2: Check reset logic uses agent decision")
    print("=" * 60)
    
    hieros_file = pathlib.Path(__file__).parent / 'hieros' / 'hieros.py'
    content = hieros_file.read_text()
    
    # Check for the new reset logic
    if 'acts["reset"] = acts["reset"] | obs["is_last"].copy()' in content:
        print("✓ PASS: Reset logic uses agent decision OR is_last")
        return True
    else:
        print("✗ FAIL: Reset logic does not use agent decision")
        return False


def check_reset_excluded_from_masking():
    """Check if reset is excluded from masking when is_last is true."""
    print("\n" + "=" * 60)
    print("TEST 3: Check reset excluded from is_last masking")
    print("=" * 60)
    
    hieros_file = pathlib.Path(__file__).parent / 'hieros' / 'hieros.py'
    content = hieros_file.read_text()
    
    # Check that reset is excluded from masking
    if 'acts.items() if k != "reset"' in content:
        print("✓ PASS: Reset excluded from masking")
        return True
    else:
        print("✗ FAIL: Reset not excluded from masking")
        return False


def check_dreamer_has_same_fix():
    """Check if dreamer.py has the same fixes."""
    print("\n" + "=" * 60)
    print("TEST 4: Check dreamer.py has same fixes")
    print("=" * 60)
    
    dreamer_file = pathlib.Path(__file__).parent / 'hieros' / 'dreamer.py'
    content = dreamer_file.read_text()
    
    checks = []
    
    # Check for reset in policy_output
    if 'policy_output = {"action": action, "log_entropy": logprob, "reset": reset}' in content:
        print("✓ Reset found in dreamer policy_output")
        checks.append(True)
    else:
        print("✗ Reset NOT found in dreamer policy_output")
        checks.append(False)
    
    # Check for reset logic
    if 'acts["reset"] = acts["reset"] | obs["is_last"].copy()' in content:
        print("✓ Reset logic found in dreamer")
        checks.append(True)
    else:
        print("✗ Reset logic NOT found in dreamer")
        checks.append(False)
    
    return all(checks)


def check_backward_compatibility():
    """Check that backward compatibility is maintained."""
    print("\n" + "=" * 60)
    print("TEST 5: Check backward compatibility")
    print("=" * 60)
    
    hieros_file = pathlib.Path(__file__).parent / 'hieros' / 'hieros.py'
    content = hieros_file.read_text()
    
    # Check that reset defaults to False (not True)
    if 'reset = torch.zeros(action.shape[0], dtype=torch.bool' in content:
        print("✓ PASS: Reset defaults to False (backward compatible)")
        return True
    else:
        print("✗ FAIL: Reset does not default to False")
        return False


def check_all_files_updated():
    """Check if all relevant files have been updated."""
    print("\n" + "=" * 60)
    print("TEST 6: Check all files updated")
    print("=" * 60)
    
    files_to_check = [
        'hieros/hieros.py',
        'hieros/dreamer.py',
        'embodied/core/driver.py',
        'hieros/train.py',
    ]
    
    all_good = True
    for file_path in files_to_check:
        full_path = pathlib.Path(__file__).parent / file_path
        if not full_path.exists():
            print(f"✗ File not found: {file_path}")
            all_good = False
            continue
        
        content = full_path.read_text()
        
        # Check for the new reset logic pattern
        if 'acts["reset"] = acts["reset"] | obs["is_last"].copy()' in content or \
           'acts["reset"] = acts["reset"] | o["is_last"].copy()' in content:
            print(f"✓ {file_path}: Updated with agent reset logic")
        else:
            print(f"✗ {file_path}: Missing agent reset logic")
            all_good = False
    
    return all_good


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING HIEROS AGENT RESET ACTION - CODE STRUCTURE")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("Reset in policy_output", check_reset_in_policy_output()))
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        results.append(("Reset in policy_output", False))
    
    try:
        results.append(("Reset logic correct", check_reset_logic()))
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        results.append(("Reset logic correct", False))
    
    try:
        results.append(("Reset excluded from masking", check_reset_excluded_from_masking()))
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        results.append(("Reset excluded from masking", False))
    
    try:
        results.append(("Dreamer has same fixes", check_dreamer_has_same_fix()))
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        results.append(("Dreamer has same fixes", False))
    
    try:
        results.append(("Backward compatibility", check_backward_compatibility()))
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {e}")
        results.append(("Backward compatibility", False))
    
    try:
        results.append(("All files updated", check_all_files_updated()))
    except Exception as e:
        print(f"\n✗ TEST 6 FAILED: {e}")
        results.append(("All files updated", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("\nThe changes successfully enable reset action capability:")
        print("1. Agent can now output reset action")
        print("2. Reset uses agent decision OR environment is_last")
        print("3. Backward compatible (defaults to False)")
        print("4. Updated: Hieros, Dreamer, Driver, and Train files")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
