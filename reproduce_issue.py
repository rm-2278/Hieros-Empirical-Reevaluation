
import argparse
import sys
from types import SimpleNamespace

# Mocking the relevant parts of train.py logic
def infer_value_type(value_str):
    try: return int(value_str)
    except ValueError: pass
    try: return float(value_str)
    except ValueError: pass
    if value_str.lower() in ('true', 'false', 'yes', 'no'):
        return value_str.lower() in ('true', 'yes')
    return value_str

def set_nested_dict(d, parts, value):
    current = d
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value

def set_nested_namespace(ns, parts, value):
    current = ns
    for part in parts[:-1]:
        if not hasattr(current, part):
            setattr(current, part, argparse.Namespace())
        current = getattr(current, part)
    setattr(current, parts[-1], value)

def parse_unknown(args, unknown):
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--'):
            if '=' in arg:
                dotted_key, value = arg[2:].split('=', 1)
                i += 1
            else:
                dotted_key = arg[2:]
                # Just simulating the case where value is provided
                value = "decaying" 
                i += 2 
            
            if not dotted_key: continue
            
            # THE BUG: Unconditional replacement
            dotted_key = dotted_key.replace('-', '_')
            
            typed_value = infer_value_type(value)
            
            if '.' in dotted_key:
                parts = dotted_key.split('.')
                if hasattr(args, parts[0]):
                    first_obj = getattr(args, parts[0])
                    if isinstance(first_obj, dict):
                        set_nested_dict(first_obj, parts[1:], typed_value)
                    else:
                        set_nested_namespace(first_obj, parts[1:], typed_value)
                else:
                    set_nested_namespace(args, parts, typed_value)
            else:
                setattr(args, dotted_key, typed_value)
        else:
            i += 1
    return args

def main():
    # Setup initial state similar to what train.py would have from configs.yaml
    args = SimpleNamespace()
    
    # Existing config has "pinpad-easy"
    args.env = {
        "pinpad-easy": {"reward_mode": "flat"}
    }
    
    print(f"Initial state: {args.env}")
    
    # Simulate passing --env.pinpad-easy.reward_mode=decaying
    unknown = ["--env.pinpad-easy.reward_mode=decaying"]
    
    args = parse_unknown(args, unknown)
    
    print(f"Final state: {args.env}")
    
    if "pinpad-easy" in args.env and args.env["pinpad-easy"]["reward_mode"] == "decaying":
        print("SUCCESS: pinpad-easy updated correctly")
        sys.exit(0)
    elif "pinpad_easy" in args.env:
         print("FAILURE: created new key pinpad_easy instead of updating pinpad-easy")
         sys.exit(1)
    else:
        print("FAILURE: unknown state")
        sys.exit(1)

if __name__ == "__main__":
    main()
