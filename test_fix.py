
import argparse
import sys
from types import SimpleNamespace

print("Script starting...", flush=True)

# Mocking the relevant parts 
def infer_value_type(value_str):
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
    print(f"Parsing unknown: {unknown}", flush=True)
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        print(f"Processing arg: {arg}, i={i}", flush=True)
        if arg.startswith('--'):
            if '=' in arg:
                dotted_key, value = arg[2:].split('=', 1)
                i += 1
            else:
                dotted_key = arg[2:]
                value = "decaying"
                i += 2
            
            if not dotted_key: continue
            
            typed_value = infer_value_type(value)
            print(f"Key: {dotted_key}, Value: {typed_value}", flush=True)
            
            # --- START NEW LOGIC FROM train.py ---
            if '.' in dotted_key:
                parts = dotted_key.split('.')
                
                # The first part corresponds to an argparse argument, so it will have underscores
                root_part = parts[0].replace('-', '_')
                
                # Check if the root object exists
                if hasattr(args, root_part):
                    current_obj = getattr(args, root_part)
                    print(f"Root part {root_part} exists", flush=True)
                    
                    # Navigate through the rest of the parts
                    # We need to handle the path, checking for existing keys with hyphens
                    remaining_parts = parts[1:]
                    
                    # If the root is a dict, we process it differently than a Namespace
                    if isinstance(current_obj, dict):
                        target_dict = current_obj
                        for i_part, part in enumerate(remaining_parts[:-1]):
                            print(f"Processing part: {part}", flush=True)
                            # Check if part exists as is (with hyphens)
                            if part in target_dict:
                                target_dict = target_dict[part]
                            # Check if part exists with underscores
                            elif part.replace('-', '_') in target_dict:
                                target_dict = target_dict[part.replace('-', '_')]
                            else:
                                # Create new dict if not found
                                # Default to underscore for new keys to match previous behavior
                                new_key = part.replace('-', '_')
                                target_dict[new_key] = {}
                                target_dict = target_dict[new_key]
                        
                        # Set the final value
                        last_part = remaining_parts[-1]
                        print(f"Setting last part: {last_part}", flush=True)
                        # Check existance for last part too
                        if last_part in target_dict:
                            target_dict[last_part] = typed_value
                        elif last_part.replace('-', '_') in target_dict:
                            target_dict[last_part.replace('-', '_')] = typed_value
                        else:
                            # Default to underscore
                            target_dict[last_part.replace('-', '_')] = typed_value
                            
                    else:
                        clean_parts = [p.replace('-', '_') for p in remaining_parts]
                        set_nested_namespace(current_obj, clean_parts, typed_value)
                else:
                    clean_parts = [p.replace('-', '_') for p in parts]
                    set_nested_namespace(args, clean_parts, typed_value)
            else:
                dotted_key = dotted_key.replace('-', '_')
                setattr(args, dotted_key, typed_value)
            # --- END NEW LOGIC ---
        else:
            i += 1
    return args

def main():
    args = SimpleNamespace()
    args.env = {
        "pinpad-easy": {"reward_mode": "flat"}
    }
    
    print(f"Initial state: {args.env}", flush=True)
    unknown = ["--env.pinpad-easy.reward_mode=decaying"]
    args = parse_unknown(args, unknown)
    print(f"Final state: {args.env}", flush=True)
    
    if "pinpad-easy" in args.env and args.env["pinpad-easy"]["reward_mode"] == "decaying":
        # Check that we didn't also create the underscored one
        if "pinpad_easy" in args.env:
             print("WARNING: Created correct key but also incorrect key", flush=True)
        else:
             print("SUCCESS: pinpad-easy updated correctly", flush=True)
             sys.exit(0)
    
    print(f"FAILURE: state is {args.env}", flush=True)
    sys.exit(1)

if __name__ == "__main__":
    main()
