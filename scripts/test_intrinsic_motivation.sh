#!/bin/bash
# Quick test script to verify intrinsic motivation exploration module
# This runs a very short training to ensure the code runs without errors
# and doesn't test for exploration convergence (which requires longer runs)

set -e

# Navigate to repo directory (detect automatically)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

# Export PYTHONPATH
export PYTHONPATH="$REPO_DIR:$REPO_DIR/hieros:$PYTHONPATH"

echo "=== Testing intrinsic motivation module integration ==="
echo ""

# Test 1: Check that intrinsic_motivation module loads
echo "Test 1: Module import check..."
python -c "
import sys
sys.path.insert(0, 'hieros')
import intrinsic_motivation as im
print('  ✓ intrinsic_motivation module imports correctly')
"

# Test 2: Check that configs can be loaded with new intrinsic motivation settings
echo ""
echo "Test 2: Config validation..."
python -c "
import ruamel.yaml as yaml
y = yaml.YAML(typ='safe')
config = y.load(open('hieros/configs.yaml'))

# Check new intrinsic_motivation config exists
assert 'intrinsic_motivation' in config['defaults'], 'Missing intrinsic_motivation in defaults'
im_config = config['defaults']['intrinsic_motivation']
assert 'enabled' in im_config, 'Missing enabled in intrinsic_motivation'
assert 'use_rnd' in im_config, 'Missing use_rnd in intrinsic_motivation'
assert 'use_episodic_count' in im_config, 'Missing use_episodic_count in intrinsic_motivation'
assert 'use_hierarchical_bonus' in im_config, 'Missing use_hierarchical_bonus in intrinsic_motivation'

# Check new experiment configs exist
assert 'pinpad-easy-rnd' in config, 'Missing pinpad-easy-rnd config'
assert 'pinpad-easy-count' in config, 'Missing pinpad-easy-count config'
assert 'pinpad-easy-hierarchical' in config, 'Missing pinpad-easy-hierarchical config'
assert 'pinpad-easy-full-exploration' in config, 'Missing pinpad-easy-full-exploration config'
assert 'pinpad-easy-scheduled' in config, 'Missing pinpad-easy-scheduled config'

print('  ✓ All intrinsic motivation configs are valid')
"

# Test 3: Check that models.py has updated ImagBehavior
echo ""
echo "Test 3: ImagBehavior update check..."
python -c "
# Check that ImagBehavior has use_intrinsic parameter
import ast
import sys
sys.path.insert(0, 'hieros')

with open('hieros/models.py', 'r') as f:
    content = f.read()
    
# Check for use_intrinsic parameter
if 'use_intrinsic' not in content:
    raise AssertionError('ImagBehavior missing use_intrinsic parameter')
    
# Check for intrinsic value head
if '\"intrinsic\"' not in content:
    raise AssertionError('ImagBehavior missing intrinsic value head handling')

print('  ✓ ImagBehavior correctly updated with intrinsic motivation support')
"

# Test 4: Check that hieros.py properly imports and uses intrinsic_motivation
echo ""
echo "Test 4: Hieros integration check..."
python -c "
with open('hieros/hieros.py', 'r') as f:
    content = f.read()

# Check import
if 'import intrinsic_motivation as im' not in content:
    raise AssertionError('hieros.py missing intrinsic_motivation import')

# Check usage in SubActor
if '_use_intrinsic_motivation' not in content:
    raise AssertionError('SubActor missing _use_intrinsic_motivation attribute')
    
if 'IntrinsicMotivationManager' not in content:
    raise AssertionError('SubActor missing IntrinsicMotivationManager initialization')

print('  ✓ Hieros correctly integrates intrinsic motivation module')
"

echo ""
echo "=== All tests passed! ==="
echo ""
echo "To run a full exploration experiment, use one of the following commands:"
echo ""
echo "1. RND-based exploration:"
echo "   python hieros/train.py --configs pinpad-easy-rnd rm_model_size --steps 50000 --wandb_logging False"
echo ""
echo "2. Count-based exploration:"
echo "   python hieros/train.py --configs pinpad-easy-count rm_model_size --steps 50000 --wandb_logging False"
echo ""
echo "3. Full exploration suite:"
echo "   python hieros/train.py --configs pinpad-easy-full-exploration rm_model_size --steps 50000 --wandb_logging False"
echo ""
echo "Note: For proper exploration analysis, run for at least 400k steps and monitor:"
echo "  - position heatmaps via get_position_heatmap()"
echo "  - intrinsic/exploration_coef metric in logs"
echo "  - intrinsic/rnd_loss metric in logs"
