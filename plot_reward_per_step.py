import json
import matplotlib.pyplot as plt

lengths = []
scores = []
steps = []

with open('logs/atari_battle_zone-20251223-210503/metrics.jsonl', 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
        except Exception:
            continue
        # Look for episode metrics
        for k, v in data.items():
            if k.startswith('episode/'):
                if 'episode/length' in data and 'episode/score' in data:
                    lengths.append(data['episode/length'])
                    scores.append(data['episode/score'])
                    # Optionally, use step if available
                    steps.append(data.get('step', len(lengths)))
                break

reward_per_step = [s / l if l != 0 else 0 for s, l in zip(scores, lengths)]

plt.figure(figsize=(10, 5))
plt.plot(steps, reward_per_step, marker='o')
plt.xlabel('Episode')
plt.ylabel('Reward per Step')
plt.title('Reward per Step over Episodes')
plt.grid(True)
plt.tight_layout()
plt.show()
