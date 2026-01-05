import collections

import embodied
import numpy as np


class PinPadEasy(embodied.Env):
    COLORS = {
        "1": (255, 0, 0),
        "2": (0, 255, 0),
        "3": (0, 0, 255),
        "4": (255, 255, 0),
        "5": (255, 0, 255),
        "6": (0, 255, 255),
        "7": (128, 0, 128),
        "8": (0, 128, 128),
    }

    # Reward modes for experimentation
    REWARD_MODES = {
        "flat": "Flat +1.0 for each correct intermediate tile (original)",
        "progressive": "Exponentially increasing rewards for later tiles",
        "sequence_bonus": "Base reward + bonus based on sequence length",
        "decaying": "Time-decaying intermediate rewards",
        "sparse": "Only reward for completing full sequence",
        "progressive_steep": "Steeper exponential increase for later tiles",
        "dense_guidance": "Step-wise rewards: +0.1 for moving toward target, -0.1 for wrong tile",
        "progress_any": "Reward any tile that contributes to sequence progress",
    }

    # Dense guidance reward constants (can be tuned)
    DENSE_MOVE_TOWARD_REWARD = 0.1
    DENSE_MOVE_AWAY_PENALTY = 0.05
    DENSE_WRONG_TILE_PENALTY = 0.1
    DENSE_CORRECT_TILE_BONUS = 1.0

    def __init__(self, task, length=1000, seed=None, reward_mode="flat"):
        assert length > 0
        assert reward_mode in self.REWARD_MODES, f"Invalid reward_mode: {reward_mode}. Valid modes: {list(self.REWARD_MODES.keys())}"
        layout = {
            "three": LAYOUT_THREE,
            "four": LAYOUT_FOUR,
            "five": LAYOUT_FIVE,
            "six": LAYOUT_SIX,
            "seven": LAYOUT_SEVEN,
            "eight": LAYOUT_EIGHT,
        }[task]
        self.layout = np.array([list(line) for line in layout.split("\n")]).T
        assert self.layout.shape == (16, 14), self.layout.shape
        self.length = length
        self._seed = seed
        self.random = np.random.RandomState(seed)
        self.pads = set(self.layout.flatten().tolist()) - set("* #\n")
        self.target = tuple(sorted(self.pads))
        self.spawns = []
        # Precompute pad center positions for distance-based rewards
        self.pad_positions = {}
        for (x, y), char in np.ndenumerate(self.layout):
            if char != "#":
                self.spawns.append((x, y))
            if char in self.pads:
                if char not in self.pad_positions:
                    self.pad_positions[char] = []
                self.pad_positions[char].append((x, y))
        # Compute center of each pad
        self.pad_centers = {}
        for pad, positions in self.pad_positions.items():
            positions = np.array(positions)
            self.pad_centers[pad] = (positions[:, 0].mean(), positions[:, 1].mean())
        self.reward_mode = reward_mode
        print(f'Created PinPadEasy env with sequence: {"->".join(self.target)}, reward_mode: {reward_mode}')
        self.sequence = collections.deque(maxlen=len(self.target))
        # Track correctly visited tiles for progress_any mode
        self.visited_correct_tiles = set()
        self.player = None
        self.steps = None
        self.done = None
        self.countdown = None
        # Track tile visits for decaying reward mode
        self.tile_visits = {tile: 0 for tile in self.pads}
        # Position visit tracking for visualization
        self.position_visit_counts = np.zeros((16, 14), dtype=np.int64)
        # Cache spaces with seed
        self._act_space = {
            "action": embodied.Space(np.int64, (), 0, 5, seed=seed),
            "reset": embodied.Space(bool, seed=seed),
        }
        self._obs_space = {
            "image": embodied.Space(np.uint8, (64, 64, 3), seed=seed),
            "reward": embodied.Space(np.float32, seed=seed),
            "is_first": embodied.Space(bool, seed=seed),
            "is_last": embodied.Space(bool, seed=seed),
            "is_terminal": embodied.Space(bool, seed=seed),
        }

    @property
    def act_space(self):
        return self._act_space

    @property
    def obs_space(self):
        return self._obs_space

    def step(self, action):
        if self.done or action["reset"]:
            self.player = self.spawns[self.random.randint(len(self.spawns))]
            self.sequence.clear()
            self.visited_correct_tiles = set()
            self.steps = 0
            self.done = False
            self.countdown = 0
            # Reset tile visits for decaying mode
            self.tile_visits = {tile: 0 for tile in self.pads}
            return self._obs(reward=0.0, is_first=True)
        if self.countdown:
            self.countdown -= 1
            if self.countdown == 0:
                self.player = self.spawns[self.random.randint(len(self.spawns))]
                self.sequence.clear()
                self.visited_correct_tiles = set()
        
        reward = 0.0
        old_pos = self.player
        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action["action"]]
        x = np.clip(self.player[0] + move[0], 0, 15)
        y = np.clip(self.player[1] + move[1], 0, 13)
        tile = self.layout[x][y]
        
        if tile != "#":
            self.player = (x, y)
            # Track position visits
            self.position_visit_counts[x, y] += 1
        
        # For dense_guidance mode, compute distance-based rewards
        if self.reward_mode == "dense_guidance":
            reward += self._compute_dense_guidance_reward(old_pos, self.player, tile)
        
        if tile in self.pads:
            if not self.sequence or self.sequence[-1] != tile:
                self.sequence.append(tile)
                
                # Handle different reward modes for tile visits
                if self.reward_mode == "progress_any":
                    # Give reward for any tile that's part of the target sequence
                    # and hasn't been rewarded yet in this episode
                    if tile in self.target and tile not in self.visited_correct_tiles:
                        # Calculate reward based on count before adding tile
                        tiles_before = len(self.visited_correct_tiles)
                        self.visited_correct_tiles.add(tile)
                        # Give reward based on how many tiles collected so far (1.0, 1.5, 2.0, ...)
                        reward += 1.0 + 0.5 * tiles_before
                elif self.reward_mode != "dense_guidance":
                    # Standard intermediate reward for correct sequence
                    if len(self.sequence) < len(self.target) and tile == self.target[len(self.sequence) - 1]:
                        reward += self._compute_intermediate_reward(tile, len(self.sequence))
        
        if tuple(self.sequence) == self.target and not self.countdown:
            reward += 10.0
            self.countdown = 10
        self.steps += 1
        self.done = self.done or (self.steps >= self.length)
        return self._obs(reward=reward, is_last=self.done)

    def _compute_dense_guidance_reward(self, old_pos, new_pos, tile):
        """
        Compute step-wise guidance reward based on movement toward target tile.
        
        Args:
            old_pos: Previous player position
            new_pos: Current player position  
            tile: The tile at the new position
        
        Returns:
            float: Small positive/negative reward based on movement
        """
        # Determine the next target tile in the sequence
        next_target_idx = len(self.sequence)
        if next_target_idx >= len(self.target):
            return 0.0  # Already completed
        
        next_target = self.target[next_target_idx]
        target_center = self.pad_centers[next_target]
        
        # Calculate distances
        old_dist = np.sqrt((old_pos[0] - target_center[0])**2 + (old_pos[1] - target_center[1])**2)
        new_dist = np.sqrt((new_pos[0] - target_center[0])**2 + (new_pos[1] - target_center[1])**2)
        
        reward = 0.0
        
        # Small reward for moving closer to target
        if new_dist < old_dist:
            reward += self.DENSE_MOVE_TOWARD_REWARD
        elif new_dist > old_dist:
            reward -= self.DENSE_MOVE_AWAY_PENALTY
        
        # Penalty for stepping on wrong tile
        if tile in self.pads and tile != next_target:
            reward -= self.DENSE_WRONG_TILE_PENALTY
        
        # Bonus for reaching the correct target tile
        if tile == next_target:
            reward += self.DENSE_CORRECT_TILE_BONUS
        
        return reward

    def _compute_intermediate_reward(self, tile, sequence_position):
        """
        Compute intermediate reward based on the reward mode.
        
        Args:
            tile: The tile character that was just reached
            sequence_position: The current position in the sequence (1-indexed)
        
        Returns:
            float: The intermediate reward for reaching this tile
        """
        if self.reward_mode == "flat":
            # Original: flat +1.0 for each correct intermediate tile
            return 1.0
        
        elif self.reward_mode == "progressive":
            # Exponentially increasing rewards: 1.0, 2.0, 4.0, 8.0, ...
            # This makes later tiles much more valuable to incentivize progression
            return 2.0 ** (sequence_position - 1)
        
        elif self.reward_mode == "progressive_steep":
            # Steeper exponential: 1.0, 3.0, 9.0, 27.0, ...
            # Even stronger incentive to reach later tiles
            return 3.0 ** (sequence_position - 1)
        
        elif self.reward_mode == "sequence_bonus":
            # Base reward + multiplicative bonus based on sequence length
            # e.g., 1.0 + 0.5*1 = 1.5, 1.0 + 0.5*2 = 2.0, 1.0 + 0.5*3 = 2.5
            base_reward = 1.0
            sequence_bonus = 0.5 * sequence_position
            return base_reward + sequence_bonus
        
        elif self.reward_mode == "decaying":
            # Time-decaying intermediate rewards based on tile visits
            # First visit (tile_visits[tile]=0): full reward (decay_factor=1.0)
            # Subsequent visits: decayed reward (decay_factor < 1.0)
            # Increment happens after calculation so first visit gets full reward
            decay_factor = 1.0 / (1.0 + 0.1 * self.tile_visits[tile])
            self.tile_visits[tile] += 1
            return 1.0 * decay_factor
        
        elif self.reward_mode == "sparse":
            # No intermediate rewards - only completion bonus
            return 0.0
        
        else:
            # Default to flat
            return 1.0

    def render(self):
        grid = np.zeros((16, 16, 3), np.uint8) + 255
        white = np.array([255, 255, 255])
        if self.countdown:
            grid[:] = (223, 255, 223)
        current = self.layout[self.player[0]][self.player[1]]
        for (x, y), char in np.ndenumerate(self.layout):
            if char == "#":
                grid[x, y] = (192, 192, 192)
            elif char in self.pads:
                color = np.array(self.COLORS[char])
                color = color if char == current else (10 * color + 90 * white) / 100
                grid[x, y] = color
        grid[self.player] = (0, 0, 0)
        grid[:, -2:] = (192, 192, 192)
        for i, char in enumerate(self.sequence):
            grid[2 * i + 1, -2] = self.COLORS[char]
        image = np.repeat(np.repeat(grid, 4, 0), 4, 1)
        return image.transpose((1, 0, 2))

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        return dict(
            image=self.render(),
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def get_position_heatmap(self):
        """
        Generate a heatmap visualization of position visit counts.
        Returns a numpy array suitable for logging as an image.
        """
        # Create a normalized heatmap (16x14 grid)
        visit_counts = self.position_visit_counts.copy().astype(np.float32)
        
        # Avoid division by zero
        max_visits = visit_counts.max()
        if max_visits > 0:
            normalized_counts = visit_counts / max_visits
        else:
            normalized_counts = visit_counts
        
        # Create RGB heatmap (red = high visits, blue = low visits)
        # Use a color gradient from blue (cold/low) to red (hot/high)
        heatmap = np.zeros((16, 14, 3), dtype=np.uint8)
        
        for (x, y), count in np.ndenumerate(normalized_counts):
            char = self.layout[x, y]
            if char == "#":
                # Walls are gray
                heatmap[x, y] = (192, 192, 192)
            else:
                # Color based on visit frequency
                # Blue (low) -> Green -> Yellow -> Red (high)
                intensity = normalized_counts[x, y]
                if intensity < 0.25:
                    # Blue to Cyan
                    r = 0
                    g = int(intensity * 4 * 255)
                    b = 255
                elif intensity < 0.5:
                    # Cyan to Green
                    r = 0
                    g = 255
                    b = int((0.5 - intensity) * 4 * 255)
                elif intensity < 0.75:
                    # Green to Yellow
                    r = int((intensity - 0.5) * 4 * 255)
                    g = 255
                    b = 0
                else:
                    # Yellow to Red
                    r = 255
                    g = int((1.0 - intensity) * 4 * 255)
                    b = 0
                heatmap[x, y] = (r, g, b)
        
        # Scale up the heatmap for better visibility (4x)
        heatmap_scaled = np.repeat(np.repeat(heatmap, 4, 0), 4, 1)
        # Transpose to match render() format
        return heatmap_scaled.transpose((1, 0, 2))

    def get_position_stats(self):
        """
        Get statistics about position visits.
        Returns a dictionary with visit statistics.
        """
        # Count only non-wall positions
        valid_positions = np.zeros_like(self.position_visit_counts, dtype=bool)
        for (x, y), char in np.ndenumerate(self.layout):
            valid_positions[x, y] = (char != "#")
        
        valid_visits = self.position_visit_counts[valid_positions]
        total_valid_positions = valid_positions.sum()
        visited_positions = (valid_visits > 0).sum()
        
        return {
            "total_visits": int(self.position_visit_counts.sum()),
            "unique_positions_visited": int(visited_positions),
            "total_valid_positions": int(total_valid_positions),
            "coverage_ratio": float(visited_positions) / float(total_valid_positions) if total_valid_positions > 0 else 0.0,
            "max_visits_single_position": int(self.position_visit_counts.max()),
            "mean_visits_per_visited_position": float(valid_visits[valid_visits > 0].mean()) if visited_positions > 0 else 0.0,
        }


LAYOUT_THREE = """
################
#1111      3333#
#1111      3333#
#1111      3333#
#1111      3333#
#              #
#              #
#              #
#              #
#     2222     #
#     2222     #
#     2222     #
#     2222     #
################
""".strip('\n')

LAYOUT_FOUR = """
################
#1111      4444#
#1111      4444#
#1111      4444#
#1111      4444#
#              #
#              #
#              #
#              #
#3333      2222#
#3333      2222#
#3333      2222#
#3333      2222#
################
""".strip('\n')

LAYOUT_FIVE = """
################
#          4444#
#111       4444#
#111       4444#
#111           #
#111        555#
#           555#
#           555#
#333        555#
#333           #
#333       2222#
#333       2222#
#          2222#
################
""".strip('\n')

LAYOUT_SIX = """
################
#111        555#
#111        555#
#111        555#
#              #
#33          66#
#33          66#
#33          66#
#33          66#
#              #
#444        222#
#444        222#
#444        222#
################
""".strip('\n')

LAYOUT_SEVEN = """
################
#111        444#
#111        444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')

LAYOUT_EIGHT = """
################
#111  8888  444#
#111  8888  444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')
