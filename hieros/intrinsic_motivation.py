"""
Intrinsic Motivation Module for Enhanced Exploration in Hieros.

This module implements various intrinsic reward mechanisms to address
exploration collapse in sequential tasks like pinpad-easy:

1. RND (Random Network Distillation) - Persistent curiosity that doesn't diminish
   with world model improvement
2. Episodic Count-Based Exploration - Explicit visitation bonus in latent space
3. Hierarchical Exploration Bonus - Rewards higher-level actors for diverse subgoals

References:
- RND: Burda et al., "Exploration by Random Network Distillation" (2018)
- Count-Based: Bellemare et al., "Count-Based Exploration with Neural Density Models" (2016)
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import tools


class RNDModule(nn.Module):
    """
    Random Network Distillation for intrinsic motivation.
    
    RND provides a curiosity signal by training a predictor network to match
    a fixed random target network. The prediction error serves as intrinsic reward.
    Unlike reconstruction-based novelty, RND doesn't diminish as the world model improves.
    
    Args:
        input_dim: Dimension of the input features (latent state dimension)
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        num_layers: Number of hidden layers in predictor network
        config: Configuration object with device and optimizer settings
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=128,
        num_layers=3,
        config=None,
    ):
        super(RNDModule, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        
        # Fixed random target network (not trained)
        target_layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            target_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                target_layers.append(nn.ReLU())
            in_dim = out_dim
        self.target_network = nn.Sequential(*target_layers)
        
        # Initialize target network with fixed random weights
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Predictor network (trained to match target)
        predictor_layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            predictor_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                predictor_layers.append(nn.ReLU())
            in_dim = out_dim
        self.predictor_network = nn.Sequential(*predictor_layers)
        self.predictor_network.apply(tools.weight_init)
        
        # Running statistics for normalization
        self.obs_rms = RunningMeanStd(shape=(input_dim,), device=config.device)
        self.reward_rms = RunningMeanStd(shape=(1,), device=config.device)
        
        # Get RND config with fallbacks
        rnd_config = getattr(config, 'rnd', {})
        if not isinstance(rnd_config, dict):
            rnd_config = {}
        
        # Optimizer for predictor network
        self._optimizer = tools.Optimizer(
            "rnd_predictor",
            self.predictor_network.parameters(),
            rnd_config.get("lr", 1e-4),
            rnd_config.get("opt_eps", 1e-8),
            rnd_config.get("grad_clip", 100.0),
            rnd_config.get("weight_decay", 0),
            opt=config.opt,
            use_amp=self._use_amp,
        )
    
    def forward(self, features):
        """
        Compute RND intrinsic reward.
        
        Args:
            features: Latent state features [batch, ..., feat_dim]
            
        Returns:
            Intrinsic reward based on prediction error [batch, ..., 1]
        """
        # Normalize input features
        normalized_features = self._normalize_obs(features)
        
        # Get target and predictor outputs
        with torch.no_grad():
            target_output = self.target_network(normalized_features)
        predictor_output = self.predictor_network(normalized_features)
        
        # Compute prediction error as intrinsic reward
        prediction_error = torch.mean((predictor_output - target_output) ** 2, dim=-1, keepdim=True)
        
        # Normalize reward
        intrinsic_reward = self._normalize_reward(prediction_error)
        
        return intrinsic_reward
    
    def _normalize_obs(self, obs):
        """Normalize observations using running statistics."""
        # CRITICAL: Detach to prevent infinite computation graph
        self.obs_rms.update(obs.reshape(-1, obs.shape[-1]).detach())
        # Use detached statistics to prevent backprop through running mean
        mu = self.obs_rms.mean.detach()
        sigma = torch.sqrt(self.obs_rms.var + 1e-8).detach()
        return (obs - mu) / sigma
    
    def _normalize_reward(self, reward):
        """Normalize rewards using running statistics."""
        # CRITICAL: Detach to prevent infinite computation graph
        self.reward_rms.update(reward.reshape(-1, 1).detach())
        return reward / torch.sqrt(self.reward_rms.var.detach() + 1e-8)
    
    def train_step(self, features):
        """
        Train the predictor network to match the target network.
        
        Args:
            features: Latent state features [batch, seq, feat_dim]
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {}
        
        with tools.RequiresGrad(self.predictor_network):
            with torch.cuda.amp.autocast(self._use_amp):
                # Normalize features
                normalized_features = self._normalize_obs(features.detach())
                
                # Compute target (no gradient)
                with torch.no_grad():
                    target_output = self.target_network(normalized_features)
                
                # Compute predictor output
                predictor_output = self.predictor_network(normalized_features)
                
                # MSE loss
                loss = F.mse_loss(predictor_output, target_output)
            
            metrics.update(self._optimizer(loss, self.predictor_network.parameters()))
            metrics["rnd_loss"] = loss.detach()
        
        return metrics


class EpisodicCountModule(nn.Module):
    """
    Episodic count-based exploration in latent space.
    
    Maintains a count of state visitations within each episode and provides
    intrinsic reward inversely proportional to visit counts. Uses locality-sensitive
    hashing (LSH) or k-nearest neighbors in latent space.
    
    Args:
        input_dim: Dimension of the input features
        num_bins: Number of bins for discretization (for hash-based counting)
        k_neighbors: Number of neighbors for kernel-based counting
        config: Configuration object
    """
    
    def __init__(
        self,
        input_dim,
        num_bins=32,
        k_neighbors=10,
        kernel_epsilon=0.001,
        config=None,
    ):
        super(EpisodicCountModule, self).__init__()
        self._config = config
        self._input_dim = input_dim
        self._num_bins = num_bins
        self._k_neighbors = k_neighbors
        self._kernel_epsilon = kernel_epsilon
        
        # Memory limits to prevent memory exhaustion - reduced defaults
        im_config = getattr(config, 'intrinsic_motivation', {})
        if not isinstance(im_config, dict):
            im_config = {}
        self._max_memory_per_env = im_config.get("max_memory_per_env", 5000)
        self._max_hash_entries_per_env = im_config.get("max_hash_entries_per_env", 10000)
        
        # Random projection for locality-sensitive hashing
        self.register_buffer(
            'random_projection',
            torch.randn(input_dim, num_bins, device=config.device) / np.sqrt(input_dim)
        )
        
        # OPTIMIZATION: Pre-compute powers of two for vectorized bit-packing
        # Limit to 63 bits to fit in signed int64
        safe_bins = min(num_bins, 63)
        self.register_buffer(
            'powers_of_two',
            (2 ** torch.arange(safe_bins, device=config.device)).long()
        )
        
        # Episodic memory - stores latent states from current episode
        # Will be reset at episode boundaries
        self._episodic_memory = defaultdict(list)
        self._visit_counts = defaultdict(lambda: defaultdict(int))
        
        # Running statistics for feature normalization
        self.feature_rms = RunningMeanStd(shape=(input_dim,), device=config.device)
        
    def reset_episode(self, env_indices=None):
        """
        Reset episodic memory at episode boundaries.
        
        Args:
            env_indices: List of environment indices to reset. If None, reset all.
        """
        if env_indices is None:
            self._episodic_memory.clear()
            self._visit_counts.clear()
        else:
            for idx in env_indices:
                if idx in self._episodic_memory:
                    del self._episodic_memory[idx]
                if idx in self._visit_counts:
                    del self._visit_counts[idx]
    
    def forward(self, features, env_indices=None):
        """
        Compute count-based intrinsic reward.
        
        Args:
            features: Latent state features [batch, feat_dim] or [batch, seq, feat_dim]
            env_indices: Environment indices for tracking per-environment counts
            
        Returns:
            Intrinsic reward based on inverse visit count [batch, 1] or [batch, seq, 1]
        """
        original_shape = features.shape
        has_seq = len(original_shape) == 3
        
        if has_seq:
            batch, seq, feat = features.shape
            features_flat = features.reshape(-1, feat)
        else:
            features_flat = features
            batch = features.shape[0]
            seq = 1
        
        # --- FIX 1: MEMORY LEAK PREVENTION ---
        # CRITICAL: Detach features before updating statistics to prevent infinite graph
        self.feature_rms.update(features_flat.detach())
        
        # Use detached stats to ensure we don't backprop through running mean
        mu = self.feature_rms.mean.detach()
        sigma = torch.sqrt(self.feature_rms.var + 1e-8).detach()
        normalized = (features_flat - mu) / sigma
        
        # --- FIX 2: VECTORIZED HASHING (Optimized) ---
        # Project to lower dimension
        projected = torch.matmul(normalized, self.random_projection)
        
        # Create binary hash codes (0 or 1)
        hash_bits = (projected > 0).long()
        
        # Bit-packing: Turn [Batch, num_bins] bits into [Batch] integers
        # This operation is fully vectorized on GPU
        num_bins = hash_bits.shape[-1]
        if num_bins <= 63:
            hash_ints = (hash_bits * self.powers_of_two).sum(dim=-1)
        else:
            # Fallback for large bins (use first 63 bits)
            hash_ints = (hash_bits[..., :63] * self.powers_of_two).sum(dim=-1)
        
        # Single CPU transfer for hash keys
        hash_ints_cpu = hash_ints.cpu().numpy()
        
        # Pre-calculate rewards on CPU then move back once
        rewards_cpu = np.zeros(features_flat.shape[0], dtype=np.float32)
        
        # Pre-calculate environment indices
        if env_indices is None:
            env_indices_cpu = np.arange(features_flat.shape[0]) % batch
        else:
            # Handle tensor env_indices
            if hasattr(env_indices, 'cpu'):
                env_indices_np = env_indices.cpu().numpy()
            else:
                env_indices_np = np.array(env_indices)
            env_indices_cpu = np.tile(env_indices_np, seq) if has_seq else env_indices_np
        
        # Counting loop (Pure CPU - fast)
        for i in range(features_flat.shape[0]):
            env_idx = int(env_indices_cpu[i % len(env_indices_cpu)])
            key = int(hash_ints_cpu[i])
            
            # Enforce memory limit per environment
            if len(self._visit_counts[env_idx]) >= self._max_hash_entries_per_env:
                keys_to_remove = list(self._visit_counts[env_idx].keys())[:len(self._visit_counts[env_idx]) // 4]
                for k in keys_to_remove:
                    del self._visit_counts[env_idx][k]
            
            # Increment count
            self._visit_counts[env_idx][key] += 1
            count = self._visit_counts[env_idx][key]
            
            # Inverse sqrt reward
            rewards_cpu[i] = 1.0 / np.sqrt(count)
        
        # Move rewards back to GPU (single transfer)
        intrinsic_rewards = torch.from_numpy(rewards_cpu).to(features.device, dtype=features.dtype)
        intrinsic_rewards = intrinsic_rewards.unsqueeze(-1)
        
        if has_seq:
            intrinsic_rewards = intrinsic_rewards.reshape(batch, seq, 1)
        
        return intrinsic_rewards
    
    def add_to_memory(self, features, env_indices=None):
        """
        Add features to episodic memory for kernel-based counting.
        
        Args:
            features: Latent state features [batch, feat_dim]
            env_indices: Environment indices for per-environment memory
        """
        batch = features.shape[0]
        for i in range(batch):
            env_idx = env_indices[i] if env_indices is not None else i
            self._episodic_memory[env_idx].append(features[i].detach().cpu())


class HierarchicalExplorationBonus(nn.Module):
    """
    Hierarchical exploration bonus for higher-level actors.
    
    Rewards higher-level actors for setting diverse subgoals that lead to
    novel states. This encourages the hierarchical structure to explore
    the full state space rather than converging to local optima.
    
    Args:
        subgoal_dim: Dimension of the subgoal space
        config: Configuration object
    """
    
    def __init__(
        self,
        subgoal_dim,
        hidden_dim=128,
        config=None,
    ):
        super(HierarchicalExplorationBonus, self).__init__()
        self._config = config
        self._subgoal_dim = subgoal_dim
        
        # Track subgoal diversity using a simple embedding approach
        self.subgoal_embedder = nn.Sequential(
            nn.Linear(subgoal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.subgoal_embedder.apply(tools.weight_init)
        
        # Get max_history from config with fallback
        im_config = getattr(config, 'intrinsic_motivation', {})
        if not isinstance(im_config, dict):
            im_config = {}
        hierarchical_config = getattr(config, 'hierarchical_exploration', {})
        if not isinstance(hierarchical_config, dict):
            hierarchical_config = {}
        
        # History of subgoal embeddings for diversity computation
        # Reduced default to prevent OOM - stores on CPU
        self._subgoal_history = []
        self._max_history = (
            im_config.get("max_history", 500) 
            or hierarchical_config.get("max_history", 500)
        )
        
        # Running statistics
        self.subgoal_rms = RunningMeanStd(shape=(subgoal_dim,), device=config.device)
    
    def forward(self, subgoals, achieved_states=None):
        """
        Compute hierarchical exploration bonus for subgoals.
        
        Args:
            subgoals: Proposed subgoals [batch, subgoal_dim]
            achieved_states: States achieved by following subgoals (optional)
            
        Returns:
            Exploration bonus [batch, 1]
        """
        batch = subgoals.shape[0]
        
        # Flatten subgoals if needed
        if len(subgoals.shape) > 2:
            subgoals = subgoals.reshape(batch, -1)
        
        # Normalize subgoals
        self.subgoal_rms.update(subgoals)
        normalized = (subgoals - self.subgoal_rms.mean) / torch.sqrt(self.subgoal_rms.var + 1e-8)
        
        # Compute embedding
        embedding = self.subgoal_embedder(normalized)
        
        # Compute diversity bonus based on distance to previous subgoals
        if len(self._subgoal_history) > 0:
            # Move history to same device as embedding for computation
            history_tensor = torch.stack(self._subgoal_history[-self._max_history:]).to(embedding.device)
            
            # Compute pairwise distances
            distances = torch.cdist(embedding.unsqueeze(0), history_tensor.unsqueeze(0))[0]
            
            # Bonus is based on minimum distance to any previous subgoal
            min_distances, _ = distances.min(dim=-1)
            diversity_bonus = min_distances.unsqueeze(-1)
            
            # Clean up
            del history_tensor
        else:
            # First subgoals get maximum bonus
            diversity_bonus = torch.ones(batch, 1, device=subgoals.device)
        
        # Add current subgoals to history (store on CPU to save GPU memory)
        for i in range(batch):
            self._subgoal_history.append(embedding[i].detach().cpu())
            if len(self._subgoal_history) > self._max_history:
                self._subgoal_history.pop(0)
        
        return diversity_bonus
    
    def reset(self):
        """Reset subgoal history (e.g., at the start of training)."""
        self._subgoal_history = []


class IntrinsicMotivationManager(nn.Module):
    """
    Manager class that combines multiple intrinsic motivation modules.
    
    Provides a unified interface for computing intrinsic rewards from
    multiple sources (RND, count-based, hierarchical) and manages their
    relative weights and training.
    
    Args:
        feat_size: Feature dimension from world model
        subgoal_shape: Shape of subgoals for hierarchical bonus
        config: Configuration object with intrinsic motivation settings
    """
    
    def __init__(
        self,
        feat_size,
        subgoal_shape,
        config,
    ):
        super(IntrinsicMotivationManager, self).__init__()
        self._config = config
        self._feat_size = feat_size
        
        # Get intrinsic motivation config with fallback
        im_config = getattr(config, 'intrinsic_motivation', {})
        if not isinstance(im_config, dict):
            im_config = {}
        
        # Initialize modules based on config
        self.use_rnd = im_config.get("use_rnd", True)
        self.use_episodic_count = im_config.get("use_episodic_count", True)
        self.use_hierarchical_bonus = im_config.get("use_hierarchical_bonus", True)
        
        if self.use_rnd:
            self.rnd_module = RNDModule(
                input_dim=feat_size,
                hidden_dim=im_config.get("rnd_hidden_dim", 128),  # Reduced default
                output_dim=im_config.get("rnd_output_dim", 64),   # Reduced default
                num_layers=im_config.get("rnd_num_layers", 2),    # Reduced default
                config=config,
            )
        
        if self.use_episodic_count:
            self.count_module = EpisodicCountModule(
                input_dim=feat_size,
                num_bins=im_config.get("count_num_bins", 32),
                k_neighbors=im_config.get("count_k_neighbors", 10),
                config=config,
            )
        
        if self.use_hierarchical_bonus:
            # Handle subgoal_shape which might be a tuple/list or accessed from config
            if subgoal_shape is not None:
                sg_dim = int(np.prod(subgoal_shape))
            elif hasattr(config, 'subgoal_shape'):
                sg_dim = int(np.prod(config.subgoal_shape))
            else:
                # Fallback default
                sg_dim = 64
            
            self.hierarchical_module = HierarchicalExplorationBonus(
                subgoal_dim=sg_dim,
                hidden_dim=im_config.get("hierarchical_hidden_dim", 64),  # Reduced default
                config=config,
            )
        
        # Reward weights
        self._rnd_weight = im_config.get("rnd_weight", 0.5)
        self._count_weight = im_config.get("count_weight", 0.3)
        self._hierarchical_weight = im_config.get("hierarchical_weight", 0.2)
        
        # Exploration schedule (optional decay)
        self._exploration_schedule = im_config.get("exploration_schedule", "constant")
        self._exploration_decay_steps = im_config.get("exploration_decay_steps", 100000)
        self._step = 0
    
    def compute_intrinsic_reward(self, features, subgoals=None, env_indices=None):
        """
        Compute combined intrinsic reward from all active modules.
        
        Args:
            features: Latent state features from world model
            subgoals: Subgoals from higher-level actor (for hierarchical bonus)
            env_indices: Environment indices for episodic counting
            
        Returns:
            Combined intrinsic reward [batch, seq, 1] or [batch, 1]
        """
        total_reward = torch.zeros_like(features[..., :1])
        
        # Get exploration coefficient based on schedule
        exploration_coef = self._get_exploration_coefficient()
        
        if self.use_rnd:
            rnd_reward = self.rnd_module(features)
            total_reward = total_reward + self._rnd_weight * rnd_reward * exploration_coef
        
        if self.use_episodic_count:
            count_reward = self.count_module(features, env_indices)
            total_reward = total_reward + self._count_weight * count_reward * exploration_coef
        
        if self.use_hierarchical_bonus and subgoals is not None:
            # Flatten subgoals if needed
            flat_subgoals = subgoals.reshape(subgoals.shape[0], -1)
            hierarchical_reward = self.hierarchical_module(flat_subgoals)
            # Expand to match feature dimensions
            if len(features.shape) == 3 and len(hierarchical_reward.shape) == 2:
                hierarchical_reward = hierarchical_reward.unsqueeze(1).expand(-1, features.shape[1], -1)
            total_reward = total_reward + self._hierarchical_weight * hierarchical_reward * exploration_coef
        
        self._step += 1
        return total_reward
    
    def _get_exploration_coefficient(self):
        """Get exploration coefficient based on schedule."""
        if self._exploration_schedule == "constant":
            return 1.0
        elif self._exploration_schedule == "linear_decay":
            return max(0.1, 1.0 - self._step / self._exploration_decay_steps)
        elif self._exploration_schedule == "exponential_decay":
            return max(0.1, np.exp(-self._step / self._exploration_decay_steps))
        else:
            return 1.0
    
    def train_step(self, features):
        """
        Train intrinsic motivation modules.
        
        Args:
            features: Latent state features
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {}
        
        if self.use_rnd:
            rnd_metrics = self.rnd_module.train_step(features)
            metrics.update({f"intrinsic/{k}": v for k, v in rnd_metrics.items()})
        
        # Periodically clean up GPU memory
        if self._step % 1000 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return metrics
    
    def reset_episode(self, env_indices=None):
        """Reset episodic components at episode boundaries."""
        if self.use_episodic_count:
            self.count_module.reset_episode(env_indices)
    
    def get_metrics(self):
        """Get current exploration metrics."""
        return {
            "intrinsic/exploration_coef": self._get_exploration_coefficient(),
            "intrinsic/step": self._step,
        }


class RunningMeanStd:
    """
    Running mean and standard deviation tracker for normalization.
    Uses Welford's algorithm for numerical stability.
    """
    
    def __init__(self, shape, device, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self._device = device
    
    def update(self, x):
        """Update running statistics with new batch of data."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update using batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
