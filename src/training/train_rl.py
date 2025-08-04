"""
Reinforcement Learning fine-tuning for chip placement optimization.
Uses PPO/SAC with GNN-based policy and value networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging
from tqdm import tqdm
import wandb

from stable_baselines3.common.buffers import RolloutBuffer
from torch.distributions import Normal
import gym
from gym import spaces

from ..models.gnn_placer import PlacementGNN
from ..models.ppa_estimator import PPAEstimator
from ..data.netlist_parser import NetlistGraph
from ..utils.metrics import calculate_hpwl, estimate_congestion

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL training."""
    # Environment
    grid_size: int = 1000
    max_steps_per_episode: int = 1000
    
    # PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Reward weights
    reward_weights: Dict[str, float] = None
    
    # Training
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    save_freq: int = 50_000
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[int] = None  # [5000, 10000, 50000, 100000, 200000] cells
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'wirelength': 1.0,
                'timing': 2.0,
                'power': 0.5,
                'congestion': 1.5,
                'overlap': 10.0  # Penalty for cell overlaps
            }
        if self.curriculum_stages is None:
            self.curriculum_stages = [5000, 10000, 50000, 100000, 200000]


class PlacementEnv(gym.Env):
    """Gym environment for chip placement."""
    
    def __init__(self, 
                 netlist_graphs: List[NetlistGraph],
                 ppa_estimator: PPAEstimator,
                 config: RLConfig):
        super().__init__()
        
        self.graphs = netlist_graphs
        self.ppa_estimator = ppa_estimator
        self.config = config
        
        # Action space: continuous placement adjustments or discrete swaps
        self.action_mode = 'continuous'  # or 'discrete'
        
        if self.action_mode == 'continuous':
            # Action: [cell_idx, dx, dy] - which cell to move and by how much
            self.action_space = spaces.Box(
                low=np.array([0, -10, -10]),
                high=np.array([1, 10, 10]),
                dtype=np.float32
            )
        else:
            # Action: swap two cells
            self.action_space = spaces.Discrete(1000)  # Top-K candidate swaps
            
        # Observation space: current placement + graph features
        self.observation_space = spaces.Dict({
            'graph_features': spaces.Box(low=-np.inf, high=np.inf, shape=(256,)),
            'placement': spaces.Box(low=0, high=config.grid_size, shape=(10000, 2)),
            'ppa_metrics': spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        })
        
        self.current_graph_idx = 0
        self.current_graph = None
        self.current_placement = None
        self.step_count = 0
        self.best_placement = None
        self.best_reward = -float('inf')
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment with new design."""
        # Select next graph (or random for training)
        self.current_graph_idx = np.random.randint(len(self.graphs))
        self.current_graph = self.graphs[self.current_graph_idx]
        
        # Initialize with a reasonable placement (e.g., from IL model)
        self.current_placement = self._get_initial_placement()
        self.step_count = 0
        self.best_placement = self.current_placement.copy()
        self.best_reward = -float('inf')
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute placement action."""
        self.step_count += 1
        
        # Apply action
        if self.action_mode == 'continuous':
            self._apply_continuous_action(action)
        else:
            self._apply_discrete_action(action)
            
        # Calculate reward
        reward, info = self._calculate_reward()
        
        # Update best placement
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_placement = self.current_placement.copy()
            
        # Check termination
        done = self.step_count >= self.config.max_steps_per_episode
        
        # Add step info
        info['step'] = self.step_count
        info['best_reward'] = self.best_reward
        
        return self._get_observation(), reward, done, info
    
    def _get_initial_placement(self) -> np.ndarray:
        """Get initial placement from IL model or random."""
        num_cells = len(self.current_graph.cells)
        # Simple grid placement as baseline
        grid_size = int(np.sqrt(num_cells)) + 1
        placement = np.zeros((num_cells, 2))
        
        for i in range(num_cells):
            row = i // grid_size
            col = i % grid_size
            placement[i] = [
                col * self.config.grid_size / grid_size,
                row * self.config.grid_size / grid_size
            ]
            
        return placement
    
    def _apply_continuous_action(self, action: np.ndarray):
        """Apply continuous placement adjustment."""
        # Decode action
        cell_idx = int(action[0] * len(self.current_graph.cells))
        dx, dy = action[1:3]
        
        # Apply movement with bounds checking
        self.current_placement[cell_idx, 0] = np.clip(
            self.current_placement[cell_idx, 0] + dx,
            0, self.config.grid_size
        )
        self.current_placement[cell_idx, 1] = np.clip(
            self.current_placement[cell_idx, 1] + dy,
            0, self.config.grid_size
        )
        
    def _apply_discrete_action(self, action: int):
        """Apply discrete cell swap."""
        # Get swap candidates based on current metrics
        candidates = self._get_swap_candidates()
        if action < len(candidates):
            cell1, cell2 = candidates[action]
            # Swap positions
            self.current_placement[[cell1, cell2]] = self.current_placement[[cell2, cell1]]
    
    def _calculate_reward(self) -> Tuple[float, Dict[str, float]]:
        """Calculate multi-objective reward."""
        # Get PPA metrics from estimator
        with torch.no_grad():
            placement_tensor = torch.FloatTensor(self.current_placement)
            graph_data = self._graph_to_hetero_data()
            
            ppa_metrics = self.ppa_estimator(graph_data, placement_tensor)
            
        # Extract metrics
        hpwl = ppa_metrics['hpwl'].item()
        wns = ppa_metrics['timing'][0].item()
        power = ppa_metrics['power'].item()
        congestion = ppa_metrics['congestion'].item()
        
        # Calculate overlap penalty
        overlap = self._calculate_overlap_penalty()
        
        # Weighted reward (negative because we minimize)
        reward = -(
            self.config.reward_weights['wirelength'] * hpwl +
            self.config.reward_weights['timing'] * max(0, -wns) +
            self.config.reward_weights['power'] * power +
            self.config.reward_weights['congestion'] * congestion +
            self.config.reward_weights['overlap'] * overlap
        )
        
        info = {
            'hpwl': hpwl,
            'wns': wns,
            'power': power,
            'congestion': congestion,
            'overlap': overlap
        }
        
        return reward, info
    
    def _calculate_overlap_penalty(self) -> float:
        """Calculate penalty for overlapping cells."""
        overlap = 0.0
        cell_size = 10  # Approximate cell size
        
        for i in range(len(self.current_placement)):
            for j in range(i + 1, len(self.current_placement)):
                dist = np.linalg.norm(self.current_placement[i] - self.current_placement[j])
                if dist < cell_size:
                    overlap += (cell_size - dist) / cell_size
                    
        return overlap
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current environment observation."""
        # Simplified observation - in practice, would use GNN embeddings
        graph_features = np.random.randn(256).astype(np.float32)  # Placeholder
        
        # Get current PPA metrics
        _, info = self._calculate_reward()
        ppa_metrics = np.array([
            info['hpwl'], info['wns'], info['power'], 
            info['congestion'], info['overlap']
        ], dtype=np.float32)
        
        return {
            'graph_features': graph_features,
            'placement': self.current_placement.astype(np.float32),
            'ppa_metrics': ppa_metrics
        }
    
    def _get_swap_candidates(self, k: int = 1000) -> List[Tuple[int, int]]:
        """Get top-k swap candidates based on heuristics."""
        # Simple heuristic: cells with high congestion
        candidates = []
        num_cells = len(self.current_placement)
        
        for _ in range(k):
            i = np.random.randint(num_cells)
            j = np.random.randint(num_cells)
            if i != j:
                candidates.append((i, j))
                
        return candidates
    
    def _graph_to_hetero_data(self):
        """Convert current state to HeteroData for GNN."""
        # Placeholder - would convert graph + placement to HeteroData
        return self.current_graph


class PPOPolicy(nn.Module):
    """Policy network for PPO algorithm."""
    
    def __init__(self, 
                 gnn_model: PlacementGNN,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.gnn = gnn_model
        self.action_dim = action_dim
        
        # Policy head
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),  # +5 for PPA metrics
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # Mean and log_std
        )
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Fixed log std (can be learned)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        # Get GNN embeddings
        graph_features = obs['graph_features']
        ppa_metrics = obs['ppa_metrics']
        
        # Concatenate features
        features = torch.cat([graph_features, ppa_metrics], dim=-1)
        
        # Get policy
        policy_out = self.policy_net(features)
        mean = policy_out[..., :self.action_dim]
        log_std = self.log_std.expand_as(mean)
        
        # Get value
        value = self.value_net(features)
        
        return mean, log_std, value
    
    def get_action(self, obs: Dict[str, torch.Tensor], 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std, value = self.forward(obs)
        
        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample()
            
        # Calculate log probability
        std = log_std.exp()
        var = std.pow(2)
        log_prob = -0.5 * (((action - mean).pow(2) / var) + log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob, value


class PPOTrainer:
    """PPO trainer for placement optimization."""
    
    def __init__(self,
                 env: PlacementEnv,
                 policy: PPOPolicy,
                 config: RLConfig,
                 checkpoint_dir: str = "checkpoints/"):
        
        self.env = env
        self.policy = policy
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Optimizers
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Buffers
        self.rollout_buffer = []
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Logging
        self.total_timesteps = 0
        self.num_episodes = 0
        
    def collect_rollouts(self, n_steps: int) -> bool:
        """Collect experience for n_steps."""
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(n_steps):
            # Convert observation to tensor
            obs_tensor = {
                k: torch.FloatTensor(v).unsqueeze(0)
                for k, v in obs.items()
            }
            
            # Get action
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs_tensor)
                
            action_np = action.squeeze(0).numpy()
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action_np)
            
            # Store transition
            self.rollout_buffer.append({
                'obs': obs,
                'action': action_np,
                'reward': reward,
                'done': done,
                'value': value.item(),
                'log_prob': log_prob.item()
            })
            
            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.num_episodes += 1
                
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Log episode stats
                if self.num_episodes % 10 == 0:
                    self._log_episode_stats()
            else:
                obs = next_obs
                
        return True
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting PPO training...")
        
        for iteration in range(self.config.total_timesteps // self.config.n_steps):
            # Collect rollouts
            self.collect_rollouts(self.config.n_steps)
            
            # Compute advantages
            advantages, returns = self._compute_gae()
            
            # PPO update
            self._ppo_update(advantages, returns)
            
            # Evaluation
            if iteration % (self.config.eval_freq // self.config.n_steps) == 0:
                self._evaluate()
                
            # Save checkpoint
            if iteration % (self.config.save_freq // self.config.n_steps) == 0:
                self._save_checkpoint(iteration)
                
            # Curriculum learning
            if self.config.use_curriculum:
                self._update_curriculum(iteration)
                
    def _compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        rewards = torch.FloatTensor([t['reward'] for t in self.rollout_buffer])
        values = torch.FloatTensor([t['value'] for t in self.rollout_buffer])
        dones = torch.FloatTensor([t['done'] for t in self.rollout_buffer])
        
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _ppo_update(self, advantages: torch.Tensor, returns: torch.Tensor) -> None:
        """Update policy using PPO."""
        # Convert buffer to tensors
        obs_batch = {
            k: torch.FloatTensor(np.array([t['obs'][k] for t in self.rollout_buffer]))
            for k in self.rollout_buffer[0]['obs'].keys()
        }
        actions = torch.FloatTensor([t['action'] for t in self.rollout_buffer])
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.rollout_buffer])
        
        # Multiple epochs
        for epoch in range(self.config.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(self.rollout_buffer))
            
            for start in range(0, len(indices), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                obs_mini = {
                    k: v[batch_indices] for k, v in obs_batch.items()
                }
                actions_mini = actions[batch_indices]
                old_log_probs_mini = old_log_probs[batch_indices]
                advantages_mini = advantages[batch_indices]
                returns_mini = returns[batch_indices]
                
                # Forward pass
                mean, log_std, values = self.policy(obs_mini)
                
                # Calculate log probs
                std = log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions_mini).sum(dim=-1)
                
                # Ratio
                ratio = (new_log_probs - old_log_probs_mini).exp()
                
                # Clipped surrogate loss
                surr1 = ratio * advantages_mini
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages_mini
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), returns_mini)
                
                # Entropy loss
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
        # Clear buffer
        self.rollout_buffer = []
        
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        eval_rewards = []
        eval_metrics = {
            'hpwl': [],
            'wns': [],
            'power': [],
            'congestion': []
        }
        
        for _ in range(10):  # 10 evaluation episodes
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                obs_tensor = {
                    k: torch.FloatTensor(v).unsqueeze(0)
                    for k, v in obs.items()
                }
                
                with torch.no_grad():
                    action, _, _ = self.policy.get_action(obs_tensor, deterministic=True)
                    
                action_np = action.squeeze(0).numpy()
                obs, reward, done, info = self.env.step(action_np)
                episode_reward += reward
                
                # Collect metrics
                for metric in eval_metrics:
                    if metric in info:
                        eval_metrics[metric].append(info[metric])
                        
            eval_rewards.append(episode_reward)
            
        # Log evaluation results
        logger.info(f"Evaluation - Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        for metric, values in eval_metrics.items():
            if values:
                logger.info(f"  {metric}: {np.mean(values):.4f}")
                
        return {
            'eval_reward': np.mean(eval_rewards),
            **{f'eval_{k}': np.mean(v) for k, v in eval_metrics.items() if v}
        }
        
    def _save_checkpoint(self, iteration: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'config': self.config
        }
        
        path = f"{self.checkpoint_dir}/ppo_checkpoint_{iteration}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def _log_episode_stats(self) -> None:
        """Log episode statistics."""
        if self.episode_rewards:
            logger.info(
                f"Episodes: {self.num_episodes}, "
                f"Timesteps: {self.total_timesteps}, "
                f"Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}, "
                f"Length: {np.mean(self.episode_lengths):.0f}"
            )
            
    def _update_curriculum(self, iteration: int) -> None:
        """Update curriculum learning stage."""
        # Determine current stage based on iteration
        stage_idx = min(
            iteration // (self.config.total_timesteps // len(self.config.curriculum_stages)),
            len(self.config.curriculum_stages) - 1
        )
        
        target_cells = self.config.curriculum_stages[stage_idx]
        
        # Update environment with new complexity
        # This would filter graphs to appropriate size
        logger.info(f"Curriculum stage: {target_cells} cells")


def main():
    """Main training script."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--surrogate', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config = RLConfig(**config_dict['rl'])
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project="eda-sim-ai",
            config=config_dict,
            name=f"rl_training_{wandb.util.generate_id()}"
        )
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained GNN
    gnn_model = PlacementGNN(
        num_cell_types=config_dict['model']['num_cell_types'],
        cell_feat_dim=config_dict['model']['cell_feat_dim'],
        io_feat_dim=config_dict['model']['io_feat_dim'],
        hidden_dim=config_dict['model']['hidden_dim'],
        num_layers=config_dict['model']['num_layers']
    ).to(device)
    
    checkpoint = torch.load(args.pretrained, map_location=device)
    gnn_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load PPA estimator
    ppa_estimator = PPAEstimator().to(device)
    ppa_checkpoint = torch.load(args.surrogate, map_location=device)
    ppa_estimator.load_state_dict(ppa_checkpoint['model_state_dict'])
    
    # Load training data
    # This would load actual netlist graphs
    from ..data.dataset import load_netlist_graphs
    graphs = load_netlist_graphs(config_dict['data']['train_dir'])
    
    # Create environment
    env = PlacementEnv(graphs, ppa_estimator, config)
    
    # Create policy
    policy = PPOPolicy(gnn_model, env.action_space.shape[0]).to(device)
    
    # Create trainer
    trainer = PPOTrainer(env, policy, config, args.output)
    
    # Train
    trainer.train()
    
    # Save final model
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'config': config
    }, f"{args.output}/final_model.pt")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()