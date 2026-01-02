"""
Reinforcement Learning Ranking Agent

Uses Deep Reinforcement Learning to learn optimal ranking policies
from user interactions (clicks, hires, ratings).
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")

# Try to import stable-baselines3 for advanced RL
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logger.warning("stable-baselines3 not available. Using custom RL implementation.")


class RankingAgent(nn.Module):
    """
    Deep Reinforcement Learning agent for candidate ranking
    
    Uses Actor-Critic architecture:
    - Policy network: Selects which candidate to rank first
    - Value network: Estimates expected reward
    """
    
    def __init__(self, state_dim: int = 50, action_dim: int = 100, hidden_dim: int = 256):
        """
        Initialize Ranking Agent
        
        Args:
            state_dim: Dimension of state vector (query + candidate features)
            action_dim: Maximum number of candidates to rank
            hidden_dim: Hidden layer dimension
        """
        super(RankingAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Policy network (Actor) - outputs probability distribution over actions
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network (Critic) - estimates expected reward
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        
        # Training statistics
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': []
        }
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both networks
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (action_probs, value_estimate)
        """
        action_probs = self.policy_net(state)
        value = self.value_net(state)
        return action_probs, value
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Select action (ranking) based on current state
        
        Args:
            state: State tensor [state_dim]
            deterministic: If True, select highest probability action
            
        Returns:
            Tuple of (action_idx, log_prob)
        """
        state = state.unsqueeze(0) if len(state.shape) == 1 else state
        probs, _ = self.forward(state)
        probs = probs.squeeze(0) if probs.shape[0] == 1 else probs
        
        dist = Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate for state"""
        state = state.unsqueeze(0) if len(state.shape) == 1 else state
        _, value = self.forward(state)
        return value.squeeze()
    
    def update(self, states: List[torch.Tensor], actions: List[int], 
               rewards: List[float], next_states: Optional[List[torch.Tensor]] = None,
               gamma: float = 0.99):
        """
        Update policy using policy gradient
        
        Args:
            states: List of state tensors
            actions: List of actions taken
            rewards: List of rewards received
            next_states: List of next states (for value learning)
            gamma: Discount factor
        """
        if not states or not actions or not rewards:
            return
        
        # Convert to tensors
        states_tensor = torch.stack(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Compute returns (discounted rewards)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Get current policy and value estimates
        action_probs, values = self.forward(states_tensor)
        values = values.squeeze()
        
        # Compute advantages
        advantages = returns_tensor - values.detach()
        
        # Policy loss (actor)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss (critic)
        value_loss = nn.MSELoss()(values, returns_tensor)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)  # Gradient clipping
        self.optimizer.step()
        
        # Store training history
        self.training_history['episodes'].append(len(self.training_history['episodes']) + 1)
        self.training_history['rewards'].append(sum(rewards))
        self.training_history['losses'].append(total_loss.item())
        
        return total_loss.item()


class RLRankingAgent:
    """
    High-level wrapper for RL Ranking Agent
    
    Handles:
    - State extraction from query-candidate pairs
    - Reward computation from user feedback
    - Training loop
    - Model persistence
    """
    
    def __init__(self, model_path: Optional[str] = None, state_dim: int = 50):
        """
        Initialize RL Ranking Agent
        
        Args:
            model_path: Path to saved model (optional)
            state_dim: Dimension of state vector
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        self.model_path = model_path or os.path.join("model", "rl_ranking_agent.pkl")
        self.state_dim = state_dim
        self.is_trained = False
        
        # Initialize agent
        self.agent = RankingAgent(state_dim=state_dim, action_dim=100)
        
        # Training buffer
        self.training_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': []
        }
        
        # Load saved model if exists
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_state(self, query: str, candidate: Dict[str, Any], 
                     feature_scores: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Extract state vector from query-candidate pair
        
        Args:
            query: Search query
            candidate: Candidate data dictionary
            feature_scores: Optional pre-computed feature scores
            
        Returns:
            State tensor [state_dim]
        """
        # Extract features
        features = []
        
        # Feature scores (if provided)
        if feature_scores:
            features.extend([
                feature_scores.get('keyword_score', 0.0),
                feature_scores.get('semantic_score', 0.0),
                feature_scores.get('cross_encoder_score', 0.0),
                feature_scores.get('skill_overlap', 0.0),
                feature_scores.get('experience_match', 0.0),
                feature_scores.get('domain_match', 0.0)
            ])
        else:
            features.extend([0.0] * 6)
        
        # Candidate features
        skills = candidate.get('skills', [])
        experience_years = candidate.get('total_experience_years', 0)
        resume_text = candidate.get('resume_text', '')
        
        features.extend([
            len(skills) / 20.0,  # Normalized skill count
            min(experience_years / 20.0, 1.0),  # Normalized experience
            len(query.split()) / 10.0,  # Normalized query length
            len(resume_text.split()) / 1000.0  # Normalized resume length
        ])
        
        # Query features (simple encoding)
        query_lower = query.lower()
        query_features = [
            1.0 if 'senior' in query_lower else 0.0,
            1.0 if 'junior' in query_lower else 0.0,
            1.0 if 'python' in query_lower else 0.0,
            1.0 if 'java' in query_lower else 0.0,
            1.0 if 'javascript' in query_lower else 0.0,
            1.0 if 'aws' in query_lower else 0.0,
            1.0 if 'developer' in query_lower else 0.0,
            1.0 if 'engineer' in query_lower else 0.0,
            1.0 if 'manager' in query_lower else 0.0,
            1.0 if 'nurse' in query_lower else 0.0,
            1.0 if 'doctor' in query_lower else 0.0,
        ]
        features.extend(query_features)
        
        # Pad or truncate to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        features = features[:self.state_dim]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def compute_reward(self, user_feedback: Dict[str, Any]) -> float:
        """
        Compute reward from user feedback
        
        Args:
            user_feedback: Dictionary containing feedback:
                - 'clicked': bool (user clicked on candidate)
                - 'hired': bool (candidate was hired)
                - 'interviewed': bool (candidate was interviewed)
                - 'rating': float (user rating 0-5)
                - 'time_spent': float (time spent viewing in seconds)
                
        Returns:
            Reward value (higher is better)
        """
        reward = 0.0
        
        # High reward for hires
        if user_feedback.get('hired', False):
            reward += 10.0
        
        # Medium reward for interviews
        if user_feedback.get('interviewed', False):
            reward += 5.0
        
        # Lower reward for clicks
        if user_feedback.get('clicked', False):
            reward += 1.0
        
        # Reward based on rating
        rating = user_feedback.get('rating', 0.0)
        if rating > 0:
            reward += rating * 0.5
        
        # Small reward for time spent (engagement)
        time_spent = user_feedback.get('time_spent', 0.0)
        if time_spent > 30:  # More than 30 seconds
            reward += 0.5
        
        # Penalty for negative feedback
        if user_feedback.get('rejected', False):
            reward -= 2.0
        
        return reward
    
    def rank_candidates(self, query: str, candidates: List[Dict[str, Any]],
                       feature_scores: Optional[List[Dict[str, float]]] = None,
                       deterministic: bool = False) -> List[Tuple[int, float]]:
        """
        Rank candidates using RL agent
        
        Args:
            query: Search query
            candidates: List of candidate dictionaries
            feature_scores: Optional pre-computed feature scores
            deterministic: If True, use deterministic ranking
            
        Returns:
            List of (candidate_idx, score) tuples, sorted by score
        """
        if not self.is_trained:
            # Return original order if not trained
            return [(i, 0.5) for i in range(len(candidates))]
        
        # Extract states for all candidates
        states = []
        for i, candidate in enumerate(candidates):
            scores = feature_scores[i] if feature_scores and i < len(feature_scores) else None
            state = self.extract_state(query, candidate, scores)
            states.append(state)
        
        # Get rankings from agent
        rankings = []
        remaining_indices = list(range(len(candidates)))
        used_states = []
        
        for _ in range(min(len(candidates), self.agent.action_dim)):
            if not remaining_indices:
                break
            
            # Create state-action mapping
            state_tensor = torch.stack([states[i] for i in remaining_indices])
            
            # Get action probabilities
            probs, _ = self.agent.forward(state_tensor)
            probs = probs.mean(dim=0)  # Average across candidates
            
            # Select action
            if deterministic:
                action_idx = torch.argmax(probs).item()
            else:
                dist = Categorical(probs)
                action_idx = dist.sample().item()
            
            # Map action to candidate index
            if action_idx < len(remaining_indices):
                candidate_idx = remaining_indices[action_idx]
                score = probs[action_idx].item()
                rankings.append((candidate_idx, score))
                remaining_indices.remove(candidate_idx)
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def train_step(self, query: str, candidates: List[Dict[str, Any]],
                   user_feedback: List[Dict[str, Any]],
                   feature_scores: Optional[List[Dict[str, float]]] = None):
        """
        Perform one training step
        
        Args:
            query: Search query
            candidates: List of candidates
            user_feedback: List of feedback dictionaries for each candidate
            feature_scores: Optional pre-computed feature scores
        """
        if len(candidates) != len(user_feedback):
            logger.warning("Mismatch between candidates and feedback")
            return
        
        # Extract states
        states = []
        actions = []
        rewards = []
        
        for i, candidate in enumerate(candidates):
            scores = feature_scores[i] if feature_scores and i < len(feature_scores) else None
            state = self.extract_state(query, candidate, scores)
            states.append(state)
            
            # Compute reward
            reward = self.compute_reward(user_feedback[i])
            rewards.append(reward)
            
            # Action is the ranking position (simplified)
            actions.append(i)
        
        # Update agent
        loss = self.agent.update(states, actions, rewards)
        
        logger.info(f"RL training step completed. Loss: {loss:.4f}, Avg reward: {np.mean(rewards):.4f}")
    
    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            torch.save({
                'agent_state_dict': self.agent.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'state_dim': self.state_dim,
                'is_trained': self.is_trained,
                'training_history': self.agent.training_history
            }, path)
            logger.info(f"Saved RL agent to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, path: Optional[str] = None):
        """Load trained model"""
        path = path or self.model_path
        
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.state_dim = checkpoint.get('state_dim', self.state_dim)
            self.is_trained = checkpoint.get('is_trained', False)
            self.agent.training_history = checkpoint.get('training_history', {
                'episodes': [], 'rewards': [], 'losses': []
            })
            logger.info(f"Loaded RL agent from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


# Global instance
_rl_ranking_agent = None


def get_rl_ranking_agent(model_path: Optional[str] = None) -> RLRankingAgent:
    """Get or create global RL ranking agent instance"""
    global _rl_ranking_agent
    if _rl_ranking_agent is None:
        _rl_ranking_agent = RLRankingAgent(model_path=model_path)
    return _rl_ranking_agent

