"""
Multi-Armed Bandit for A/B Testing

Uses Thompson Sampling to automatically select the best ranking strategy
by balancing exploration vs. exploitation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from scipy.stats import beta
import logging

logger = logging.getLogger(__name__)


class RankingStrategyBandit:
    """
    Multi-Armed Bandit for ranking strategy selection
    
    Uses Thompson Sampling to automatically find the best ranking strategy
    """
    
    def __init__(self, strategies: List[str], alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize Multi-Armed Bandit
        
        Args:
            strategies: List of ranking strategy names
            alpha_prior: Prior alpha parameter (successes)
            beta_prior: Prior beta parameter (failures)
        """
        self.strategies = strategies
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Track success/failure for each strategy
        self.alpha = {s: alpha_prior for s in strategies}  # Successes
        self.beta = {s: beta_prior for s in strategies}  # Failures
        
        # Statistics
        self.total_pulls = {s: 0 for s in strategies}
        self.total_rewards = {s: 0.0 for s in strategies}
        self.avg_rewards = {s: 0.0 for s in strategies}
    
    def select_strategy(self, deterministic: bool = False) -> str:
        """
        Select strategy using Thompson Sampling
        
        Args:
            deterministic: If True, select strategy with highest success rate
            
        Returns:
            Selected strategy name
        """
        if deterministic:
            # Select strategy with highest success rate
            success_rates = {
                s: self.alpha[s] / (self.alpha[s] + self.beta[s])
                for s in self.strategies
            }
            return max(success_rates, key=success_rates.get)
        
        # Thompson Sampling: sample from beta distribution
        samples = {}
        for strategy in self.strategies:
            # Sample from beta distribution
            sample = np.random.beta(self.alpha[strategy], self.beta[strategy])
            samples[strategy] = sample
        
        # Select strategy with highest sample
        selected = max(samples, key=samples.get)
        return selected
    
    def update(self, strategy: str, success: bool, reward: Optional[float] = None):
        """
        Update strategy statistics based on outcome
        
        Args:
            strategy: Strategy name that was used
            success: Whether the strategy was successful
            reward: Optional reward value (0-1)
        """
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy}")
            return
        
        # Update alpha/beta
        if success:
            self.alpha[strategy] += 1
        else:
            self.beta[strategy] += 1
        
        # Update statistics
        self.total_pulls[strategy] += 1
        
        if reward is not None:
            self.total_rewards[strategy] += reward
            self.avg_rewards[strategy] = self.total_rewards[strategy] / self.total_pulls[strategy]
        else:
            # Use success/failure as reward
            reward_value = 1.0 if success else 0.0
            self.total_rewards[strategy] += reward_value
            self.avg_rewards[strategy] = self.total_rewards[strategy] / self.total_pulls[strategy]
    
    def get_success_rate(self, strategy: str) -> float:
        """Get success rate for a strategy"""
        if strategy not in self.strategies:
            return 0.0
        
        total = self.alpha[strategy] + self.beta[strategy]
        if total == 0:
            return 0.5  # Default prior
        
        return self.alpha[strategy] / total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all strategies"""
        stats = {}
        for strategy in self.strategies:
            stats[strategy] = {
                'success_rate': self.get_success_rate(strategy),
                'total_pulls': self.total_pulls[strategy],
                'avg_reward': self.avg_rewards[strategy],
                'alpha': self.alpha[strategy],
                'beta': self.beta[strategy]
            }
        return stats
    
    def get_best_strategy(self) -> str:
        """Get strategy with highest success rate"""
        success_rates = {
            s: self.get_success_rate(s)
            for s in self.strategies
        }
        return max(success_rates, key=success_rates.get)
    
    def reset(self):
        """Reset all statistics"""
        self.alpha = {s: self.alpha_prior for s in self.strategies}
        self.beta = {s: self.beta_prior for s in self.strategies}
        self.total_pulls = {s: 0 for s in self.strategies}
        self.total_rewards = {s: 0.0 for s in self.strategies}
        self.avg_rewards = {s: 0.0 for s in self.strategies}


# Global instance
_bandit = None


def get_bandit(strategies: Optional[List[str]] = None) -> RankingStrategyBandit:
    """Get or create global bandit instance"""
    global _bandit
    if _bandit is None:
        if strategies is None:
            strategies = ['default', 'ltr', 'rl', 'hybrid']
        _bandit = RankingStrategyBandit(strategies)
    return _bandit

