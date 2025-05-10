import gymnasium as gym
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from random import random
from collections import deque
from IPython import display
import matplotlib.pyplot as plt
from typing import Dict, Any

from .vizualize_grid import display_game, get_tile_value
from .features import extract_features, feature_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseAgent:
    """Abstract base class; every agent implements ``act(observation)``."""
    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, observation):  # noqa: D401  (simple interface)
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Uniform‑random moves (baseline)."""
    def act(self, observation):
        return self.env.action_space.sample()
    
    
class EnhancedDQN(nn.Module):
    """Enhanced neural network for DQN based on the paper's architecture"""
    def __init__(self, input_size, output_size):
        super(EnhancedDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for more efficient learning"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return [], [], [], [], [], [], []
        
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -beta
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class StrongBaselineAgent(BaseAgent):
    """Enhanced DQN agent implementing multiple DQN improvements"""
    def __init__(self, env,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.9995,
                memory_size=100000,
                batch_size=128,
                lr=0.0003,
                target_update_freq=1000,
                double_dqn=True,
                prioritized_replay=True,
                n_step_returns=3,  # N-step returns from the paper
                model_dir="models/strong_baseline"):
        super().__init__(env)
        
        self.state_size = len(feature_names)
        self.action_size = env.action_space.n
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.n_step_returns = n_step_returns
        
        # Step counter
        self.steps_done = 0
        
        # Memory
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
        
        # N-step return buffer
        self.n_step_buffer = deque(maxlen=n_step_returns)
        
        # Neural networks
        self.policy_net = EnhancedDQN(self.state_size, self.action_size).to(device)
        self.target_net = EnhancedDQN(self.state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber loss
        
        # Model directory
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def act(self, observation):
        board = get_tile_value(observation)
        state = self._state_to_tensor(board)
        
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def _compute_n_step_returns(self, reward, done):
        """Compute n-step returns when a new experience is added"""
        self.n_step_buffer.append((reward, done))
        
        if len(self.n_step_buffer) < self.n_step_returns:
            return None
        
        # Calculate n-step discounted return
        n_reward = 0
        for i in range(self.n_step_returns):
            n_reward += self.gamma**i * self.n_step_buffer[i][0]
        
        # Check if any experience in the buffer is terminal
        n_done = any(done for _, done in self.n_step_buffer)
        
        return n_reward, n_done
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience with n-step returns"""
        # Process n-step returns
        n_step_result = self._compute_n_step_returns(reward, done)
        
        if n_step_result is not None:
            n_reward, n_done = n_step_result
            
            # Get the state-action pair from n steps ago
            old_state, old_action = self.n_step_buffer[0][2], self.n_step_buffer[0][3]
            
            # Store in memory
            if self.prioritized_replay:
                self.memory.push(old_state, old_action, n_reward, next_state, n_done)
            else:
                self.memory.append((old_state, old_action, n_reward, next_state, n_done))
        
        # Add current experience to n-step buffer
        self.n_step_buffer.append((reward, done, state, action))
    
    def replay(self):
        """Learn from experiences using prioritized replay if enabled"""
        if self.prioritized_replay:
            if len(self.memory) < self.batch_size:
                return
            
            # Anneal beta parameter for importance sampling
            beta = min(1.0, 0.4 + 0.6 * (self.steps_done / 100000))
            
            # Sample batch with priorities
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, beta)
            
            states = torch.cat(states).to(device)
            next_states = torch.cat(next_states).to(device)
            actions = torch.tensor(actions, dtype=torch.long, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device)
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
            
            # Current Q values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            # Next Q values with Double DQN
            if self.double_dqn:
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q = self.target_net(next_states).max(1)[0]
            
            # Target Q values
            target_q = rewards + (1 - dones) * self.gamma**self.n_step_returns * next_q
            
            # Compute TD errors for priority updates
            td_errors = torch.abs(current_q - target_q).detach()
            
            # Compute loss with importance sampling weights
            loss = self.loss_fn(current_q, target_q) * weights
            loss = loss.mean()
            
            # Update priorities
            self.memory.update_priorities(indices, td_errors.cpu().numpy() + 1e-6)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
        else:
            # Standard experience replay
            if len(self.memory) < self.batch_size:
                return
            
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.cat(states).to(device)
            next_states = torch.cat(next_states).to(device)
            actions = torch.tensor(actions, dtype=torch.long, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device)
            
            # Current Q values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            # Next Q values with Double DQN
            if self.double_dqn:
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q = self.target_net(next_states).max(1)[0]
            
            # Target Q values
            target_q = rewards + (1 - dones) * self.gamma**self.n_step_returns * next_q
            
            # Compute loss
            loss = self.loss_fn(current_q, target_q).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
    
    def update_target_net(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _shape_reward(self, reward, board, next_board):
        """Apply reward shaping based on the paper's recommendations"""
        shaped_reward = reward
        
        # Reward for maintaining empty cells
        empty_diff = np.count_nonzero(next_board == 0) - np.count_nonzero(board == 0)
        shaped_reward += 0.1 * empty_diff
        
        # Reward for monotonicity (tiles arranged in ascending/descending order)
        if next_board.max() == board.max():
            features = extract_features(next_board)
            shaped_reward += 0.05 * features["monotonicity"]
        
        # Reward for keeping max tile in corner
        if next_board.max() > 64:  # Only care about corner placement for high tiles
            features = extract_features(next_board)
            shaped_reward += 0.2 * features["corner_max"]
        
        # Reward for smoothness (adjacent tiles have similar values)
        features = extract_features(next_board)
        shaped_reward += 0.05 * (features["smoothness"] / -16)  # Normalize and invert (higher is better)
        
        return shaped_reward
    
    def _state_to_tensor(self, board):
        """Convert state to tensor with feature extraction"""
        features = extract_features(board)
        state = np.array(list(features.values()), dtype=np.float32)
        return torch.FloatTensor(state).unsqueeze(0).to(device)
    
    def save(self, filename):
        """Save model"""
        torch.save({
            'steps_done': self.steps_done,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """Load model"""
        checkpoint = torch.load(filename)
        
        self.steps_done = checkpoint.get('steps_done', 0)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        

def run_episode(
    env,
    agent,
    *,
    max_steps: int | None = None,
    delay: float = 0.25,
    visualize: bool = True
) -> int:
    """
    Запускает одну партию 2048 и корректно отображает накопленный счёт.

    Если max_steps=None (по умолчанию), игра идёт до завершения среды.
    Если указан max_steps, партия прерывается не позже этого количества ходов.
    """
    obs, _ = env.reset()
    cum_score = 0
    step = 0
    done = False         
    
    # первая отрисовка
    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        display_game(obs, ax=ax)

    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        prev_score = cum_score
        cum_score += int(reward)      # <‑‑ избавляемся от numpy‑типов

        if visualize:
            display_game(
                obs,
                action=action,
                step_num=step,
                prev_score=prev_score,
                curr_score=cum_score,
                ax=ax,
            )
            time.sleep(delay)

        step += 1
        # проверяем условия выхода
        if done:
            break
        if max_steps is not None and step >= max_steps:
            break

    # -- ФИНАЛ --
    if visualize:
        # подчёркиваем, что это финальное состояние (повторно выводить поле не обязательно)
        print("\n===== Финал =====")
        if done:
            print("Игра завершена (ходов сыграно:", step, ")")
        else:
            print(f"Прервано по лимиту {max_steps} ходов (сыграно: {step})")
        print(f"Итоговый счёт: {cum_score}")

    return cum_score

