import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch

from .agents import StrongBaselineAgent
from .features import MetricsRecorder
from .vizualize_grid import get_tile_value


def train_strong_baseline(episodes=20000,
                         batch_size=128,
                         update_target_every=1000,
                         save_model_every=1000,
                         evaluation_episodes=100,
                         model_dir="models/strong_baseline",
                         metrics_dir="metrics/strong_baseline"):
    """Train the strong baseline agent based on the paper's approach"""
    env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")
    
    agent = StrongBaselineAgent(
        env=env,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        memory_size=100000,
        batch_size=batch_size,
        lr=0.0003,
        target_update_freq=update_target_every,
        double_dqn=True,
        prioritized_replay=True,
        n_step_returns=3,
        model_dir=model_dir
    )
    
    metrics = MetricsRecorder(save_dir=metrics_dir)
    
    best_score = 0
    
    for episode in tqdm(range(episodes)):
        observation, _ = env.reset()
        board = get_tile_value(observation)
        state = agent._state_to_tensor(board)
        
        done = False
        score = 0
        max_tile = 0
        steps = 0
        
        while not done:
            steps += 1
            action = agent.act(observation)
            
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_board = get_tile_value(next_observation)
            next_state = agent._state_to_tensor(next_board)
            
            # Apply reward shaping
            shaped_reward = agent._shape_reward(reward, board, next_board)
            
            # Store experience
            agent.remember(state, action, shaped_reward, next_state, done)
            
            # Learn from experiences
            agent.replay()
            
            # Update variables for next iteration
            state = next_state
            observation = next_observation
            board = next_board
            score += reward
            max_tile = max(max_tile, next_board.max())
            
            # Increment step counter
            agent.steps_done += 1
        
        # Record metrics
        metrics.record(score, max_tile, steps)
        
        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_net()
        
        # Save model periodically
        if episode % save_model_every == 0 or episode == episodes - 1:
            agent.save(f"{model_dir}/strong_baseline_ep{episode}.pt")
            
            # Save best model based on score
            if score > best_score:
                best_score = score
                agent.save(f"{model_dir}/strong_baseline_best.pt")
        
        # Log progress
        if episode % 100 == 0 and episode > 0:
            last_100 = metrics.scores[-100:]
            avg_score = np.mean(last_100)
            avg_max_tile = np.mean(metrics.max_tiles[-100:])
            
            print(f"Episode: {episode}, Avg Score: {avg_score:.2f}, Avg Max Tile: {avg_max_tile:.2f}, Epsilon: {agent.epsilon:.4f}")
            
            # Update learning rate based on performance
            agent.scheduler.step(avg_score)
    
    # Final metrics
    metrics.save()
    metrics.plot()
    
    print("Training completed!")