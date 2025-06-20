import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, List, Any, Optional, Callable
import math
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
from sklearn.preprocessing import StandardScaler
import random

from .vizualize_grid import get_tile_value
from .features import extract_features


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
    
    
class NNAgent(BaseAgent):
    def __init__(self, env: gym.Env, lr: float = 1e-3, device: str = None):
        super().__init__(env)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(16, 128, kernel_size=2, padding=1)
                self.conv2 = nn.Conv2d(128, 128, kernel_size=2)
                # Adjust input dimension to 128*4*4 = 2048
                self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Fix here
                self.fc2 = nn.Linear(256, 4)
        
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)  # Flatten to [batch_size, 2048]
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        self.model = Net().to(self.device)
        self.target_model = Net().to(self.device)  # Целевая сеть
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)  # Регуляризация
        self.loss_fn = nn.SmoothL1Loss()  # Лучше для DQN чем MSE
        self.update_count = 0

    def _preprocess(self, board: np.ndarray) -> torch.Tensor:
        arr = np.array(board)
        if arr.ndim == 3 and arr.shape == (4, 4, 16):
            x = arr.astype(np.float32).transpose(2, 0, 1)
            return torch.from_numpy(x)
        x = np.zeros((16, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                v = int(arr[i, j])
                idx = int(math.log2(v)) if v > 0 else 0
                x[idx, i, j] = 1.0
        return torch.from_numpy(x)

    def act(self, observation: np.ndarray) -> int:
        self.model.eval()
        state = self._preprocess(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qs = self.model(state)
        return int(torch.argmax(qs, dim=1).item())

    def train_batch(self,
                    states: np.ndarray,
                    targets: np.ndarray,
                    epochs: int = 1,
                    batch_size: int = 64):
        X = torch.stack([self._preprocess(s) for s in states])
        y = torch.from_numpy(targets.astype(np.float32))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def learn(self,
              num_episodes: int = 5000,
              batch_size: int = 128,  # Увеличен размер батча
              gamma: float = 0.99,
              train_start: int = 5000,  # Больше примеров перед обучением
              buffer_size: int = 100000,
              eps_start: float = 1.0,
              eps_min: float = 0.01,  # Минимальное исследование
              eps_decay: float = 0.9995,  # Экспоненциальное затухание
              target_update: int = 1000):  # Частота обновления цели

        replay_buffer = deque(maxlen=buffer_size)
        epsilon = eps_start
        total_steps = 0

        for ep in range(1, num_episodes + 1):
            obs = self.env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            done = False
            ep_rewards = 0
            ep_max_tile = 2

            while not done:
                # Эпсилон-жадная стратегия с экспоненциальным затуханием
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.act(state)
                
                res = self.env.step(action)
                if len(res) == 5:
                    next_state, reward, term, trunc, info = res
                    done = term or trunc
                else:
                    next_state, reward, done, info = res
                
                # Сохраняем переход в буфере
                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                ep_rewards += reward
                ep_max_tile = max(ep_max_tile, int(np.max(state)))
                total_steps += 1
                
                # Экспоненциальное уменьшение epsilon
                epsilon = max(eps_min, epsilon * eps_decay)
                
                # Начинаем обучение после накопления опыта
                if total_steps > train_start and len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    
                    states = np.array(states)
                    next_states = np.array(next_states)
                    rewards = np.array(rewards, dtype=np.float32)
                    dones = np.array(dones, dtype=np.bool_)
                    
                    # Вычисляем Q-значения с использованием целевой сети
                    with torch.no_grad():
                        next_q = self.target_model(
                            torch.stack([self._preprocess(ns) for ns in next_states]).to(self.device)
                        )
                        next_q = next_q.max(1)[0].cpu().numpy()
                    
                    targets = self.model(
                        torch.stack([self._preprocess(s) for s in states]).to(self.device)
                    ).detach().cpu().numpy().copy()
                    
                    # Обновляем только выбранные действия
                    batch_idx = np.arange(batch_size)
                    targets[batch_idx, actions] = rewards + gamma * next_q * ~dones
                    
                    # Обучаем модель
                    self.train_batch(states, targets, epochs=1, batch_size=batch_size)
                    self.update_count += 1
                    
                    # Периодическое обновление целевой сети
                    if self.update_count % target_update == 0:
                        self.target_model.load_state_dict(self.model.state_dict())

            # Логирование после эпизода
            print(f"Ep {ep:4d} | TotalR {ep_rewards:8.1f} | Eps {epsilon:.4f}")

            # Сохранение весов каждые 100 эпизодов
            if ep % 100 == 0:
                self.save_weights(f"weights_ep{ep}.pth")

        print("Training completed.")
        
class DualFeatures:
    """Container for raw and normalized feature dicts."""
    def __init__(self, raw: Dict[str, float], norm: Dict[str, float]) -> None:
        self.raw = raw
        self.norm = norm
    def __getitem__(self, key: str) -> float:
        return self.norm[key]
    def items(self): return self.norm.items()


class MonteCarloAgent(BaseAgent):
    """
    Rollout agent for 2048, using a custom formula or per-action formulas,
    with optional feature scaling if data provided.
    """
    def __init__(
        self,
        env,
        feature_list: List[str],
        rollouts: int = 30,
        rollout_depth: int = 7,
        seed: Optional[int] = None,
        formula: Optional[Callable[[DualFeatures, int], float]] = None,
        formulas: Optional[List[Callable[[DualFeatures], float]]] = None,
        data: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(env)
        # Exactly one of formula or formulas must be provided
        if formula is None and formulas is not None and callable(formulas):
            # treat single callable as per-action list
            formulas = formulas  # noqa
        if (formula is None) == (formulas is None):  # both or neither
            raise ValueError("Provide exactly one of 'formula' or 'formulas'.")

        self.feature_list = feature_list
        self.formula = formula
        self.formulas = formulas
        self.rollouts = rollouts
        self.rollout_depth = rollout_depth
        self.rng = np.random.default_rng(seed)

        if data is not None:
            self._initialize_scaler(data)

    def _initialize_scaler(self, data: np.ndarray) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def _extract_dual_features(self, board: np.ndarray) -> DualFeatures:
        # raw features
        raw = extract_features(board, self.feature_list)
        # normalized
        if hasattr(self, 'scaler'):
            arr = self.scaler.transform(
                np.array([list(raw.values())], dtype=float)
            ).flatten()
            norm = {k: arr[i] for i, k in enumerate(raw.keys())}
        else:
            norm = raw.copy()
        return DualFeatures(raw, norm)

    @staticmethod
    def _compress(row: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        nonzero = row[row != 0]
        merged, reward = [], 0
        i = 0
        while i < len(nonzero):
            if i+1 < len(nonzero) and nonzero[i] == nonzero[i+1]:
                val = nonzero[i]*2
                merged.append(val)
                reward += val
                i += 2
            else:
                merged.append(nonzero[i])
                i += 1
        new_row = np.zeros_like(row)
        new_row[:len(merged)] = merged
        return new_row, reward, not np.array_equal(new_row, row)

    def _apply_move(self, board: np.ndarray, action: int):
        b = board.copy()
        total_r, changed = 0, False
        # ... same as before, omitted for brevity ...
        return b, total_r, changed

    def _spawn_tile(self, board: np.ndarray) -> bool:
        # same as before
        empties = np.argwhere(board == 0)
        if empties.size == 0:
            return False
        r, c = empties[self.rng.integers(len(empties))]
        board[r, c] = 4 if self.rng.random() < .1 else 2
        return True

    def _heuristic(self, board: np.ndarray, action: int) -> float:
        feats = self._extract_dual_features(board)
        if self.formulas is not None:
            # one formula per action
            return float(self.formulas[action](feats))
        else:
            return float(self.formula(feats, action))

    def _rollout(self, board: np.ndarray, root_action: int) -> float:
        b = board.copy()
        total_reward = 0
        for _ in range(self.rollout_depth):
            legal = [a for a in range(4) if self._apply_move(b, a)[2]]
            if not legal:
                break
            a = self.rng.choice(legal)
            b, r, _ = self._apply_move(b, a)
            total_reward += r
            if not self._spawn_tile(b):
                break
        return total_reward + self._heuristic(b, root_action)

    def act(self, observation, mask: Optional[np.ndarray] = None) -> int:
        board = get_tile_value(observation)
        best_a, best_v = 0, -np.inf
        for a in range(self.env.action_space.n):
            if mask is not None and not mask[a]:
                continue
            nb, r0, ch = self._apply_move(board, a)
            if not ch:
                continue
            vals = [self._rollout(nb, a) for _ in range(self.rollouts)]
            score = r0 + np.mean(vals)
            if score > best_v:
                best_v, best_a = score, a
        if mask is not None and not mask[best_a]:
            legal = np.flatnonzero(mask)
            return int(self.rng.choice(legal)) if legal.size else 0
        return best_a

class ExpectimaxAgent(BaseAgent):
    TILE_PROBS: Tuple[Tuple[int, float], ...] = ((2, 0.9), (4, 0.1))

    def __init__(
        self, 
        depth: int = 3, 
        *, 
        discount: float = 1.0,
        heuristic_weights: Optional[dict] = None
    ):
        super().__init__(env=None)
        self.depth = depth
        self.discount = discount
        # If weights provided, use linear heuristic; otherwise use tile sum only
        self.weights = heuristic_weights or None

    def act(self, observation):  # noqa: D401  (simple interface)
        board = get_tile_value(observation)
        score, action = self._expectimax(board, self.depth, is_player=True)
        # If no action was possible (terminal), return 0 by convention
        return int(action if action is not None else 0)

    def _expectimax(self, board: np.ndarray, depth: int, *, is_player: bool) -> Tuple[float, Optional[int]]:
        legal_moves = self._legal_moves(board)
        if depth == 0 or not legal_moves.any():
            return self._heuristic(board), None

        if is_player:
            best_val = -np.inf
            best_action = None
            for action in range(4):
                if not legal_moves[action]:
                    continue
                new_board, _ = self._move(board, action)
                val, _ = self._expectimax(new_board, depth - 1, is_player=False)
                if val > best_val:
                    best_val, best_action = val, action
            return best_val * self.discount, best_action

        # Chance node – enumerate all empty positions and both tile types
        empty = np.argwhere(board == 0)
        exp_val = 0.0
        for (r, c) in empty:
            for tile_val, prob in self.TILE_PROBS:
                board[r, c] = tile_val
                val, _ = self._expectimax(board, depth - 1, is_player=True)
                exp_val += prob * val / len(empty)
            board[r, c] = 0  # Undo
        return exp_val * self.discount, None

    def _heuristic(self, board: np.ndarray) -> float:
        """Evaluate board: linear combination of features + tile sum or just tile sum."""
        # Manual tile_sum: sum of all tile values
        board_sum = float(board.sum())
        if self.weights:
            feats = extract_features(board)
            weighted_sum = sum(
                self.weights.get(fname, 0.0) * feats.get(fname, 0.0)
                for fname in self.weights
            )
            return weighted_sum + board_sum
        # fallback: use tile sum only
        return board_sum

    @staticmethod
    def _merge_row_left(row: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Return (merged_row, changed_flag). No score accumulation – not needed for search."""
        nonzero = row[row != 0]
        merged = []
        skip = False
        for i in range(len(nonzero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                merged.append(nonzero[i] * 2)
                skip = True
            else:
                merged.append(nonzero[i])
        new = np.array(merged + [0] * (4 - len(merged)), dtype=int)
        return new, not np.array_equal(new, row)

    def _move(self, board: np.ndarray, action: int) -> Tuple[np.ndarray, bool]:
        """Apply *action* (0 ↑, 1 →, 2 ↓, 3 ←) and return (new_board, changed?)."""
        b = board.copy()
        moved_any = False
        if action == 0:  # UP
            b = b.T
            for i in range(4):
                new_col, moved = self._merge_row_left(b[i])
                b[i] = new_col
                moved_any |= moved
            b = b.T
        elif action == 2:  # DOWN
            b = b.T
            for i in range(4):
                new_col, moved = self._merge_row_left(b[i][::-1])
                b[i] = new_col[::-1]
                moved_any |= moved
            b = b.T
        elif action == 3:  # LEFT
            for i in range(4):
                b[i], moved = self._merge_row_left(b[i])
                moved_any |= moved
        elif action == 1:  # RIGHT
            for i in range(4):
                new_row, moved = self._merge_row_left(b[i][::-1])
                b[i] = new_row[::-1]
                moved_any |= moved
        else:
            raise ValueError("Invalid action; must be 0,1,2,3")
        return b, moved_any

    def _legal_moves(self, board: np.ndarray) -> np.ndarray:
        """Vector bool length‑4: legal player moves from *board*."""
        legal = np.zeros(4, dtype=bool)
        for a in range(4):
            new_board, changed = self._move(board, a)
            legal[a] = changed
        return legal
