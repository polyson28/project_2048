import gymnasium as gym
import time
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.serialization import safe_globals
import torch.optim as optim
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from random import random
from collections import deque
from IPython import display
import matplotlib.pyplot as plt
import copy
import math
from pathlib import Path
from tqdm.auto import trange, tqdm 
from typing import Dict, Any, Optional, Sequence, List, Callable, Union

from .vizualize_grid import get_tile_value
from .features import extract_features, feature_names
from .base_agents import DQNAgentWrapper

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
    

class ViperAgent(BaseAgent):
    """VIPER (Verifiable Policy Extraction) agent for the 2048 game.

    The agent extracts an interpretable **decision‑tree** policy by imitating a
    high‑performing DQN oracle using the VIPER algorithm (Bastani et al., NeurIPS
    2018). At prediction time the agent is *extremely* fast because it only
    needs one pass through a shallow tree.

    Parameters
    ----------
    env : gym.Env
        The 2048 environment.
    oracle : DQNAgentWrapper
        Pre‑trained deep‑Q agent acting as the expert (π★) and providing Q‑values.
    num_iters : int, default 15
        Number of VIPER iterations.
    num_trajs : int, default 50
        Number of oracle rollouts collected per iteration.
    max_depth : int, default 8
        Maximum depth of the extracted decision tree.
    min_samples_leaf : int, default 1
        Minimum number of samples required at each leaf.
    random_state : int | None, default 42
        Random seed for the decision‑tree learner.
    auto_train : bool, default True
        If *True*, ``fit()`` is executed from the constructor.
    """

    def __init__(
        self,
        env: gym.Env,
        oracle: DQNAgentWrapper,
        *,
        num_iters: int = 15,
        num_trajs: int = 50,
        max_depth: int = 8,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        auto_train: bool = True,
    ) -> None:
        super().__init__(env)
        self.oracle = oracle
        self.num_iters = num_iters
        self.num_trajs = num_trajs
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        # Internal containers (aggregated over VIPER iterations)
        self._xs: List[np.ndarray] = []  # feature vectors
        self._ys: List[int] = []         # oracle actions
        self._ws: List[float] = []       # importance weights (\tilde ℓ)

        self.tree: Optional[DecisionTreeClassifier] = None

        if auto_train:
            self.fit()

    def fit(self) -> None:
        """Run VIPER to extract the decision‑tree policy from the oracle."""

        best_tree: Optional[DecisionTreeClassifier] = None
        best_score: float = -np.inf

        it_iter = trange(1, self.num_iters + 1, desc="VIPER it")
        for _it in it_iter:
            self._collect_dataset(self.num_trajs)
            tree = self._train_tree()
            score = self._evaluate(tree, n_episodes=20)

            if score > best_score:
                best_score, best_tree = score, tree
                # De‑refit ensures picklability / small size
                best_tree.n_outputs_  # touch attribute to silence mypy

            # Optional: early stopping if agent already matches oracle
            if score >= self._oracle_mean_score():
                break

        self.tree = best_tree if best_tree is not None else tree

    def act(self, observation) -> int:
        """Choose an action using the learned decision tree.

        Falls back to the oracle if the tree has not been fitted yet.
        """
        if self.tree is None:
            # Safety fallback – use oracle directly
            return int(self.oracle.act(observation))

        x = self._featurize_obs(observation).reshape(1, -1)
        action_pred = int(self.tree.predict(x)[0])
        return action_pred

    def save(self, path: str) -> None:
        """Persist the extracted decision tree to *path* (joblib)."""
        if self.tree is None:
            raise RuntimeError("Cannot save – the model has not been trained yet.")
        joblib.dump(self.tree, path)

    def load(self, path: str) -> None:
        """Load a previously saved decision tree from *path*."""
        self.tree = joblib.load(path)

    def _collect_dataset(self, n_episodes: int) -> None:
        """Run *n_episodes* rollouts with the *current* extracted policy and
        aggregate states labelled by the oracle. On the very first call the
        rollouts are performed **purely** with the oracle – this mirrors the
        behaviour of DAGGER / VIPER.
        """
        ep_iter = trange(n_episodes, desc="rollouts", leave=False)
        for _ in ep_iter:
            obs, _ = self.env.reset()
            done = False
            while not done:
                # Oracle guidance ------------------------------------------------
                q_values = self._oracle_q_values(obs)
                oracle_action = int(np.argmax(q_values))

                # Importance weight \tilde ℓ(s) = max_a Q – min_a Q
                weight = float(q_values.max() - q_values.min())

                # Feature extraction -----------------------------------------
                x_vec = self._featurize_obs(obs)

                # Store ------------------------------------------------------
                self._xs.append(x_vec)
                self._ys.append(oracle_action)
                self._ws.append(weight)

                # Execute either current policy or oracle --------------------
                if self.tree is None:
                    action_to_play = oracle_action  # cold start – oracle only
                else:
                    # Roll‑in with student, oracle used only for labelling
                    action_to_play = int(self.tree.predict(x_vec.reshape(1, -1))[0])

                obs, _, terminated, truncated, _ = self.env.step(action_to_play)
                done = terminated or truncated

    def _train_tree(self) -> DecisionTreeClassifier:
        X = np.vstack(self._xs)
        y = np.asarray(self._ys, dtype=int)
        w = np.asarray(self._ws, dtype=float)

        clf = DecisionTreeClassifier(
            criterion="gini",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        clf.fit(X, y, sample_weight=w)
        return clf

    def _evaluate(self, tree: DecisionTreeClassifier, n_episodes: int = 10) -> float:
        """Return the average game score over *n_episodes* using *tree*."""
        total_score = 0.0
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            cum = 0.0
            while not done:
                x = self._featurize_obs(obs).reshape(1, -1)
                action = int(tree.predict(x)[0])
                obs, reward, terminated, truncated, _ = self.env.step(action)
                cum += float(reward)
                done = terminated or truncated
            total_score += cum
        return total_score / n_episodes

    def _oracle_mean_score(self, n_episodes: int = 10) -> float:
        """Utility – mean reward of the oracle on *n_episodes* fresh games."""
        total = 0.0
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            cmp = 0.0
            while not done:
                action = self.oracle.act(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                cmp += float(reward)
                done = terminated or truncated
            total += cmp
        return total / n_episodes

    def _featurize_obs(self, obs) -> np.ndarray:
        """Convert raw env *obs* into the fixed‑length feature vector that the
        decision tree expects (order corresponds to ``feature_names``).
        """
        board = get_tile_value(obs)
        feats = extract_features(board)
        return np.asarray([feats[name] for name in feature_names], dtype=np.float32)

    def _oracle_q_values(self, obs) -> np.ndarray:
        """Compute the oracle's Q‑values for the given observation (shape = 4)."""
        board = get_tile_value(obs)
        state_t = self.oracle._preprocess_board(board).to(self.oracle.device)
        with torch.no_grad():
            q = self.oracle.net(state_t).cpu().numpy().ravel()
        return q

    def __repr__(self) -> str:  # pragma: no cover – diagnostic only
        if self.tree is None:
            return "<ViperAgent (untrained)>"
        return (
            f"<ViperAgent depth={self.tree.get_depth()}"
            f" leaves={self.tree.get_n_leaves()}>"
        )
