import gymnasium as gym
import gymnasium_2048
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Gumbel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import random 
from typing import Dict, List, Optional, Callable
from types import SimpleNamespace
import copy
import sys
import warnings 
import math
from dataclasses import dataclass

from .vizualize_grid import get_tile_value
from .features import extract_features, feature_names, MetricsRecorder
from .run import create_dataset, EpisodeRunner, run_episode
from .base_agents import NNAgent, MonteCarloAgent, ExpectimaxAgent

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
class DualFeatures(SimpleNamespace):
    """Container that exposes *both* raw and normalised feature dictionaries.

    Attributes
    ----------
    raw : dict[str, float]
        Features computed directly from the game board ("physics" space).
    norm : dict[str, float]
        The same features after ``StandardScaler`` normalisation.
    """

    def __init__(self, raw: Dict[str, float], norm: Dict[str, float]) -> None:
        # Store both copies explicitly
        super().__init__()
        self.raw = raw
        self.norm = norm

    # Optional convenience so that `df["name"]` accesses the *normalised* copy
    def __getitem__(self, item: str) -> float:
        return self.norm[item]

    def __iter__(self):  # keeps ``dict()`` happy if user really wants one
        return iter(self.norm)

    def keys(self):
        return self.norm.keys()

    def values(self):
        return self.norm.values()

    def items(self):
        return self.norm.items()
    
    
def make_formulas(models, features_list: list=feature_names): 
    formulas = []
    for a, model in enumerate(models):
        def make_formula(m):
            return lambda feats: float(
                m.predict(
                    np.array([[feats[name] for name in features_list]])
                )[0,0]
            )
        formulas.append(make_formula(model))
    return formulas


class BaseAgent:
    """Abstract base class; every agent implements ``act(observation)``."""

    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, observation, mask: Optional[np.ndarray] = None):  # noqa: D401
        raise NotImplementedError

class FormulaAgent(BaseAgent):
    def __init__(
        self,
        env: gym.Env,
        feature_list: List[str],
        formula: Optional[Callable[[DualFeatures, int], float]] = None,
        formulas: Optional[List[Callable[[DualFeatures], float]]] = None,
        data: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(env)
        if formula is None and formulas is not None and callable(formulas):
            formula = formulas
            formulas = None
        if (formula is None) == (formulas is None):
            raise ValueError("Provide exactly one of 'formula' or 'formulas'.")

        self.feature_list = feature_list
        self.formula = formula
        self.formulas = formulas
        if data is not None:
            self._initialize_scaler(data)

    def _initialize_scaler(self, data: np.ndarray) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def _get_legit_mask(self) -> np.ndarray:
        ui = self.env.unwrapped
        if hasattr(ui, "legit_actions"):
            return np.asarray(ui.legit_actions, dtype=bool)
        if hasattr(ui, "get_legit_actions"):
            return np.asarray(ui.get_legit_actions(), dtype=bool)
        return np.ones(self.env.action_space.n, dtype=bool)

    def _extract_dual_features(self, observation) -> DualFeatures:
        board = get_tile_value(observation)
        raw = extract_features(board, self.feature_list)
        if hasattr(self, 'scaler'):
            arr = self.scaler.transform(
                np.array(list(raw.values()), dtype=float).reshape(1, -1)
            ).flatten()
            norm = {k: arr[i] for i, k in enumerate(raw.keys())}
        else:
            norm = raw 

        return DualFeatures(raw, norm)

    def act(self, observation, mask: Optional[np.ndarray] = None) -> int:
        feats = self._extract_dual_features(observation)
        if mask is None:
            mask = self._get_legit_mask()

        if self.formulas is not None:
            scores = [f(feats) for f in self.formulas]
        else:
            scores = [self.formula(feats, a) for a in range(self.env.action_space.n)]

        scores = [s if mask[i] else -np.inf for i, s in enumerate(scores)]
        return int(np.argmax(scores))

    def get_ranked_actions(self, observation) -> List[tuple[int, float]]:
        feats = self._extract_dual_features(observation)
        if self.formulas is not None:
            scores = [f(feats) for f in self.formulas]
        else:
            scores = [self.formula(feats, a) for a in range(self.env.action_space.n)]
        return sorted([(i, scores[i]) for i in range(len(scores))], key=lambda x: x[1], reverse=True)


def _sanitize(v: np.ndarray, clip: float = 1e3) -> np.ndarray:
    """Replace NaN/Inf and clip extreme values"""
    v = np.nan_to_num(v, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(v, -clip, clip)

class FeatureAffine(nn.Module):
    """Per-feature linear transform with clamping"""

    def __init__(self, n_feats: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_feats))
        self.bias = nn.Parameter(torch.zeros(n_feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # x:(B,14)
        return torch.clamp(x * self.weight + self.bias, -1e4, 1e4)

def _safe_div(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    denom = torch.where(torch.abs(y) < eps, torch.full_like(y, eps), y)
    z = x / denom
    return torch.clamp(z, -1e2, 1e2)                              # жёсткий лимит

OP_LIBRARY: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": _safe_div,
    "max": torch.maximum,
    "min": torch.minimum,
    "abs": lambda x, _y: torch.abs(x),
}
OP_NAMES: List[str] = list(OP_LIBRARY.keys())


class _SymbolicNode(nn.Module):
    """DAG-узел с gumbel-softmax и tanh-нормировкой результата"""

    def __init__(self, num_inputs: int, temperature: float = 5.0):
        super().__init__()
        self.temperature = temperature
        self.op_logits = nn.Parameter(torch.randn(len(OP_NAMES)))
        self.a_logits = nn.Parameter(torch.randn(num_inputs))
        self.b_logits = nn.Parameter(torch.randn(num_inputs))

    # — gumbel-softmax —
    def _gs(self, logits: torch.Tensor, hard: bool):
        t = torch.clamp(torch.tensor(self.temperature, device=logits.device), min=0.3)
        noise = Gumbel(0, 1).sample(logits.shape).to(logits.device)
        y = F.softmax((logits + noise) / t, dim=-1)
        if hard:
            y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        return y

    def forward(self, inputs: list[torch.Tensor], *, hard: bool = False):
        op_w = self._gs(self.op_logits, hard)
        a_w  = self._gs(self.a_logits, hard)
        b_w  = self._gs(self.b_logits, hard)

        stacked = torch.stack(inputs, 0)              # (N,B)
        a = (stacked * a_w.unsqueeze(1)).sum(0)
        b = (stacked * b_w.unsqueeze(1)).sum(0)

        res = torch.stack([OP_LIBRARY[n](a, b) for n in OP_NAMES], 0)
        y   = (op_w.unsqueeze(1) * res).sum(0)
        return torch.tanh(y)                           # стабилизируем диапазон

    def discrete_spec(self):
        return (
            OP_NAMES[self.op_logits.argmax().item()],
            self.a_logits.argmax().item(),
            self.b_logits.argmax().item(),
        )

class SymbolicNet(nn.Module):
    def __init__(self, *, layers: int = 6, temperature: float = 5.0):
        super().__init__()
        self.temperature = temperature
        self.layers = nn.ModuleList(
            [_SymbolicNode(14 + i, temperature) for i in range(layers)]
        )

    def anneal(self, *, factor: float = .98, min_temp: float = .5):
        self.temperature = max(min_temp, self.temperature * factor)
        for n in self.layers:
            n.temperature = self.temperature

    def forward(self, feats: torch.Tensor, *, hard: bool = False):
        inter = [feats[:, i] for i in range(14)]          # (B,)
        for node in self.layers:
            inter.append(node(inter, hard=hard))
        out = inter[-1].unsqueeze(1)                      # (B,1)
        return torch.clamp(out, -1e2, 1e2)

    def export_expression(self):
        exprs = [f"f{i}" for i in range(14)]
        for node in self.layers:
            op, ia, ib = node.discrete_spec()
            a, b = exprs[ia], exprs[ib]
            match op:
                case "add": e = f"({a}+{b})"
                case "sub": e = f"({a}-{b})"
                case "mul": e = f"({a}*{b})"
                case "div": e = f"({a}/({b}+1e-4))"
                case "max": e = f"max({a},{b})"
                case "min": e = f"min({a},{b})"
                case "abs": e = f"abs({a})"
            exprs.append(e)
        return exprs[-1]

@dataclass
class Transition:
    s: np.ndarray; r: float; s_next: np.ndarray; done: bool

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity, self.data, self.ptr = capacity, [], 0
    def add(self, t: Transition):
        if len(self.data) < self.capacity: self.data.append(t)
        else: self.data[self.ptr] = t
        self.ptr = (self.ptr + 1) % self.capacity
    def sample(self, batch: int):
        idxs = np.random.choice(len(self.data), batch, replace=batch > len(self.data))
        return [self.data[i] for i in idxs]

class ESPLLearner:
    def __init__(
        self, env: gym.Env,
        *, layers: int = 6, lr: float = 1e-4, gamma: float = .99,
        batch_size: int = 512, buffer_capacity: int | float = 1e5,
        warmup_steps: int = 10_000, updates_per_step: int = 1,
        anneal_start: int = 50_000, anneal_factor: float = .98, min_temp: float = .5,
        eps_start: float = .7, eps_end: float = .01, eps_decay: float = .995,
        device: str = "cpu", seed: int | None = None, verbose: bool = True,
        scaler: StandardScaler | None = None, feature_list: list[str] = feature_names,
    ):
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        self.env, self.gamma = env, gamma
        self.batch, self.upd_per = batch_size, updates_per_step
        self.verbose, self.device = verbose, torch.device(device)
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay

        self.affine = FeatureAffine(len(feature_list)).to(self.device)
        self.net    = SymbolicNet(layers=layers).to(self.device)

        # L2-регуляризация
        self.optim = torch.optim.Adam(
            list(self.affine.parameters()) + list(self.net.parameters()),
            lr=lr,
            weight_decay=1e-5,
        )

        self.target_affine = copy.deepcopy(self.affine).eval()
        self.target_net    = copy.deepcopy(self.net).eval()

        self.target_tau, self.grad_clip_norm = .005, 1.0
        self.env_steps  = 0
        self.grad_steps = 0
        self.anneal_start, self.anneal_factor, self.min_temp = anneal_start, anneal_factor, min_temp
        self.buf, self.warm = ReplayBuffer(int(buffer_capacity)), warmup_steps
        self.scaler, self.feature_list = scaler, feature_list

    def _feats(self, obs):
        board = get_tile_value(obs)
        raw   = extract_features(board, self.feature_list)
        vec   = np.array([raw[f] for f in self.feature_list], np.float32)
        return _sanitize(vec)

    def _value(self, board):
        raw = extract_features(board, self.feature_list)
        vec = _sanitize(np.array([raw[f] for f in self.feature_list], np.float32))
        t   = torch.tensor([vec], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.net(self.affine(t)).item()

    def train_episodes(self, num_episodes: int = 10_000):
        for ep in range(1, num_episodes + 1):
            obs, _ = self.env.reset(); done = False; ep_rew = 0.0
            while not done:
                board = get_tile_value(obs)
                a = self._random_legal_move(board) if random.random() < self.eps else self._select_action(obs)
                if a is None: break
                nxt, r, term, trunc, _ = self.env.step(a); done = term or trunc

                self.buf.add(Transition(self._feats(obs), r, self._feats(nxt), done))
                self.env_steps += 1

                if len(self.buf.data) >= self.warm:
                    for _ in range(self.upd_per): self._update()

                obs, ep_rew = nxt, ep_rew + r

            if self.grad_steps > self.anneal_start:
                self.net.anneal(factor=self.anneal_factor, min_temp=self.min_temp)
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

            if self.verbose:
                print(
                    f"Ep {ep:5d}  reward={ep_rew:6.1f}  env={self.env_steps:7d}  grad={self.grad_steps:7d}  eps={self.eps:.4f}  T={self.net.temperature:.2f}"
                )

    def _random_legal_move(self, board):
        legal = []
        for a in range(self.env.action_space.n):
            env_cpy = copy.deepcopy(self.env.unwrapped)
            if not np.array_equal(board, get_tile_value(env_cpy.step(a)[0])):
                legal.append(a)
        return random.choice(legal) if legal else None

    def _select_action(self, obs):
        board = get_tile_value(obs); best_val, best_act = -math.inf, None
        for a in range(self.env.action_space.n):
            env_cpy = copy.deepcopy(self.env.unwrapped)
            obs2, r, term, trunc, _ = env_cpy.step(a)
            if np.array_equal(board, get_tile_value(obs2)): continue
            val = r + self.gamma * self._value(get_tile_value(obs2))
            if val > best_val: best_val, best_act = val, a
        return best_act if best_act is not None else self._random_legal_move(board)

    def _update(self):
        batch = self.buf.sample(self.batch)
        s  = torch.from_numpy(np.stack([t.s for t in batch])).to(self.device)
        sp = torch.from_numpy(np.stack([t.s_next for t in batch])).to(self.device)
        r  = torch.tensor([t.r   for t in batch], dtype=torch.float32, device=self.device)
        d  = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            tgt = r + self.gamma * (1.0 - d) * self.target_net(self.target_affine(sp)).squeeze(1)

        pred = self.net(self.affine(s)).squeeze(1)
        loss = F.mse_loss(pred, tgt)
        if not torch.isfinite(loss):
            if self.verbose:
                print("non-finite loss", loss.item())
            return

        self.optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(self.net.parameters()) + list(self.affine.parameters()), self.grad_clip_norm)
        self.optim.step()

        with torch.no_grad():
            for p, p_t in zip(self.net.parameters(), self.target_net.parameters()):
                p_t.mul_(1 - self.target_tau).add_(self.target_tau * p)
            for p, p_t in zip(self.affine.parameters(), self.target_affine.parameters()):
                p_t.mul_(1 - self.target_tau).add_(self.target_tau * p)

        self.grad_steps += 1

    def export_formula(self) -> str:
        _ = self.net(torch.zeros(1, 14, device=self.device), hard=True)  # fix choices
        tree = self.net.export_expression()
        w = self.affine.weight.detach().cpu().numpy()
        b = self.affine.bias.detach().cpu().numpy()
        affine = " + ".join([f"({w[i]:+.3f}*f{i}{b[i]:+.3f})" for i in range(len(w))])
        return affine + "  →  " + tree