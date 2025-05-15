import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .vizualize_grid import get_tile_value

from drl2048.modules.agents import ConvDQN_Agent
from drl2048.modules.architectures import BigConvolutionalNetwork
from drl2048.modules import architectures as real_architectures

class BaseAgent:
    """Abstract base class; every agent implements ``act(observation)``."""
    def __init__(self, env: gym.Env):
        self.env = env                 

    def act(self, observation):  # noqa: D401  (simple interface)
        raise NotImplementedError
    
    
class DQNAgentWrapper(BaseAgent):
    """
    Обёртка для готового агента из репозитория YangRui2015/2048_env.
    Ожидает CNN-сеть, сохранённую через ``torch.save(model.state_dict(), *.pth)``.
    """

    def __init__(self, model_path: str, device: str | None = None):
        super().__init__(env=None)     # среду передаём позже
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = self._build_cnn().to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()                # только инференс

    @staticmethod
    def _build_cnn(conv_size=(32, 64), fc_size=(512, 128)) -> nn.Module:
        """Точная копия CNN_Net из NN_module.py (репозиторий YangRui2015)."""
        class CNN_Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(1, conv_size[0], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(conv_size[0], conv_size[1], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.fc1 = nn.Linear(conv_size[1] * 4 * 4, fc_size[0])
                self.fc2 = nn.Linear(fc_size[0], fc_size[1])
                self.head = nn.Linear(fc_size[1], 4)      # 4 действия

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.head(x)

        return CNN_Net()

    @staticmethod
    def _preprocess_board(board: np.ndarray) -> torch.Tensor:
        """
        board (4×4, реальные числа) → 1×1×4×4 tensor; масштаб как в оригинале:
        log₂(value+1) / 16  (см. README и utils.py репозитория).”"
        """
        state = np.log2(board + 1, where=board >= 0) / 16.0
        return torch.from_numpy(state.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    def act(self, observation: np.ndarray) -> int:
        board = get_tile_value(observation)
        legal = ConvDQNAgentWrapper._legal_moves(board)

        with torch.no_grad():
            state_t = self._preprocess_board(board).to(self.device)
            q = self.net(state_t).cpu().numpy().ravel()

        q[~legal] = -np.inf             # обнуляем запрещённые
        return int(np.argmax(q))
    
    
class ConvDQNAgentWrapper(BaseAgent):
    def __init__(
        self,
        model_path, 
        agent_params={
            'gamma': 0.995,
            'replay_memory_size': 10000,
            'batch_size': 384,
            'eps_start': 0.8,
            'eps_end': 0.02, 
            'eps_decay': 1000,             
            'tau': 5e-3,
            'kind_action': "entropy",
            'lr': 1e-3
        }, 
        model_params={'l2_regularization': 3e-4}
    ):
        self.model_path = model_path
        self.agent_params = agent_params
        self.model_params = model_params 
        self._load_model()
        
    def _load_model(self) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BigConvolutionalNetwork(**self.model_params).to(device)

        agent = ConvDQN_Agent(
            model=self.model,
            device=device,
            **self.agent_params
            )
        agent.load(self.model_path)
        self.agent = agent 
    
    @staticmethod
    def _encode_board(obs: np.ndarray) -> torch.Tensor:
        """
        Возвращает one-hot-тензор (12, 4, 4), который ждёт сеть.
        Вход obs может быть:
            • (4, 4)  — числа 0, 2, 4, 8…          (log2 <= 15)
            • (4, 4, 16) — one-hot                 (последний dim = 16)
        """
        board = obs
        if board.ndim == 3 and board.shape[2] == 16:
            # переводим one-hot ➜ индексы (0…15)
            board = board.argmax(axis=2)
        else:
            board = board.copy()

        # переводим реальные значения ➜ log2
        if board.max() > 15:                       # приходят 2, 4, 8…
            board = np.where(board > 0,
                             np.log2(board).astype(int),
                             0)

        planes = np.zeros((12, 4, 4), dtype=np.float32)
        for k in range(12):
            planes[k] = (board == k)
        return torch.from_numpy(planes)

    @staticmethod
    def _legal_moves(board: np.ndarray) -> np.ndarray:
        """
        Возвращает вектор bool длиной 4: [up, down, left, right].
        Любой алгоритм 2048 подойдёт; ниже — короткая проверка
        «можно ли сделать сдвиг/слияние» для каждого направления.
        """
        def can_shift(mat):
            for row in mat:
                nz = row[row != 0]
                if len(nz) < len(row):              # есть пустая клетка слева
                    return True
                if (nz[:-1] == nz[1:]).any():       # есть пара для слияния
                    return True
            return False

        left  = can_shift(board)
        right = can_shift(np.fliplr(board))
        up = can_shift(board.T)
        down = can_shift(np.flipud(board).T)
        return np.array([up, down, left, right], dtype=bool)

    def act(self, obs: np.ndarray) -> int:
        # приводим к числовому представлению
        numeric_board = (
            obs.argmax(axis=2) if obs.ndim == 3 and obs.shape[2] == 16
            else np.asarray(obs)
        )

        legal = self._legal_moves(numeric_board)
        if not legal.any():
            return 0

        state = self._encode_board(numeric_board).unsqueeze(0).to(self.agent.device)

        with torch.no_grad():
            action_tensor = self.agent.select_action(
                state,
                legal,
                kind=self.agent.kind_action,
                train=True,
            )
        return int(action_tensor.item())