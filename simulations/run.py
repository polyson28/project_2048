import gymnasium as gym
import matplotlib.pyplot as plt 
import time
from typing import Callable, Dict, List, Union, Optional, Literal, Tuple
import numpy as np 
import torch
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

from .vizualize_grid import display_game, get_tile_value
from .features import extract_features, feature_names

class EpisodeRunner:
    """Запускает одну партию 2048, оберегая среду от недопустимых действий.

    Parameters
    ----------
    env : gym.Env
        Среда 2048 (должна предоставлять ``env.unwrapped.legit_actions``
        или метод ``env.unwrapped.get_legit_actions()``).
    agent : AgentLike
        Объект с методом ``act(obs, mask) -> int``.
    delay : float, default 0.25
        Пауза между ходами при визуализации.
    visualize : bool, default True
        Визуализировать ли партию.
    on_illegal : {"random", "ask"} | Callable[[np.ndarray], int], default "random"
        Стратегия, если агент выдал нелегальный ход:
        * ``"random"`` – бесшумно берётся случайное легальное действие.
        * ``"ask"``    – запрашивать ход у агента, передавая маску,
          максимум 10 попыток.
        * функция      – пользовательский колбэк ``mask -> action``.
    """

    def __init__(
        self,
        env,
        agent,
        *,
        delay: float = 0.05,
        visualize: bool = True,
        on_illegal: str | Callable[[np.ndarray], int] = "random",
    ) -> None:
        self.env = env
        self.agent = agent
        self.delay = delay
        self.visualize = visualize
        self.on_illegal = on_illegal

    def _get_legit_mask(self) -> np.ndarray:
        """Возвращает булеву маску допустимых действий текущего состояния."""
        ui = self.env.unwrapped
        if hasattr(ui, "legit_actions"):
            return np.asarray(ui.legit_actions, dtype=bool)
        if hasattr(ui, "get_legit_actions"):
            return np.asarray(ui.get_legit_actions(), dtype=bool)
        # если среда не поддерживает – считаем все действия допустимыми
        return np.ones(self.env.action_space.n, dtype=bool)

    def _resolve_action(self, obs, proposed: int) -> int:
        mask = self._get_legit_mask()
        if mask[proposed]:
            return proposed

        # ---- действие недопустимо ----
        if self.on_illegal == "random":
            return int(np.random.choice(np.flatnonzero(mask)))

        if self.on_illegal == "ask":
            for _ in range(10):
                new_action = self.agent.act(obs, mask=mask)
                if mask[new_action]:
                    return new_action
            # не смог – fallback random
            return int(np.random.choice(np.flatnonzero(mask)))

        # пользовательский колбэк
        return int(self.on_illegal(mask))
    
    def _get_next_best_action(self, obs, used_actions):
        """Получить следующее лучшее действие от агента, исключая уже использованные."""
        
        # Проверяем, поддерживает ли агент ранжированные действия
        if hasattr(self.agent, 'get_ranked_actions'):
            try:
                ranked_actions = self.agent.get_ranked_actions(obs)
                # Находим первое действие, которое еще не использовалось
                for action, score in ranked_actions:
                    if action not in used_actions:
                        mask = self._get_legit_mask()
                        if mask[action]:  # Проверяем, что действие легально
                            return action
            except (AttributeError, NotImplementedError):
                pass
        
        # Fallback: случайный выбор из неиспользованных легальных действий
        mask = self._get_legit_mask()
        available_actions = []
        for a in range(4):
            if mask[a] and a not in used_actions:
                available_actions.append(a)
        
        if available_actions:
            return int(np.random.choice(available_actions))
        
        return None  # Нет доступных действий

    def run(self, max_steps: int | None = None, **display_params) -> int:
        obs, _ = self.env.reset()
        cum_score = 0
        step = 0
        done = False

        if self.visualize:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            display_game(obs, ax=ax, **display_params)

        while not done and (max_steps is None or step < max_steps):
            # Получаем действие от агента
            raw_action = self.agent.act(obs)
            action = self._resolve_action(obs, raw_action)
            
            # Сохраняем состояние до выполнения действия
            prev_obs = obs.copy()
            used_actions = {action}  # Отслеживаем использованные действия
            
            # Выполняем действие
            obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Проверяем, изменилось ли состояние после действия
            attempts = 0
            max_attempts = 4  # Максимум 4 попытки (по числу действий)
            
            while np.array_equal(prev_obs, obs) and attempts < max_attempts:
                # Действие не изменило состояние - пробуем следующее
                next_action = self._get_next_best_action(obs, used_actions)
                
                if next_action is None:
                    # Нет больше доступных действий - игра должна завершиться
                    terminated = True
                    break
                
                used_actions.add(next_action)
                action = next_action
                
                # Пробуем новое действие
                obs, reward, terminated, truncated, _ = self.env.step(action)
                attempts += 1

            done = terminated or truncated
            prev_score = cum_score
            cum_score += int(reward)

            if self.visualize:
                display_game(
                    obs,
                    action=action,
                    step_num=step,
                    prev_score=prev_score,
                    curr_score=cum_score,
                    ax=ax,
                    **display_params
                )
                time.sleep(self.delay)

            step += 1

        if self.visualize:
            print("\n===== Финал =====")
            if done:
                print(f"Игра завершена (ходов сыграно: {step})")
            else:
                print(f"Прервано по лимиту {max_steps} ходов (сыграно: {step})")
            print(f"Итоговый счёт: {cum_score}")

        return cum_score


def run_episode(
    env,
    agent,
    *,
    max_steps: int | None = None,
    delay: float = 0.05,
    visualize: bool = True, 
    on_illegal: str | Callable[[np.ndarray], int] = "random",
    **display_params
) -> int:
    """Обёртка, сохраняющая старый интерфейс ``run_episode``."""
    return EpisodeRunner(
        env,
        agent,
        delay=delay,
        visualize=visualize,
        on_illegal=on_illegal,
    ).run(max_steps=max_steps, **display_params)
    
    
def create_dataset(
    env: gym.Env,
    agent,
    n_episodes: int,
    *,
    max_steps: int | None = None,
    visualize: bool = False,
    on_illegal: Union[str, Callable[[np.ndarray], int]] = "random",
    delay: float = 0.05,
    save_path: str | None = None,
) -> Dict[str, List]:
    """
    Записывает **состояние ДО хода**.  
    Для каждого board сохраняет:
        • interpretable-features текущей доски  
        • действие, выполненное с этой доски  
        • вознаграждение за ход  
        • Q-values (4 значения) ← модель агента на той же доске  
    """
    class DataCollectionWrapper(gym.Wrapper):
        def __init__(self, env, dataset):
            super().__init__(env)
            self.ds: Dict[str, List] = dataset
            self.ep_id   = 0
            self.step_id = 0
            self._prev_board: np.ndarray | None = None   # доска ДО очередного шага

        # при начале эпизода сообщаем номер
        def begin_episode(self, ep_id: int):
            self.ep_id = ep_id

        # reset: фиксируем первую доску
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self.step_id   = 0
            self._prev_board = get_tile_value(obs)
            return obs, info

        # step:   1) пишем prev_board, 2) делаем env.step,
        #         3) сохраняем reward и новую prev_board
        def step(self, action: int):
            assert self._prev_board is not None, "reset() не был вызван"

            # ----------- 1. log состояние ДО действия -----------
            b = self._prev_board.copy()
            self.ds["boards"]      .append(b)
            self.ds["features"]    .append(extract_features(b))
            self.ds["actions"]     .append(int(action))
            self.ds["episode_ids"] .append(self.ep_id)
            self.ds["step_ids"]    .append(self.step_id)

            # ----------- 2. выполняем действие -----------
            obs, reward, terminated, truncated, info = self.env.step(action)

            # ----------- 3. дописываем reward & готовимся к след. ходу -----------
            self.ds["rewards"].append(float(reward))
            self.step_id   += 1
            self._prev_board = get_tile_value(obs)
            return obs, reward, terminated, truncated, info

    dataset = { # списки-столбцы одинаковой длины
        "boards":      [],  # np.ndarray (4×4), ДО действия t
        "features":    [], # dict{feature→val} для boards[t]
        "actions":     [], # int ∈ {0,1,2,3}, действие из boards[t]
        "rewards":     [], # float, reward, полученный за действие t
        "episode_ids": [], # int, номер эпизода
        "step_ids":    [], # int, порядковый номер шага в эпизоде
    }

    wrapped_env = DataCollectionWrapper(env, dataset)

    scores = []
    for ep in tqdm(range(n_episodes)):
        wrapped_env.begin_episode(ep)
        score = run_episode(
            wrapped_env, 
            agent,
            max_steps=max_steps, 
            visualize=visualize,
            delay=delay if visualize else 0.0,
            on_illegal=on_illegal,
            add_to_title=f"эпизод {ep+1}",
        )
        scores.append(score)

    states = torch.stack([agent._preprocess(b) for b in dataset["boards"]]).to(agent.device)
    with torch.no_grad():
        q_mat = agent.model(states).cpu().numpy()      # shape = (N, 4)
    dataset["Q-values"] = [row for row in q_mat]       # сериализуем как list[list[float]]

    if save_path:
        for path in save_path: 
            with open(path, "wb") as f:
                pickle.dump(dataset, f)
            print(f"Датасет сохранён в: {path}")

    print(f"Собрано {len(dataset['boards'])} ходов из {n_episodes} эпизодов.")
    return dataset, scores 


def simulate_move(board: np.ndarray, action: int, env_id: str = "gymnasium_2048/TwentyFortyEight-v0") -> np.ndarray:
    """
    Симулировать один шаг 2048 на доске `board` и вернуть новую доску (4x4) после хода `action`.
    Требует, чтобы внутри env.unwrapped была возможность задать состояние доски напрямую.
    """
    # Создаем окружение и сбрасываем состояние
    env = gym.make(env_id)
    env.reset()
    # Перекодируем числовую доску в экспоненциальное представление для env
    exp_board = np.zeros_like(board, dtype=int)
    non_zero = board > 0
    # логарифм по основанию 2 от значений плиток дает экспоненты
    exp_board[non_zero] = np.log2(board[non_zero]).astype(int)
    # Устанавливаем внутреннюю доску среды
    env.unwrapped.board = exp_board.copy()
    # Выполняем действие и получаем новое наблюдение
    obs, _, _, _, _ = env.step(action)
    # Преобразуем наблюдение обратно в числовую доску
    new_board = get_tile_value(obs)
    return new_board


def preprocess_dataset(
    data_path: str, 
    keep_best: Union[int, None]=None, 
    features_list: list=feature_names, 
    add_board_features: bool=False, 
    expand: bool=False, 
    transform: Literal['divide_by_actions', 'divide_by_state', None]=None, 
    target_type: Literal['Q-values', 'actions']='Q-values', 
    normalize: bool=False,
):
    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)
        
    data = raw_data.copy()
    if keep_best:
        # Преобразуем списки в массивы для векторных вычислений
        episode_ids = np.array(data["episode_ids"], dtype=int)
        rewards = np.array(data["rewards"], dtype=float)

        ep_scores = {}
        for ep in np.unique(episode_ids):
            ep_scores[ep] = rewards[episode_ids == ep].sum()

        best_eps = sorted(ep_scores, key=lambda ep: ep_scores[ep], reverse=True)[:keep_best]
        mask = np.isin(episode_ids, best_eps)

        for key in list(data.keys()):
            # Для каждого ключа – заменяем список на «обрезанный» по mask
            values = data[key]
            data[key] = [values[i] for i in range(len(values)) if mask[i]]
        
    raw_features = np.array(
        [[row[f] for f in features_list] for row in data["features"]], 
        dtype=np.float32
    )
    
    if add_board_features: 
        boards = np.array(data['boards']).reshape(-1, 16)
        middle_features = np.array([np.concat([b, f]) for b, f in zip(boards, raw_features)])
    else: 
        middle_features = raw_features.copy()
        
    target = np.asarray(data[target_type], dtype=np.float32) 
    
    features = middle_features.copy()
    # if expand:
    #     actions = np.arange(4)
    #     features_rep = np.repeat(features, 4, axis=0)
    #     act_rep = np.tile(actions, len(data['boards'])).reshape(-1, 1) 
    #     features_expanded = np.hstack([features_rep, act_rep]) 
    #     target_expanded = target.reshape(-1)
        
    #     features = features_expanded.copy()
    #     target = target_expanded.copy()
    
    if expand:
        # Expand dataset: for each board, apply all 4 actions and extract features of resulting board
        boards = data["boards"]
        # original targets assumed shape (N, 4) for Q-values
        original_targets = target.copy()
        new_features = []
        new_targets = []
        for idx, board in enumerate(boards):
            for action in range(4):
                # simulate move to get next board state
                next_board = simulate_move(board, action)
                # extract features for new board
                feats_dict = extract_features(next_board, features_list)
                feats_row = [feats_dict[name] for name in features_list]
                new_features.append(feats_row)
                # q-value for this action from original targets
                if original_targets.ndim > 1:
                    new_targets.append(original_targets[idx][action])
                else:
                    new_targets.append(float(original_targets[idx]))
        features = np.array(new_features, dtype=np.float32)
        target = np.array(new_targets, dtype=np.float32)
        
    # Опциональная нормализация
    if normalize:
        scaler = StandardScaler()
        scaler.fit(features)
    
    if transform == 'divide_by_empty' and 'num_empty' in features_list:
        idx = features_list.index('num_empty')
        count_empty = np.array([f[idx] for f in features])
        
        mask_dict = {
            'easy': (count_empty >= 8), 
            'medium': np.array(count_empty < 8) & np.array(count_empty >= 4), 
            'hard': (count_empty < 4)
        }
        
        res = []
        for mask in mask_dict.values():
            res_i = features[mask]
            target_i = target[mask]
            
            # Опциональная нормализация
            if normalize:
                res_i = scaler.transform(res_i)
            
            res.append((res_i, target_i))
            
        return (res[0], res[1], res[2])
        
    # if transform == 'divide_by_actions':
    #     actions = np.array(data['actions']).reshape(-1, 1)
    #     features = np.array([np.concat([f, a]) for f, a in zip(features, actions)])
        
    #     res = []
    #     for act in range(4):
    #         mask = [f[-1] == act for f in features]
    #         res_i = features[mask]
    #         target_i = target[mask]
            
    #         # Опциональная нормализация
    #         if normalize:
    #             res_i = scaler.transform(res_i)
            
    #         res.append((res_i, target_i))
            
    #     return (res[0], res[1], res[2], res[3])
    
    # features = scaler.transform(features[:, :-1])
    if normalize:
        if transform == 'divide_by_actions':
            features = scaler.transform(features[:, :-1])
        else:
            features = scaler.transform(features)
    return features, target