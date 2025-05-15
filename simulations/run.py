import gymnasium as gym
import matplotlib.pyplot as plt 
import time
from typing import Callable
import numpy as np 

from .vizualize_grid import display_game
from .features import extract_features, feature_names

# def run_episode(
#     env,
#     agent,
#     *,
#     max_steps: int | None = None,
#     delay: float = 0.25,
#     visualize: bool = True
# ) -> int:
#     """
#     Запускает одну партию 2048 и корректно отображает накопленный счёт.

#     Если max_steps=None (по умолчанию), игра идёт до завершения среды.
#     Если указан max_steps, партия прерывается не позже этого количества ходов.
#     """
#     obs, _ = env.reset()
#     cum_score = 0
#     step = 0
#     done = False         
    
#     # первая отрисовка
#     if visualize:
#         fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#         display_game(obs, ax=ax)

#     while True:
#         action = agent.act(obs)
#         obs, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#         prev_score = cum_score
#         cum_score += int(reward)      # <‑‑ избавляемся от numpy‑типов

#         if visualize:
#             display_game(
#                 obs,
#                 action=action,
#                 step_num=step,
#                 prev_score=prev_score,
#                 curr_score=cum_score,
#                 ax=ax,
#             )
#             time.sleep(delay)

#         step += 1
#         # проверяем условия выхода
#         if done:
#             break
#         if max_steps is not None and step >= max_steps:
#             break

#     # -- ФИНАЛ --
#     if visualize:
#         # подчёркиваем, что это финальное состояние (повторно выводить поле не обязательно)
#         print("\n===== Финал =====")
#         if done:
#             print("Игра завершена (ходов сыграно:", step, ")")
#         else:
#             print(f"Прервано по лимиту {max_steps} ходов (сыграно: {step})")
#         print(f"Итоговый счёт: {cum_score}")

#     return cum_score


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
        delay: float = 0.25,
        visualize: bool = True,
        on_illegal: str | Callable[[np.ndarray], int] = "random",
    ) -> None:
        self.env = env
        self.agent = agent
        self.delay = delay
        self.visualize = visualize
        self.on_illegal = on_illegal

    # ---------- helpers ----------

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

    # ---------- main loop ----------

    # def run(self, max_steps: int | None = None) -> int:
    #     obs, _ = self.env.reset()
    #     cum_score = 0
    #     step = 0
    #     done = False

    #     # первая отрисовка
    #     if self.visualize:
    #         fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #         display_game(obs, ax=ax)

    #     while not done and (max_steps is None or step < max_steps):
    #         raw_action = self.agent.act(obs)
    #         action = self._resolve_action(obs, raw_action)

    #         obs, reward, terminated, truncated, _ = self.env.step(action)
    #         done = terminated or truncated

    #         prev_score = cum_score
    #         cum_score += int(reward)

    #         if self.visualize:
    #             display_game(
    #                 obs,
    #                 action=action,
    #                 step_num=step,
    #                 prev_score=prev_score,
    #                 curr_score=cum_score,
    #                 ax=ax,
    #             )
    #             time.sleep(self.delay)

    #         step += 1

    #     if self.visualize:
    #         print("\n===== Финал =====")
    #         if done:
    #             print(f"Игра завершена (ходов сыграно: {step})")
    #         else:
    #             print(f"Прервано по лимиту {max_steps} ходов (сыграно: {step})")
    #         print(f"Итоговый счёт: {cum_score}")

    #     return cum_score
    
    def run(self, max_steps: int | None = None) -> int:
        obs, _ = self.env.reset()
        cum_score = 0
        step = 0
        done = False
        
        # первая отрисовка
        if self.visualize:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            display_game(obs, ax=ax)
        
        while not done and (max_steps is None or step < max_steps):
            # Получаем действие от агента
            raw_action = self.agent.act(obs)
            action = self._resolve_action(obs, raw_action)
            
            # Сохраняем состояние до выполнения действия
            prev_obs = obs.copy()
            
            # Выполняем действие
            obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Проверяем, изменилось ли состояние после действия
            if np.array_equal(prev_obs, obs):
                # Действие не изменило состояние - неэффективный ход
                mask = self._get_legit_mask()
                mask[action] = False  # Исключаем неэффективное действие
                
                # Если остались другие легальные действия
                if np.any(mask):
                    new_action = int(np.random.choice(np.flatnonzero(mask)))
                    # Пробуем новое действие
                    obs, reward, terminated, truncated, _ = self.env.step(new_action)
                    action = new_action  # Обновляем для визуализации
                else:
                    # Нет действий, которые изменили бы состояние - игра должна завершиться
                    terminated = True
            
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



# ---- совместимость со старым API ----

def run_episode(
    env,
    agent,
    *,
    max_steps: int | None = None,
    delay: float = 0.25,
    visualize: bool = True,
    on_illegal: str | Callable[[np.ndarray], int] = "random",
) -> int:
    """Обёртка, сохраняющая старый интерфейс ``run_episode``."""
    return EpisodeRunner(
        env,
        agent,
        delay=delay,
        visualize=visualize,
        on_illegal=on_illegal,
    ).run(max_steps=max_steps)
