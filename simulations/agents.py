import gymnasium as gym
import time
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from random import random

from .vizualize_grid import display_game, get_tile_value


class BaseAgent:
    """Базовый класс для агентов"""
    def __init__(self, env):
        self.env = env
        
    def act(self, observation):
        raise NotImplementedError

class RandomAgent(BaseAgent):
    """Агент со случайными действиями"""
    def act(self, observation):
        return self.env.action_space.sample()
    
def run_episode(env, agent, max_steps=1000, delay=0.5, visualize=True):
    """Запускает эпизод игры с указанным агентом"""
    observation, _ = env.reset()
    total_reward = 0
    prev_score = 0
    done = False
    fig = None
    
    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        display_game(observation, ax=ax)
    
    for step in range(max_steps+1):
        # Получаем действие от агента
        action = agent.act(observation)
        
        # Сохраняем текущее состояние доски
        prev_observation = observation.copy()
        
        # Применяем действие
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Проверяем, изменилось ли состояние доски
        if np.array_equal(prev_observation, observation) and not done:
            # Пробуем другие действия, пока не найдем то, которое изменит доску
            actions_to_try = list(range(env.action_space.n))
            actions_to_try.remove(action)  # Удаляем исходное действие
            random.shuffle(actions_to_try)  # Пробуем в случайном порядке
            
            for alt_action in actions_to_try:
                # Пробуем это действие
                alt_observation, alt_reward, alt_terminated, alt_truncated, _ = env.step(alt_action)
                alt_done = alt_terminated or alt_truncated
                
                # Проверяем, изменилось ли состояние доски
                if not np.array_equal(observation, alt_observation) or alt_done:
                    # Это действие сработало, обновляем все переменные
                    observation = alt_observation
                    reward = alt_reward
                    terminated = alt_terminated
                    truncated = alt_truncated
                    done = alt_done
                    action = alt_action  # Обновляем для визуализации
                    break
        
        total_reward += reward
        
        if visualize:
            display_game(
                observation, 
                action=action,
                step_num=step,
                prev_score=prev_score,
                curr_score=total_reward,
                ax=ax
            )
            time.sleep(delay)
            
        prev_score = total_reward
        if done:
            break
            
    if visualize:
        print("Игра завершена!" if terminated else "Достигнут лимит шагов")
        print(f"Итоговый счет: {total_reward}")
        
    return total_reward

# def run_episode(env, agent, max_steps=1000, delay=0.5, visualize=True):
#     """Запускает эпизод игры с указанным агентом"""
#     observation, _ = env.reset()
#     total_reward = 0
#     prev_score = 0
#     done = False
#     fig = None
    
#     if visualize:
#         fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#         display_game(observation, ax=ax)
    
#     for step in range(max_steps+1):
#         action = agent.act(observation)
#         observation, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         total_reward += reward
        
#         if visualize:
#             display_game(
#                 observation, 
#                 action=action,
#                 step_num=step,
#                 prev_score=prev_score,
#                 curr_score=total_reward,
#                 ax=ax
#             )
#             time.sleep(delay)
            
#         prev_score = total_reward
#         if done:
#             break
            
#     if visualize:
#         print("Игра завершена!" if terminated else "Достигнут лимит шагов")
#         print(f"Итоговый счет: {total_reward}")
        
#     return total_reward



# ------------------------------------------------------------------

# class BaseAgent:
#     """Базовый класс для агентов"""
#     def __init__(self, env):
#         self.env = env
        
#     def _simulate_move(self, board, direction):
#         """Симулирует движение без изменения оригинала"""
#         temp_board = board.copy()
#         changed = False
        
#         if direction == 0:  # Вверх
#             for j in range(4):
#                 col = temp_board[:, j]
#                 new_col, col_changed = self._process_column(col)
#                 if col_changed: 
#                     changed = True
#                     temp_board[:, j] = new_col
                    
#         elif direction == 1:  # Вправо
#             for i in range(4):
#                 row = temp_board[i, :][::-1]
#                 new_row, row_changed = self._process_row(row)
#                 if row_changed:
#                     changed = True
#                     temp_board[i, :] = new_row[::-1]
                    
#         elif direction == 2:  # Вниз
#             for j in range(4):
#                 col = temp_board[:, j][::-1]
#                 new_col, col_changed = self._process_column(col)
#                 if col_changed:
#                     changed = True
#                     temp_board[:, j] = new_col[::-1]
                    
#         elif direction == 3:  # Влево
#             for i in range(4):
#                 row = temp_board[i, :]
#                 new_row, row_changed = self._process_row(row)
#                 if row_changed:
#                     changed = True
#                     temp_board[i, :] = new_row
                    
#         return temp_board, changed

#     def _process_row(self, row):
#         """Обработка ряда с учетом слияний"""
#         new_row = [x for x in row if x != 0]
#         changed = len(new_row) != len(row)
        
#         i = 0
#         while i < len(new_row)-1:
#             if new_row[i] == new_row[i+1]:
#                 new_row[i] *= 2
#                 new_row.pop(i+1)
#                 new_row.append(0)
#                 changed = True
#             i += 1
            
#         new_row += [0]*(4 - len(new_row))
#         return np.array(new_row), changed

#     def _process_column(self, col):
#         """Обработка колонки (аналогично рядам)"""
#         return self._process_row(col)

#     # def get_valid_actions(self, board):
#     #     """Возвращает допустимые действия с учетом ТОЛЬКО сдвига"""
#     #     valid = []
#     #     for action in range(4):
#     #         # Подсчитываем ненулевые элементы до хода
#     #         count_before = np.count_nonzero(board)
            
#     #         # Симулируем ход
#     #         temp_board, _ = self._simulate_move(board.copy(), action)
            
#     #         # Подсчитываем ненулевые элементы после хода
#     #         count_after = np.count_nonzero(temp_board)
            
#     #         # Действие считается допустимым, если число ненулевых элементов уменьшилось
#     #         if count_after < count_before:
#     #             valid.append(action)
        
#     #     return np.array(valid)
    
#     def get_valid_actions(self, board):
#         """Возвращает допустимые действия с учетом слияний"""
#         valid = []
#         for action in range(4):
#             # Симулируем ход
#             temp_board, changed = self._simulate_move(board.copy(), action)
            
#             # Действие считается допустимым, если доска изменилась и произошло слияние
#             if changed and np.count_nonzero(temp_board) < np.count_nonzero(board):
#                 valid.append(action)
        
#         return np.array(valid)
        
#     def act(self, observation):
#         raise NotImplementedError

# class RandomAgent(BaseAgent):
#     """Агент со случайными действиями"""
#     def act(self, observation):
#         board = get_tile_value(observation)  # Используем функцию из visualize_grid
#         valid_actions = self.get_valid_actions(board)
        
#         if valid_actions.size == 0:
#             raise ValueError("Нет допустимых действий")
            
#         return np.random.choice(valid_actions)

# def run_episode(env, agent, max_steps=1000, delay=0.5, visualize=True):
#     """Запускает эпизод игры с указанным агентом"""
#     observation, _ = env.reset()
#     total_reward = 0
#     prev_score = 0
#     done = False
#     fig = None
#     step_cnt = 0
#     terminated = False
#     truncated = False
    
#     if visualize:
#         fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#         display_game(observation, valid_actions=range(4), ax=ax, step_num=0)
    
#     while step_cnt <= max_steps:
#         board = get_tile_value(observation)
#         valid_actions = agent.get_valid_actions(board)
        
#         print(valid_actions)
        
#         if valid_actions.size == 0:
#             print('Игра завершена')
#             break
        
#         action = agent.act(observation)
    
#         prev_board = board.copy()
#         observation, reward, terminated, truncated, _ = env.step(action)
        
#         new_board = get_tile_value(observation)
#         if np.array_equal(prev_board, new_board):
#             print(f'Недопустимое действие {action}, шаг {step_cnt}')
#             continue
        
#         done = terminated or truncated
#         total_reward += reward
        
#         if visualize:
#             display_game(
#                 observation, 
#                 action=action,
#                 step_num=step_cnt,
#                 prev_score=prev_score,
#                 curr_score=total_reward,
#                 valid_actions=valid_actions, 
#                 ax=ax
#             )
#             time.sleep(delay)
            
#         prev_score = total_reward
#         if done:
#             break
#         step_cnt += 1
            
#     if visualize:
#         print("Игра завершена!" if terminated else "Достигнут лимит шагов")
#         print(f"Итоговый счет: {total_reward}")
        
#     return total_reward