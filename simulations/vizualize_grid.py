import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython import display

# расшифровка доступных действий  
DIRECTIONS = {
    0: 'UP', 
    1: 'RIGHT', 
    2: 'DOWN', 
    3: 'LEFT', 
    'None': 'None'
}
COLORS = {
    0: '#CCC0B3',
    2: '#EEE4DA',
    4: '#EDE0C8',
    8: '#F2B179',
    16: '#F59563',
    32: '#F67C5F',
    64: '#F65E3B',
    128: '#EDCF72',
    256: '#EDCC61',
    512: '#EDC850',
    1024: '#EDC53F',
    2048: '#EDC22E',
}

def get_tile_value(observation):
    """Преобразует observation среды в числовое представление игрового поля"""
    if observation is None:
        return np.zeros((4, 4), dtype=int)
    
    board = np.zeros((4, 4), dtype=int)
    for i in range(1, min(16, observation.shape[2])):
        positions = np.where(observation[:, :, i] == 1)
        for j in range(len(positions[0])):
            row, col = positions[0][j], positions[1][j]
            board[row, col] = 2 ** i
    return board

def display_game(observation, action=None, step_num=0, prev_score=0, curr_score=0, ax=None, clear_output=True):
    """Визуализирует игровое поле с информацией о действии и счете"""
    # Создаем упорядоченный список цветов
    color_keys = sorted(COLORS.keys())
    color_list = [COLORS[key] for key in color_keys]
    cmap = ListedColormap(color_list)
    
    board = get_tile_value(observation)
    
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
        fig.subplots_adjust(wspace=0.3)
    else:
        fig = ax[0].figure
        ax[0].clear()
        ax[1].clear()

    # Нормализация значений
    norm = plt.Normalize(0, len(color_list)-1)
    
    # Создаем матрицу индексов для цветов
    board_indices = np.zeros_like(board, dtype=int)
    for idx, key in enumerate(color_keys):
        board_indices[board == key] = idx
    
    ax[0].imshow(board_indices, cmap=cmap, norm=norm)
    
    # Добавление значений в ячейки
    for i in range(4):
        for j in range(4):
            if board[i, j] > 0:
                text_color = 'white' if board[i, j] > 8 else 'black'
                ax[0].text(j, i, str(board[i, j]), 
                          ha='center', va='center', 
                          fontsize=20, color=text_color)
    
    ax[0].grid(True, color='black', linewidth=1.5)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Игровое поле 2048')
    
    # Информационная панель (остается без изменений)
    action_names = {0: "↑ ВВЕРХ", 1: "→ ВПРАВО", 2: "↓ ВНИЗ", 3: "← ВЛЕВО"}
    action_text = f"Шаг {step_num}" + (
        "\nНачальное состояние" if action is None else f"\nПоследнее действие: {action_names.get(action, 'Неизвестно')}"
    ) # + (
    #     f"\nДопустимые действия: {valid_actions}"
    # )
    score_text = f"Текущий счет: {curr_score}\nПрирост: +{curr_score - prev_score}"
    
    ax[1].axis('off')
    ax[1].text(0.5, 0.7, action_text, 
              ha='center', va='center', 
              fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgrey'))
    ax[1].text(0.5, 0.3, score_text, 
              ha='center', va='center', 
              fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
    
    score_text = f"Счет: {curr_score}\n(+{curr_score - prev_score})"
    
    ax[1].axis('off')
    ax[1].text(0.5, 0.7, action_text,
              ha='center', va='center',
              fontsize=14,
              color='#776E65',
              bbox=dict(boxstyle="round,pad=0.4", facecolor='#EEE4DA', edgecolor='#BBADA0'))
    
    if clear_output:
        display.clear_output(wait=True)
    display.display(fig)
    plt.close(fig)