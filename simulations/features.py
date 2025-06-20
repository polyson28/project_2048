import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import convolve2d
from typing import Dict, Any, List

all = [
    "extract_features",
    "feature_names",
]

EDGE_MASK = np.ones((4, 4), dtype=bool)
EDGE_MASK[1:3, 1:3] = False  # only outer ring is True

# Helper – logarithm base‑2 that gracefully handles zeros
log2 = np.vectorize(lambda x: np.log2(x) if x > 0 else 0.0)


def max_tile(board: np.ndarray) -> float:
    """Largest tile value on the board."""
    return float(board.max())


def second_max_tile(board: np.ndarray) -> float:
    """Second‑largest distinct tile value (0 if none)."""
    uniques = np.unique(board)
    if uniques.size < 2:
        return 0.0
    return float(sorted(uniques)[-2])


def num_empty(board: np.ndarray) -> float:
    """Number of empty (zero) cells."""
    return float(np.count_nonzero(board == 0))


def empty_ratio(board: np.ndarray) -> float:
    """Proportion of empty cells (0–1)."""
    return num_empty(board) / 16.0


def tile_sum(board: np.ndarray) -> float:
    """Sum of all tile values (game score proxy)."""
    return float(board.sum())


def log_sum(board: np.ndarray) -> float:
    """Sum of log₂(tile) values – smoother growth than raw sum."""
    return float(log2(board).sum())


def max_tile_ratio(board: np.ndarray) -> float:
    """max_tile / tile_sum – how dominant the biggest tile is."""
    total = board.sum()
    return float(board.max() / total) if total else 0.0


def corner_max(board: np.ndarray) -> float:
    """1 if the highest tile sits in a corner, else 0."""
    r, c = np.unravel_index(board.argmax(), board.shape)
    return 1.0 if (r in {0, 3} and c in {0, 3}) else 0.0


def corner_sum(board: np.ndarray) -> float:
    """Total value of the four corner cells."""
    return float(board[0, 0] + board[0, 3] + board[3, 0] + board[3, 3])


def edge_occupancy(board: np.ndarray) -> float:
    """Number of occupied edge cells (outer ring)."""
    return float(np.count_nonzero(board[EDGE_MASK] > 0))


def smoothness(board: np.ndarray) -> float:
    """Negative sum of |log₂ diff| for all adjacent pairs (higher is smoother)."""
    diff_h = np.abs(log2(board[:, :-1]) - log2(board[:, 1:]))
    diff_v = np.abs(log2(board[:-1, :]) - log2(board[1:, :]))
    # Empty cells (0) were turned into 0 by log2() – treat them as smooth
    return -float(diff_h.sum() + diff_v.sum())


def monotonicity(board: np.ndarray) -> float:
    """Higher when rows & cols are consistently increasing or decreasing."""
    mono_row = 0.0
    for row in board:
        diffs = np.diff(log2(row))
        mono_row += np.sum(diffs >= 0) ** 2 + np.sum(diffs <= 0) ** 2
    mono_col = 0.0
    for col in board.T:
        diffs = np.diff(log2(col))
        mono_col += np.sum(diffs >= 0) ** 2 + np.sum(diffs <= 0) ** 2
    return float(mono_row + mono_col)


def entropy(board: np.ndarray) -> float:
    """Shannon entropy of tile distribution (zero‑masked)."""
    tiles = board[board > 0].flatten()
    if tiles.size == 0:
        return 0.0
    values, counts = np.unique(tiles, return_counts=True)
    probs = counts / counts.sum()
    return -float(np.sum(probs * np.log2(probs)))


def std_dev(board: np.ndarray) -> float:
    """Standard deviation of tile values (measure of spread)."""
    return float(board.std())


def high_value_tiles(board: np.ndarray, threshold: int = 128) -> float:
    """Count tiles ≥ *threshold* (default 128)."""
    return float(np.count_nonzero(board >= threshold))


def cluster_score(board: np.ndarray) -> float:
    """Sum of |value diff| between each tile and its neighbours (lower is better)."""
    score = 0.0
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                continue
            v = board[i, j]
            for (ni, nj) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] != 0:
                    score += abs(v - board[ni, nj])
    return float(score)


def potential_merge_chains(board: np.ndarray) -> float:
    """Count potential merge chains in the board."""
    # This is similar to potential_merges but with different interpretation
    merges = 0
    # horizontal pairs
    merges += np.sum(board[:, :-1] == board[:, 1:])
    # vertical pairs
    merges += np.sum(board[:-1, :] == board[1:, :])
    return float(merges)


def corner_stability(board: np.ndarray) -> float:
    """Measure how stable the largest tile is in the corner."""
    r, c = np.unravel_index(board.argmax(), board.shape)
    if (r in {0, 3} and c in {0, 3}):
        # Check if the largest tile is in a corner and surrounded by smaller tiles
        corner_value = board[r, c]
        neighbors = []
        for (nr, nc) in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if 0 <= nr < 4 and 0 <= nc < 4:
                neighbors.append(board[nr, nc])
        stability = sum(1 for n in neighbors if n < corner_value) / len(neighbors) if neighbors else 0
        return float(stability)
    return 0.0


def monotonic_consistency(board: np.ndarray) -> float:
    """Check monotonic row/column consistency."""
    def is_monotonic(arr):
        return all(x <= y for x, y in zip(arr, arr[1:])) or all(x >= y for x, y in zip(arr, arr[1:]))

    score = 0
    for row in board:
        if is_monotonic(row):
            score += 1
    for col in board.T:
        if is_monotonic(col):
            score += 1
    return float(score) / 8.0  # Normalize by total rows + cols


def snake_weighted_sum(board: np.ndarray) -> float:
    """
    Стратегия "змейки" - высокие значения в левом верхнем углу, 
    затем зигзагообразно распределяются по доске.
    
    Значения в левом верхнем углу имеют наибольший вес, затем вес
    экспоненциально уменьшается по змеевидному пути.
    """
    # Генерируем веса программно вместо хардкода
    weights = np.zeros((4, 4), dtype=float)
    
    # Заполняем веса для змеевидного пути:
    # Ряд 1: слева направо
    weights[0, 0] = 2**15  # 32768
    weights[0, 1] = 2**14  # 16384
    weights[0, 2] = 2**13  # 8192
    weights[0, 3] = 2**12  # 4096
    
    # Ряд 2: справа налево
    weights[1, 3] = 2**11  # 2048
    weights[1, 2] = 2**10  # 1024
    weights[1, 1] = 2**9   # 512
    weights[1, 0] = 2**8   # 256
    
    # Ряд 3: слева направо
    weights[2, 0] = 2**7   # 128
    weights[2, 1] = 2**6   # 64
    weights[2, 2] = 2**5   # 32
    weights[2, 3] = 2**4   # 16
    
    # Ряд 4: справа налево
    weights[3, 3] = 2**3   # 8
    weights[3, 2] = 2**2   # 4
    weights[3, 1] = 2**1   # 2
    weights[3, 0] = 2**0   # 1
    
    max_possible = (weights * 65536).sum()       # 65536 – теоретический максимум тайла
    return float((board * weights).sum() / max_possible)


def corner_weighted_sum(board: np.ndarray) -> float:
    """
    Придаем больший вес углам, меньший вес краям, и наименьший вес центру.
    Веса генерируются на основе расстояния от точки до ближайшего угла.
    """
    weights = np.zeros((4, 4), dtype=float)
    
    # Для каждой клетки вычисляем расстояние до ближайшего угла
    for i in range(4):
        for j in range(4):
            # Расстояние до четырех углов
            d1 = abs(i - 0) + abs(j - 0)  # верхний левый
            d2 = abs(i - 0) + abs(j - 3)  # верхний правый
            d3 = abs(i - 3) + abs(j - 0)  # нижний левый
            d4 = abs(i - 3) + abs(j - 3)  # нижний правый
            
            # Минимальное расстояние до угла
            min_dist = min(d1, d2, d3, d4)
            
            # Преобразуем расстояние в вес: ближе к углу = выше вес
            weights[i, j] = 1.0 / (1.0 + min_dist)
    
    max_possible = 65536 * weights.max() * 4     # 4 угла по максимуму
    return float((board * weights).sum() / max_possible)


# 1) Horizontal & vertical signed gradient (monotone direction)
H_GRAD_KERNEL = np.array([[1, -1, 0, 0]])
V_GRAD_KERNEL = H_GRAD_KERNEL.T


def conv_horiz_gradient(board: np.ndarray) -> float:
    """Sum of log₂‑value horizontal gradients: >0 means rows generally increase left→right."""
    log_b = log2(board)
    conv = convolve2d(log_b, H_GRAD_KERNEL, mode="valid")
    return float(conv.sum())


def conv_vert_gradient(board: np.ndarray) -> float:
    """Sum of log₂‑value vertical gradients: >0 means columns increase top→bottom."""
    log_b = log2(board)
    conv = convolve2d(log_b, V_GRAD_KERNEL, mode="valid")
    return float(conv.sum())

# 2) Counts of 2×2 uniform blocks (all four equal & non‑zero)

BLOCK2_KERNEL = np.ones((2, 2), dtype=int)


def conv_2x2_same_tiles(board: np.ndarray) -> float:
    """Number of 2×2 sub‑grids fully filled with the **same** non‑zero tile value."""
    same_tiles = 0
    for i in range(3):
        for j in range(3):
            sub = board[i : i + 2, j : j + 2]
            if sub[0, 0] != 0 and np.all(sub == sub[0, 0]):
                same_tiles += 1
    return float(same_tiles)


def conv_3_in_row_same_tiles(board: np.ndarray) -> float:
    """Counts horizontal length‑3 sequences with equal non‑zero tiles."""
    cnt = 0
    for i in range(4):
        for j in range(2):
            slice_ = board[i, j : j + 3]
            if slice_[0] != 0 and np.all(slice_ == slice_[0]):
                cnt += 1
    return float(cnt)


def conv_3_in_col_same_tiles(board: np.ndarray) -> float:
    """Counts vertical length‑3 sequences with equal non‑zero tiles."""
    cnt = 0
    for i in range(2):
        for j in range(4):
            slice_ = board[i : i + 3, j]
            if slice_[0] != 0 and np.all(slice_ == slice_[0]):
                cnt += 1
    return float(cnt)


# Обновляем словарь feature_funcs с новыми функциями
feature_funcs = {
    "max_tile": max_tile,
    "second_max_tile": second_max_tile,
    "num_empty": num_empty,
    "empty_ratio": empty_ratio,
    "tile_sum": tile_sum,
    "log_sum": log_sum,
    "max_tile_ratio": max_tile_ratio,
    "corner_max": corner_max,
    "corner_sum": corner_sum,
    "edge_occupancy": edge_occupancy,
    "potential_merges": potential_merge_chains,
    "smoothness": smoothness,
    "monotonicity": monotonicity,
    "snake_weighted_sum": snake_weighted_sum,
    "corner_weighted_sum": corner_weighted_sum,
    "entropy": entropy,
    "std_dev": std_dev,
    "high_value_tiles_128+": high_value_tiles,
    "cluster_score": cluster_score,
    "conv_horiz_gradient": conv_horiz_gradient,
    "conv_vert_gradient": conv_vert_gradient,
    "conv_2x2_same_tiles": conv_2x2_same_tiles,
    "conv_3_in_row_same_tiles": conv_3_in_row_same_tiles,
    "conv_3_in_col_same_tiles": conv_3_in_col_same_tiles,
}

feature_names = list(feature_funcs.keys())

def extract_features(board: np.ndarray, features: List[str]=feature_names) -> Dict[str, Any]:
    """Compute *all* interpretable features for the given board.
    
    Parameters
    ----------
    board : np.ndarray, shape (4, 4)
        Numerical board produced by ``get_tile_value``.
        
    Returns
    -------
    dict[str, float]
        Mapping *feature name → value*.
    """
    get_feature_funcs = {val: feature_funcs[val] for val in features}
    return {name: func(board) for name, func in get_feature_funcs.items()}


class MetricsRecorder:
    """Класс для записи и анализа метрик производительности агента"""
    
    def __init__(self, save_dir="metrics"):
        """Инициализация рекордера метрик"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Метрики
        self.scores = []          # игровые очки
        self.max_tiles = []       # максимальная плитка в игре
        self.steps = []           # длительность эпизодов
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def record(self, score, max_tile, episode_length):
        """Запись метрик одного эпизода"""
        self.scores.append(score)
        self.max_tiles.append(max_tile)
        self.steps.append(episode_length)
    
    def save(self):
        """Сохранение метрик в файл"""
        filename = f"{self.save_dir}/metrics_{self.timestamp}.npz"
        np.savez(
            filename,
            scores=np.array(self.scores),
            max_tiles=np.array(self.max_tiles),
            steps=np.array(self.steps)
        )
        
        # Сохраняем также текстовый отчет
        with open(f"{self.save_dir}/report_{self.timestamp}.txt", 'w') as f:
            f.write(f"Всего эпизодов: {len(self.scores)}\n")
            f.write(f"Средний счет: {np.mean(self.scores):.2f}\n")
            f.write(f"Максимальный счет: {np.max(self.scores)}\n")
            f.write(f"Средняя максимальная плитка: {np.mean(self.max_tiles):.2f}\n")
            
            # Распределение максимальных плиток
            unique_tiles, counts = np.unique(self.max_tiles, return_counts=True)
            f.write("\nРаспределение максимальных плиток:\n")
            for tile, count in zip(unique_tiles, counts):
                percentage = count / len(self.max_tiles) * 100
                f.write(f"  {int(tile)}: {count} ({percentage:.2f}%)\n")
    
    def plot(self, window=100):
        """Построение графиков метрик с использованием скользящего среднего"""
        plt.figure(figsize=(15, 10))
        
        # Скользящее среднее
        def moving_average(data, window_size):
            if len(data) < window_size:
                return np.array(data)
            cumsum = np.cumsum(np.insert(data, 0, 0)) 
            return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        
        # График счета
        plt.subplot(2, 2, 1)
        plt.plot(self.scores, alpha=0.3, label='Данные по эпизодам')
        if len(self.scores) >= window:
            avg_scores = moving_average(self.scores, window)
            plt.plot(range(window-1, window-1+len(avg_scores)), 
                    avg_scores, label=f'Среднее за {window} игр')
        plt.title('Счет')
        plt.xlabel('Эпизод')
        plt.ylabel('Счет')
        plt.legend()
        
        # График максимальной плитки
        plt.subplot(2, 2, 2)
        plt.plot(self.max_tiles, alpha=0.3, label='Данные по эпизодам')
        if len(self.max_tiles) >= window:
            avg_tiles = moving_average(self.max_tiles, window)
            plt.plot(range(window-1, window-1+len(avg_tiles)), 
                    avg_tiles, label=f'Среднее за {window} игр')
        plt.title('Максимальная плитка')
        plt.xlabel('Эпизод')
        plt.ylabel('Значение')
        plt.legend()
        
        # График длительности эпизодов
        plt.subplot(2, 2, 3)
        plt.plot(self.steps, alpha=0.3, label='Данные по эпизодам')
        if len(self.steps) >= window:
            avg_steps = moving_average(self.steps, window)
            plt.plot(range(window-1, window-1+len(avg_steps)), 
                    avg_steps, label=f'Среднее за {window} игр')
        plt.title('Длительность эпизодов')
        plt.xlabel('Эпизод')
        plt.ylabel('Шаги')
        plt.legend()
        
        # Гистограмма максимальных плиток
        plt.subplot(2, 2, 4)
        unique_tiles, counts = np.unique(self.max_tiles, return_counts=True)
        plt.bar([str(int(t)) for t in unique_tiles], counts)
        plt.title('Распределение максимальных плиток')
        plt.xlabel('Значение плитки')
        plt.ylabel('Количество эпизодов')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/metrics_plot_{self.timestamp}.png")
        plt.show()
