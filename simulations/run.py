import gymnasium as gym

from .agents import RandomAgent, run_episode

def configure_simulation(env_params=None):
    """Конфигурация параметров симуляции"""
    env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", **(env_params or {}))
    return env

def run_simulation(agent_type='random', delay=0.5, max_steps=1000, env_params=None):
    """
    Запуск симуляции с выбранными параметрами
    
    Параметры:
    agent_type: 'random', 'human' или пользовательский агент
    delay: задержка между шагами (в секундах)
    max_steps: максимальное количество шагов
    env_params: параметры среды
    """
    env = configure_simulation(env_params)
    
    agents = {
        'random': RandomAgent(env),
    }
    
    if agent_type not in agents:
        raise ValueError(f"Неизвестный тип агента: {agent_type}. Доступные варианты: {list(agents.keys())}")
    
    return run_episode(
        env=env,
        agent=agents[agent_type],
        max_steps=max_steps,
        delay=delay,
        visualize=True
    )