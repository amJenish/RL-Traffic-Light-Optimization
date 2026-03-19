from .base import BasePolicy

def get_dqn():
    from .dqn import DQNPolicy
    return DQNPolicy