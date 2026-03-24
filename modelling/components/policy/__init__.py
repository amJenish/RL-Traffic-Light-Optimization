from .base import BasePolicy

def get_dqn():
    from .dqn import DQNPolicy
    return DQNPolicy

def get_double_dqn():
    from .double_dqn import DoubleDQNPolicy
    return DoubleDQNPolicy