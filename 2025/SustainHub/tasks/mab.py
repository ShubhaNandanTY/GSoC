
#Multi-Armed-Bandits for Task Allocation

import numpy as np
from scipy.stats import beta

class MABAllocator:
    def __init__(self, agents, exploration_factor=0.2):
        self.agents = agents
        self.exploration_factor = exploration_factor

    @staticmethod
    def _is_available(agent):
        # Agents that dropped out have task_load reset to 0, so check inactive as well
        return agent.task_load < agent.max_load and not getattr(agent, "inactive", False)

    def select_agent(self, task):
        task_type = task.task_type

        # Exploration
        if np.random.random() < self.exploration_factor:
            available_agents = [a for a in self.agents if self._is_available(a)]
            if available_agents:
                return np.random.choice(available_agents)
            return None

        # Exploitation (Thompson Sampling)
        max_sample = -1
        best_agent = None

        for agent in self.agents:
            if not self._is_available(agent):
                continue

            task_idx = agent._task_type_to_idx(task_type)
            successes = agent.success_counts[task_idx]
            failures = agent.total_counts[task_idx] - successes

            sample = beta.rvs(successes + 1, failures + 1)

            if sample > max_sample:
                max_sample = sample
                best_agent = agent

        return best_agent
