import threading

import pytest

from simulation.simulation import Simulation
from tasks.generator import Task
from tasks.mab import MABAllocator


@pytest.mark.parametrize("exploration_factor", [0.0, 1.0])
def test_allocator_skips_inactive_agents(exploration_factor):
    sim = Simulation(agent_count=4)
    for agent in sim.agents[:3]:
        agent.inactive = True
    allocator = MABAllocator(sim.agents, exploration_factor=exploration_factor)

    for _ in range(50):
        assert allocator.select_agent(Task("bug", 1)) is sim.agents[3]


@pytest.mark.parametrize("exploration_factor", [0.0, 1.0])
def test_allocator_returns_none_when_everyone_dropped_out(exploration_factor):
    sim = Simulation(agent_count=3)
    for agent in sim.agents:
        agent.inactive = True

    assert MABAllocator(sim.agents, exploration_factor).select_agent(Task("docs", 1)) is None


def test_assign_tasks_returns_when_all_agents_dropped_out():
    sim = Simulation(agent_count=2)
    for agent in sim.agents:
        agent.inactive = True
    sim.task_queue = [Task("bug", 1), Task("feature", 1)]

    # assign_tasks used to spin forever here; run it in a daemon thread so a regression fails instead of hanging
    worker = threading.Thread(target=sim.assign_tasks, daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert sim.task_queue == []


def test_dropout_count_larger_than_active_agents():
    sim = Simulation(agent_count=3, dropouts_per_step=2)
    sim.simulate_dropout(count=2)

    sim.simulate_dropout(count=2)  # only one agent is still active

    assert all(agent.inactive for agent in sim.agents)
