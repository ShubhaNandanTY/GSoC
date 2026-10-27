import random

import numpy as np
import pytest

import simulation.simulation as simulation_module
from simulation.simulation import Simulation
from tasks.generator import Task


@pytest.fixture(autouse=True)
def no_plots_or_saves(monkeypatch):
    monkeypatch.setattr(simulation_module, "plot_sarsa_agents", lambda *args, **kwargs: None)
    monkeypatch.setattr(Simulation, "save_agents", lambda self: None)


def test_new_tasks_without_dropouts_are_not_reassignments():
    sim = Simulation(agent_count=3)
    sim.task_queue = [Task("bug", 1), Task("docs", 1)]

    sim.assign_tasks()

    assert sim.total_reassigned_tasks_step == 0
    assert all(getattr(agent, "reassigned_tasks", 0) == 0 for agent in sim.agents)


def test_task_taking_over_dropped_task_counts_once():
    sim = Simulation(agent_count=3)
    sim.dropped_task_log = [
        {"task_type": "docs", "original_agent": "C9", "reassigned_to": None, "completion_status": None}
    ]
    sim.task_queue = [Task("docs", 1), Task("docs", 1), Task("bug", 1)]

    sim.assign_tasks()

    assert sim.total_reassigned_tasks_step == 1
    assert sum(getattr(agent, "reassigned_tasks", 0) for agent in sim.agents) == 1
    assert sim.dropped_task_log[0]["completion_status"] == "Pending"


@pytest.mark.parametrize("dropouts_per_step", [0, 1])
def test_metrics_stay_in_documented_range(dropouts_per_step):
    random.seed(3)
    np.random.seed(3)
    sim = Simulation(agent_count=15, tasks_per_step=3, dropouts_per_step=dropouts_per_step)

    sim.run(steps=10)

    assert all(0 <= ro <= 1 for ro in sim.ro_history)
    assert all(0 <= rq <= 1 for rq in sim.rq_history)
    if dropouts_per_step == 0:
        assert sim.ro_history == [0.0] * 10
