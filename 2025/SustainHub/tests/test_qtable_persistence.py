import json
import random

import numpy as np
import pytest

import simulation.simulation as simulation_module
from agents.sarsa import SARSAAgentLogic
from simulation.simulation import Simulation

# Captured at import, before the autouse fixture in conftest.py stubs file I/O
REAL_SAVE = SARSAAgentLogic._save_q_table


@pytest.fixture
def save_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(SARSAAgentLogic, "_save_q_table", lambda self: calls.append(self.agent_name))
    monkeypatch.setattr(simulation_module, "plot_sarsa_agents", lambda *args, **kwargs: None)
    return calls


def test_updates_do_not_write_to_disk(save_calls):
    logic = SARSAAgentLogic("Z1")

    for _ in range(10):
        logic.update(("bug",), "do_task", 1, ("bug",), "do_task")

    assert save_calls == []
    assert logic.get_q(("bug",), "do_task") > 0


def test_run_saves_each_agent_once_at_the_end(save_calls):
    random.seed(0)
    np.random.seed(0)
    sim = Simulation(agent_count=4, tasks_per_step=6)

    sim.run(steps=5)

    assert sorted(save_calls) == ["C1", "C2", "C3", "C4"]


def test_save_writes_the_learned_q_table(tmp_path, monkeypatch):
    monkeypatch.setattr(SARSAAgentLogic, "_save_q_table", REAL_SAVE)
    logic = SARSAAgentLogic("Z1")
    logic.path = str(tmp_path / "data" / "qtable_Z1.json")
    logic.update(("docs",), "do_task", 1, ("docs",), "skip_task")

    logic.save()

    with open(logic.path) as f:
        assert json.load(f) == logic.q_table
