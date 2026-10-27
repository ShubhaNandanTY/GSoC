import os
import sys

import pytest

# SustainHub modules import each other as top-level packages (agents, simulation, tasks)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def no_qtable_files(monkeypatch, tmp_path):
    """Keep tests from reading or writing data/qtable_*.json and data/trained_agents.json."""
    from agents.sarsa import SARSAAgentLogic
    import simulation.simulation as simulation_module

    monkeypatch.setattr(SARSAAgentLogic, "_load_q_table", lambda self: None)
    monkeypatch.setattr(SARSAAgentLogic, "_save_q_table", lambda self: None)
    monkeypatch.setattr(simulation_module, "SAVE_FILE", str(tmp_path / "trained_agents.json"))
