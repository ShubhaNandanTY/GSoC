import random

from simulation.simulation import Simulation
from tasks.generator import Task


def make_sim_with_tasks(task_types, action="do_task"):
    sim = Simulation(agent_count=1)
    agent = sim.agents[0]
    agent.sarsa.choose_action = lambda state: action
    for task_type in task_types:
        assert agent.assign_task(Task(task_type, 1))
    return sim, agent


def test_each_assigned_task_is_processed_once():
    random.seed(0)
    sim, agent = make_sim_with_tasks(["bug", "feature", "docs"])

    sim.simulate_task_completion()

    assert [r["task_type"] for r in agent.rewards] == ["bug", "feature", "docs"]
    assert agent.total_counts.tolist() == [1, 1, 1]
    assert agent.current_tasks == []
    assert agent.task_load == 0


def test_queue_does_not_grow_across_steps():
    random.seed(0)
    sim, agent = make_sim_with_tasks(["bug"])
    sim.simulate_task_completion()

    agent.assign_task(Task("docs", 1))
    agent.assign_task(Task("docs", 1))
    sim.simulate_task_completion()

    assert [r["task_type"] for r in agent.rewards] == ["bug", "docs", "docs"]
    assert agent.current_tasks == []


def test_skipped_task_leaves_queue():
    sim, agent = make_sim_with_tasks(["feature"], action="skip_task")

    sim.simulate_task_completion()

    assert agent.current_tasks == []
    assert agent.task_load == 0
    assert agent.total_counts.tolist() == [0, 0, 0]


def test_complete_task_reports_whether_task_was_done():
    _, agent = make_sim_with_tasks(["bug", "bug"])
    assert agent.complete_task(True, "bug") is True

    agent.sarsa.choose_action = lambda state: "skip_task"
    assert agent.complete_task(True, "bug") is False


def test_reassigned_dropped_task_gets_completion_status():
    random.seed(0)
    sim, agent = make_sim_with_tasks(["docs"])
    sim.dropped_task_log = [
        {"task_type": "docs", "original_agent": "C9", "reassigned_to": agent.name, "completion_status": "Pending"}
    ]

    sim.simulate_task_completion()

    assert sim.dropped_task_log[0]["completion_status"] in ("Success", "Fail")
