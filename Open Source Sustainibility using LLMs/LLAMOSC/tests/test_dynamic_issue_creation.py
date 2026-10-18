"""Tests for dynamic issue creation (issue #63). No Ollama server is needed."""
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from LLAMOSC.agents import issue_creator as issue_creator_module
from LLAMOSC.agents.issue_creator import IssueCreatorAgent
from LLAMOSC.simulation.issue import Issue
from LLAMOSC.simulation.sim import Simulation


@pytest.fixture
def issues_folder(tmp_path):
    pending = tmp_path / "issues" / "pending"
    pending.mkdir(parents=True)
    return pending


def make_issues(folder, count):
    issues = []
    for i in range(1, count + 1):
        path = folder / f"task_{i}.md"
        path.write_text(f"Issue {i}. Some description")
        issues.append(Issue(i, (i + 1) % 5, str(path)))
    return issues


def make_sim(n_available=2):
    contributors = [SimpleNamespace(available=True) for _ in range(n_available)]
    contributors.append(SimpleNamespace(available=False))
    return Simulation(contributors)


def test_pending_issues_excludes_solved(issues_folder, tmp_path):
    issues = make_issues(issues_folder, 5)
    solved = tmp_path / "issues" / "solved"
    solved.mkdir()
    for issue in issues[:4]:
        os.rename(issue.filepath, solved / os.path.basename(issue.filepath))

    creator = IssueCreatorAgent("Issue Creator")
    with mock.patch.object(creator, "should_create_issue", return_value=False) as should_create:
        make_sim().try_dynamic_issue_creation(creator, issues, issues, "", str(issues_folder))

    assert should_create.call_args[0][0]["pending_issues"] == 1


def test_simulation_state_reaches_llm_prompt(issues_folder):
    sim = make_sim(n_available=2)
    sim.issues_solved = 4
    sim.time_step = 7
    creator = IssueCreatorAgent("Issue Creator")
    creator.steps_since_last_creation = creator.CREATION_COOLDOWN

    with mock.patch.object(issue_creator_module, "query_ollama", return_value="NO") as query:
        sim.try_dynamic_issue_creation(creator, [], [], "", str(issues_folder))

    prompt = query.call_args[1]["prompt"]
    assert "Solved Issues: 4" in prompt
    assert "Contributors: 2" in prompt
    assert "Time Step: 7" in prompt


def test_new_issue_queued_once_when_lists_are_shared(issues_folder):
    issues = []
    new_issue = Issue(1, 2, str(issues_folder / "task_1.md"))
    creator = IssueCreatorAgent("Issue Creator")

    with mock.patch.object(creator, "should_create_issue", return_value=True), \
            mock.patch.object(creator, "create_issue", return_value=new_issue):
        make_sim().try_dynamic_issue_creation(creator, issues, issues, "", str(issues_folder))

    assert issues == [new_issue]
    assert creator.dynamic_issues_created == 1
    assert creator.steps_since_last_creation == 0


def test_new_issue_added_to_both_lists_when_separate(issues_folder):
    queue, known = [], []
    new_issue = Issue(1, 2, str(issues_folder / "task_1.md"))
    creator = IssueCreatorAgent("Issue Creator")

    with mock.patch.object(creator, "should_create_issue", return_value=True), \
            mock.patch.object(creator, "create_issue", return_value=new_issue):
        make_sim().try_dynamic_issue_creation(creator, queue, known, "", str(issues_folder))

    assert queue == [new_issue]
    assert known == [new_issue]


def test_cooldown_advances_once_per_step(issues_folder):
    sim = make_sim()
    creator = IssueCreatorAgent("Issue Creator")

    for step in range(1, creator.CREATION_COOLDOWN + 1):
        with mock.patch.object(issue_creator_module, "query_ollama") as query:
            sim.try_dynamic_issue_creation(creator, [], [], "", str(issues_folder))
        query.assert_not_called()
        assert creator.steps_since_last_creation == step

    with mock.patch.object(issue_creator_module, "query_ollama", return_value="NO") as query:
        sim.try_dynamic_issue_creation(creator, [], [], "", str(issues_folder))
    query.assert_called_once()
