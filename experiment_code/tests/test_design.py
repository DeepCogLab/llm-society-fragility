import asyncio
from pathlib import Path

import pytest

from society.core import Spec, graph_for, neighbors_for, task_for
from society.experiment import injected_message, prompt_for, run_arm


class EvidenceClient:
    model = {"id": "test", "family": "test"}

    async def complete(self, system, prompt, seed, temperature):
        import re
        # An intentionally simple deterministic decision maker for design invariants only.
        node = int(re.search(r"You are peer (\d+)", prompt)[1])
        report = re.search(r"observes ([AB])", prompt)
        choice = report[1] if report else "A"
        messages = re.findall(r"answer=([AB])", prompt)
        if messages and messages.count("B") > messages.count("A"):
            choice = "B"
        return {"choice": choice, "confidence": 0.8, "message": f"My observation suggests {choice}.",
                "cites": [f"E{node:03d}"], "call_key": str(seed)}


def test_edge_budgets_and_exposure_are_preassigned():
    for topology in ["WS", "ER", "SBM"]:
        spec = Spec(topology=topology)
        g = graph_for(spec)
        assert g.number_of_edges() == spec.n * spec.degree // 2
        for node in g:
            assert len(neighbors_for(spec, g, node, 3)) <= spec.exposure
            assert neighbors_for(spec, g, node, 3) == neighbors_for(spec, g, node, 3)


def test_holdout_tasks_and_balanced_labels():
    assert task_for(Spec(replicate=0))["truth"] == "A"
    assert task_for(Spec(replicate=1))["truth"] == "B"
    assert task_for(Spec(phase="pilot"))["task_id"] != task_for(Spec(phase="confirm"))["task_id"]


def test_prompt_has_no_truth_or_treatment_metadata():
    spec = Spec()
    task = task_for(spec)
    prompt = prompt_for(spec, task, 0, 0, [], [], "natural")
    assert "truth" not in prompt.lower()
    assert "injected" not in prompt.lower()
    assert "treatment" not in prompt.lower()
    assert task["question"] in prompt


def test_injection_changes_only_communications():
    spec = Spec()
    task = task_for(spec)
    original = {"id": "M01_001", "sender": 1, "round": 1, "choice": "A", "text": "original", "roots": ["E001"]}
    modified = injected_message(original, task, spec)
    assert original["text"] == "original"
    assert modified["choice"] != task["truth"]
    assert modified["roots"] == ["E900"]


def test_paired_histories_and_no_relay_sham(tmp_path):
    async def check():
        client, spec = EvidenceClient(), Spec(n=12, rounds=7)
        control = await run_arm(client, spec, "natural", False, None, tmp_path / "control.json")
        treated = await run_arm(client, spec, "natural", True, control, tmp_path / "treated.json")
        sham = await run_arm(client, spec, "no_relay", False, control, tmp_path / "sham.json")
        assert control["events"] == sham["events"]
        for c, t in zip(control["events"], treated["events"]):
            if c["round"] < spec.onset:
                assert c == t
        for event in treated["events"]:
            assert event["direct_injected_exposure"] is False or spec.onset <= event["round"] < spec.onset + spec.duration
        assert not any(e["direct_injected_exposure"] for e in treated["events"] if e["round"] >= spec.onset + spec.duration)
        # Internal choices of seeded agents still come from client, not the intervention.
        for event in treated["events"]:
            if event["round"] == 0:
                assert event["choice"] == treated["task"]["private"][event["node"]]
    asyncio.run(check())


def test_no_substitution_of_failed_or_invalid_outputs():
    from society.client import LLMClient
    with pytest.raises(ValueError):
        LLMClient.validate({"choice": "A"})
    with pytest.raises(ValueError):
        LLMClient.validate({"choice": "A", "confidence": 1.5, "message": "", "cites": []})
