from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx

PROTOCOL = "local-disruptions-v2-explanation-first"


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def seed_for(*parts: Any) -> int:
    return int(digest(parts)[:8], 16) % (2**31 - 1)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class Spec:
    phase: str = "pilot"
    scenario: str = "cascade"
    topology: str = "WS"
    replicate: int = 0
    n: int = 24
    degree: int = 4
    rounds: int = 9
    reliability: float = 0.7
    fraction: float = 0.125
    onset: int = 2
    duration: int = 2
    placement: str = "random"
    temperature: float = 0.7
    wording: int = 0
    exposure: int = 4
    memory: int = 1

    @property
    def block_id(self) -> str:
        # All arms of a block share task, graph, exposure order and sampling seeds.
        return digest(asdict(self))[:16]


def graph_for(spec: Spec) -> nx.Graph:
    sd = seed_for(PROTOCOL, spec.phase, spec.topology, spec.replicate, spec.n, spec.degree)
    n, k = spec.n, spec.degree
    if spec.topology == "WS":
        g = nx.watts_strogatz_graph(n, k, 0.18, seed=sd)
    elif spec.topology == "ER":
        g = nx.gnm_random_graph(n, n * k // 2, seed=sd)
    elif spec.topology == "SBM":
        # Match the edge count exactly, retain a four-community preference.
        rng = random.Random(sd)
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
        def key(edge):
            weight = 7 if edge[0] * 4 // n == edge[1] * 4 // n else 1
            return rng.random() ** (1 / weight)
        edges = sorted(candidates, key=key, reverse=True)[: n * k // 2]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
    else:
        raise ValueError(spec.topology)
    # Keep disconnected realizations; record them rather than conditioning on connectivity.
    return g


def task_for(spec: Spec) -> dict:
    rng = random.Random(seed_for(PROTOCOL, "task", spec.phase, spec.replicate, spec.n, spec.scenario, spec.wording))
    truth = ["A", "B"][spec.replicate % 2]
    private = [truth if rng.random() < spec.reliability else opposite(truth) for _ in range(spec.n)]
    if spec.scenario == "asch":
        # Directly inspectable objective task; arithmetic, counting and comparison variants.
        a, b, c = rng.randrange(13, 89), rng.randrange(13, 89), rng.randrange(3, 15)
        correct = (a + b) * c
        wrong = correct + rng.choice([-c, c, 2 * c])
        options = {truth: correct, opposite(truth): wrong}
        if spec.wording % 3 == 0:
            question = f"Compute ({a} + {b}) * {c}. Options A={options['A']}, B={options['B']}."
        elif spec.wording % 3 == 1:
            correct = max(a, b) + c
            wrong = min(a, b) - c
            options = {truth: correct, opposite(truth): wrong}
            question = f"Take the larger of {a} and {b}, then add {c}. A={options['A']}, B={options['B']}."
        else:
            bits = [rng.randrange(2) for _ in range(18)]
            correct = sum(bits)
            options = {truth: correct, opposite(truth): correct + rng.choice([-1, 1])}
            question = f"Count ones in {''.join(map(str, bits))}. A={options['A']}, B={options['B']}."
        private = [truth] * spec.n
    elif spec.scenario == "norm":
        # A normative preference task; truth means majority preference, not moral correctness.
        private = [truth if i < math.ceil(spec.n * 0.75) else opposite(truth) for i in range(spec.n)]
        rng.shuffle(private)
        question = "Choose a group work rule: A=short daily check-in, B=one longer weekly check-in. Your private preference is supplied below. No objectively correct rule exists."
    elif spec.scenario == "cooperation":
        truth = "A"
        private = ["A"] * spec.n
        question = f"Repeated public-goods game with {spec.n} players. Each has 10 tokens per round. A=contribute all 10; B=contribute 0. Each token contributed yields 1.6 tokens divided equally among ALL {spec.n} players. Payoff = unspent tokens + your equal share. Choose freely. Public claims are not verified actual contributions."
    else:
        labels = ["a sealed urn", "an unknown transmitter", "a hidden production line"]
        question = f"Infer hidden state A or B of {labels[spec.wording % 3]}. Prior P(A)=P(B)=0.5. Each original sensor report independently matches the state with probability {spec.reliability:.2f}. Reports are conditionally independent given the state. Peers may report observations or interpretations; their claims are not guaranteed accurate."
    return {"truth": truth, "private": private, "question": question, "task_id": digest([question, private, truth])[:16]}


def opposite(x: str) -> str:
    return "B" if x == "A" else "A"


def selected_nodes(spec: Spec, g: nx.Graph) -> list[int]:
    count = max(1, round(spec.n * spec.fraction)) if spec.fraction else 0
    rng = random.Random(seed_for(PROTOCOL, "placement", spec.phase, spec.replicate, spec.n, spec.topology))
    nodes = list(g.nodes)
    rng.shuffle(nodes)
    if spec.placement == "hub":
        nodes.sort(key=lambda i: g.degree(i), reverse=True)
    elif spec.placement == "peripheral":
        nodes.sort(key=lambda i: g.degree(i))
    return sorted(nodes[:count])


def neighbors_for(spec: Spec, g: nx.Graph, node: int, t: int) -> list[int]:
    neighbors = sorted(g.neighbors(node))
    random.Random(seed_for(PROTOCOL, "exposure", spec.block_id, node, t)).shuffle(neighbors)
    return neighbors[: spec.exposure]


def graph_summary(g: nx.Graph) -> dict:
    return {"edges": sorted([list(e) for e in g.edges]), "mean_degree": sum(dict(g.degree).values()) / len(g), "clustering": nx.average_clustering(g), "components": nx.number_connected_components(g)}
