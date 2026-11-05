"""trajcode.measures — medida de camino, ley del significado, entropías, visitas y KL (T12, T15).

Referencias: theory/00-modelo.tex (Defs. 2–5), theory/02-desalineacion.tex (Teorema 1).
"""
from __future__ import annotations

import math
import random
from typing import Iterable

import networkx as nx

from .graphs import TrajectoryNetwork, walks


def path_measure(net: TrajectoryNetwork) -> dict:
    """P(π) para cada camino completo (Ec. 1, Parte 0)."""
    return dict(walks(net))


def meaning_law(P: dict, mu: dict) -> dict:
    """P_M(m) = Σ_{π ∈ Π(m)} P(π) (Ec. 2, Parte 0)."""
    PM: dict = {}
    for w, x in P.items():
        PM[mu[w]] = PM.get(mu[w], 0.0) + x
    return PM


def entropy(values: Iterable[float]) -> float:
    """Entropía en bits de una ley dada por sus probabilidades."""
    return -sum(x * math.log2(x) for x in values if x > 0)


def cond_entropy_walk_given_meaning(P: dict, mu: dict) -> float:
    """H(Π | M) = H(Π) − H(M), válido porque M es función de Π (Ec. 3, Parte 0)."""
    return entropy(P.values()) - entropy(meaning_law(P, mu).values())


def visit_probabilities(net: TrajectoryNetwork) -> dict:
    """ν(v) = Pr[v ∈ Π] (Def. 5, Parte 0), por propagación en orden topológico."""
    g = nx.DiGraph([(v, u) for v, s in net.succ.items() for u in s])
    g.add_nodes_from(net.succ)
    nu = {v: 0.0 for v in net.succ}
    nu[net.root] = 1.0
    for v in nx.topological_sort(g):
        for u, q in net.p.get(v, {}).items():
            nu[u] += nu[v] * q
    return nu


def kl(p: dict, q: dict) -> float:
    """D(p ‖ q) en bits; infinito si q no cubre el soporte de p."""
    total = 0.0
    for k, x in p.items():
        if x <= 0:
            continue
        y = q.get(k, 0.0)
        if y <= 0:
            return math.inf
        total += x * math.log2(x / y)
    return total


def kl_paths(P_S: dict, Q_R: dict) -> float:
    """Desalineación A = D(P_S ‖ Q_R) entre medidas de camino (Def. 4, Parte 0)."""
    return kl(P_S, Q_R)


def kl_nodewise(net_S: TrajectoryNetwork, net_R: TrajectoryNetwork) -> dict:
    """Contribución de cada nodo ν_S(v) · D(p_S(·|v) ‖ q_R(·|v)) (Teorema 1(b), Parte 2)."""
    nu = visit_probabilities(net_S)
    return {v: nu[v] * kl(row, net_R.p.get(v, {})) for v, row in net_S.p.items() if nu[v] > 0}


def perturb_kernel(net: TrajectoryNetwork, fraction: float, seed: int = 0,
                   low: float = 0.2, high: float = 4.0) -> TrajectoryNetwork:
    """Edición de ruteo (Parte 2): reescala los núcleos de una fracción de nodos sin cambiar su soporte."""
    rng = random.Random(seed)
    p = {}
    for v, row in net.p.items():
        if rng.random() < fraction:
            w = {u: x * rng.uniform(low, high) for u, x in row.items()}
            z = sum(w.values())
            p[v] = {u: x / z for u, x in w.items()}
        else:
            p[v] = dict(row)
    out = TrajectoryNetwork(dict(net.succ), p, net.root, dict(net.labels))
    out.validate()
    return out
