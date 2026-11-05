"""trajcode.alignment — desalineación entre emisor y receptor y decisión de enseñar (T15).

Referencia: theory/02-desalineacion.tex. El receptor se representa por su núcleo q (dict nodo -> {sucesor: prob});
una arista ausente es una entrada faltante o con probabilidad 0 (edición topológica).
"""
from __future__ import annotations

import math
import random

from .graphs import TrajectoryNetwork, walks
from .measures import entropy, kl, meaning_law, path_measure, visit_probabilities


def receiver_measure(net: TrajectoryNetwork, q: dict) -> dict:
    """Q_R(π) sobre los caminos del emisor (0 si usan una arista que el receptor no tiene)."""
    out = {}
    for w, _ in walks(net):
        pr = 1.0
        for a, b in zip(w, w[1:]):
            pr *= q.get(a, {}).get(b, 0.0)
        out[w] = pr
    return out


def misalignment(net: TrajectoryNetwork, q: dict) -> float:
    """A = D(P_S ‖ Q_R) (Teorema 1(a)); infinito si el emisor usa caminos ausentes en el receptor."""
    return kl(path_measure(net), receiver_measure(net, q))


def _node_terms(net: TrajectoryNetwork, q: dict):
    nu = visit_probabilities(net)
    for v, row in net.p.items():
        if nu[v] <= 0:
            continue
        qv = q.get(v, {})
        yield v, nu[v], row, qv, kl(row, qv)


def local_bounds(net: TrajectoryNetwork, q: dict) -> tuple[float, float]:
    """Cotas de Pinsker (inferior) y de probabilidad mínima (superior) del Teorema 1(d)."""
    lo = hi = 0.0
    for v, nu, row, qv, Dv in _node_terms(net, q):
        if Dv <= 1e-15:
            continue
        tv = sum(abs(row.get(u, 0.0) - qv.get(u, 0.0)) for u in set(row) | set(qv))
        lo += nu * tv ** 2 / (2 * math.log(2))
        pos = [x for x in qv.values() if x > 0]
        hi += math.inf if Dv == math.inf or not pos else nu * math.log2(1 / min(pos))
    return lo, hi


def meaning_phrasing_split(net: TrajectoryNetwork, q: dict, mu: dict) -> tuple[float, float]:
    """A = D_s(P_{S,M} ‖ Q_{R,M}) + E[D(P_{Π|M} ‖ Q_{Π|M})] (Teorema 1(e))."""
    P = path_measure(net); Q = receiver_measure(net, q)
    PM, QM = meaning_law(P, mu), meaning_law(Q, mu)
    ds = kl(PM, QM)
    cond = 0.0
    for w, x in P.items():
        if x <= 0:
            continue
        if Q[w] <= 0:
            return ds, math.inf
        cond += x * math.log2((x / PM[mu[w]]) / (Q[w] / QM[mu[w]]))
    return ds, cond


def drop_edges(net: TrajectoryNetwork, fraction: float, seed: int = 0) -> tuple[dict, list]:
    """Edición topológica: el receptor pierde una arista en una fracción de los nodos con ≥ 2 sucesores."""
    rng = random.Random(seed)
    q, removed = {}, []
    for v, row in net.p.items():
        row = dict(row)
        if len(row) >= 2 and rng.random() < fraction:
            u = rng.choice(sorted(row, key=repr))
            removed.append((v, u))
            del row[u]
            z = sum(row.values())
            row = {k: x / z for k, x in row.items()}
        q[v] = row
    return q, removed


def detection_probability(net: TrajectoryNetwork, q: dict) -> float:
    """Pr[Π ∉ supp Q_R]: probabilidad de que el receptor detecte una arista ausente (Proposición 1(ii))."""
    Q = receiver_measure(net, q)
    return sum(x for w, x in path_measure(net).items() if Q[w] <= 0)


def silent_error_probability(net: TrajectoryNetwork, mu_S: dict, mu_R: dict) -> float:
    """Pr[μ_R(Π) ≠ μ_S(Π)]: error semántico silencioso por edición de etiquetas (Proposición 1(iii))."""
    return sum(x for w, x in path_measure(net).items() if mu_S[w] != mu_R[w])


def uniform_receiver(net: TrajectoryNetwork) -> dict:
    """Receptor con la misma topología pero sin estadísticas (Corolario 1(b))."""
    return {v: {u: 1 / len(s) for u in s} for v, s in net.succ.items() if s}


def teach_set(net: TrajectoryNetwork, q: dict, cost: dict, N: int) -> set:
    """T* = {v : N ν_S(v) D_v > c_v}: nodos que conviene enseñar (Corolario 2)."""
    return {v for v, nu, row, qv, Dv in _node_terms(net, q) if N * nu * Dv > cost[v]}


def teach_cost(net: TrajectoryNetwork, q: dict, cost: dict, N: int, T: set) -> float:
    """Σ_{v∈T} c_v + N (H(P_S) + Σ_{v∉T} ν_S(v) D_v)."""
    talk = sum(nu * Dv for v, nu, row, qv, Dv in _node_terms(net, q) if v not in T)
    return sum(cost[v] for v in T) + N * (entropy(path_measure(net).values()) + talk)
