"""trajcode.kbupdate — actualización del conocimiento compartido (T38, Opción C).

Referencia: theory/06-actualizacion.tex (Proposiciones C1 y C2) y theory/02-desalineacion.tex (Corolario 2).

Modelo de costos (en bits, sin el término común N·H(P_S)):
- el receptor usa núcleos desactualizados q; el emisor conoce q y los núcleos vigentes p_S (Supuesto A5);
- cada visita del emisor al nodo v mientras v no se ha enseñado cuesta, en promedio, D_v = D(p_S(·|v) ‖ q(·|v))
  bits ("renta"); enseñar v cuesta c_v bits una sola vez y anula las rentas futuras de v ("compra").
Una secuencia de visitas es una lista de diccionarios {nodo: número de visitas en el mensaje t}.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np

from .graphs import TrajectoryNetwork, walks
from .measures import kl, visit_probabilities


# ------------------------------------------------------------------ términos por nodo
def node_terms(net: TrajectoryNetwork, q: dict) -> dict:
    """{v: (ν_S(v), D_v)} para los nodos no terminales visitados con D_v > 0."""
    nu = visit_probabilities(net)
    out = {}
    for v, row in net.p.items():
        if nu[v] > 0:
            D = kl(row, q.get(v, {}))
            if D > 0:
                out[v] = (nu[v], D)
    return out


def teach_costs(net: TrajectoryNetwork, bits_per_prob: int = 8) -> dict:
    """c_v = ⌈log2 |V|⌉ (dirección del nodo) + (deg⁺(v) − 1)·b (núcleo cuantizado con b bits por probabilidad)."""
    addr = math.ceil(math.log2(len(net.succ)))
    return {v: addr + (len(s) - 1) * bits_per_prob for v, s in net.succ.items() if s}


# ------------------------------------------------------------------ Proposición C1 (N conocido)
def separable_rule(nu: dict, D: dict, c: dict, N: int) -> set:
    """T* = {v : N ν(v) D_v > c_v} (Proposición C1); O(|V|)."""
    return {v for v in D if N * nu[v] * D[v] > c[v]}


def expected_cost(nu: dict, D: dict, c: dict, N: int, T: set) -> float:
    """Σ_{v∈T} c_v + N Σ_{v∉T} ν(v) D_v."""
    return sum(c[v] for v in T) + N * sum(nu[v] * D[v] for v in D if v not in T)


def exhaustive_offline(nu: dict, D: dict, c: dict, N: int) -> tuple[float, set]:
    """Mínimo por búsqueda exhaustiva sobre los 2^|V| conjuntos (solo para pruebas)."""
    keys = list(D)
    best = None
    for bits in itertools.product((0, 1), repeat=len(keys)):
        T = {v for v, b in zip(keys, bits) if b}
        e = expected_cost(nu, D, c, N, T)
        if best is None or e < best[0] - 1e-12:
            best = (e, T)
    return best


# ------------------------------------------------------------------ costos sobre una secuencia realizada
def realized_cost_fixed(stream: list, D: dict, c: dict, T: set) -> float:
    """Enseñar T antes del primer mensaje y pagar la renta del resto."""
    return sum(c[v] for v in T) + sum(n * D[v] for visits in stream for v, n in visits.items()
                                      if v in D and v not in T)


def hindsight_opt(stream: list, D: dict, c: dict) -> float:
    """Óptimo fuera de línea con la secuencia conocida: Σ_v min(c_v, R_v(N))."""
    R = {v: 0.0 for v in D}
    for visits in stream:
        for v, n in visits.items():
            if v in D:
                R[v] += n * D[v]
    return sum(min(c[v], R[v]) for v in D)


def online_rule(stream: list, D: dict, c: dict) -> tuple[float, dict]:
    """Regla en línea por nodo (Proposición C2).

    Antes del mensaje t, el emisor conoce el camino que va a enviar. Si la renta acumulada de v más la renta de
    este mensaje alcanzaría c_v, enseña v (paga c_v) y envía el mensaje ya alineado en v; si no, paga la renta.
    Devuelve (costo total, {v: índice del mensaje en que se enseñó}).
    """
    R = {v: 0.0 for v in D}
    taught, cost = {}, 0.0
    for t, visits in enumerate(stream):
        for v, n in visits.items():
            if v not in D or v in taught or n == 0:
                continue
            r = n * D[v]
            if R[v] + r >= c[v]:
                taught[v] = t
                cost += c[v]
            else:
                R[v] += r
                cost += r
    return cost, taught


def sync_on_first_use(stream: list, D: dict, c: dict) -> float:
    """Heurística "sincronizar todo lo desalineado que se use" (cf. Chen et al. 2025): enseñar v en su primera visita."""
    seen = set()
    for visits in stream:
        seen.update(v for v, n in visits.items() if v in D and n > 0)
    return sum(c[v] for v in seen)


def global_threshold(stream: list, D: dict, c: dict) -> float:
    """Alquiler de esquís agregado: enseñar todos los nodos editados a la vez cuando la renta total alcanzaría Σ c_v."""
    C = sum(c.values())
    R, cost = 0.0, 0.0
    for visits in stream:
        r = sum(n * D[v] for v, n in visits.items() if v in D)
        if r == 0:
            continue
        if R + r >= C:
            return cost + C
        R += r
        cost += r
    return cost


# ------------------------------------------------------------------ versiones vectorizadas (E6)
def visit_matrix(net: TrajectoryNetwork, nodes: list) -> tuple[list, np.ndarray, np.ndarray]:
    """Caminos, su probabilidad y la matriz de visitas (camino × nodo) para los nodos dados."""
    W = list(walks(net))
    idx = {v: i for i, v in enumerate(nodes)}
    X = np.zeros((len(W), len(nodes)))
    for i, (w, _) in enumerate(W):
        for v in w:
            if v in idx:
                X[i, idx[v]] += 1
    return [w for w, _ in W], np.array([p for _, p in W]), X


def stream_costs(X: np.ndarray, draws: np.ndarray, D: np.ndarray, c: np.ndarray) -> dict:
    """Costos de todas las políticas sobre la secuencia draws (índices de camino); vectorizado sobre nodos.

    Coincide con online_rule, hindsight_opt, sync_on_first_use y global_threshold (prueba en tests/test_kbupdate.py).
    """
    rent = X[draws] * D                      # N × |V|
    cum = np.cumsum(rent, axis=0)
    total = cum[-1]
    out = {"talk_only": float(total.sum()), "hindsight": float(np.minimum(c, total).sum())}
    # regla por nodo: compra en el primer t con cum_t ≥ c (cum_t incluye la renta del mensaje t)
    on = 0.0
    for j in range(len(c)):
        hit = np.flatnonzero(cum[:, j] >= c[j])
        on += total[j] if hit.size == 0 else (cum[hit[0], j] - rent[hit[0], j]) + c[j]
    out["online"] = float(on)
    out["sync_first_use"] = float(c[total > 0].sum())
    agg = np.cumsum(rent.sum(axis=1))
    C = float(c.sum())
    hit = np.flatnonzero(agg >= C)
    out["global"] = float(agg[-1]) if hit.size == 0 else float(agg[hit[0]] - rent[hit[0]].sum() + C)
    return out


# ------------------------------------------------------------------ escenario RTLS
def make_site_graph(buildings: int = 2, floors: int = 3, wings: int = 2, rooms: int = 4,
                    seed: int = 0) -> tuple[TrajectoryNetwork, dict, dict]:
    """Jerarquía de un sitio: sitio → edificio → piso → (ala → sala | vestíbulo → sala).

    Cada piso tiene un vestíbulo (punto de referencia) desde el que se nombran las dos primeras salas de cada ala;
    esas salas tienen dos descripciones (sinonimia). El significado es la sala (nodo terminal).
    Devuelve (red, mapa camino → significado, {nodo no terminal: piso o None}).
    """
    rng = random.Random(seed)
    succ, floor_of = {"site": []}, {"site": None}
    for b in range(buildings):
        B = f"b{b}"
        succ["site"].append(B); succ[B] = []; floor_of[B] = None
        for f in range(floors):
            F = f"{B}f{f}"
            succ[B].append(F); succ[F] = []; floor_of[F] = F
            lobby = f"{F}:lobby"
            succ[lobby] = []; floor_of[lobby] = F
            for w in range(wings):
                Wn = f"{F}w{w}"
                succ[F].append(Wn); succ[Wn] = []; floor_of[Wn] = F
                for r in range(rooms):
                    R = f"{Wn}r{r}"
                    succ[Wn].append(R); succ[R] = []
                    if r < 2:
                        succ[lobby].append(R)
            succ[F].append(lobby)

    def kernel(s):
        w = [rng.gammavariate(1.0, 1.0) + 0.05 for _ in s]
        z = sum(w)
        return {u: x / z for u, x in zip(s, w)}

    p = {v: kernel(s) for v, s in succ.items() if s}
    net = TrajectoryNetwork(succ, p, "site")
    net.validate()
    mu = {w: w[-1] for w, _ in walks(net)}
    return net, mu, {v: f for v, f in floor_of.items() if succ[v]}


def remodel(net: TrajectoryNetwork, nodes: set, seed: int = 0, low: float = 0.2, high: float = 4.0) -> TrajectoryNetwork:
    """Edición de ruteo en los nodos dados (plano remodelado): reescala sus núcleos sin cambiar el soporte."""
    rng = random.Random(seed)
    p = {}
    for v, row in net.p.items():
        if v in nodes:
            w = {u: x * rng.uniform(low, high) for u, x in row.items()}
            z = sum(w.values())
            p[v] = {u: x / z for u, x in w.items()}
        else:
            p[v] = dict(row)
    out = TrajectoryNetwork(dict(net.succ), p, net.root, dict(net.labels))
    out.validate()
    return out
