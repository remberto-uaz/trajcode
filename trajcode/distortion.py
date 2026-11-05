"""trajcode.distortion — distorsión geodésica, códigos de cociente, Blahut–Arimoto y refinabilidad (T16).

Referencia: theory/03-distorsion.tex. Pensado para |𝓜| ≤ 8 (enumeración de particiones; constitution P8).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse.csgraph import shortest_path


def geodesic_matrix(adjacency) -> np.ndarray:
    """Distancia de camino más corto en el grafo de conceptos O (aristas de longitud 1)."""
    A = (np.asarray(adjacency, float) > 0).astype(float)
    return shortest_path(A, method="D", directed=False, unweighted=True)


def _partitions(items: list):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def quotient_points(p, d) -> list[tuple[float, float]]:
    """Puntos (D_Q, H(Q)) de todos los códigos de cociente con el mejor representante por clase."""
    p, d = np.asarray(p, float), np.asarray(d, float)
    pts = []
    for part in _partitions(list(range(len(p)))):
        R = -sum(p[B].sum() * math.log2(p[B].sum()) for B in part if p[B].sum() > 0)
        D = sum(min(sum(p[i] * d[i, b] for i in B) for b in B) for B in part)
        pts.append((float(D), float(R)))
    return pts


def lower_envelope(points) -> list[tuple[float, float]]:
    """Envolvente convexa inferior de los puntos (D, R), ordenada por D."""
    pts = sorted(set((round(a, 12), round(b, 12)) for a, b in points))
    hull: list = []
    for P in pts:
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (P[1] - hull[-2][1]) - \
                (hull[-1][1] - hull[-2][1]) * (P[0] - hull[-2][0]) <= 0:
            hull.pop()
        hull.append(P)
    # recorta la parte creciente: la envolvente operativa es no creciente
    k = min(range(len(hull)), key=lambda i: (hull[i][1], hull[i][0]))
    return hull[:k + 1]


def envelope_at(hull, D: float) -> float:
    """Valor de la envolvente en D (horizontal a la derecha del último punto)."""
    if D <= hull[0][0]:
        return hull[0][1]
    for (a, b), (c, e) in zip(hull, hull[1:]):
        if a <= D <= c:
            return b + (e - b) * (D - a) / (c - a)
    return hull[-1][1]


def blahut_arimoto(p, d, beta: float, tol: float = 1e-13, max_iter: int = 200_000):
    """Blahut–Arimoto con pendiente β. Devuelve (R en bits, D, ley conjunta J[x, x̂])."""
    p, d = np.asarray(p, float), np.asarray(d, float)
    q = np.ones(d.shape[1]) / d.shape[1]
    K = np.exp(-beta * d)
    for _ in range(max_iter):
        W = q * K
        W /= W.sum(1, keepdims=True)
        qn = p @ W
        if np.max(np.abs(qn - q)) < tol:
            q = qn
            break
        q = qn
    W = q * K
    W /= W.sum(1, keepdims=True)
    J = p[:, None] * W
    ref = np.maximum(p[:, None] * q[None, :], 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(J > 1e-300, J * np.log2(np.maximum(J, 1e-300) / ref), 0.0)
    return float(terms.sum()), float((J * d).sum()), J


def rd_at(p, d, D: float, iters: int = 80) -> tuple[float, float]:
    """R(D) por bisección sobre la pendiente β. Devuelve (R, D alcanzada)."""
    lo, hi = 1e-3, 80.0
    for _ in range(iters):
        b = (lo + hi) / 2
        _, Db, _ = blahut_arimoto(p, d, b)
        lo, hi = (b, hi) if Db > D else (lo, b)
    R, Dr, _ = blahut_arimoto(p, d, (lo + hi) / 2)
    return R, Dr


def refinability_residual(J1, J2) -> float:
    """Residuo ℓ1 mínimo de J2 W = J1 con W estocástica (criterio de Equitz–Cover); 0 ⇒ refinable."""
    J1, J2 = np.asarray(J1), np.asarray(J2)
    n, m = J2.shape
    k = J1.shape[1]
    rows, rhs = [], []
    for x in range(n):
        for a in range(k):
            row = np.zeros(m * k)
            for b in range(m):
                row[b * k + a] = J2[x, b]
            rows.append(row); rhs.append(J1[x, a])
    for b in range(m):
        row = np.zeros(m * k); row[b * k:(b + 1) * k] = 1
        rows.append(row); rhs.append(1.0)
    A = np.array(rows); nE = len(rhs)
    c = np.r_[np.zeros(m * k), np.ones(2 * nE)]
    res = linprog(c, A_eq=np.c_[A, np.eye(nE), -np.eye(nE)], b_eq=rhs,
                  bounds=[(0, None)] * (m * k + 2 * nE), method="highs")
    return float(res.fun)


def diameter(d, classes) -> float:
    """Diámetro de una partición: máxima distancia dentro de una clase (Def. 7, Parte 0)."""
    d = np.asarray(d, float)
    return max((d[i, j] for B in classes for i in B for j in B), default=0.0)


def nested_quotient_rates(leaves: list, p) -> list[float]:
    """H(Q_ℓ) para ℓ = 0..h en una jerarquía (Proposición 1, Parte 3)."""
    p = np.asarray(p, float)
    depth = len(leaves[0]) if leaves else 0
    rates = []
    for l in range(depth + 1):
        mass: dict = {}
        for lf, x in zip(leaves, p):
            mass[lf[:l]] = mass.get(lf[:l], 0.0) + x
        rates.append(-sum(x * math.log2(x) for x in mass.values() if x > 0))
    return rates
