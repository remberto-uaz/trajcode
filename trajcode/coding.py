"""trajcode.coding — códigos de camino: nativos, adaptados (con y sin memoria) y bits-back (T13–T14).

Referencias: theory/00-modelo.tex (Lema 1) y theory/01-memoria.tex (Teoremas 1–2, Proposición 1).
Las longitudes "ideales" son −log2 de la probabilidad de codificación; ArithmeticPathCoder valida
con aritmética racional exacta que son alcanzables con menos de 2 bits de exceso (ADR-04).
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Callable, Iterator

from .graphs import TrajectoryNetwork, Walk
from .measures import meaning_law

Kernel = Callable[[Walk], dict]   # prefijo -> {siguiente nodo: probabilidad}


# ---------------------------------------------------------------- núcleos de codificación
def native_kernel(net: TrajectoryNetwork) -> Kernel:
    """Núcleo compartido p(·|v): depende solo del nodo actual (código nativo, Def. 2 de la Parte 1)."""
    return lambda prefix: net.p[prefix[-1]]


def prefix_kernel(Q: dict) -> Kernel:
    """Núcleo con memoria: ley de la siguiente decisión dada todo el prefijo, bajo la ley de caminos Q."""
    nxt: dict = {}
    for w, q in Q.items():
        if q <= 0:
            continue
        for k in range(1, len(w)):
            row = nxt.setdefault(w[:k], {})
            row[w[k]] = row.get(w[k], 0.0) + q
    table = {pre: {u: x / sum(row.values()) for u, x in row.items()} for pre, row in nxt.items()}
    return lambda prefix: table[tuple(prefix)]


# ---------------------------------------------------------------- codificador aritmético exacto
class ArithmeticPathCoder:
    """Codificador aritmético de caminos con intervalos racionales exactos (Lema 1, Parte 0).

    encode(walk) -> cadena de bits; decode(bits) -> walk. Garantía: len(bits) < −log2 Q(walk) + 2.
    """

    def __init__(self, net: TrajectoryNetwork, kernel: Kernel):
        self.net, self.kernel = net, kernel

    def _row(self, prefix: Walk) -> list[tuple]:
        dist = self.kernel(prefix)
        items = sorted(((u, Fraction(x)) for u, x in dist.items() if x > 0), key=lambda t: repr(t[0]))
        z = sum(x for _, x in items)
        return [(u, x / z) for u, x in items]

    def _interval(self, walk: Walk) -> tuple[Fraction, Fraction]:
        low, width = Fraction(0), Fraction(1)
        for t in range(1, len(walk)):
            cum = Fraction(0)
            for u, q in self._row(walk[:t]):
                if u == walk[t]:
                    low, width = low + width * cum, width * q
                    break
                cum += q
            else:
                raise ValueError(f"la decisión {walk[t]!r} tiene probabilidad 0 en el núcleo")
        return low, width

    def ideal_length(self, walk: Walk) -> float:
        return sum(-math.log2(self.kernel(walk[:t])[walk[t]]) for t in range(1, len(walk)))

    def encode(self, walk: Walk) -> str:
        low, width = self._interval(walk)
        k = 0
        while True:
            scale = 2 ** k
            c = math.ceil(low * scale)
            if Fraction(c + 1, scale) <= low + width:
                return format(c, f"0{k}b") if k else ""
            k += 1

    def decode(self, bits: str) -> Walk:
        k = len(bits)
        a = Fraction(int(bits, 2) if k else 0, 2 ** k)
        b = a + Fraction(1, 2 ** k)
        walk, low, width = (self.net.root,), Fraction(0), Fraction(1)
        while self.net.succ[walk[-1]]:
            cum = Fraction(0)
            for u, q in self._row(walk):
                lo, hi = low + width * cum, low + width * (cum + q)
                if lo <= a and b <= hi:
                    walk, low, width = walk + (u,), lo, width * q
                    break
                cum += q
            else:
                raise ValueError("los bits no determinan un camino (código truncado o desalineado)")
        return walk


# ---------------------------------------------------------------- selecciones
def _preimages(P: dict, mu: dict) -> dict:
    pre: dict = {}
    for w in sorted(P, key=repr):
        pre.setdefault(mu[w], []).append(w)
    return pre


def viterbi_selection(P: dict, mu: dict) -> dict:
    """f*(m) = camino más probable de Π(m), con desempate fijo (Teorema 1(a) y 2(c), Parte 1)."""
    return {m: max(ws, key=lambda w: (P[w], repr(w))) for m, ws in _preimages(P, mu).items()}


def all_selections(P: dict, mu: dict, limit: int | None = None) -> Iterator[dict]:
    """Enumera selecciones f con f(m) ∈ Π(m) (hasta `limit`)."""
    pre = _preimages(P, mu)
    ms = sorted(pre, key=repr)
    for i, choice in enumerate(itertools.product(*(pre[m] for m in ms))):
        if limit is not None and i >= limit:
            return
        yield dict(zip(ms, choice))


def selection_law(f: dict, PM: dict) -> dict:
    """Q_f: la ley P_M transportada a los caminos seleccionados (Def. 1, Parte 1)."""
    return {f[m]: x for m, x in PM.items() if x > 0}


# ---------------------------------------------------------------- tasas (Teoremas 1–2, Proposición 1)
def native_length(f: dict, P: dict, PM: dict) -> float:
    """L_nat(f) = E[−log2 P(f(M))] (Teorema 1(a))."""
    return sum(x * -math.log2(P[f[m]]) for m, x in PM.items() if x > 0)


def synonymy_gap(P: dict, mu: dict) -> dict:
    """g(m) = log2(P_M(m) / max_{π∈Π(m)} P(π)) = H_∞(Π | M = m) (Ec. 1, Parte 1)."""
    PM = meaning_law(P, mu)
    best: dict = {}
    for w, x in P.items():
        best[mu[w]] = max(best.get(mu[w], 0.0), x)
    return {m: math.log2(PM[m] / best[m]) for m in PM}


def _expected_length(Q: dict, kernel: Kernel) -> float:
    return sum(q * sum(-math.log2(kernel(w[:t])[w[t]]) for t in range(1, len(w))) for w, q in Q.items())


def adapted_length(Q: dict) -> float:
    """Código adaptado con memoria completa del prefijo; debe valer H(M) (Teorema 2(a))."""
    return _expected_length(Q, prefix_kernel(Q))


def memoryless_kernel(Q: dict) -> Kernel:
    """Mejor núcleo que depende solo del nodo, para la selección que induce Q."""
    mass: dict = {}
    for w, q in Q.items():
        for a, b in zip(w, w[1:]):
            row = mass.setdefault(a, {})
            row[b] = row.get(b, 0.0) + q
    table = {a: {b: x / sum(row.values()) for b, x in row.items()} for a, row in mass.items()}
    return lambda prefix: table[prefix[-1]]


def memoryless_adapted_length(Q: dict) -> float:
    """Tasa del mejor código adaptado sin memoria para la selección de Q (Teorema 2(b)–(d))."""
    return _expected_length(Q, memoryless_kernel(Q))


def min_memoryless_adapted_length(P: dict, mu: dict) -> float:
    """Mínimo sobre todas las selecciones de la tasa adaptada sin memoria."""
    PM = meaning_law(P, mu)
    return min(memoryless_adapted_length(selection_law(f, PM)) for f in all_selections(P, mu))


def future_law_counts(Q: dict, ndigits: int = 12) -> dict:
    """N_v(f): número de leyes futuras distintas en cada nodo (Teorema 2(b)); N(f) = máximo."""
    laws: dict = {}
    for w in Q:
        for k in range(1, len(w)):
            pre = w[:k]
            ext = {y: q for y, q in Q.items() if y[:k] == pre and q > 0}
            z = sum(ext.values())
            law = tuple(sorted(((y[k:], round(q / z, ndigits)) for y, q in ext.items()), key=repr))
            laws.setdefault(pre[-1], set()).add(law)
    return {v: len(s) for v, s in laws.items()}


def bitsback_net_length(P: dict, mu: dict) -> float:
    """Costo neto ideal de bits-back con el núcleo nativo: E[−log2 P(π) + log2 P(π|m)] = H(M) (Prop. 1)."""
    PM = meaning_law(P, mu)
    return sum(x * (-math.log2(x) + math.log2(x / PM[mu[w]])) for w, x in P.items() if x > 0)


def memory_states(Q: dict) -> int:
    """N(f) = max_v N_v(f); vale 1 si no hay decisiones (un solo camino trivial)."""
    return max(future_law_counts(Q).values(), default=1)
