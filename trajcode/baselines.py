"""trajcode.baselines — esquemas de referencia y algoritmos lineales para la comparación de TCOM (T39).

Referencias de tasa (sin pérdidas, un significado por mensaje salvo que se indique):
- índice del significado con Huffman: E[ℓ_Huff(M)] ∈ [H(M), H(M) + 1);
- índice con aritmética de un solo símbolo (Shannon–Fano–Elias): E[⌈−log2 P_M(M)⌉ + 1];
- índice con aritmética sobre bloques largos, que es también la codificación aritmética semántica sobre conjuntos
  sinónimos (Liang et al. 2025): H(M) bits por significado;
- camino sintáctico completo (se envía el camino tal como se dijo, sin selección): H(Π) = H(M) + H(Π|M);
- código embebido jerárquico: Huffman por nivel sobre la cadena de clases (Parte 5).
Además, el tamaño del conocimiento que necesita el decodificador (número de probabilidades libres).
"""
from __future__ import annotations

import heapq
import math

import networkx as nx

from .graphs import TrajectoryNetwork


def huffman_lengths(probs) -> list:
    """Longitudes de un código de Huffman binario (0 si hay un solo símbolo con probabilidad positiva)."""
    probs = list(probs)
    live = [i for i, x in enumerate(probs) if x > 0]
    L = [0] * len(probs)
    if len(live) <= 1:
        return L
    heap = [(probs[i], k, [i]) for k, i in enumerate(live)]
    heapq.heapify(heap)
    tie = len(heap)
    while len(heap) > 1:
        a, _, A = heapq.heappop(heap)
        b, _, B = heapq.heappop(heap)
        for i in A + B:
            L[i] += 1
        heapq.heappush(heap, (a + b, tie, A + B))
        tie += 1
    return L


def huffman_expected_length(probs) -> float:
    probs = list(probs)
    return sum(x * l for x, l in zip(probs, huffman_lengths(probs)))


def sfe_expected_length(probs) -> float:
    """Aritmética de un solo símbolo (Shannon–Fano–Elias): E[⌈−log2 p⌉ + 1]."""
    return sum(x * (math.ceil(-math.log2(x) - 1e-12) + 1) for x in probs if x > 0)


def entropy_bits(probs) -> float:
    return -sum(x * math.log2(x) for x in probs if x > 0)


def decoder_parameters(net: TrajectoryNetwork, n_meanings: int) -> dict:
    """Probabilidades libres que el decodificador debe conocer.

    índice: |M| − 1 (la tabla de P_M) más la lista de significados;
    camino: Σ_v (deg⁺(v) − 1) sobre los nodos con más de un sucesor (los núcleos del grafo).
    """
    return {"index": n_meanings - 1, "path": sum(len(s) - 1 for s in net.succ.values() if len(s) > 1)}


def embedded_huffman_lengths(chain, p) -> list:
    """Longitud esperada acumulada del código embebido jerárquico hasta cada nivel j = 0..h.

    En el nivel j, cada clase de Q_{j−1} codifica cuál de sus hijas contiene a M con un Huffman propio
    construido con las probabilidades condicionadas.
    """
    out, acc = [0.0], 0.0
    for j in range(1, len(chain.classes)):
        parent = {k[:-1] for k in chain.keys[j]}
        for pk in parent:
            kids = [sum(p[i] for i in B) for k, B in zip(chain.keys[j], chain.classes[j]) if k[:-1] == pk]
            z = sum(kids)
            if z > 0:
                acc += z * huffman_expected_length([x / z for x in kids])
        out.append(acc)
    return out


def viterbi_dp(net: TrajectoryNetwork, meaning_of_terminal: dict) -> dict:
    """Selección de Viterbi por programación dinámica en orden topológico, O(|V| + |E|).

    Válida cuando el significado depende solo del terminal (Parte 1, Teorema 2(c)). Devuelve, por significado,
    (probabilidad del camino más probable, camino); los empates se rompen por repr del nodo predecesor.
    """
    g = nx.DiGraph([(v, u) for v, s in net.succ.items() for u in s])
    g.add_nodes_from(net.succ)
    best = {net.root: (1.0, None)}
    for v in nx.topological_sort(g):
        if v not in best:
            continue
        pv = best[v][0]
        for u, q in net.p.get(v, {}).items():
            cand = (pv * q, v)
            if u not in best or (cand[0], repr(cand[1])) > (best[u][0], repr(best[u][1])):
                best[u] = cand
    out = {}
    for t in net.terminals:
        if t not in best:
            continue
        m = meaning_of_terminal[t]
        if m not in out or best[t][0] > out[m][0]:
            path, x = [t], t
            while best[x][1] is not None:
                x = best[x][1]
                path.append(x)
            out[m] = (best[t][0], tuple(reversed(path)))
    return out
