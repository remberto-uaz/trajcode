"""trajcode.graphs — redes de trayectoria, generadores y ontologías (T11; plan.md § 7; theory/00-modelo.tex).

Representación deliberadamente simple (constitution P8): diccionarios de sucesores y de núcleos de ruteo.
NetworkX se usa solo para validar la aciclicidad.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Hashable, Iterator

import networkx as nx

Node = Hashable
Walk = tuple

_TOL = 1e-9


@dataclass
class TrajectoryNetwork:
    """Red de trayectoria G = (V, E, r, T, ℓ, p) de la Def. 1 (Parte 0).

    succ:   nodo -> lista de sucesores (lista vacía en los terminales)
    p:      nodo no terminal -> {sucesor: probabilidad de ruteo}
    root:   nodo raíz r
    labels: etiqueta ℓ(v) opcional
    """
    succ: dict
    p: dict
    root: Node
    labels: dict = field(default_factory=dict)

    @property
    def terminals(self) -> set:
        return {v for v, s in self.succ.items() if not s}

    def validate(self) -> None:
        """Comprueba los Supuestos 1–2 de la Parte 0; lanza ValueError si alguno falla."""
        if self.root not in self.succ:
            raise ValueError("la raíz no está en el grafo")
        for v, s in self.succ.items():
            for u in s:
                if u not in self.succ:
                    raise ValueError(f"sucesor {u!r} de {v!r} no declarado")
        g = nx.DiGraph([(v, u) for v, s in self.succ.items() for u in s])
        g.add_nodes_from(self.succ)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("el grafo tiene ciclos (Supuesto 1)")
        if not self.terminals:
            raise ValueError("no hay terminales")
        for v, s in self.succ.items():
            if not s:
                continue
            row = self.p.get(v, {})
            if set(row) != set(s):
                raise ValueError(f"el núcleo de {v!r} no coincide con sus sucesores")
            if any(x <= 0 for x in row.values()):
                raise ValueError(f"probabilidad no positiva en {v!r}")
            if abs(sum(row.values()) - 1) > _TOL:
                raise ValueError(f"el núcleo de {v!r} no suma 1")


def walks(net: TrajectoryNetwork) -> Iterator[tuple[Walk, float]]:
    """Enumera los caminos completos con su probabilidad P(π), Ec. (1) de la Parte 0."""
    stack = [((net.root,), 1.0)]
    while stack:
        walk, pr = stack.pop()
        v = walk[-1]
        if not net.succ[v]:
            yield walk, pr
            continue
        for u, q in net.p[v].items():
            stack.append((walk + (u,), pr * q))


def make_toy_week_graph() -> tuple[TrajectoryNetwork, dict]:
    """Ejemplo 'qué día es hoy' de la Parte 0 (Fig. 1). Devuelve (red, mapa camino -> significado)."""
    succ = {"r": ["c"], "c": ["a", "b"], "a": ["Mon", "Tue", "Wed"], "b": ["Tue'", "Wed'", "Thu"],
            "Mon": [], "Tue": [], "Wed": [], "Tue'": [], "Wed'": [], "Thu": []}
    p = {"r": {"c": 1.0}, "c": {"a": 0.6, "b": 0.4},
         "a": {"Mon": 0.5, "Tue": 0.3, "Wed": 0.2}, "b": {"Tue'": 0.5, "Wed'": 0.3, "Thu": 0.2}}
    labels = {"c": "according to the calendar", "a": "today is", "b": "tomorrow is"}
    net = TrajectoryNetwork(succ, p, "r", labels)
    today = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
             "Tue'": "Monday", "Wed'": "Tuesday", "Thu": "Wednesday"}
    mu = {w: today[w[-1]] for w, _ in walks(net)}
    return net, mu


def make_random_dag(n: int, max_out: int = 3, seed: int = 0) -> TrajectoryNetwork:
    """DAG aleatorio con raíz 0 y aristas hacia adelante; se conserva solo la parte alcanzable desde la raíz."""
    rng = random.Random(seed)
    succ = {v: [] for v in range(n)}
    for v in range(n - 1):
        k = rng.randint(1, max_out) if v < n - 3 else rng.randint(0, max_out)
        cand = list(range(v + 1, n))
        rng.shuffle(cand)
        succ[v] = sorted(cand[:k])
    reach, frontier = {0}, [0]
    while frontier:
        v = frontier.pop()
        for u in succ[v]:
            if u not in reach:
                reach.add(u); frontier.append(u)
    succ = {v: [u for u in succ[v]] for v in sorted(reach)}
    p = {}
    for v, s in succ.items():
        if s:
            w = [rng.random() + 0.05 for _ in s]
            z = sum(w)
            p[v] = {u: x / z for u, x in zip(s, w)}
    net = TrajectoryNetwork(succ, p, 0)
    net.validate()
    return net


def make_meaning_map(net: TrajectoryNetwork, n_meanings: int, kind: str = "endpoint", seed: int = 0) -> dict:
    """Mapa camino -> significado.

    kind="endpoint": el significado depende solo del nodo final (Teorema 2(c), Parte 1).
    kind="context":  depende también del primer paso (polisemia; Teorema 2(d), Parte 1).
    """
    if kind not in ("endpoint", "context"):
        raise ValueError("kind debe ser 'endpoint' o 'context'")
    rng = random.Random(seed)
    lab = {v: rng.randrange(n_meanings) for v in net.succ}
    mu = {}
    for w, _ in walks(net):
        if kind == "endpoint" or len(w) < 2:
            mu[w] = lab[w[-1]]
        else:
            mu[w] = (lab[w[1]] + lab[w[-1]]) % n_meanings
    return mu


def make_ontology_tree(depth: int, max_branch: int = 3, seed: int = 0) -> tuple[list, list]:
    """Árbol de ontología aleatorio. Devuelve (hojas, matriz de distancia ultramétrica d = h − profundidad(lca)).

    Ver Def. 6 de la Parte 0 y § 3 de la Parte 3.
    """
    rng = random.Random(seed)
    leaves = [()]
    for _ in range(depth):
        leaves = [l + (b,) for l in leaves for b in range(rng.randint(1, max_branch))]

    def lca_depth(x, y):
        k = 0
        while k < depth and x[k] == y[k]:
            k += 1
        return k

    d = [[depth - lca_depth(x, y) for y in leaves] for x in leaves]
    return leaves, d


def make_polysemy_example() -> tuple[TrajectoryNetwork, dict]:
    """Ejemplo 2 de la Parte 1: la 'palabra' x significa cosas distintas según el contexto a o b.

    r -> a (0.6) | b (0.4);  a, b -> x;  x -> t1 (0.5) | t2 (0.3) | t3 (0.2).
    Significados: (a,t1)=A, (a,t2)=A, (a,t3)=B, (b,t1)=A, (b,t2)=C, (b,t3)=C.
    """
    succ = {"r": ["a", "b"], "a": ["x"], "b": ["x"], "x": ["t1", "t2", "t3"], "t1": [], "t2": [], "t3": []}
    p = {"r": {"a": 0.6, "b": 0.4}, "a": {"x": 1.0}, "b": {"x": 1.0}, "x": {"t1": 0.5, "t2": 0.3, "t3": 0.2}}
    net = TrajectoryNetwork(succ, p, "r", {"x": "polysemous word"})
    table = {("a", "t1"): "A", ("a", "t2"): "A", ("a", "t3"): "B",
             ("b", "t1"): "A", ("b", "t2"): "C", ("b", "t3"): "C"}
    mu = {w: table[(w[1], w[-1])] for w, _ in walks(net)}
    return net, mu


def make_layered_dag(layers: int, width: int, max_out: int = 3, seed: int = 0) -> TrajectoryNetwork:
    """DAG por capas con raíz única y todos los nodos alcanzables (para medir tiempos en grafos grandes, T39).

    Capa 0 = raíz; capas 1..layers con `width` nodos; cada nodo enlaza con 1..max_out nodos de la capa siguiente
    y cada nodo de una capa recibe al menos un arco. La última capa son los terminales.
    """
    rng = random.Random(seed)
    L = [[(0, 0)]] + [[(i, j) for j in range(width)] for i in range(1, layers + 1)]
    succ = {v: [] for layer in L for v in layer}
    for a, b in zip(L, L[1:]):
        for v in a:
            succ[v] = rng.sample(b, min(len(b), rng.randint(1, max_out)))
        orphans = set(b) - {u for v in a for u in succ[v]}
        for u in orphans:
            succ[rng.choice(a)].append(u)
    p = {}
    for v, s in succ.items():
        if s:
            w = [rng.random() + 0.05 for _ in s]
            z = sum(w)
            p[v] = {u: x / z for u, x in zip(s, w)}
    net = TrajectoryNetwork(succ, p, (0, 0))
    net.validate()
    return net
