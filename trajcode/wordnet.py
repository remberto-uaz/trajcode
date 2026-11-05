"""Subárboles de la jerarquía de sustantivos de WordNet 3.0 como redes de trayectoria (T37, ADR-T03).

Lectura con la biblioteca estándar desde data/wordnet/wordnet.zip (make data):
- data.noun: punteros de hiperónimo `@` y de hiperónimo de instancia `@i`;
- index.noun: sentidos de cada lema, en orden de frecuencia;
- index.sense: número de etiquetas en SemCor por sentido (tag_cnt), que da la masa de cada synset.

Construcción de un subárbol (ADR-T03):
1. nodos alcanzables desde la raíz por hipónimos en a lo más `depth` pasos; terminales = nodos sin hijos dentro;
2. masa de cada terminal w(t) = α + etiquetas de t y de todos sus descendientes en WordNet (cada uno una vez);
3. se conservan los `max_leaves` terminales de mayor masa y los nodos que llegan a alguno;
4. ruteo p(u | v) ∝ masa de las hojas conservadas alcanzables desde u; significado = terminal (determinado por el
   extremo). La herencia múltiple da varios caminos al mismo terminal: sinonimia.
"""
from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from trajcode.graphs import TrajectoryNetwork, walks
from trajcode.measures import path_measure
from trajcode.coding import viterbi_selection


@dataclass
class WordNetNouns:
    hypernyms: dict          # offset -> [offsets]
    lemma: dict              # offset -> primer lema
    count: dict              # offset -> etiquetas en SemCor (suma de sus sentidos)
    senses: dict             # lema -> [offsets] en orden de sentido
    _children: dict = field(default_factory=dict, repr=False)

    def synset(self, lemma: str, sense: int = 1) -> int:
        return self.senses[lemma][sense - 1]

    @property
    def children(self) -> dict:
        if not self._children:
            ch: dict = {}
            for s, hs in self.hypernyms.items():
                for h in hs:
                    ch.setdefault(h, []).append(s)
            self._children = {k: sorted(v) for k, v in ch.items()}
        return self._children


def load_nouns(zip_path) -> WordNetNouns:
    hyp, lem, cnt, senses = {}, {}, {}, {}
    with zipfile.ZipFile(zip_path) as z:
        for line in z.read("wordnet/data.noun").decode("latin-1").splitlines():
            if line.startswith("  "):
                continue
            f = line.split(" | ")[0].split()
            off, wc = int(f[0]), int(f[3], 16)
            lem[off] = f[4]
            i = 4 + 2 * wc
            nptr = int(f[i]); i += 1
            hs = []
            for _ in range(nptr):
                sym, tgt, pos = f[i], int(f[i + 1]), f[i + 2]
                if sym in ("@", "@i") and pos == "n":
                    hs.append(tgt)
                i += 4
            hyp[off] = hs
        for line in z.read("wordnet/index.noun").decode("latin-1").splitlines():
            if line.startswith("  "):
                continue
            f = line.split()
            n_syn = int(f[2])
            senses[f[0]] = [int(x) for x in f[-n_syn:]]
        for line in z.read("wordnet/index.sense").decode("latin-1").splitlines():
            key, off, _, tag = line.split()
            if "%1:" in key:
                cnt[int(off)] = cnt.get(int(off), 0) + int(tag)
    return WordNetNouns(hyp, lem, cnt, senses)


@dataclass
class WordNetSubtree:
    root: int
    net: TrajectoryNetwork
    mu: dict                 # camino -> significado (offset del terminal)
    meanings: list           # offsets de los significados, ordenados
    weight: dict             # offset del terminal -> masa w(t)
    reach: dict              # nodo -> conjunto de terminales alcanzables
    labels: dict


def _cumulative_count(wn: WordNetNouns, s: int, memo: dict) -> int:
    """Etiquetas de s y de sus descendientes en WordNet, cada synset una sola vez."""
    if s in memo:
        return memo[s]
    seen, stack, tot = {s}, [s], 0
    while stack:
        v = stack.pop()
        tot += wn.count.get(v, 0)
        for u in wn.children.get(v, []):
            if u not in seen:
                seen.add(u); stack.append(u)
    memo[s] = tot
    return tot


def subtree(wn: WordNetNouns, root: int, depth: int, max_leaves: int, alpha: float = 1.0) -> WordNetSubtree:
    dist = {root: 0}
    frontier = [root]
    while frontier:
        nxt = []
        for v in frontier:
            if dist[v] == depth:
                continue
            for u in wn.children.get(v, []):
                if u not in dist:
                    dist[u] = dist[v] + 1; nxt.append(u)
        frontier = nxt
    S = set(dist)
    kids = {v: [u for u in wn.children.get(v, []) if u in S] for v in S}
    terms = [v for v in S if not kids[v]]
    memo: dict = {}
    w = {t: alpha + _cumulative_count(wn, t, memo) for t in terms}
    keep = set(sorted(terms, key=lambda t: (-w[t], t))[:max_leaves])
    # nodos que alcanzan alguna hoja conservada
    g = nx.DiGraph()
    g.add_nodes_from(S)
    g.add_edges_from((v, u) for v in S for u in kids[v])
    order = list(nx.topological_sort(g))
    reach = {}
    for v in reversed(order):
        r = {v} if v in keep else set()
        for u in kids[v]:
            r |= reach.get(u, set())
        reach[v] = r
    V = {v for v in S if reach[v]}
    succ = {v: [u for u in kids[v] if u in V] for v in V}
    mass = {v: sum(w[t] for t in reach[v]) for v in V}
    p = {v: {u: mass[u] / sum(mass[x] for x in s) for u in s} for v, s in succ.items() if s}
    labels = {v: wn.lemma[v] for v in V}
    net = TrajectoryNetwork(succ=succ, p=p, root=root, labels=labels)
    mu = {walk: walk[-1] for walk, _ in walks(net)}
    meanings = sorted(keep)
    return WordNetSubtree(root, net, mu, meanings, {t: w[t] for t in meanings}, {v: reach[v] for v in V}, labels)


def viterbi_leaves(st: WordNetSubtree) -> list:
    """Hojas como tuplas de índices de hijo a lo largo del camino más probable de cada significado (rellenadas con 0
    hasta la misma longitud), en el orden de st.meanings. Es la cadena de clases que usa la Parte 5."""
    P = path_measure(st.net)
    f = viterbi_selection(P, st.mu)
    tuples = []
    for m in st.meanings:
        w = f[m]
        tuples.append(tuple(st.net.succ[a].index(b) for a, b in zip(w, w[1:])))
    L = max(len(t) for t in tuples)
    return [t + (0,) * (L - len(t)) for t in tuples]


def geodesic_distortion(st: WordNetSubtree) -> np.ndarray:
    """Distancia de camino más corto entre significados en el grafo no dirigido del subárbol."""
    g = nx.Graph([(v, u) for v, s in st.net.succ.items() for u in s])
    n = len(st.meanings)
    d = np.zeros((n, n))
    for i, a in enumerate(st.meanings):
        L = nx.single_source_shortest_path_length(g, a)
        for j, b in enumerate(st.meanings):
            d[i, j] = L[b]
    return d


def resnik_distortion(st: WordNetSubtree, tree: list | None = None) -> np.ndarray:
    """d(m, m̂) = IC_max − IC(ancestro común más informativo), 0 en la diagonal [Resnik 1999].

    Se calcula sobre el árbol de la selección de Viterbi (por omisión), donde la masa de un nodo es la de las hojas
    cuyo camino seleccionado pasa por él; así la distorsión es una ultramétrica."""
    leaves = tree if tree is not None else viterbi_leaves(st)
    wts = np.array([st.weight[m] for m in st.meanings], float)
    total = wts.sum()
    L = len(leaves[0])
    ic = {}
    for j in range(L + 1):
        mass: dict = {}
        for lf, x in zip(leaves, wts):
            mass[lf[:j]] = mass.get(lf[:j], 0.0) + x
        for k, x in mass.items():
            ic[k] = -math.log2(x / total)
    ic_max = max(-math.log2(x / total) for x in wts)
    n = len(leaves)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            k = 0
            while k < L and leaves[i][k] == leaves[j][k]:
                k += 1
            d[i, j] = ic_max - ic[leaves[i][:k]]
    return d
