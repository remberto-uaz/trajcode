"""Pruebas de trajcode.baselines (T39)."""
import math

import numpy as np
from hypothesis import given, settings, strategies as st

from trajcode.graphs import make_random_dag, make_meaning_map, make_ontology_tree, walks
from trajcode.measures import path_measure, meaning_law, entropy
from trajcode.coding import viterbi_selection
from trajcode.progressive import level_chain
from trajcode.baselines import (huffman_lengths, huffman_expected_length, sfe_expected_length, entropy_bits,
                                decoder_parameters, embedded_huffman_lengths, viterbi_dp)


@given(st.lists(st.floats(0.001, 1.0), min_size=1, max_size=40))
@settings(max_examples=200, deadline=None)
def test_huffman_kraft_y_cotas(w):
    p = [x / sum(w) for x in w]
    L = huffman_lengths(p)
    if len(p) > 1:
        assert abs(sum(2.0 ** -l for l in L) - 1) < 1e-9          # Huffman es completo
    H = entropy_bits(p)
    assert H - 1e-9 <= huffman_expected_length(p) < H + 1 + 1e-9
    assert huffman_expected_length(p) <= sfe_expected_length(p) + 1e-9
    assert sfe_expected_length(p) < H + 2 + 1e-9


def test_huffman_valor_conocido():
    assert huffman_lengths([0.4, 0.2, 0.2, 0.1, 0.1]) in ([1, 2, 3, 4, 4], [1, 3, 3, 3, 3], [2, 2, 2, 3, 3])
    assert abs(huffman_expected_length([0.4, 0.2, 0.2, 0.1, 0.1]) - 2.2) < 1e-12


@given(st.integers(0, 400), st.integers(5, 14))
@settings(max_examples=80, deadline=None)
def test_viterbi_lineal_igual_a_la_enumeracion(seed, n):
    net = make_random_dag(n, seed=seed)
    mu = make_meaning_map(net, 3, "endpoint", seed)
    P = path_measure(net)
    f = viterbi_selection(P, mu)
    lab = {w[-1]: m for w, m in mu.items()}
    dp = viterbi_dp(net, lab)
    assert set(dp) == set(f)
    for m, (pr, path) in dp.items():
        assert abs(pr - P[f[m]]) < 1e-12
        assert mu[path] == m and abs(P[path] - pr) < 1e-12


@given(st.integers(0, 200))
@settings(max_examples=40, deadline=None)
def test_codigo_embebido_entre_entropia_y_entropia_mas_niveles(seed):
    leaves, d = make_ontology_tree(3, 3, seed=seed)
    p = np.random.default_rng(seed).dirichlet(np.ones(len(leaves)))
    ch = level_chain(leaves, p, d)
    L = embedded_huffman_lengths(ch, p)
    h = len(ch.H) - 1
    for j in range(h + 1):
        assert ch.H[j] - 1e-9 <= L[j] <= ch.H[j] + j + 1e-9
    assert all(a <= b + 1e-12 for a, b in zip(L, L[1:]))


def test_parametros_del_decodificador():
    net = make_random_dag(10, seed=1)
    assert decoder_parameters(net, 4)["index"] == 3
    assert decoder_parameters(net, 4)["path"] == sum(len(s) - 1 for s in net.succ.values() if s)


def test_dag_por_capas_alcanzable_y_valido():
    from trajcode.graphs import make_layered_dag
    from trajcode.measures import visit_probabilities
    net = make_layered_dag(6, 20, seed=3)
    nu = visit_probabilities(net)
    assert all(x > 0 for x in nu.values())
    assert len(net.terminals) == 20
