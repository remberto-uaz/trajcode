"""Pruebas de trajcode.progressive (T35): Teorema 1, Proposición 2 y Proposición 3 de la Parte 5."""
import itertools

import numpy as np
from hypothesis import given, settings, strategies as st

from trajcode.graphs import make_ontology_tree
from trajcode.progressive import (level_chain, expected_distortion, optimal_allocation, progressivity_cost,
                                  uniform_index_distortion, level_bits, simulate_layered)
from trajcode.channel import BSC, BIAWGN, PolarCode, eps_normal, simulate_blocks


def _tree(seed, depth=3):
    leaves, d = make_ontology_tree(depth, 3, seed=seed)
    p = np.random.default_rng(seed).dirichlet(np.ones(len(leaves)))
    return leaves, np.asarray(d, float), p


def test_cadena_de_niveles_es_monotona_y_suma_hm():
    leaves, d, p = _tree(7)
    ch = level_chain(leaves, p, d)
    assert ch.dbar[0] >= ch.dbar[-1] == 0.0
    assert all(a >= b - 1e-12 for a, b in zip(ch.dbar, ch.dbar[1:]))
    assert abs(sum(ch.Hcond) - ch.H[-1]) < 1e-12
    assert ch.H[0] == 0.0


@given(st.integers(0, 50), st.lists(st.floats(0, 1), min_size=3, max_size=3))
@settings(max_examples=60, deadline=None)
def test_distorsion_esperada_extremos_y_monotonia(seed, eps):
    leaves, d, p = _tree(seed)
    ch = level_chain(leaves, p, d)
    h = len(ch.dbar) - 1
    eps = (eps + [0.0] * h)[:h]
    assert abs(expected_distortion(ch.dbar, [0.0] * h) - ch.dbar[-1]) < 1e-12
    assert abs(expected_distortion(ch.dbar, [1.0] * h) - ch.dbar[0]) < 1e-12
    base = expected_distortion(ch.dbar, eps)
    for j in range(h):
        up = list(eps); up[j] = min(1.0, up[j] + 0.1)
        assert expected_distortion(ch.dbar, up) >= base - 1e-12


@given(st.integers(0, 60), st.integers(2, 12), st.floats(0.02, 0.15))
@settings(max_examples=25, deadline=None)
def test_programacion_dinamica_igual_a_busqueda_exhaustiva(seed, extra, q):
    leaves, d, p = _tree(seed)
    ch = level_chain(leaves, p, d)
    h = len(ch.dbar) - 1
    k = level_bits(leaves)
    n = sum(k) + extra
    f = lambda nj, kj: eps_normal(nj, kj, BSC(q))
    best, alloc = optimal_allocation(ch.dbar, k, n, f)
    assert sum(alloc) == n
    brute = min(expected_distortion(ch.dbar, [f(a, kk) for a, kk in zip(al, k)])
                for al in itertools.product(range(n + 1), repeat=h) if sum(al) == n)
    assert abs(best - brute) < 1e-12
    assert abs(expected_distortion(ch.dbar, [f(a, kk) for a, kk in zip(alloc, k)]) - best) < 1e-12


def test_costo_de_identificar_clases():
    for seed in (201, 205, 210):
        leaves, d, p = _tree(seed, depth=2)
        ch = level_chain(leaves, p, d)
        delta = progressivity_cost(ch, p, d)
        assert abs(delta[0]) < 1e-6 and abs(delta[-1]) < 1e-6
        for j, cls in enumerate(ch.classes):
            if 1 < len(cls) < len(leaves):
                assert delta[j] > 0


def test_linea_base_uniforme():
    assert uniform_index_distortion(0.4, 0.25) == 0.1


def test_monte_carlo_con_codigos_polares_coincide_con_el_teorema_1():
    """Integración: transmisión por capas con códigos polares reales y CRC en BI-AWGN."""
    leaves, d, p = _tree(7)
    ch = level_chain(leaves, p, d)
    k = level_bits(leaves)
    awgn = BIAWGN(10 ** (1.0 / 10))
    codes = [PolarCode(16, kj, awgn, crc=4) for kj in k]
    eps = [simulate_blocks(c, awgn, frames=3000, seed=10 + j)[0] for j, c in enumerate(codes)]
    mc, se = simulate_layered(leaves, p, d, ch, codes, awgn, trials=3000, seed=5)
    assert abs(mc - expected_distortion(ch.dbar, eps)) < 4 * se + 0.01


# ---------------------------------------------------------------- simulación por lotes (T36)
def test_lotes_reproducen_el_teorema_1_y_la_linea_base_uniforme():
    from trajcode.progressive import simulate_layered_batch, simulate_uniform_batch, index_bits
    leaves, d, p = _tree(7)
    ch = level_chain(leaves, p, d)
    k = level_bits(leaves)
    B = 8
    awgn = BIAWGN(10 ** (-1.0 / 10))
    codes = [PolarCode(32, B * kj, awgn, crc=8) for kj in k]
    eps = [simulate_blocks(c, awgn, frames=3000, seed=20 + j)[0] for j, c in enumerate(codes)]
    mc, se, und = simulate_layered_batch(leaves, p, d, ch, codes, awgn, B=B, trials=3000, seed=6)
    assert abs(mc - expected_distortion(ch.dbar, eps)) < 4 * se + 0.01
    assert 0 <= und <= 0.01
    ku = index_bits(len(leaves))
    code_u = PolarCode(64, B * ku, awgn, crc=8)
    eps_u = simulate_blocks(code_u, awgn, frames=3000, seed=30)[0]
    mu_, se_u, _ = simulate_uniform_batch(p, d, ch, code_u, awgn, B=B, trials=3000, seed=7)
    assert abs(mu_ - uniform_index_distortion(ch.dbar[0], eps_u)) < 4 * se_u + 0.01
