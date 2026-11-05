"""T13–T14: pruebas del módulo trajcode.coding (constitution P4.2). Respaldo: theory/01-memoria.tex."""
import math
from hypothesis import given, settings, strategies as st

from trajcode.graphs import make_random_dag, make_meaning_map, make_toy_week_graph, make_polysemy_example
from trajcode.measures import path_measure, meaning_law, entropy, cond_entropy_walk_given_meaning
from trajcode.coding import (ArithmeticPathCoder, native_kernel, prefix_kernel, viterbi_selection,
                             selection_law, native_length, synonymy_gap, adapted_length,
                             memoryless_adapted_length, future_law_counts, memory_states, bitsback_net_length,
                             min_memoryless_adapted_length, all_selections)

SEEDS = st.integers(0, 10**6)
SIZES = st.integers(3, 10)
TOL = 1e-9


# ---------- T13: codificador aritmético exacto (Lema 1, Parte 0) ----------
@given(seed=SEEDS, n=SIZES)
@settings(max_examples=60, deadline=None)
def test_aritmetico_ida_y_vuelta_y_longitud_nativo(seed, n):
    net = make_random_dag(n, seed=seed)
    coder = ArithmeticPathCoder(net, native_kernel(net))
    for walk, pr in path_measure(net).items():
        bits = coder.encode(walk)
        assert coder.decode(bits) == walk
        assert len(bits) < -math.log2(pr) + 2


@given(seed=SEEDS, n=SIZES, k=st.integers(2, 4), kind=st.sampled_from(["endpoint", "context"]))
@settings(max_examples=60, deadline=None)
def test_aritmetico_ida_y_vuelta_adaptado(seed, n, k, kind):
    net = make_random_dag(n, seed=seed)
    mu = make_meaning_map(net, k, kind, seed)
    P = path_measure(net); f = viterbi_selection(P, mu); Q = selection_law(f, meaning_law(P, mu))
    coder = ArithmeticPathCoder(net, prefix_kernel(Q))
    for walk, q in Q.items():
        bits = coder.encode(walk)
        assert coder.decode(bits) == walk
        assert len(bits) < -math.log2(q) + 2


def test_nodo_de_grado_uno_cuesta_cero():
    net, _ = make_toy_week_graph()                         # r -> c tiene grado de salida 1
    coder = ArithmeticPathCoder(net, native_kernel(net))
    w = ("r", "c", "a", "Mon")
    assert abs(coder.ideal_length(w) - (-math.log2(0.6) - math.log2(0.5))) < 1e-12


# ---------- T14: Teorema 1 (códigos nativos) ----------
@given(seed=SEEDS, n=SIZES, k=st.integers(2, 4), kind=st.sampled_from(["endpoint", "context"]))
@settings(max_examples=120, deadline=None)
def test_teorema1_precio_del_ruteo_nativo(seed, n, k, kind):
    net = make_random_dag(n, seed=seed); mu = make_meaning_map(net, k, kind, seed)
    P = path_measure(net); PM = meaning_law(P, mu); HM = entropy(PM.values())
    f = viterbi_selection(P, mu); g = synonymy_gap(P, mu)
    Eg = sum(PM[m] * g[m] for m in PM)
    assert abs(native_length(f, P, PM) - (HM + Eg)) < TOL                  # (a)
    assert -TOL <= Eg <= cond_entropy_walk_given_meaning(P, mu) + TOL     # (b)
    for m in PM:
        assert g[m] <= math.log2(sum(1 for w in P if mu[w] == m)) + TOL
    for sel in all_selections(P, mu, limit=200):                          # (a): Viterbi es óptima
        assert native_length(sel, P, PM) >= native_length(f, P, PM) - TOL


# ---------- T14: Teorema 2 (códigos adaptados y memoria) ----------
@given(seed=SEEDS, n=SIZES, k=st.integers(2, 4), kind=st.sampled_from(["endpoint", "context"]))
@settings(max_examples=120, deadline=None)
def test_teorema2a_adaptado_alcanza_HM(seed, n, k, kind):
    net = make_random_dag(n, seed=seed); mu = make_meaning_map(net, k, kind, seed)
    P = path_measure(net); PM = meaning_law(P, mu)
    for sel in all_selections(P, mu, limit=50):
        assert abs(adapted_length(selection_law(sel, PM)) - entropy(PM.values())) < TOL


@given(seed=SEEDS, n=SIZES, k=st.integers(2, 4))
@settings(max_examples=120, deadline=None)
def test_teorema2c_significado_por_nodo_final_no_necesita_memoria(seed, n, k):
    net = make_random_dag(n, seed=seed); mu = make_meaning_map(net, k, "endpoint", seed)
    P = path_measure(net); PM = meaning_law(P, mu)
    Q = selection_law(viterbi_selection(P, mu), PM)
    assert memory_states(Q) == 1
    assert abs(memoryless_adapted_length(Q) - entropy(PM.values())) < TOL


def test_teorema2d_polisemia_obliga_a_llevar_memoria():
    net, mu = make_polysemy_example()
    P = path_measure(net); PM = meaning_law(P, mu); HM = entropy(PM.values())
    for sel in all_selections(P, mu):
        assert memory_states(selection_law(sel, PM)) == 2
    best = min_memoryless_adapted_length(P, mu)
    assert round(HM, 3) == 1.210 and round(best - HM, 3) == 0.041           # Ejemplo 2, Parte 1
    Qv = selection_law(viterbi_selection(P, mu), PM)
    assert round(memoryless_adapted_length(Qv) - HM, 3) == 0.722
    assert round(native_length(viterbi_selection(P, mu), P, PM) - HM, 3) == 0.950


# ---------- Proposición 1 (bits-back) ----------
@given(seed=SEEDS, n=SIZES, k=st.integers(2, 4), kind=st.sampled_from(["endpoint", "context"]))
@settings(max_examples=120, deadline=None)
def test_bitsback_neto_igual_HM(seed, n, k, kind):
    net = make_random_dag(n, seed=seed); mu = make_meaning_map(net, k, kind, seed)
    P = path_measure(net)
    assert abs(bitsback_net_length(P, mu) - entropy(meaning_law(P, mu).values())) < TOL


def test_ejemplo_semana_brecha_uniforme():
    net, mu = make_toy_week_graph()
    P = path_measure(net); g = synonymy_gap(P, mu)
    assert {round(x, 3) for x in g.values()} == {0.737}
