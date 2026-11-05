"""T15–T16: pruebas de trajcode.alignment (Parte 2) y trajcode.distortion (Parte 3)."""
import itertools, math
import numpy as np
from hypothesis import given, settings, strategies as st

from trajcode.graphs import make_random_dag, make_meaning_map, make_toy_week_graph, make_ontology_tree
from trajcode.measures import path_measure, meaning_law, entropy, visit_probabilities, kl_nodewise, perturb_kernel
from trajcode.alignment import (receiver_measure, drop_edges, misalignment, local_bounds, meaning_phrasing_split,
                                detection_probability, silent_error_probability, uniform_receiver,
                                teach_set, teach_cost)
from trajcode.distortion import (geodesic_matrix, quotient_points, lower_envelope, envelope_at, blahut_arimoto,
                                 rd_at, refinability_residual, diameter, nested_quotient_rates)

SEEDS = st.integers(0, 10**6)
SIZES = st.integers(4, 10)
TOL = 1e-9


# =============================== T15 — Parte 2 ===============================
@given(seed=SEEDS, n=SIZES, frac=st.floats(0.05, 1.0))
@settings(max_examples=100, deadline=None)
def test_cotas_locales_y_separacion(seed, n, frac):
    net = make_random_dag(n, seed=seed); netR = perturb_kernel(net, frac, seed + 7)
    mu = make_meaning_map(net, 3, "context", seed)
    A = misalignment(net, netR.p)
    lo, hi = local_bounds(net, netR.p)
    assert lo - TOL <= A <= hi + TOL                                      # Teorema 1(d)
    ds, cond = meaning_phrasing_split(net, netR.p, mu)
    assert abs(A - (ds + cond)) < TOL and ds >= -TOL and cond >= -TOL      # Teorema 1(e)


@given(seed=SEEDS, n=SIZES)
@settings(max_examples=100, deadline=None)
def test_aristas_faltantes_se_detectan(seed, n):
    net = make_random_dag(n, seed=seed)
    qR, removed = drop_edges(net, fraction=0.3, seed=seed + 3)
    P = path_measure(net); Q = receiver_measure(net, qR)
    pdet = detection_probability(net, qR)
    assert abs(pdet - sum(x for w, x in P.items() if Q[w] == 0)) < TOL     # Proposición 1(ii)
    assert (misalignment(net, qR) == math.inf) == (pdet > 0)              # Teorema 1(c)


def test_ediciones_de_etiqueta_son_silenciosas():
    net, mu = make_toy_week_graph()
    muR = {w: ({"Tue'": "Tuesday", "Wed'": "Wednesday", "Thu": "Thursday"}[w[-1]] if w[2] == "b" else m)
           for w, m in mu.items()}
    assert misalignment(net, net.p) == 0                                  # sin costo en bits
    assert round(silent_error_probability(net, mu, muR), 3) == 0.400      # Proposición 1(iii)


@given(seed=SEEDS, n=SIZES)
@settings(max_examples=100, deadline=None)
def test_receptor_sin_estadisticas(seed, n):
    net = make_random_dag(n, seed=seed); nu = visit_probabilities(net)
    qU = uniform_receiver(net)
    rate = entropy(path_measure(net).values()) + misalignment(net, qU)
    assert abs(rate - sum(nu[v] * math.log2(len(net.succ[v])) for v in net.p)) < TOL   # Corolario 1(b)


@given(seed=SEEDS, n=st.integers(4, 8), N=st.integers(1, 200))
@settings(max_examples=60, deadline=None)
def test_ensenar_es_separable_y_optimo(seed, n, N):
    net = make_random_dag(n, seed=seed); netR = perturb_kernel(net, 1.0, seed + 1)
    rng = np.random.default_rng(seed); cost = {v: float(rng.uniform(0.5, 20)) for v in net.p}
    T = teach_set(net, netR.p, cost, N)
    best = min(teach_cost(net, netR.p, cost, N, set(S))
               for r in range(len(net.p) + 1) for S in itertools.combinations(net.p, r))
    assert abs(teach_cost(net, netR.p, cost, N, T) - best) < 1e-7          # Corolario 2


def test_ejemplo_semana_parte2():
    net, mu = make_toy_week_graph()
    qR = {v: dict(r) for v, r in net.p.items()}; qR["a"] = {"Mon": 1/3, "Tue": 1/3, "Wed": 1/3}
    lo, hi = local_bounds(net, qR); ds, cond = meaning_phrasing_split(net, qR, mu)
    assert [round(x, 3) for x in (misalignment(net, qR), lo, hi, ds, cond)] == [0.060, 0.048, 0.951, 0.036, 0.024]
    q2 = {v: dict(r) for v, r in net.p.items()}; q2["b"] = {"Tue'": 0.5, "Wed'": 0.5, "Thu": 0.0}
    assert round(detection_probability(net, q2), 3) == 0.080
    assert round(entropy(path_measure(net).values()) + misalignment(net, uniform_receiver(net)), 3) == 2.585


# =============================== T16 — Parte 3 ===============================
def test_blahut_arimoto_reproduce_hamming_binario():
    p = np.array([0.3, 0.7]); d = 1 - np.eye(2)
    for D in (0.05, 0.1, 0.2):
        R, Dr = rd_at(p, d, D)
        h = lambda x: -x * math.log2(x) - (1 - x) * math.log2(1 - x)
        assert abs(Dr - D) < 1e-4 and abs(R - (h(0.3) - h(D))) < 1e-4


@given(seed=SEEDS, n=st.integers(3, 6))
@settings(max_examples=25, deadline=None)
def test_cocientes_por_encima_de_RD_y_extremos(seed, n):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n)); d = geodesic_matrix(np.ones((n, n)) - np.eye(n) if seed % 2 else
                                                       np.eye(n, k=1) + np.eye(n, k=-1))
    H = lower_envelope(quotient_points(p, d))
    assert abs(H[0][1] - entropy(p)) < 1e-9 and abs(H[-1][1]) < 1e-9       # Teorema 1(b)
    for beta in (0.7, 2.0, 5.0):
        R, D, _ = blahut_arimoto(p, d, beta)
        assert R <= envelope_at(H, D) + 1e-6                               # Teorema 1(a)


@given(seed=SEEDS, depth=st.integers(1, 3))
@settings(max_examples=40, deadline=None)
def test_cocientes_anidados_y_diametro(seed, depth):
    leaves, d = make_ontology_tree(depth, 3, seed)
    rng = np.random.default_rng(seed); p = rng.dirichlet(np.ones(len(leaves)))
    rates = nested_quotient_rates(leaves, p)
    for l in range(depth + 1):
        classes = {}
        for i, lf in enumerate(leaves):
            classes.setdefault(lf[:l], []).append(i)
        assert diameter(np.array(d), list(classes.values())) <= depth - l        # diam(Q_l) <= h - l
    assert all(a <= b + TOL for a, b in zip(rates, rates[1:]))                   # H(Q_l) crece con l
    assert abs(rates[-1] - entropy(p)) < 1e-9                                   # Proposición 1


def test_control_hamming_y_conjetura_uniforme_refinables():
    rng = np.random.default_rng(3)
    for _ in range(8):
        p = rng.dirichlet(np.ones(5)); d = 1 - np.eye(5)
        b1, b2 = sorted(rng.uniform(0.5, 4, 2))
        assert refinability_residual(blahut_arimoto(p, d, b1)[2], blahut_arimoto(p, d, b2)[2]) < 1e-6
    leaves, d = make_ontology_tree(3, 2, seed=1); n = len(leaves); p = np.ones(n) / n
    for b1, b2 in [(0.8, 2.0), (1.5, 4.0)]:
        assert refinability_residual(blahut_arimoto(p, np.array(d, float), b1)[2],
                                     blahut_arimoto(p, np.array(d, float), b2)[2]) < 1e-6


def test_ejemplo_semana_parte3():
    p = np.array([0.5, 0.3, 0.2]); d = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], float)
    H = lower_envelope(quotient_points(p, d))
    assert [(round(a, 3), round(b, 3)) for a, b in H] == [(0.0, 1.485), (0.3, 0.722), (0.7, 0.0)]
    R, D = rd_at(p, d, 0.3)
    assert round(R, 3) == 0.439 and round(envelope_at(H, 0.3) - R, 3) == 0.283


@given(n=st.integers(2, 9))
@settings(max_examples=20, deadline=None)
def test_distancias_geodesicas_camino_y_ciclo(n):
    path = np.eye(n, k=1) + np.eye(n, k=-1)
    assert np.array_equal(geodesic_matrix(path), np.abs(np.subtract.outer(np.arange(n), np.arange(n))))
    cyc = path.copy(); cyc[0, n - 1] = cyc[n - 1, 0] = 1
    a = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    assert np.array_equal(geodesic_matrix(cyc), np.minimum(a, n - a))
