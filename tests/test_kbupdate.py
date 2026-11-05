"""Pruebas de trajcode.kbupdate (T38): Proposiciones C1 y C2 de la Parte 6."""
import random

import numpy as np
from hypothesis import given, settings, strategies as st

from trajcode.graphs import make_random_dag
from trajcode.measures import perturb_kernel, visit_probabilities
from trajcode.alignment import teach_set
from trajcode.kbupdate import (node_terms, teach_costs, separable_rule, expected_cost, exhaustive_offline,
                               realized_cost_fixed, hindsight_opt, online_rule, sync_on_first_use,
                               global_threshold, visit_matrix, stream_costs, make_site_graph, remodel)


def _instance(seed, n=12, frac=0.6):
    net_q = make_random_dag(n, seed=seed)
    net_s = perturb_kernel(net_q, frac, seed=seed + 1)
    terms = node_terms(net_s, net_q.p)
    nu = {v: a for v, (a, _) in terms.items()}
    D = {v: b for v, (_, b) in terms.items()}
    return net_s, net_q, nu, D


@given(st.integers(0, 200), st.integers(1, 5000))
@settings(max_examples=60, deadline=None)
def test_c1_regla_separable_igual_a_busqueda_exhaustiva(seed, N):
    net_s, net_q, nu, D = _instance(seed)
    if not D or len(D) > 12:
        return
    c = {v: random.Random(seed).uniform(1, 40) for v in D}
    best, _ = exhaustive_offline(nu, D, c, N)
    T = separable_rule(nu, D, c, N)
    assert abs(expected_cost(nu, D, c, N, T) - best) < 1e-9 * max(1.0, best)


def test_c1_coincide_con_el_corolario_2_de_la_parte_2():
    net_s, net_q, nu, D = _instance(3)
    c = teach_costs(net_s)
    for N in (1, 10, 100, 1000):
        assert separable_rule(nu, D, c, N) == teach_set(net_s, net_q.p, c, N)


def _random_stream(rng, D, N):
    keys = list(D)
    return [{v: rng.randint(0, 2) for v in rng.sample(keys, rng.randint(0, len(keys)))} for _ in range(N)]


@given(st.integers(0, 300), st.integers(1, 300))
@settings(max_examples=80, deadline=None)
def test_c2_a_lo_mas_el_doble_del_optimo_fuera_de_linea(seed, N):
    rng = random.Random(seed)
    D = {v: rng.uniform(0.01, 3.0) for v in range(rng.randint(1, 8))}
    c = {v: rng.uniform(0.5, 30.0) for v in D}
    stream = _random_stream(rng, D, N)
    on, _ = online_rule(stream, D, c)
    opt = hindsight_opt(stream, D, c)
    assert on <= 2 * opt + 1e-9
    assert on >= opt - 1e-9


def test_c2_cota_ajustada_en_el_peor_caso():
    """Rentas pequeñas y horizonte que termina justo al enseñar: el cociente tiende a 2."""
    D, c = {0: 2.0 ** -10}, {0: 1.0}          # potencias de 2: sumas exactas en coma flotante
    stream = [{0: 1}] * 1024
    on, taught = online_rule(stream, D, c)
    assert taught == {0: 1023}
    assert on / hindsight_opt(stream, D, c) == 2.0 - 2.0 ** -10


def test_heuristicas_con_cociente_no_acotado():
    D, c = {0: 1e-3}, {0: 50.0}
    assert sync_on_first_use([{0: 1}], D, c) / hindsight_opt([{0: 1}], D, c) > 1e4
    long = [{0: 1}] * 10 ** 6
    assert realized_cost_fixed(long, D, c, set()) / hindsight_opt(long, D, c) == 20.0
    # umbral global: un nodo caro con renta alta y muchos baratos con renta casi nula
    D2 = {0: 1.0, **{i: 1e-6 for i in range(1, 51)}}
    c2 = {0: 1.0, **{i: 1.0 for i in range(1, 51)}}
    s2 = [{v: 1 for v in D2}] * 30
    assert global_threshold(s2, D2, c2) / hindsight_opt(s2, D2, c2) > 10


@given(st.integers(0, 100))
@settings(max_examples=30, deadline=None)
def test_version_vectorizada_igual_a_la_de_referencia(seed):
    net, mu, floor_of = make_site_graph(seed=seed % 5)
    new = remodel(net, {v for v, f in floor_of.items() if f in ("b0f0", "b1f2")}, seed=seed)
    terms = node_terms(new, net.p)
    nodes = sorted(terms)
    D = {v: terms[v][1] for v in nodes}
    c = teach_costs(new)
    c = {v: c[v] for v in nodes}
    W, P, X = visit_matrix(new, nodes)
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(W), size=rng.integers(1, 3000), p=P / P.sum())
    fast = stream_costs(X, draws, np.array([D[v] for v in nodes]), np.array([c[v] for v in nodes]))
    stream = [{v: 1 for v in W[i] if v in D} for i in draws]
    assert abs(fast["online"] - online_rule(stream, D, c)[0]) < 1e-6
    assert abs(fast["hindsight"] - hindsight_opt(stream, D, c)) < 1e-6
    assert abs(fast["sync_first_use"] - sync_on_first_use(stream, D, c)) < 1e-6
    assert abs(fast["global"] - global_threshold(stream, D, c)) < 1e-6
    assert abs(fast["talk_only"] - realized_cost_fixed(stream, D, c, set())) < 1e-6


def test_sitio_rtls_tiene_sinonimia_y_la_remodelacion_es_local():
    net, mu, floor_of = make_site_graph()
    meanings = list(mu.values())
    assert len(set(meanings)) < len(meanings)                  # salas con dos descripciones
    edited = {v for v, f in floor_of.items() if f == "b0f1"}
    new = remodel(net, edited, seed=1)
    assert set(node_terms(new, net.p)) <= edited
    nu = visit_probabilities(new)
    assert abs(sum(nu[v] for v in new.terminals) - 1) < 1e-12
