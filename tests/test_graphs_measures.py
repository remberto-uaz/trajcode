"""T11–T12: pruebas de propiedades de trajcode.graphs y trajcode.measures (constitution P4.2).
Cada prueba nombra el enunciado de theory/ que respalda."""
import math
import pytest
from hypothesis import given, settings, strategies as st

from trajcode.graphs import (TrajectoryNetwork, make_toy_week_graph, make_random_dag,
                             make_meaning_map, make_ontology_tree, walks)
from trajcode.measures import (path_measure, meaning_law, entropy, cond_entropy_walk_given_meaning,
                               visit_probabilities, kl_paths, kl_nodewise, perturb_kernel)

SEEDS = st.integers(min_value=0, max_value=10**6)
SIZES = st.integers(min_value=3, max_value=11)
TOL = 1e-9


# ---------- graphs (T11) ----------
@given(seed=SEEDS, n=SIZES)
@settings(max_examples=150, deadline=None)
def test_random_dag_invariantes(seed, n):
    net = make_random_dag(n, max_out=3, seed=seed)
    net.validate()                                         # DAG, raíz, núcleos válidos
    for v, row in net.p.items():
        assert abs(sum(row.values()) - 1) < TOL            # Def. 1 (Parte 0)
        assert all(x > 0 for x in row.values())
        assert set(row) == set(net.succ[v])
    reach = {w for walk, _ in walks(net) for w in walk}
    assert reach == set(net.succ)                          # todo nodo es alcanzable (Supuesto 1)


def test_validate_rechaza_ciclos_y_nucleos_malos():
    with pytest.raises(ValueError):
        TrajectoryNetwork(succ={0: [1], 1: [0]}, p={0: {1: 1.0}, 1: {0: 1.0}}, root=0).validate()
    with pytest.raises(ValueError):
        TrajectoryNetwork(succ={0: [1, 2], 1: [], 2: []}, p={0: {1: 0.5, 2: 0.4}}, root=0).validate()


@given(seed=SEEDS, depth=st.integers(1, 4))
@settings(max_examples=80, deadline=None)
def test_ontologia_es_ultrametrica(seed, depth):
    leaves, d = make_ontology_tree(depth, max_branch=3, seed=seed)    # Def. 6 (Parte 0)
    n = len(leaves)
    for i in range(n):
        assert d[i][i] == 0
        for j in range(n):
            assert d[i][j] == d[j][i]
            for k in range(n):
                assert d[i][k] <= max(d[i][j], d[j][k])


# ---------- measures (T12) ----------
@given(seed=SEEDS, n=SIZES, k=st.integers(2, 4), kind=st.sampled_from(["endpoint", "context"]))
@settings(max_examples=150, deadline=None)
def test_medida_y_regla_de_la_cadena(seed, n, k, kind):
    net = make_random_dag(n, max_out=3, seed=seed)
    mu = make_meaning_map(net, n_meanings=k, kind=kind, seed=seed)
    P = path_measure(net)
    assert abs(sum(P.values()) - 1) < TOL
    PM = meaning_law(P, mu)
    assert abs(sum(PM.values()) - 1) < TOL
    lhs = entropy(P.values())
    rhs = entropy(PM.values()) + cond_entropy_walk_given_meaning(P, mu)
    assert abs(lhs - rhs) < 1e-9                           # Ec. (3), Parte 0


@given(seed=SEEDS, n=SIZES)
@settings(max_examples=150, deadline=None)
def test_visitas_satisfacen_balance(seed, n):
    net = make_random_dag(n, max_out=3, seed=seed)
    nu = visit_probabilities(net)
    assert abs(nu[net.root] - 1) < TOL                     # Def. 5 (Parte 0)
    for v in net.succ:
        if v == net.root:
            continue
        inflow = sum(nu[u] * net.p[u][v] for u in net.p if v in net.p[u])
        assert abs(nu[v] - inflow) < TOL


@given(seed=SEEDS, n=SIZES, frac=st.floats(0.0, 1.0))
@settings(max_examples=150, deadline=None)
def test_kl_se_localiza_por_nodo(seed, n, frac):
    net = make_random_dag(n, max_out=3, seed=seed)
    netR = perturb_kernel(net, fraction=frac, seed=seed + 1)
    A = kl_paths(path_measure(net), path_measure(netR))
    local = kl_nodewise(net, netR)
    assert A >= -TOL
    assert abs(A - sum(local.values())) < 1e-9             # Teorema 1(b), Parte 2


# ---------- ejemplo del texto ----------
def test_ejemplo_semana_reproduce_parte0():
    net, mu = make_toy_week_graph()
    P = path_measure(net); PM = meaning_law(P, mu)
    assert round(entropy(P.values()), 3) == 2.456
    assert round(entropy(PM.values()), 3) == 1.485
    assert round(cond_entropy_walk_given_meaning(P, mu), 3) == 0.971
    nu = visit_probabilities(net)
    assert (round(nu["a"], 3), round(nu["b"], 3)) == (0.6, 0.4)
    assert len(list(walks(net))) == 6
