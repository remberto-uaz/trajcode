"""Pruebas de trajcode.wordnet (T37, ADR-T03). Requieren `make data`; sin el archivo, se omiten."""
import pathlib

import numpy as np
import pytest

ZIP = pathlib.Path(__file__).resolve().parents[1] / "data/wordnet/wordnet.zip"
pytestmark = pytest.mark.skipif(not ZIP.exists(), reason="falta data/wordnet/wordnet.zip (make data)")

from trajcode.wordnet import load_nouns, subtree, geodesic_distortion, resnik_distortion, viterbi_leaves
from trajcode.measures import path_measure, meaning_law


@pytest.fixture(scope="module")
def wn():
    return load_nouns(ZIP)


def test_lector_reproduce_hechos_del_archivo(wn):
    dog = wn.synset("dog")
    assert dog == 2084071
    assert set(wn.hypernyms[dog]) == {2083346, 1317541}      # canine.n.02, domestic_animal.n.01
    assert wn.count[dog] >= 42                                # dog%1:05:00:: tiene 42 etiquetas en SemCor
    assert wn.synset("location") == 27167


@pytest.fixture(scope="module")
def loc(wn):
    return subtree(wn, wn.synset("location"), depth=4, max_leaves=40)


def test_subarbol_es_una_red_de_trayectoria_valida(loc):
    net, mu = loc.net, loc.mu
    net.validate()
    P = path_measure(net)
    assert abs(sum(P.values()) - 1) < 1e-9
    PM = meaning_law(P, mu)
    assert len(PM) == len(loc.meanings) <= 40
    assert abs(sum(PM.values()) - 1) < 1e-9


def test_herencia_multiple_produce_sinonimia(wn):
    st = subtree(wn, wn.synset("animal"), depth=6, max_leaves=60)
    P = path_measure(st.net)
    walks_per = {}
    for w in P:
        walks_per[st.mu[w]] = walks_per.get(st.mu[w], 0) + 1
    assert max(walks_per.values()) >= 2


def test_distorsiones(loc):
    for d in (geodesic_distortion(loc), resnik_distortion(loc)):
        d = np.asarray(d)
        n = len(loc.meanings)
        assert d.shape == (n, n)
        assert np.allclose(d, d.T)
        assert np.all(np.diag(d) == 0)
        off = d[~np.eye(n, dtype=bool)]
        assert np.all(off > 0)
        # desigualdad triangular
        assert np.all(d[:, None, :] <= d[:, :, None] + d[None, :, :] + 1e-9)


def test_resnik_es_ultrametrica_en_la_seleccion_de_viterbi(loc):
    leaves = viterbi_leaves(loc)
    d = np.asarray(resnik_distortion(loc, tree=leaves))
    n = len(d)
    assert np.all(d[:, None, :] <= np.maximum(d[:, :, None], d[None, :, :]) + 1e-9)


def test_hojas_de_viterbi_son_prefijos_coherentes(loc):
    leaves = viterbi_leaves(loc)
    assert len(leaves) == len(loc.meanings)
    assert len({len(l) for l in leaves}) == 1               # misma longitud (con relleno)
    assert len(set(leaves)) == len(leaves)
