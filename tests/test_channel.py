"""Pruebas de trajcode.channel (T35, ADR-T02): canales binarios, aproximación normal, códigos polares con SC y CRC."""
import math

import numpy as np
from hypothesis import given, settings, strategies as st

from trajcode.channel import (BEC, BSC, BIAWGN, capacity, dispersion, eps_normal, PolarCode, crc_bits,
                              crc_ok, simulate_blocks)


# ---------------------------------------------------------------- capacidad y dispersión
def test_capacidades_conocidas():
    assert capacity(BEC(0.3)) == np.float64(0.7) or abs(capacity(BEC(0.3)) - 0.7) < 1e-12
    assert abs(capacity(BSC(0.0)) - 1.0) < 1e-12
    assert abs(capacity(BSC(0.11)) - (1 - (-0.11 * math.log2(0.11) - 0.89 * math.log2(0.89)))) < 1e-12
    assert abs(dispersion(BEC(0.3)) - 0.21) < 1e-12


def test_capacidad_awgn_monotona_y_acotada():
    cs = [capacity(BIAWGN(10 ** (g / 10))) for g in (-10, -5, 0, 5, 10)]
    assert all(0 <= c <= 1 for c in cs)
    assert all(a < b for a, b in zip(cs, cs[1:]))
    assert cs[-1] > 0.999 and cs[0] < 0.15
    # referencias con σ² = 1/(2·snr): C = 0.5 bits por uso cerca de −2.82 dB (Eb/N0 ≈ 0.19 dB a tasa 1/2),
    # y C ≈ 0.486 con σ² = 1 (snr = 0.5)
    assert abs(capacity(BIAWGN(10 ** (-2.82 / 10))) - 0.5) < 0.005
    assert abs(capacity(BIAWGN(0.5)) - 0.486) < 0.005


@given(st.integers(1, 400), st.integers(0, 200), st.floats(0.01, 0.2))
@settings(max_examples=60, deadline=None)
def test_aproximacion_normal_decrece_con_n(n, k, q):
    ch = BSC(q)
    if k == 0:
        assert eps_normal(n, 0, ch) == 0.0
        return
    assert eps_normal(0, k, ch) == 1.0
    assert eps_normal(n + 1, k, ch) <= eps_normal(n, k, ch) + 1e-12


# ---------------------------------------------------------------- CRC
@given(st.lists(st.integers(0, 1), min_size=1, max_size=40), st.data())
@settings(max_examples=80, deadline=None)
def test_crc_detecta_todo_error_de_un_bit(bits, data):
    word = np.array(bits + list(crc_bits(np.array(bits), 8)), dtype=np.uint8)
    assert crc_ok(word, 8)
    i = data.draw(st.integers(0, len(word) - 1))
    word[i] ^= 1
    assert not crc_ok(word, 8)


# ---------------------------------------------------------------- códigos polares
@given(st.sampled_from([8, 16, 32, 64]), st.data())
@settings(max_examples=40, deadline=None)
def test_ida_y_vuelta_sin_ruido(N, data):
    K = data.draw(st.integers(1, N))
    code = PolarCode(N, K, BEC(0.3))
    rng = np.random.default_rng(data.draw(st.integers(0, 10_000)))
    u = rng.integers(0, 2, size=(5, K)).astype(np.uint8)
    x = code.encode(u)
    llr = np.where(x == 0, 30.0, -30.0)
    assert np.array_equal(code.decode(llr), u)


def test_construccion_bec_es_exacta_y_esta_en_rango():
    code = PolarCode(64, 16, BEC(0.4))
    z = code.reliability
    assert np.all((z >= 0) & (z <= 1))
    # conservación de la capacidad en el BEC: Σ (1 − Z_i) = N·C
    assert abs(np.sum(1 - z) - 64 * 0.6) < 1e-9
    assert set(code.info) == set(np.argsort(z)[:16].tolist())


def test_error_de_bloque_en_bec_bajo_la_cota_de_union():
    """En el BEC, Z_i es la probabilidad de borrado del canal sintético i; la cota de unión es exacta como cota."""
    code = PolarCode(64, 16, BEC(0.3))
    bound = float(np.sum(code.reliability[code.info]))
    bler, _ = simulate_blocks(code, BEC(0.3), frames=4000, seed=1)
    assert bler <= bound + 3 * math.sqrt(bound * (1 - min(bound, 1)) / 4000) + 1e-9


def test_error_de_bloque_awgn_decrece_con_la_snr():
    blers = []
    for g_db in (-1.0, 1.0, 3.0, 5.0):
        ch = BIAWGN(10 ** (g_db / 10))
        code = PolarCode(128, 40, ch)
        blers.append(simulate_blocks(code, ch, frames=2000, seed=2)[0])
    assert all(a >= b - 0.02 for a, b in zip(blers, blers[1:]))
    assert blers[0] > blers[-1]


def test_crc_convierte_errores_en_fallos_detectados():
    ch = BIAWGN(10 ** (-4.0 / 10))   # C ≈ 0.41, apenas sobre la tasa 48/128 = 0.375: con N = 128 hay muchos errores
    code = PolarCode(128, 40, ch, crc=8)
    bler, undetected = simulate_blocks(code, ch, frames=3000, seed=3)
    assert bler > 0.05            # el punto de operación tiene errores
    assert undetected <= 0.01     # casi todos se detectan (del orden de 2^−8 de los errores)


# ---------------------------------------------------------------- valores conocidos (añadidas tras la prueba de mutación)
def test_boxplus_es_la_regla_exacta():
    from trajcode.channel import _boxplus
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 4, 200), rng.normal(0, 4, 200)
    exact = 2 * np.arctanh(np.tanh(a / 2) * np.tanh(b / 2))
    assert np.allclose(_boxplus(a, b), exact, atol=1e-9)


def test_crc8_valor_publicado():
    """CRC-8 con polinomio 0x07, sin reflejo, registro inicial 0 (CRC-8/SMBUS): "123456789" → 0xF4."""
    msg = np.array([int(b) for ch in b"123456789" for b in format(ch, "08b")], dtype=np.uint8)
    r = crc_bits(msg, 8)
    assert int("".join(map(str, r)), 2) == 0xF4


def test_aproximacion_normal_valor_de_referencia():
    """Definición adoptada: ε = Q((nC − k + ½ log2 n)/√(nV)) en BSC."""
    from scipy.stats import norm
    ch = BSC(0.11)
    n, k = 500, 200
    C, V = capacity(ch), dispersion(ch)
    ref = norm.sf((n * C - k + 0.5 * math.log2(n)) / math.sqrt(n * V))
    assert abs(eps_normal(n, k, ch) - ref) < 1e-15
    assert eps_normal(n, k, ch) < norm.sf((n * C - k) / math.sqrt(n * V))
