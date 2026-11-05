"""Canales binarios sin memoria, aproximación normal y códigos polares con decodificación SC y CRC (T35, ADR-T02).

Convenciones:
- BEC(e): borrado con probabilidad e. BSC(q): cruce con probabilidad q. BIAWGN(snr): BPSK (0 → +1, 1 → −1) con
  ruido gaussiano de varianza σ² = 1/(2·snr), snr = Es/N0 lineal por uso de canal.
- Las LLR son log P(y|0)/P(y|1): positivas favorecen el bit 0.
- Código polar con matriz G_N = F^{⊗m}, F = [[1,0],[1,1]], sin permutación de inversión de bits; la decodificación
  por cancelación sucesiva (SC) sigue la misma recursión: x = [v1 ⊕ v2, v2] con v1 = u1·G_{N/2}, v2 = u2·G_{N/2}.
- Construcción: BEC exacta (parámetros de Bhattacharyya), BSC por la cota de Bhattacharyya con la misma recursión,
  BI-AWGN por aproximación gaussiana [Trifonov 2012].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(120)  # cuadratura para esperanzas gaussianas
_GH_W = _GH_W / _GH_W.sum()
_BIG = 50.0  # |LLR| de un bit conocido


@dataclass(frozen=True)
class BEC:
    e: float


@dataclass(frozen=True)
class BSC:
    q: float


@dataclass(frozen=True)
class BIAWGN:
    snr: float

    @property
    def sigma2(self) -> float:
        return 1.0 / (2.0 * self.snr)


def _h2(x: float) -> float:
    return 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def _awgn_info_density(ch: BIAWGN) -> np.ndarray:
    """Valores de i(x;y) = 1 − log2(1 + e^{−L}) en los nodos de cuadratura, con L ~ N(μ, 2μ), μ = 2/σ²."""
    mu = 2.0 / ch.sigma2
    L = mu + math.sqrt(2 * mu) * _GH_X
    return 1.0 - np.logaddexp(0.0, -L) / math.log(2)


def capacity(ch) -> float:
    if isinstance(ch, BEC):
        return 1.0 - ch.e
    if isinstance(ch, BSC):
        return 1.0 - _h2(ch.q)
    if isinstance(ch, BIAWGN):
        return float(_GH_W @ _awgn_info_density(ch))
    raise TypeError(ch)


def dispersion(ch) -> float:
    if isinstance(ch, BEC):
        return ch.e * (1.0 - ch.e)
    if isinstance(ch, BSC):
        q = ch.q
        return 0.0 if q <= 0 or q >= 1 else q * (1 - q) * math.log2((1 - q) / q) ** 2
    if isinstance(ch, BIAWGN):
        i = _awgn_info_density(ch)
        return float(_GH_W @ (i - _GH_W @ i) ** 2)
    raise TypeError(ch)


def eps_normal(n: int, k: int, ch) -> float:
    """Aproximación normal del error de bloque [Polyanskiy–Poor–Verdú 2010]:
    ε ≈ Q((nC − k + ½·log2 n)/√(nV)); sin el término logarítmico en el BEC. ε(0, k>0) = 1; ε(n, 0) = 0."""
    if k <= 0:
        return 0.0
    if n <= 0:
        return 1.0
    C, V = capacity(ch), dispersion(ch)
    third = 0.0 if isinstance(ch, BEC) else 0.5 * math.log2(n)
    if V <= 0:
        return 0.0 if k <= n * C else 1.0
    return float(norm.sf((n * C - k + third) / math.sqrt(n * V)))


# ---------------------------------------------------------------- CRC
_CRC_POLY = {4: 0b10011, 8: 0b100000111, 16: 0b10001000000100001}  # x^4+x+1; x^8+x^2+x+1; CRC-16-CCITT


def _crc_remainder(bits: np.ndarray, c: int) -> np.ndarray:
    """Residuo de bits(x)·x^c módulo el polinomio de grado c; bits de forma (..., L)."""
    poly = np.array([(_CRC_POLY[c] >> (c - i)) & 1 for i in range(c + 1)], dtype=np.uint8)
    b = np.atleast_2d(np.asarray(bits, dtype=np.uint8))
    reg = np.concatenate([b, np.zeros((b.shape[0], c), np.uint8)], axis=1)
    for i in range(b.shape[1]):
        lead = reg[:, i].astype(bool)
        reg[lead, i:i + c + 1] ^= poly
    return reg[:, -c:]


def crc_bits(bits: np.ndarray, c: int) -> np.ndarray:
    r = _crc_remainder(bits, c)
    return r[0] if np.ndim(bits) == 1 else r


def crc_ok(word: np.ndarray, c: int) -> np.ndarray | bool:
    """True si los últimos c bits son el CRC de los anteriores."""
    w = np.atleast_2d(np.asarray(word, dtype=np.uint8))
    ok = np.all(_crc_remainder(w[:, :-c], c) == w[:, -c:], axis=1)
    return bool(ok[0]) if np.ndim(word) == 1 else ok


# ---------------------------------------------------------------- construcción
def _phi(x):
    x = np.asarray(x, float)
    out = np.where(x < 10, np.exp(-0.4527 * np.power(np.maximum(x, 1e-12), 0.86) + 0.0218),
                   np.sqrt(np.pi / np.maximum(x, 1e-12)) * np.exp(-x / 4) * (1 - 10 / (7 * np.maximum(x, 1e-12))))
    return np.where(x <= 0, 1.0, out)


def _phi_inv(y: float) -> float:
    if y >= 1.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while _phi(hi) > y:
        hi *= 2
        if hi > 1e6:
            return hi
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _phi(mid) > y else (lo, mid)
    return (lo + hi) / 2


def _reliability(N: int, ch) -> np.ndarray:
    """Probabilidad de error (o de borrado) estimada de cada canal sintético, en el orden de u."""
    if isinstance(ch, (BEC, BSC)):
        z0 = ch.e if isinstance(ch, BEC) else 2 * math.sqrt(ch.q * (1 - ch.q))

        def rec(n, z):
            if n == 1:
                return [z]
            return rec(n // 2, 2 * z - z * z) + rec(n // 2, z * z)
        return np.array(rec(N, z0))
    if isinstance(ch, BIAWGN):
        def rec(n, m):
            if n == 1:
                return [m]
            return rec(n // 2, _phi_inv(1 - (1 - float(_phi(m))) ** 2)) + rec(n // 2, 2 * m)
        means = np.array(rec(N, 2.0 / ch.sigma2))
        return norm.sf(np.sqrt(means / 2))
    raise TypeError(ch)


# ---------------------------------------------------------------- código polar
def _transform(u: np.ndarray) -> np.ndarray:
    n = u.shape[1]
    if n == 1:
        return u.copy()
    v1, v2 = _transform(u[:, :n // 2]), _transform(u[:, n // 2:])
    return np.concatenate([v1 ^ v2, v2], axis=1)


def _boxplus(a, b):
    return (np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
            + np.log1p(np.exp(-np.abs(a + b))) - np.log1p(np.exp(-np.abs(a - b))))


def _sc(llr: np.ndarray, frozen: np.ndarray):
    """Decodificación SC; devuelve (û, x̂ reencodificado)."""
    n = llr.shape[1]
    if n == 1:
        u = np.zeros((llr.shape[0], 1), np.uint8) if frozen[0] else (llr < 0).astype(np.uint8)
        return u, u
    a, b = llr[:, :n // 2], llr[:, n // 2:]
    u1, v1 = _sc(_boxplus(a, b), frozen[:n // 2])
    u2, v2 = _sc(b + (1 - 2 * v1.astype(float)) * a, frozen[n // 2:])
    return np.concatenate([u1, u2], axis=1), np.concatenate([v1 ^ v2, v2], axis=1)


@dataclass
class PolarCode:
    N: int
    K: int
    channel: object
    crc: int = 0
    reliability: np.ndarray = field(init=False)
    info: list = field(init=False)
    frozen: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.N & (self.N - 1) or self.N < 1:
            raise ValueError("N debe ser potencia de 2")
        if not 0 < self.K + self.crc <= self.N:
            raise ValueError("se requiere 0 < K + crc ≤ N")
        self.reliability = _reliability(self.N, self.channel)
        self.info = sorted(np.argsort(self.reliability, kind="stable")[:self.K + self.crc].tolist())
        self.frozen = np.ones(self.N, bool)
        self.frozen[self.info] = False

    def encode(self, payload: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(payload, np.uint8))
        if self.crc:
            p = np.concatenate([p, crc_bits(p, self.crc).reshape(p.shape[0], -1)], axis=1)
        u = np.zeros((p.shape[0], self.N), np.uint8)
        u[:, self.info] = p
        return _transform(u)

    def decode_full(self, llr: np.ndarray):
        """Devuelve (carga útil decodificada, CRC aceptado); sin CRC, aceptado siempre."""
        u, _ = _sc(np.atleast_2d(np.asarray(llr, float)), self.frozen)
        word = u[:, self.info]
        ok = crc_ok(word, self.crc) if self.crc else np.ones(word.shape[0], bool)
        return word[:, :self.K], np.atleast_1d(ok)

    def decode(self, llr: np.ndarray) -> np.ndarray:
        return self.decode_full(llr)[0]


def transmit(x: np.ndarray, ch, rng: np.random.Generator) -> np.ndarray:
    """LLR recibidas para palabras x (F × N)."""
    x = np.asarray(x, np.uint8)
    if isinstance(ch, BEC):
        llr = np.where(x == 0, _BIG, -_BIG)
        llr[rng.random(x.shape) < ch.e] = 0.0
        return llr
    if isinstance(ch, BSC):
        y = x ^ (rng.random(x.shape) < ch.q).astype(np.uint8)
        L = math.log((1 - ch.q) / ch.q) if 0 < ch.q < 0.5 else _BIG
        return np.where(y == 0, L, -L)
    if isinstance(ch, BIAWGN):
        s = 1.0 - 2.0 * x
        y = s + math.sqrt(ch.sigma2) * rng.standard_normal(x.shape)
        return 2.0 * y / ch.sigma2
    raise TypeError(ch)


def simulate_blocks(code: PolarCode, ch, frames: int, seed: int = 0, batch: int = 1000):
    """(tasa de bloques no aceptados correctamente, tasa de errores no detectados).

    Sin CRC se supone detección perfecta (supuesto L2): todo error cuenta como fallo detectado."""
    rng = np.random.default_rng(seed)
    bad = undet = 0
    done = 0
    while done < frames:
        F = min(batch, frames - done)
        pay = rng.integers(0, 2, size=(F, code.K)).astype(np.uint8)
        dec, ok = code.decode_full(transmit(code.encode(pay), ch, rng))
        wrong = np.any(dec != pay, axis=1)
        bad += int(np.sum(wrong | ~ok))
        undet += int(np.sum(wrong & ok)) if code.crc else 0
        done += F
    return bad / frames, undet / frames
