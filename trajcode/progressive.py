"""Transmisión progresiva por niveles (Parte 5, T35): cadena de clases, Teorema 1, Proposiciones 2 y 3.

Una jerarquía se da por sus hojas (tuplas de índices de hijo, como en graphs.make_ontology_tree); el nivel j es la
partición por prefijos de longitud j. El bloque del nivel j codifica el índice del hijo con k_j bits de longitud fija.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trajcode.distortion import rd_at
from trajcode.channel import transmit


@dataclass
class LevelChain:
    classes: list   # por nivel: lista de clases (listas de índices de significados)
    keys: list      # por nivel: prefijo de cada clase, en el mismo orden
    H: list         # H(C_j)
    Hcond: list     # H(C_j | C_{j−1}), j = 1..h
    dbar: list      # d̄_j
    rep: list       # por nivel: dict prefijo → mejor representante


def level_chain(leaves, p, d) -> LevelChain:
    p, d = np.asarray(p, float), np.asarray(d, float)
    h = len(leaves[0])
    classes, keys, H, dbar, rep = [], [], [], [], []
    for j in range(h + 1):
        cls: dict = {}
        for i, lf in enumerate(leaves):
            cls.setdefault(tuple(lf[:j]), []).append(i)
        ks = sorted(cls)
        classes.append([cls[k] for k in ks]); keys.append(ks)
        H.append(-sum(p[B].sum() * math.log2(p[B].sum()) for B in cls.values() if p[B].sum() > 0))
        tot, r = 0.0, {}
        for k in ks:
            B = cls[k]
            costs = p[B] @ d[B, :]          # representante en cualquier significado
            b = int(np.argmin(costs)); r[k] = b; tot += float(costs[b])
        dbar.append(tot); rep.append(r)
    Hcond = [H[j] - H[j - 1] for j in range(1, h + 1)]
    return LevelChain(classes, keys, H, Hcond, dbar, rep)


def level_bits(leaves) -> list:
    """k_j = ⌈log2(máximo número de hijos en el nivel j)⌉ (0 si todo nodo tiene un solo hijo)."""
    h = len(leaves[0])
    out = []
    for j in range(1, h + 1):
        kids: dict = {}
        for lf in leaves:
            kids.setdefault(tuple(lf[:j - 1]), set()).add(lf[j - 1])
        out.append(int(math.ceil(math.log2(max(len(s) for s in kids.values())))))
    return out


def expected_distortion(dbar, eps) -> float:
    """Teorema 1 (Parte 5): Σ_j Pr[J = j+1] d̄_j, con ε_{h+1} := 1."""
    tot, ok = 0.0, 1.0
    for j, e in enumerate(eps):
        tot += ok * e * dbar[j]
        ok *= 1 - e
    return tot + ok * dbar[len(eps)]


def optimal_allocation(dbar, k, n, eps_fn):
    """Proposición 2: programación dinámica exacta. Devuelve (E[d] mínima, asignación n_1..n_h)."""
    h = len(k)
    V = np.full((h + 1, n + 1), dbar[h], float)
    arg = np.zeros((h, n + 1), int)
    E = [[eps_fn(nj, k[j]) for nj in range(n + 1)] for j in range(h)]
    for j in range(h - 1, -1, -1):
        for r in range(n + 1):
            # el último nivel recibe todo lo que queda, de modo que Σ n_j = n (ε no crece con n_j)
            choices = [r] if j == h - 1 else range(r + 1)
            vals = {nj: E[j][nj] * dbar[j] + (1 - E[j][nj]) * V[j + 1][r - nj] for nj in choices}
            arg[j][r] = min(vals, key=vals.get); V[j][r] = vals[arg[j][r]]
    alloc, r = [], n
    for j in range(h):
        alloc.append(int(arg[j][r])); r -= alloc[-1]
    return float(V[0][n]), alloc


def _ba(p, d, beta, q, tol, max_iter):
    """Blahut–Arimoto con pendiente β desde la ley de salida q; devuelve (R, D, q final)."""
    K = np.exp(-beta * (d - d.min(axis=1, keepdims=True)))
    for _ in range(max_iter):
        W = q * K
        W /= W.sum(1, keepdims=True)
        qn = np.maximum(p @ W, 1e-300)
        done = np.max(np.abs(qn - q)) < tol
        q = qn
        if done:
            break
    W = q * K
    W /= W.sum(1, keepdims=True)
    J = p[:, None] * W
    ref = np.maximum(p[:, None] * q[None, :], 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = float(np.where(J > 1e-300, J * np.log2(np.maximum(J, 1e-300) / ref), 0.0).sum())
    return R, float((J * d).sum()), q


def rd_curve(p, d, n_beta: int = 200, beta_max: float = 200.0, tol: float = 1e-11, max_iter: int = 200_000):
    """Barrido de β de mayor a menor con arranque en caliente: lista de (β, D, R, q) con β creciente.

    El barrido va hacia β pequeño porque Blahut–Arimoto actualiza q de forma multiplicativa: una salida con masa
    casi nula no se recupera, así que el arranque en caliente solo es fiable cuando el soporte se reduce."""
    p, d = np.asarray(p, float), np.asarray(d, float)
    q = np.ones(d.shape[1]) / d.shape[1]
    out = []
    for b in np.geomspace(beta_max, 1e-3, n_beta):
        R, D, q = _ba(p, d, float(b), q, tol, max_iter)
        out.append((float(b), D, R, q.copy()))
    return out[::-1]


def rd_from_curve(p, d, curve, D, iters: int = 40, tol: float = 1e-12, max_iter: int = 200_000):
    """R(D) exacto: bisección local en β dentro del intervalo del barrido que encierra a D, siempre desde la ley de
    salida convergida en el extremo de β mayor (soporte más amplio). Fuera del rango barrido devuelve None."""
    p, d = np.asarray(p, float), np.asarray(d, float)
    for (b0, D0, _, _), (b1, D1, _, q1) in zip(curve, curve[1:]):
        if D1 <= D <= D0:
            lo, hi = math.log(b0), math.log(b1)
            for _ in range(iters):
                mid = (lo + hi) / 2
                _, Dm, _ = _ba(p, d, math.exp(mid), q1.copy(), tol, max_iter)
                lo, hi = (mid, hi) if Dm > D else (lo, mid)
            R, _, _ = _ba(p, d, math.exp((lo + hi) / 2), q1.copy(), tol, max_iter)
            return R
    return None


def progressivity_cost(chain: LevelChain, p, d, iters: int = 120) -> list:
    """Proposición 3: Δ_j = H(C_j) − R(d̄_j); 0 en la partición trivial y en la de singuletes.
    R(d̄_j) por barrido con arranque en caliente y bisección local; si d̄_j cae fuera del barrido, por
    distortion.rd_at (bisección desde cero)."""
    n = len(p)
    curve = rd_curve(p, d)
    out = []
    for j, cls in enumerate(chain.classes):
        if len(cls) == 1 or len(cls) == n:
            out.append(0.0)
            continue
        R = rd_from_curve(p, d, curve, chain.dbar[j])
        if R is None:
            R, _ = rd_at(np.asarray(p, float), np.asarray(d, float), chain.dbar[j], iters=iters)
        out.append(chain.H[j] - R)
    return out


def uniform_index_distortion(dbar0: float, eps_u: float) -> float:
    """Código de índice con protección uniforme y detección: E[d] = ε_U · d̄_0 (Observación 1(a))."""
    return eps_u * dbar0


def simulate_layered(leaves, p, d, chain: LevelChain, codes, ch, trials: int, seed: int = 0):
    """Monte Carlo del receptor real: cada nivel con su código polar; acepta si el CRC pasa y el índice existe,
    se detiene en el primer bloque rechazado y emite el representante de la clase alcanzada.
    Devuelve (E[d] estimada, error estándar)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    h = len(leaves[0])
    idx = rng.choice(len(leaves), size=trials, p=np.asarray(p, float))
    children: dict = {}
    for lf in leaves:
        for j in range(h):
            children.setdefault(tuple(lf[:j]), set()).add(lf[j])
    prefix = [() for _ in range(trials)]
    alive = np.ones(trials, bool)
    for j in range(h):
        kj = codes[j].K if codes[j] is not None else 0
        child = np.array([leaves[m][j] for m in idx])
        if kj > 0:
            bits = ((child[:, None] >> np.arange(kj - 1, -1, -1)) & 1).astype(np.uint8)
            dec, ok = codes[j].decode_full(transmit(codes[j].encode(bits), ch, rng))
            got = dec.astype(int) @ (1 << np.arange(kj - 1, -1, -1))
        else:
            got, ok = np.zeros(trials, int), np.ones(trials, bool)
        for t in range(trials):
            if not alive[t]:
                continue
            if ok[t] and int(got[t]) in children.get(prefix[t], set()):
                prefix[t] = prefix[t] + (int(got[t]),)
            else:
                alive[t] = False
    vals = np.array([d[m, chain.rep[len(pr)][pr]] for m, pr in zip(idx, prefix)])
    return float(vals.mean()), float(vals.std() / math.sqrt(trials))


# ---------------------------------------------------------------- lotes de B significados por bloque (T36, E5)
def index_bits(n_meanings: int) -> int:
    """Bits del código de índice de longitud fija: ⌈log2 |M|⌉."""
    return max(1, int(math.ceil(math.log2(n_meanings))))


def _bits_of(values: np.ndarray, k: int) -> np.ndarray:
    """(T, B) enteros → (T, B·k) bits, el más significativo primero."""
    return ((values[..., None] >> np.arange(k - 1, -1, -1)) & 1).reshape(values.shape[0], -1).astype(np.uint8)


def _ints_of(bits: np.ndarray, B: int, k: int) -> np.ndarray:
    return bits.reshape(bits.shape[0], B, k).astype(int) @ (1 << np.arange(k - 1, -1, -1))


def simulate_layered_batch(leaves, p, d, chain: LevelChain, codes, ch, B: int, trials: int, seed: int = 0):
    """Transmisión por niveles con B significados por bloque: el bloque del nivel j lleva los B índices de hijo
    (B·k_j bits más CRC). El receptor acepta el bloque si el CRC pasa; un índice inexistente detiene a ese
    significado; un bloque rechazado detiene a los B. Devuelve (E[d] por significado, error estándar entre lotes,
    fracción de bloques aceptados con error)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    h = len(leaves[0])
    leaf_arr = np.array(leaves)
    idx = rng.choice(len(leaves), size=(trials, B), p=np.asarray(p, float))
    children: dict = {}
    for lf in leaves:
        for j in range(h):
            children.setdefault(tuple(lf[:j]), set()).add(lf[j])
    prefix = [[() for _ in range(B)] for _ in range(trials)]
    alive = np.ones((trials, B), bool)
    undetected = blocks = 0
    for j in range(h):
        code = codes[j]
        child = leaf_arr[idx, j]
        if code is None or code.K == 0:
            got, ok, wrong = child, np.ones(trials, bool), np.zeros(trials, bool)
        else:
            kj = code.K // B
            pay = _bits_of(child, kj)
            dec, ok = code.decode_full(transmit(code.encode(pay), ch, rng))
            wrong = np.any(dec != pay, axis=1)
            got = _ints_of(dec, B, kj)
            undetected += int(np.sum(ok & wrong)); blocks += trials
        for t in range(trials):
            for b in range(B):
                if not alive[t, b]:
                    continue
                g = int(got[t, b])
                if ok[t] and g in children.get(prefix[t][b], set()):
                    prefix[t][b] = prefix[t][b] + (g,)
                else:
                    alive[t, b] = False
    per_trial = np.array([np.mean([d[idx[t, b], chain.rep[len(prefix[t][b])][prefix[t][b]]] for b in range(B)])
                          for t in range(trials)])
    return float(per_trial.mean()), float(per_trial.std() / math.sqrt(trials)), undetected / max(blocks, 1)


def simulate_uniform_batch(p, d, chain: LevelChain, code, ch, B: int, trials: int, seed: int = 0):
    """Código de índice con protección uniforme: los B índices de significado (⌈log2|M|⌉ bits cada uno) en un solo
    bloque. Rechazo → representante global; aceptado con error → significado decodificado (si existe; si no,
    representante global). Devuelve (E[d], error estándar, fracción de bloques aceptados con error)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    n = len(p)
    k = code.K // B
    idx = rng.choice(n, size=(trials, B), p=np.asarray(p, float))
    pay = _bits_of(idx, k)
    dec, ok = code.decode_full(transmit(code.encode(pay), ch, rng))
    wrong = np.any(dec != pay, axis=1)
    got = _ints_of(dec, B, k)
    g0 = chain.rep[0][()]
    est = np.where(ok[:, None] & (got < n), got, g0)
    per_trial = d[idx, est].mean(axis=1)
    return float(per_trial.mean()), float(per_trial.std() / math.sqrt(trials)), float(np.mean(ok & wrong))
