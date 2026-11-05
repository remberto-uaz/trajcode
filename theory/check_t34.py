"""T34 — verificación numérica de la Parte 5 (transmisión progresiva por canal ruidoso).

(1) Teorema A1: distorsión esperada del decodificador secuencial. Monte Carlo con códigos lineales binarios
    aleatorios por nivel sobre un canal de borrado (BEC) con decodificación ML, que detecta el fallo sin error.
(2) Proposición A2: la programación dinámica da la misma asignación óptima que la búsqueda exhaustiva.
(3) Proposición A3: Δ_j = H(C_j) − R(d̄_j) ≥ 0, con igualdad en j = 0 y j = h; ¿es estrictamente positiva a veces?
(4) Proposición A4(b): régimen de SNR desconocida (umbrales asintóticos y costo en usos de canal).

Semillas fijas; solo NumPy/SciPy y trajcode. Uso: python theory/check_t34.py
"""
import itertools, math, pathlib, sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trajcode.graphs import make_ontology_tree
from trajcode.distortion import rd_at

H2 = lambda x: 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


# ---------------------------------------------------------------- cadena de clases de una jerarquía
def chain(leaves, p, d):
    """Para cada nivel j: clases (prefijo de longitud j), H(C_j), d̄_j (mejor representante por clase)."""
    p, d = np.asarray(p, float), np.asarray(d, float)
    h = len(leaves[0])
    out = []
    for j in range(h + 1):
        cls: dict = {}
        for i, lf in enumerate(leaves):
            cls.setdefault(lf[:j], []).append(i)
        H = -sum(p[B].sum() * math.log2(p[B].sum()) for B in cls.values() if p[B].sum() > 0)
        Dbar = sum(min((p[B] * d[B, b]).sum() for b in B) for B in cls.values())
        out.append((list(cls.values()), H, float(Dbar)))
    return out


def expected_distortion(dbar, eps):
    """Teorema A1: Σ_j Pr[J = j+1] d̄_j con ε_{h+1} := 1."""
    h = len(eps)
    tot, ok = 0.0, 1.0
    for j in range(h):
        tot += ok * eps[j] * dbar[j]
        ok *= 1 - eps[j]
    return tot + ok * dbar[h]


# ---------------------------------------------------------------- (1) Monte Carlo con códigos reales sobre BEC
def gf2_rank(A):
    A = A.copy() % 2
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
        if r == rows:
            break
    return r


def check_A1(seed=1, trials=20000):
    rng = np.random.default_rng(seed)
    leaves, d = make_ontology_tree(3, 3, seed=7)
    p = rng.dirichlet(np.ones(len(leaves)))
    ch = chain(leaves, p, d)
    dbar = [c[2] for c in ch]
    # bits por nivel = ⌈log2 (máx. hijos)⌉; usos de canal por nivel; un generador aleatorio por nivel
    h = len(leaves[0])
    k = []
    for j in range(1, h + 1):
        kids = {}
        for lf in leaves:
            kids.setdefault(lf[:j - 1], set()).add(lf[j - 1])
        k.append(max(1, math.ceil(math.log2(max(len(s) for s in kids.values())))))
    n = [k[0] + 4, k[1] + 3, k[2] + 2]  # más redundancia en los niveles gruesos
    e = 0.35  # probabilidad de borrado
    G = [rng.integers(0, 2, size=(k[j], n[j])) for j in range(h)]

    def fails(j):
        keep = rng.random(n[j]) > e
        return gf2_rank(G[j][:, keep]) < k[j]

    # ε_j estimada en corridas independientes
    eps = [np.mean([fails(j) for _ in range(trials)]) for j in range(h)]
    # E[d] por Monte Carlo del decodificador secuencial
    idx = rng.choice(len(leaves), size=trials, p=p)
    vals = []
    for m in idx:
        J = h
        for j in range(h):
            if fails(j):
                J = j
                break
        cls = next(B for B in ch[J][0] if m in B)
        b = min(cls, key=lambda b: (p[cls] * np.asarray(d)[cls, b]).sum())
        vals.append(d[m][b])
    mc = float(np.mean(vals))
    formula = expected_distortion(dbar, eps)
    se = float(np.std(vals)) / math.sqrt(trials)
    print(f"(1) Teorema A1, BEC e={e}, k={k}, n={n}: ε={np.round(eps, 4).tolist()}")
    print(f"    E[d] Monte Carlo = {mc:.4f}   fórmula = {formula:.4f}   |dif| = {abs(mc - formula):.4f} (3σ = {3 * se:.4f})")
    assert abs(mc - formula) < 3 * se
    return True


# ---------------------------------------------------------------- (2) programación dinámica vs exhaustiva
def eps_normal_bsc(n, k, q):
    """Aproximación normal (sin término logarítmico) del error de bloque en BSC(q); ε = 1 si n = 0 y k > 0."""
    if k == 0:
        return 0.0
    if n == 0:
        return 1.0
    C = 1 - H2(q)
    V = q * (1 - q) * math.log2((1 - q) / q) ** 2
    return float(norm.sf((n * C - k) / math.sqrt(n * V)))


def dp_alloc(dbar, k, n, eps_fn):
    h = len(k)
    V = [[0.0] * (n + 1) for _ in range(h + 1)]
    arg = [[0] * (n + 1) for _ in range(h)]
    for r in range(n + 1):
        V[h][r] = dbar[h]
    for j in range(h - 1, -1, -1):
        for r in range(n + 1):
            best, bn = math.inf, 0
            for nj in range(r + 1):
                e = eps_fn(nj, k[j])
                val = e * dbar[j] + (1 - e) * V[j + 1][r - nj]
                if val < best - 1e-15:
                    best, bn = val, nj
            V[j][r], arg[j][r] = best, bn
    alloc, r = [], n
    for j in range(h):
        alloc.append(arg[j][r]); r -= arg[j][r]
    return V[0][n], alloc


def check_A2(seed=2, cases=40):
    rng = np.random.default_rng(seed)
    worst = 0.0
    mono = 0
    for c in range(cases):
        leaves, d = make_ontology_tree(3, 3, seed=100 + c)
        if len(leaves) < 4:
            continue
        p = rng.dirichlet(np.ones(len(leaves)))
        dbar = [x[2] for x in chain(leaves, p, d)]
        k = [int(x) for x in rng.integers(1, 4, size=3)]
        n = int(sum(k) + rng.integers(2, 10))
        q = float(rng.uniform(0.02, 0.15))
        f = lambda nj, kj: eps_normal_bsc(nj, kj, q)
        v_dp, a_dp = dp_alloc(dbar, k, n, f)
        v_bf = min(expected_distortion(dbar, [f(a, k[0]), f(b, k[1]), f(n - a - b, k[2])])
                   for a in range(n + 1) for b in range(n + 1 - a))
        worst = max(worst, abs(v_dp - v_bf))
        rates = [kj / nj if nj else math.inf for kj, nj in zip(k, a_dp)]
        mono += all(rates[i] <= rates[i + 1] + 1e-12 for i in range(2))
    print(f"(2) Proposición A2: {cases} casos, máx |DP − exhaustiva| = {worst:.2e}; "
          f"tasa de código no decreciente con el nivel en {mono} de {cases}")
    assert worst < 1e-12
    return mono, cases


# ---------------------------------------------------------------- (3) costo de la progresividad
def check_A3(seed=3, cases=30):
    """Δ_j = H(C_j) − R(d̄_j): ≥ 0 siempre; > 0 en todo nivel no trivial (ni partición trivial ni identidad)."""
    rng = np.random.default_rng(seed)
    neg, nontriv, gaps = 0, 0, []
    for c in range(cases):
        leaves, d = make_ontology_tree(2, 3, seed=200 + c)
        if len(leaves) < 3:
            continue
        p = rng.dirichlet(np.ones(len(leaves)))
        ch = chain(leaves, p, d)
        for j, (cls, H, Dbar) in enumerate(ch):
            if len(cls) == 1 or all(len(B) == 1 for B in cls):
                continue  # partición trivial o identidad: Δ_j = 0 por (b)
            nontriv += 1
            R, Dr = rd_at(p, np.asarray(d, float), Dbar, iters=120)
            gap = H - R
            neg += gap < -1e-6
            gaps.append(gap)
    print(f"(3) Proposición A3: {nontriv} niveles no triviales; Δ_j < 0 en {neg}; "
          f"mín Δ_j = {min(gaps):.4f} bits, máx Δ_j = {max(gaps):.3f} bits")
    assert neg == 0 and min(gaps) > 0
    return nontriv, min(gaps), max(gaps)


# ---------------------------------------------------------------- (4) SNR desconocida, régimen asintótico
def check_A4():
    # BPSK con decisión dura: BSC con q(γ) = Q(√(2γ)); C(γ) = 1 − h(q)
    C = lambda g_db: 1 - H2(norm.sf(math.sqrt(2 * 10 ** (g_db / 10))))
    leaves, d = make_ontology_tree(3, 3, seed=7)
    p = np.random.default_rng(1).dirichlet(np.ones(len(leaves)))
    ch = chain(leaves, p, d)
    Hc = [ch[j + 1][1] - ch[j][1] for j in range(len(ch) - 1)]  # H(C_j | C_{j−1})
    HM = ch[-1][1]
    g0, targets = 6.0, [0.0, 3.0, 6.0]  # SNR de diseño por nivel (dB), del grueso al fino
    rho_u = HM / C(g0)
    rho_p = sum(H / C(g) for H, g in zip(Hc, targets))
    print(f"(4) Proposición A4(b): H(C_j|C_j−1) = {np.round(Hc, 3).tolist()}, H(M) = {HM:.3f}")
    print(f"    usos de canal por significado: uniforme {rho_u:.3f}, por capas {rho_p:.3f} (+{100 * (rho_p / rho_u - 1):.1f} %)")
    for g in [-1.0, 1.0, 4.0, 7.0]:
        Ju = len(Hc) if C(g) > HM / rho_u else 0
        Jp = 0
        for t in targets:
            if g >= t:
                Jp += 1
            else:
                break
        print(f"    γ = {g:+.0f} dB: d̄ uniforme = {ch[Ju][2]:.3f}, d̄ por capas = {ch[Jp][2]:.3f}")
    assert rho_p >= rho_u - 1e-12
    return dict(rho_uniform=rho_u, rho_layered=rho_p, extra_pct=100 * (rho_p / rho_u - 1))


if __name__ == "__main__":
    import json, pathlib
    check_A1()
    mono, cases = check_A2()
    nontriv, dmin, dmax = check_A3()
    a4 = check_A4()
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "t34.json"
    out.write_text(json.dumps(dict(mono=mono, cases=cases, nontrivial_levels=nontriv, delta_min=dmin,
                                   delta_max=dmax, **a4), indent=1), encoding="utf-8")
    print("OK", out.name)
