"""E5 (T36, Opción A): distorsión semántica contra SNR para transmisión por niveles e índice con protección uniforme.

Fuentes: árbol sintético (altura 3, como en theory/check_t34.py) y el subárbol location de WordNet (E4, distorsión
geodésica). B significados por bloque y CRC de C bits por bloque. Canal BI-AWGN con BPSK.

Esquemas con el mismo presupuesto de n usos de canal por lote:
- ideal (aproximación normal, Proposición 2 con asignación entera exacta):
  * longitud fija: k_j = B·⌈log2 hijos⌉ + C por nivel; índice: B·⌈log2|M|⌉ + C;
  * entrópico: k_j = ⌈B·H(C_j|C_{j−1})⌉ + C; índice: ⌈B·H(M)⌉ + C (régimen de la Proposición 4);
  * por niveles diseñado en γ0 y diseñado en γ0 − 3 dB (diseño robusto);
- práctico: códigos polares con SC y CRC; longitudes potencia de 2; asignación elegida entre todas las
  combinaciones con Σ N_j ≤ n por la aproximación normal en γ0; Monte Carlo del receptor real.
γ0 es la SNR en que el índice ideal de longitud fija falla con probabilidad ε_U = 0.01.

Barrido del tamaño de lote (ideal, entrópico): B ∈ {16, 64, 256, 1024}, índice a tasa 1/4 (n = 4·k_U), γ0 con
ε_U = 0.01; la asignación usa la Proposición 2 sobre una rejilla de paso s = ⌈n/400⌉ usos (exacta en la rejilla).

Salidas: results/e5_ideal.csv, results/e5_polar.csv, results/e5_bloque.csv, figures/e5_canal.pdf.
"""
import csv, itertools, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from style import plt, COL_W, INK, MID, LIGHT, SERIES, RES, save

from trajcode.graphs import make_ontology_tree
from trajcode.channel import BIAWGN, PolarCode, eps_normal
from trajcode.progressive import (level_chain, level_bits, index_bits, expected_distortion, optimal_allocation,
                                  simulate_layered_batch, simulate_uniform_batch)
from trajcode.wordnet import load_nouns, subtree, viterbi_leaves, geodesic_distortion
from trajcode.measures import path_measure, meaning_law

B, CRC = 16, 8
B_SWEEP = (16, 64, 256, 1024)
SWEEP_OFF = (-4.0, -3.0, -2.0, -1.0, 0.0)
GRID = np.arange(-6.0, 2.5, 1.0)       # γ − γ0 en dB
ROBUST = -3.0                           # diseño robusto: γ0 − 3 dB
TRIALS = 800                            # lotes por punto en el Monte Carlo
db = lambda g: BIAWGN(10 ** (g / 10))


def sources():
    leaves, d = make_ontology_tree(3, 3, seed=7)
    p = np.random.default_rng(1).dirichlet(np.ones(len(leaves)))
    yield "synthetic", leaves, np.asarray(d, float), p
    zp = pathlib.Path(__file__).resolve().parents[1] / "data/wordnet/wordnet.zip"
    if not zp.exists():
        sys.exit("falta data/wordnet/wordnet.zip: ejecuta `make data`")
    wn = load_nouns(zp)
    st = subtree(wn, wn.synset("location"), depth=6, max_leaves=64)
    PM = meaning_law(path_measure(st.net), st.mu)
    yield "location", viterbi_leaves(st), geodesic_distortion(st), np.array([PM[m] for m in st.meanings])


def gamma0(n, kU):
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if eps_normal(n, kU, db(mid)) > 0.01 else (lo, mid)
    return (lo + hi) / 2


ideal_rows, polar_rows, block_rows = [], [], []
for name, leaves, d, p in sources():
    ch = level_chain(leaves, p, d)
    kbits = level_bits(leaves)
    payload = {"fixed": [B * k + CRC if k else 0 for k in kbits],
               "entropic": [int(math.ceil(B * hc)) + CRC if hc > 1e-12 else 0 for hc in ch.Hcond]}
    kU = {"fixed": B * index_bits(len(leaves)) + CRC, "entropic": int(math.ceil(B * ch.H[-1])) + CRC}
    n = 1 << int(math.ceil(math.log2(2 * sum(payload["fixed"]))))   # presupuesto: tasa ≤ 1/2 para el escalonado fijo
    g0 = gamma0(n, kU["fixed"])
    print(f"{name}: |M| = {len(leaves)}, niveles = {len(kbits)}, k = {payload['fixed']}, kU = {kU['fixed']}, "
          f"n = {n}, γ0 = {g0:.2f} dB")
    # ---- ideal
    for mode in ("fixed", "entropic"):
        allocs = {}
        for tag, gd in (("layered", g0), ("layered_robust", g0 + ROBUST)):
            _, allocs[tag] = optimal_allocation(ch.dbar, payload[mode], n, lambda nj, kj: eps_normal(nj, kj, db(gd)))
        for off in GRID:
            g = g0 + off
            row = dict(source=name, mode=mode, n=n, gamma0_db=g0, offset_db=float(off))
            for tag, al in allocs.items():
                row[tag] = expected_distortion(ch.dbar, [eps_normal(a, k, db(g)) for a, k in zip(al, payload[mode])])
            row["uniform"] = eps_normal(n, kU[mode], db(g)) * ch.dbar[0]
            row["dbar0"] = ch.dbar[0]
            row["alloc_layered"] = " ".join(map(str, allocs["layered"]))
            row["alloc_robust"] = " ".join(map(str, allocs["layered_robust"]))
            ideal_rows.append(row)
    # ---- práctico: polares, longitudes potencia de 2, Σ N_j ≤ n
    act = [j for j, k in enumerate(payload["fixed"]) if k > 0]
    opts = [[N for N in (16, 32, 64, 128, 256, 512) if N >= payload["fixed"][j]] for j in act]
    best = None
    for combo in itertools.product(*opts):
        if sum(combo) > n:
            continue
        al = [0] * len(kbits)
        for j, N in zip(act, combo):
            al[j] = N
        e = expected_distortion(ch.dbar, [eps_normal(a, k, db(g0)) for a, k in zip(al, payload["fixed"])])
        if best is None or e < best[0]:
            best = (e, al)
    al = best[1]
    for off in GRID:
        g = g0 + off
        chn = db(g)
        codes = [PolarCode(N, B * kbits[j], chn, crc=CRC) if N else None for j, N in enumerate(al)]
        mL, seL, uL = simulate_layered_batch(leaves, p, d, ch, codes, chn, B=B, trials=TRIALS, seed=100 + int(10 * off))
        cU = PolarCode(n, B * index_bits(len(leaves)), chn, crc=CRC)
        mU, seU, uU = simulate_uniform_batch(p, d, ch, cU, chn, B=B, trials=TRIALS, seed=200 + int(10 * off))
        polar_rows.append(dict(source=name, n=n, gamma0_db=g0, offset_db=float(off), alloc=" ".join(map(str, al)),
                               layered=mL, layered_se=seL, layered_undetected=uL, uniform=mU, uniform_se=seU,
                               uniform_undetected=uU, dbar0=ch.dbar[0]))
    print(f"   polar: asignación {al}")
    # ---- barrido del tamaño de lote (ideal, entrópico)
    for Bs in B_SWEEP:
        pay = [int(math.ceil(Bs * hc)) + CRC if hc > 1e-12 else 0 for hc in ch.Hcond]
        kUs = int(math.ceil(Bs * ch.H[-1])) + CRC
        ns = 4 * kUs
        s = max(1, -(-ns // 400))
        gs0 = gamma0(ns, kUs)
        al_s = {}
        for tag, gd in (("layered", gs0), ("layered_robust", gs0 + ROBUST)):
            _, a = optimal_allocation(ch.dbar, pay, ns // s, lambda nj, kj, gd=gd: eps_normal(nj * s, kj, db(gd)))
            al_s[tag] = [x * s for x in a]
        for off in SWEEP_OFF:
            g = gs0 + off
            block_rows.append(dict(source=name, B=Bs, n=ns, step=s, gamma0_db=gs0, offset_db=off,
                                   layered=expected_distortion(ch.dbar, [eps_normal(a, k, db(g)) for a, k in zip(al_s["layered"], pay)]) / ch.dbar[0],
                                   layered_robust=expected_distortion(ch.dbar, [eps_normal(a, k, db(g)) for a, k in zip(al_s["layered_robust"], pay)]) / ch.dbar[0],
                                   uniform=eps_normal(ns, kUs, db(g))))


def write(fn, rows):
    with open(RES / fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})


write("e5_ideal.csv", ideal_rows)
write("e5_polar.csv", polar_rows)
write("e5_bloque.csv", block_rows)

# ---- figura: arriba, E[d]/d̄_0 contra γ − γ0 (B = 16); abajo, efecto del tamaño de lote a γ0 y a γ0 − 2 dB
fig, axes = plt.subplots(2, 2, figsize=(COL_W, 0.95 * COL_W))
for ax, name in zip(axes[0], ("synthetic", "location")):
    for mode, ls in (("entropic", ":"), ("fixed", "-")):
        rows = [r for r in ideal_rows if r["source"] == name and r["mode"] == mode]
        x = [r["offset_db"] for r in rows]
        ax.plot(x, [r["layered_robust"] / r["dbar0"] for r in rows], ls=ls, color=INK, lw=0.9)
        ax.plot(x, [r["uniform"] / r["dbar0"] for r in rows], ls=ls, color=MID, lw=0.9)
    rows = [r for r in polar_rows if r["source"] == name]
    x = [r["offset_db"] for r in rows]
    ax.errorbar(x, [r["layered"] / r["dbar0"] for r in rows], yerr=[2 * r["layered_se"] / r["dbar0"] for r in rows],
                fmt="o", ms=2.5, color=INK, lw=0.6)
    ax.errorbar(x, [r["uniform"] / r["dbar0"] for r in rows], yerr=[2 * r["uniform_se"] / r["dbar0"] for r in rows],
                fmt="s", ms=2.5, color=MID, lw=0.6, mfc="white")
    ax.set_title(f"({'ab'[name == 'location']}) {name}, $B$ = {B}", fontsize=7)
    ax.set_xlabel(r"$\gamma-\gamma_0$ [dB]")
axes[0][0].set_ylabel(r"$\mathrm{E}[d]/\bar d_0$")
for ax, name in zip(axes[1], ("synthetic", "location")):
    for off, mk in ((0.0, "o"), (-2.0, "^")):
        rows = [r for r in block_rows if r["source"] == name and r["offset_db"] == off]
        Bs = [r["B"] for r in rows]
        ax.plot(Bs, [r["layered"] if off == 0.0 else r["layered_robust"] for r in rows], marker=mk, ms=2.5,
                color=INK, lw=0.9)
        ax.plot(Bs, [r["uniform"] for r in rows], marker=mk, ms=2.5, color=MID, lw=0.9, mfc="white")
    ax.set_xscale("log", base=2)
    ax.set_title(f"({'cd'[name == 'location']}) {name}, entropic", fontsize=7)
    ax.set_xlabel("meanings per block $B$")
axes[1][0].set_ylabel(r"$\mathrm{E}[d]/\bar d_0$")
fig.subplots_adjust(hspace=0.55, wspace=0.3)
save(fig, "e5_canal")
print("ok results/e5_*.csv, figures/e5_canal.pdf")
