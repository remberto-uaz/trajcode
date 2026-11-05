"""E3 (Parte 3): R(D) geodésica frente a códigos de cociente, y prueba de refinabilidad sucesiva.
Salida: results/e3_*.csv, figures/e3_distorsion.pdf. Es la fuente de los números de theory/03-distorsion.tex."""
import csv, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from style import plt, COL_W, INK, MID, LIGHT, SERIES, RES, save
from trajcode.graphs import make_ontology_tree
from trajcode.distortion import (geodesic_matrix, quotient_points, lower_envelope, envelope_at, blahut_arimoto,
                                 rd_at, refinability_residual)

rng = np.random.default_rng(2026)

# ---------------- (a) ejemplo de la semana ----------------
p_w = np.array([0.5, 0.3, 0.2]); d_w = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], float)
pts_w = quotient_points(p_w, d_w); H_w = lower_envelope(pts_w)
curve = [blahut_arimoto(p_w, d_w, b)[:2] for b in np.geomspace(0.05, 30, 120)]
curve = sorted((D, R) for R, D in curve)
table_w = [(D, *rd_at(p_w, d_w, D), envelope_at(H_w, D)) for D in (0.1, 0.2, 0.3, 0.5)]
with open(RES / "e3_semana.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["D", "R_BA", "D_reached", "envelope", "gap"])
    w.writerows([(D, R, Dr, E, E - R) for D, R, Dr, E in table_w])

# ---------------- (b) brecha por familia de métricas ----------------
n = 6
ring = np.eye(n, k=1) + np.eye(n, k=-1)
cyc = ring.copy(); cyc[0, n - 1] = cyc[n - 1, 0] = 1
families = {"path": lambda s: geodesic_matrix(ring),
            "cycle": lambda s: geodesic_matrix(cyc),
            "ultrametric": lambda s: np.array(make_ontology_tree(2, 3, seed=s)[1], float)}
betas = np.geomspace(0.05, 40, 40); gap_rows = []
for fam, gen in families.items():
    for s in range(25):
        d = gen(s)
        while d.shape[0] < 3:                          # árbol degenerado: se resiembra
            s += 1000; d = gen(s)
        p = rng.dirichlet(np.ones(d.shape[0]))
        H = lower_envelope(quotient_points(p, d)); Dmax = H[-1][0]
        for b in betas:                                 # un barrido en β por instancia
            R, D, _ = blahut_arimoto(p, d, b, tol=1e-10)
            if D < Dmax:
                gap_rows.append(dict(family=fam, trial=s, frac=D / Dmax, gap=envelope_at(H, D) - R))
with open(RES / "e3_brecha.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(gap_rows[0])); w.writeheader(); w.writerows(gap_rows)
assert all(r["gap"] > -1e-4 for r in gap_rows)       # Teorema 1(a): la envolvente nunca está por debajo de R(D)

# ---------------- (c) tabla de refinabilidad (Conjetura 1) ----------------
def balanced(br, lv):
    m = br ** lv
    return np.array([[0 if i == j else lv - next(k for k in range(lv + 1) if i // br ** (lv - k - 1) != j // br ** (lv - k - 1) or k == lv)
                      for j in range(m)] for i in range(m)], float)

cases = [("Hamming, random P_M (control)", lambda: (1 - np.eye(5), rng.dirichlet(np.ones(5)))),
         ("Random ultrametric, uniform P_M", lambda: (lambda d: (d, np.ones(len(d)) / len(d)))(np.array(make_ontology_tree(3, 2, int(rng.integers(1e6)))[1], float))),
         ("Balanced binary tree (8 leaves), uniform P_M", lambda: (balanced(2, 3), np.ones(8) / 8)),
         ("Random ultrametric, random P_M", lambda: (lambda d: (d, rng.dirichlet(np.ones(len(d)))))(np.array(make_ontology_tree(3, 2, int(rng.integers(1e6)))[1], float))),
         ("Balanced binary tree (8 leaves), random P_M", lambda: (balanced(2, 3), rng.dirichlet(np.ones(8)))),
         ("Path metric, random P_M", lambda: (geodesic_matrix(np.eye(5, k=1) + np.eye(5, k=-1)), rng.dirichlet(np.ones(5)))),
         ("Cycle metric, random P_M", lambda: (lambda c: (geodesic_matrix(c), rng.dirichlet(np.ones(5))))(np.eye(5, k=1) + np.eye(5, k=-1) + np.eye(5, k=4) + np.eye(5, k=-4)))]
TR = 60; ref_rows = []
for name, gen in cases:
    fails, worst, done, uniq, min_sig = 0, 0.0, 0, 0, np.inf
    while done < TR:
        d, p = gen()
        if d.shape[0] < 3:
            continue
        b1, b2 = sorted(rng.uniform(0.4, 5, 2))
        r = refinability_residual(blahut_arimoto(p, d, b1)[2], blahut_arimoto(p, d, b2)[2])
        fails += r > 1e-5; worst = max(worst, r); done += 1
        if r > 1e-5:
            # Unicidad (T27-v2): la ley óptima de salida solo entra vía K q con K = exp(-βd) y -log estrictamente
            # cóncava, así que K q* es única; si K es invertible, q* y el canal óptimo son únicos en esa pendiente.
            sig = min(np.linalg.svd(np.exp(-bb * d), compute_uv=False).min() for bb in (b1, b2))
            uniq += sig > 1e-9; min_sig = min(min_sig, sig)
    ref_rows.append(dict(case=name, not_refinable=fails, trials=TR, max_residual=worst,
                         unique_failures=uniq, min_sigma_failures=min_sig if fails else float("nan")))
with open(RES / "e3_refinabilidad.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ref_rows[0])); w.writeheader(); w.writerows(ref_rows)
assert ref_rows[0]["not_refinable"] == 0                 # control de Hamming

# ---------------- figura ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(2 * COL_W, 2.2))
a.plot([c[0] for c in curve], [c[1] for c in curve], color=INK, ls="-", label="$R(D)$ (Blahut-Arimoto)")
a.plot([h[0] for h in H_w], [h[1] for h in H_w], color=MID, ls="--", label="quotient envelope")
a.scatter([q[0] for q in pts_w], [q[1] for q in pts_w], s=14, color=MID, marker="s", zorder=3, label="quotient codes")
a.annotate("{Mon,Tue},{Wed}", (0.3, 0.722), xytext=(0.03, 0.22), fontsize=6.5, color=INK,
           arrowprops=dict(arrowstyle="-", lw=0.5, color=MID))
a.set_xlim(-0.02, 0.75); a.set_xlabel("expected geodesic distortion $D$"); a.set_ylabel("rate [bits]")
a.set_title('(a) "what day is today"'); a.legend(loc="upper right")
bins = np.linspace(0, 1, 11); mids = (bins[1:] + bins[:-1]) / 2
def binned(fam):
    return [np.mean([r["gap"] for r in gap_rows if r["family"] == fam and lo <= r["frac"] < hi] or [np.nan])
            for lo, hi in zip(bins, bins[1:])]
for i, fam in enumerate(families):
    b.plot(mids, binned(fam), **SERIES[i], label=fam)
b.set_xlabel(r"$D / D_{\max}$"); b.set_ylabel("gap to $R(D)$ [bits]")
b.set_title("(b) gap of quotient codes (6 meanings)"); b.legend(loc="lower center")
save(fig, "e3_distorsion")

print("E3 semana:", [(D, round(R, 3), round(E - R, 3)) for D, R, Dr, E in table_w])
print("E3 brecha media máxima por familia (bins de D/Dmax):", {fam: round(float(np.nanmax(binned(fam))), 3) for fam in families})
for r in ref_rows:
    print(f"  {r['case']:46s} {r['not_refinable']:2d}/{r['trials']}  max {r['max_residual']:.1e}  únicos {r['unique_failures']}  σmin {r['min_sigma_failures']:.1e}")
