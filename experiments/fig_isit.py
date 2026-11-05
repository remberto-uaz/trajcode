"""Fig. 2 del manuscrito ISIT (una columna): E1 polisemia y E3 brecha de los códigos de cociente.

Solo lee results/*.csv (no recalcula nada). Salida: figures/fig_isit.pdf.
"""
import csv, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from style import plt, COL_W, INK, MID, SERIES, RES, save
# (fuente matemática: la de style.py, "stix")

pol = list(csv.DictReader(open(RES / "e1_polisemia.csv")))
rho = [float(r["rho"]) for r in pol]
exc = [float(r["excess_memoryless_viterbi"]) for r in pol]
need = [float(r["frac_need_memory"]) for r in pol]
gap = list(csv.DictReader(open(RES / "e3_brecha.csv")))
bins = np.linspace(0, 1, 11); mids = (bins[1:] + bins[:-1]) / 2

fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.55), gridspec_kw=dict(wspace=0.45))
inst = list(csv.DictReader(open(RES / "e1_polisemia_inst.csv")))
q1 = [np.percentile([float(r["excess"]) for r in inst if float(r["rho"]) == x], 25) for x in rho]
q3 = [np.percentile([float(r["excess"]) for r in inst if float(r["rho"]) == x], 75) for x in rho]
a.fill_between(rho, q1, q3, color=MID, alpha=0.18, lw=0)
a.plot(rho, exc, **SERIES[0])
a.set_xlabel(r"polysemy rate $\rho$"); a.set_ylabel("excess over $H(M)$ [bits]")
a.set_ylim(0, max(max(exc), max(q3)) * 1.1)
a.set_title("(a) memoryless code, E1")
for i, (fam, lab) in enumerate([("path", "path"), ("cycle", "cycle"), ("ultrametric", "ultram.")]):
    g = [r for r in gap if r["family"] == fam]
    m = [np.mean([float(r["gap"]) for r in g if lo <= float(r["frac"]) < hi] or [np.nan]) for lo, hi in zip(bins, bins[1:])]
    lo_ = [np.percentile([float(r["gap"]) for r in g if lo <= float(r["frac"]) < hi], 25) for lo, hi in zip(bins, bins[1:])]
    hi_ = [np.percentile([float(r["gap"]) for r in g if lo <= float(r["frac"]) < hi], 75) for lo, hi in zip(bins, bins[1:])]
    if fam == "ultrametric":
        b.fill_between(mids, lo_, hi_, color=MID, alpha=0.18, lw=0)
    b.plot(mids, m, **SERIES[i], label=lab)
b.set_xlabel(r"$D/D_{\max}$"); b.set_ylabel("gap to $R(D)$ [bits]"); b.set_ylim(0, None)
b.set_title("(b) quotient codes, E3"); b.legend(loc="lower center", handlelength=1.6, fontsize=6)
save(fig, "fig_isit")
print("figures/fig_isit.pdf")
