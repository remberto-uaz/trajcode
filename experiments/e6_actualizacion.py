"""E6 (T38, Opción C): costo de actualizar el conocimiento compartido contra el número de mensajes y el tamaño de la edición.

Escenario RTLS (plano remodelado): sitio de 2 edificios × 3 pisos × 2 alas × 4 salas, con un vestíbulo por piso desde
el que se nombran las dos primeras salas de cada ala (sinonimia). Remodelar un piso es una edición de ruteo en sus
nodos (piso, alas, vestíbulo); el receptor conserva los núcleos anteriores. c_v = ⌈log2|V|⌉ + 8(deg⁺(v) − 1) bits.

Políticas (costo de actualización = bits de actualización + bits extra por desalineación; sin el término N·H(P_S)):
talk_only, sync_first_use (cf. Chen et al. 2025), global (alquiler de esquís agregado), online (Proposición C2),
offline (Proposición C1 con N conocido) y hindsight (óptimo con la secuencia conocida, denominador de los cocientes).
Réplicas: REMODELS semillas de remodelación × STREAMS semillas de secuencia.

Salidas: results/e6_horizonte.csv, results/e6_edicion.csv, figures/e6_actualizacion.pdf.
"""
import csv, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from style import plt, COL_W, INK, MID, LIGHT, SERIES, RES, save

from trajcode.kbupdate import (make_site_graph, remodel, node_terms, teach_costs, separable_rule, visit_matrix,
                               stream_costs)

REMODELS, STREAMS = 5, 4
N_GRID = np.unique(np.round(np.logspace(0, 5, 26)).astype(int))
FLOORS_N = 2                       # pisos remodelados en el barrido de N
N_EDIT = 3000                      # horizonte del barrido del tamaño de la edición
POLICIES = ("talk_only", "sync_first_use", "global", "online", "offline")

net, mu, floor_of = make_site_graph(seed=0)
floors = sorted({f for f in floor_of.values() if f})
order = floors[:]
random.Random(0).shuffle(order)
c_all = teach_costs(net)


def instance(k, rseed):
    edited = {v for v, f in floor_of.items() if f in order[:k]}
    new = remodel(net, edited, seed=rseed)
    terms = node_terms(new, net.p)
    nodes = sorted(terms)
    nu = np.array([terms[v][0] for v in nodes]); D = np.array([terms[v][1] for v in nodes])
    c = np.array([c_all[v] for v in nodes], float)
    W, P, X = visit_matrix(new, nodes)
    return nodes, nu, D, c, P / P.sum(), X


def costs_at(N, draws, nodes, nu, D, c, X):
    out = stream_costs(X, draws[:N], D, c)
    T = separable_rule(dict(zip(nodes, nu)), dict(zip(nodes, D)), dict(zip(nodes, c)), N)
    mask = np.array([v in T for v in nodes])
    tot = (X[draws[:N]] * D).sum(axis=0)
    out["offline"] = float(c[mask].sum() + tot[~mask].sum())
    return out


rows_h, rows_e = [], []
for rseed in range(REMODELS):
    nodes, nu, D, c, P, X = instance(FLOORS_N, rseed)
    if rseed == 0:
        be = c / (nu * D)
        print(f"|V| = {len(net.succ)}, nodos editados = {len(nodes)}, N de equilibrio c_v/(ν D_v): "
              f"{be.min():.0f}–{be.max():.0f}")
    for sseed in range(STREAMS):
        draws = np.random.default_rng(1000 * rseed + sseed).choice(len(P), size=int(N_GRID[-1]), p=P)
        for N in N_GRID:
            r = costs_at(int(N), draws, nodes, nu, D, c, X)
            rows_h.append(dict(remodel_seed=rseed, stream_seed=sseed, N=int(N), **r))
    for k in range(1, len(floors) + 1):
        nodes, nu, D, c, P, X = instance(k, rseed)
        for sseed in range(STREAMS):
            draws = np.random.default_rng(1000 * rseed + sseed).choice(len(P), size=N_EDIT, p=P)
            r = costs_at(N_EDIT, draws, nodes, nu, D, c, X)
            rows_e.append(dict(remodel_seed=rseed, stream_seed=sseed, floors=k, edited_nodes=len(nodes), N=N_EDIT, **r))


def write(fn, rows):
    with open(RES / fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})


write("e6_horizonte.csv", rows_h)
write("e6_edicion.csv", rows_e)


def ratio(rows, key, pol):
    vals = sorted({r[key] for r in rows})
    return vals, [np.mean([r[pol] / r["hindsight"] for r in rows if r[key] == v and r["hindsight"] > 0]) for v in vals]


worst = max(r["online"] / r["hindsight"] for r in rows_h + rows_e if r["hindsight"] > 0)
print(f"peor cociente de la regla en línea: {worst:.3f} (cota de la Proposición C2: 2)")
for pol in POLICIES:
    _, y = ratio(rows_h, "N", pol)
    print(f"  {pol:15s} cociente medio máx. en N = {max(y):.2f}")

# ---- figura: (a) cociente con el óptimo a posteriori contra N; (b) contra el número de pisos remodelados
fig, ax = plt.subplots(1, 2, figsize=(COL_W, 0.42 * COL_W))
for i, pol in enumerate(POLICIES):
    sty = SERIES[i] if i < len(SERIES) else dict(color=LIGHT, ls="-", marker="x")
    x, y = ratio(rows_h, "N", pol)
    ax[0].plot(x, y, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=2.2, lw=0.8,
               label=pol.replace("_", " "))
    x, y = ratio(rows_e, "floors", pol)
    ax[1].plot(x, y, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=2.5, lw=0.8)
for a in ax:
    a.axhline(2.0, color=MID, lw=0.6, ls="--")
ax[0].set_yscale("log"); ax[0].set_xscale("log")
ax[1].set_ylim(bottom=0.9)
ax[0].set_xlabel("messages $N$"); ax[0].set_ylabel("cost / hindsight optimum")
ax[1].set_xlabel("remodelled floors"); ax[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax[0].set_title(f"(a) {FLOORS_N} floors remodelled", fontsize=7)
ax[1].set_title(f"(b) $N$ = {N_EDIT}", fontsize=7)
ax[0].legend(fontsize=5, frameon=False)
fig.subplots_adjust(wspace=0.4)
save(fig, "e6_actualizacion")
print("ok results/e6_*.csv, figures/e6_actualizacion.pdf")
