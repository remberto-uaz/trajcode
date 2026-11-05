"""E4 (T37, Opción B): E1–E3 y la cadena de clases de la Parte 5 sobre subárboles de sustantivos de WordNet 3.0.

Subárboles (ADR-T03): location.n.01, artifact.n.01, animal.n.01; profundidad 6 desde la raíz del subárbol;
64 significados (terminales de mayor masa); masas por etiquetas de SemCor con suavizado de Laplace α = 1.
Salidas: results/e4_memoria.csv, results/e4_desalineacion.csv, results/e4_cadena.csv, figures/e4_wordnet.pdf.
Requiere `make data`.
"""
import csv, math, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from style import plt, COL_W, INK, MID, LIGHT, SERIES, RES, save

from trajcode.wordnet import load_nouns, subtree, viterbi_leaves, geodesic_distortion, resnik_distortion
from trajcode.measures import path_measure, meaning_law, entropy, cond_entropy_walk_given_meaning, kl_nodewise
from trajcode.coding import (synonymy_gap, viterbi_selection, selection_law, memoryless_adapted_length,
                             memory_states, bitsback_net_length)
from trajcode.alignment import misalignment, meaning_phrasing_split, drop_edges, detection_probability, \
    silent_error_probability
from trajcode.measures import perturb_kernel
from trajcode.progressive import level_chain, progressivity_cost

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZIP = ROOT / "data/wordnet/wordnet.zip"
SUBTREES = ["location", "artifact", "animal"]
DEPTH, LEAVES, ALPHA = 6, 64, 1.0
STRENGTHS = [0.25, 0.5, 1.0]          # fracción de nodos con ruteo editado
EDGE_DROP, RELABEL, REPS = 0.1, 0.1, 10

if not ZIP.exists():
    sys.exit("falta data/wordnet/wordnet.zip: ejecuta `make data`")
wn = load_nouns(ZIP)
mem, mis, cad = [], [], []
for name in SUBTREES:
    st = subtree(wn, wn.synset(name), depth=DEPTH, max_leaves=LEAVES, alpha=ALPHA)
    net, mu = st.net, st.mu
    P = path_measure(net); PM = meaning_law(P, mu)
    HM, HPiM = entropy(PM.values()), cond_entropy_walk_given_meaning(P, mu)
    g = synonymy_gap(P, mu); Eg = sum(PM[m] * g[m] for m in PM)
    Q = selection_law(viterbi_selection(P, mu), PM)
    mem.append(dict(subtree=name, nodes=len(net.succ), walks=len(P), meanings=len(PM),
                    multi=sum(1 for m in PM if sum(1 for w in P if mu[w] == m) > 1),
                    HM=HM, HPiM=HPiM, native_excess=Eg,
                    viterbi_memoryless_excess=memoryless_adapted_length(Q) - HM,
                    viterbi_states=memory_states(Q), bitsback_excess=bitsback_net_length(P, mu) - HM))
    # ---- E2 sobre WordNet: ediciones de ruteo (identidad por nodo), de topología y de etiquetas
    for s in STRENGTHS:
        for rep in range(REPS):
            R = perturb_kernel(net, s, seed=1000 * rep + int(100 * s))
            A = misalignment(net, R.p)
            nodal = sum(kl_nodewise(net, R).values())
            ds, dc = meaning_phrasing_split(net, R.p, mu)
            mis.append(dict(subtree=name, kind="routing", strength=s, rep=rep, A=A, nodewise=nodal,
                            meaning_term=ds, phrasing_term=dc, value=A))
    for rep in range(REPS):
        q, _ = drop_edges(net, EDGE_DROP, seed=rep)
        mis.append(dict(subtree=name, kind="topology", strength=EDGE_DROP, rep=rep, A=math.inf, nodewise=math.nan,
                        meaning_term=math.nan, phrasing_term=math.nan, value=detection_probability(net, q)))
        rng = random.Random(rep)
        moved = rng.sample(st.meanings, max(1, int(RELABEL * len(st.meanings))))
        perm = {m: rng.choice([x for x in st.meanings if x != m]) for m in moved}
        muR = {w: perm.get(m, m) for w, m in mu.items()}
        mis.append(dict(subtree=name, kind="label", strength=RELABEL, rep=rep, A=0.0, nodewise=0.0,
                        meaning_term=math.nan, phrasing_term=math.nan, value=silent_error_probability(net, mu, muR)))
    # ---- cadena de clases (Parte 5) con distorsión geodésica y de Resnik
    leaves = viterbi_leaves(st)
    p = np.array([PM[m] for m in st.meanings])
    for dname, d in (("geodesic", geodesic_distortion(st)), ("resnik", resnik_distortion(st, tree=leaves))):
        ch = level_chain(leaves, p, d)
        delta = progressivity_cost(ch, p, d)
        for j in range(len(ch.dbar)):
            cad.append(dict(subtree=name, distortion=dname, level=j, classes=len(ch.classes[j]), H=ch.H[j],
                            dbar=ch.dbar[j], dbar_rel=ch.dbar[j] / ch.dbar[0], R=ch.H[j] - delta[j], delta=delta[j]))
    print(f"{name}: |M| = {len(PM)}, H(M) = {HM:.3f}, H(Π|M) = {HPiM:.3f}, E[g] = {Eg:.3f}")


def write(fn, rows):
    with open(RES / fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})


write("e4_memoria.csv", mem)
write("e4_desalineacion.csv", mis)
write("e4_cadena.csv", cad)

# ---- figura: (a) precio de la memoria frente a los grafos sintéticos de E1; (b) Δ_j por nivel
fig, ax = plt.subplots(1, 2, figsize=(COL_W, 0.42 * COL_W))
syn = list(csv.DictReader(open(RES / "e1_memoria.csv", encoding="utf-8")))
ax[0].scatter([float(r["HPiM"]) for r in syn], [float(r["native"]) for r in syn], s=4, color=LIGHT, lw=0,
              label="synthetic (E1)")
for i, r in enumerate(mem):
    sty = SERIES[i % len(SERIES)]
    ax[0].scatter(r["HPiM"], r["native_excess"], s=18, color=sty["color"], marker=sty["marker"], label=r["subtree"],
                  zorder=3)
top = max(float(r["HPiM"]) for r in syn)
ax[0].plot([0, top], [0, top], color=MID, lw=0.6, ls="--")
ax[0].set_xlabel(r"$H(\Pi\mid M)$ [bits]"); ax[0].set_ylabel(r"native excess $\mathrm{E}[g(M)]$ [bits]")
ax[0].set_title("(a) price of memory", fontsize=7)
ax[0].legend(fontsize=5, frameon=False)
for i, name in enumerate(SUBTREES):
    for dname, ls in (("geodesic", "-"), ("resnik", ":")):
        rows = [r for r in cad if r["subtree"] == name and r["distortion"] == dname]
        sty = SERIES[i % len(SERIES)]
        ax[1].plot([r["level"] for r in rows], [r["delta"] for r in rows], ls=ls, marker=sty["marker"], ms=2.5, lw=0.8,
                   color=sty["color"])
# tipo de línea: continua = geodésica, punteada = Resnik (se explica en el pie de figura, sin leyenda)
ax[1].set_xlabel("level $j$"); ax[1].set_ylabel(r"$\Delta_j$ [bits]")
ax[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax[1].set_title("(b) cost of identifying classes", fontsize=7)
fig.subplots_adjust(wspace=0.45)
save(fig, "e4_wordnet")
print("ok results/e4_*.csv, figures/e4_wordnet.pdf")
