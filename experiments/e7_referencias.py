"""E7 (T39): esquemas de referencia para E1–E4 y complejidad medida.

Los CSV de la v2 (E1–E3) y de E4 no se modifican; todo va en archivos nuevos.
- results/e7_tasa.csv: exceso sobre H(M) de cada esquema sin pérdidas y conocimiento que necesita el decodificador,
  en las 600 instancias de E1, el ejemplo de la semana y los subárboles de WordNet de E4.
- results/e7_desalineacion.csv: bits extra por desalineación de ruteo para el índice (o la aritmética semántica)
  construido con la ley de significados del receptor y para los códigos de camino (nativo y sintáctico).
- results/e7_embebido.csv: código embebido jerárquico (Huffman por nivel) frente a H(C_j) y R(d̄_j).
- results/e7_complejidad.csv: tiempos medidos (no reproducibles bit a bit; se excluyen de la comparación de hashes)
  y results/e7_pendientes.csv: pendientes log–log ajustadas, redondeadas a 0.1.
Requiere `make data` para WordNet.
"""
import csv, math, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from style import RES

from trajcode.graphs import make_random_dag, make_layered_dag, make_meaning_map, make_toy_week_graph, make_ontology_tree
from trajcode.measures import (path_measure, meaning_law, entropy, cond_entropy_walk_given_meaning, perturb_kernel)
from trajcode.coding import (viterbi_selection, selection_law, native_length, memoryless_adapted_length,
                             bitsback_net_length, memory_states)
from trajcode.alignment import misalignment, meaning_phrasing_split
from trajcode.baselines import (huffman_expected_length, sfe_expected_length, decoder_parameters,
                                embedded_huffman_lengths, viterbi_dp)
from trajcode.progressive import level_chain, progressivity_cost, optimal_allocation
from trajcode.kbupdate import node_terms, separable_rule, online_rule
from trajcode.wordnet import load_nouns, subtree, viterbi_leaves, geodesic_distortion

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZIP = ROOT / "data/wordnet/wordnet.zip"
SEED0, TRIALS = 2026, 600            # mismas instancias que E1
MIS_TRIALS, MIS_FRACTION = 200, 0.5
TREES = 40


def rate_row(source, ident, net, mu):
    P = path_measure(net); PM = meaning_law(P, mu); HM = entropy(PM.values()); pm = list(PM.values())
    HPiM = cond_entropy_walk_given_meaning(P, mu)
    f = viterbi_selection(P, mu); Q = selection_law(f, PM)
    return dict(source=source, id=ident, meanings=len(PM), HM=HM, HPiM=HPiM,
                index_huffman=huffman_expected_length(pm) - HM, index_sfe=sfe_expected_length(pm) - HM,
                index_arithmetic_block=0.0, syntactic=HPiM, native=native_length(f, P, PM) - HM,
                adapted_memoryless=memoryless_adapted_length(Q) - HM, bitsback=bitsback_net_length(P, mu) - HM,
                memory_states=memory_states(Q), **{f"params_{k}": v for k, v in decoder_parameters(net, len(PM)).items()})


# ---------------------------------------------------------------- tasa
rate = []
for t in range(TRIALS):
    seed = SEED0 + t
    n = 5 + t % 8
    kind = "endpoint" if (t // 8) % 2 == 0 else "context"
    k = 2 + (t // 16) % 3
    net = make_random_dag(n, max_out=3, seed=seed)
    rate.append(rate_row(f"synthetic-{kind}", seed, net, make_meaning_map(net, k, kind, seed)))
net, mu = make_toy_week_graph()
rate.append(rate_row("week", 0, net, mu))
wn = load_nouns(ZIP) if ZIP.exists() else sys.exit("falta data/wordnet/wordnet.zip: ejecuta `make data`")
subs = {name: subtree(wn, wn.synset(name), depth=6, max_leaves=64) for name in ("location", "artifact", "animal")}
for name, st in subs.items():
    rate.append(rate_row("wordnet", name, st.net, st.mu))

# ---------------------------------------------------------------- desalineación
mis = []
for t in range(MIS_TRIALS):
    seed = 7000 + t
    net = make_random_dag(6 + t % 7, seed=seed)
    kind = "endpoint" if t % 2 == 0 else "context"
    mu = make_meaning_map(net, 3, kind, seed)
    R = perturb_kernel(net, MIS_FRACTION, seed=seed + 1)
    # el emisor usa los núcleos editados (R); el receptor conserva los de net
    ds, dc = meaning_phrasing_split(R, net.p, mu)
    mis.append(dict(source=f"synthetic-{kind}", id=seed, A_path=misalignment(R, net.p), index_or_semantic_arith=ds,
                    phrasing=dc))
for name, st in subs.items():
    for rep in range(10):
        R = perturb_kernel(st.net, MIS_FRACTION, seed=500 + rep)
        ds, dc = meaning_phrasing_split(R, st.net.p, st.mu)
        mis.append(dict(source="wordnet", id=f"{name}-{rep}", A_path=misalignment(R, st.net.p),
                        index_or_semantic_arith=ds, phrasing=dc))

# ---------------------------------------------------------------- código embebido
emb = []


def embed_rows(source, ident, leaves, p, d):
    ch = level_chain(leaves, p, d)
    L = embedded_huffman_lengths(ch, p)
    delta = progressivity_cost(ch, p, d)
    for j in range(len(ch.H)):
        emb.append(dict(source=source, id=ident, level=j, classes=len(ch.classes[j]), H=ch.H[j], embedded_huffman=L[j],
                        R_dbar=ch.H[j] - delta[j], dbar=ch.dbar[j]))


for s in range(TREES):
    leaves, d = make_ontology_tree(3, 3, seed=300 + s)
    if len(leaves) < 2:
        continue
    embed_rows("synthetic-tree", 300 + s, leaves, np.random.default_rng(300 + s).dirichlet(np.ones(len(leaves))),
               np.asarray(d, float))
for name, st in subs.items():
    PM = meaning_law(path_measure(st.net), st.mu)
    embed_rows("wordnet", name, viterbi_leaves(st), np.array([PM[m] for m in st.meanings]), geodesic_distortion(st))


# ---------------------------------------------------------------- complejidad
def best_time(fn, reps=3):
    out = math.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); out = min(out, time.perf_counter() - t0)
    return out


cx = []
for n in (250, 500, 1000, 2000, 4000, 8000, 16000, 32000):
    net = make_layered_dag(n // 50, 50, seed=n)
    E = sum(len(s) for s in net.succ.values())
    lab = {v: hash(v) % 7 for v in net.terminals}
    cx.append(dict(task="viterbi_dp", size_name="E", size=E, seconds=best_time(lambda: viterbi_dp(net, lab))))
    R = perturb_kernel(net, 0.3, seed=n + 1)
    cx.append(dict(task="update_rule_C1", size_name="E", size=E, seconds=best_time(
        lambda: (lambda tm: separable_rule({v: a for v, (a, _) in tm.items()}, {v: b for v, (_, b) in tm.items()},
                                           {v: 30.0 for v in tm}, 1000))(node_terms(R, net.p)))))
for visits in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
    D, c = {v: 0.01 for v in range(50)}, {v: 1e9 for v in range(50)}
    stream = [{v % 50: 1} for v in range(visits)]
    cx.append(dict(task="online_rule_C2", size_name="visits", size=visits,
                   seconds=best_time(lambda: online_rule(stream, D, c), reps=2)))
for nb in (64, 128, 256, 512, 1024):
    table = np.linspace(1.0, 0.0, nb + 1)
    cx.append(dict(task="allocation_dp_h3", size_name="n", size=nb, seconds=best_time(
        lambda: optimal_allocation([1.0, 0.6, 0.3, 0.0], [4, 4, 4], nb, lambda m, kk: table[m]), reps=2)))
for w in (8, 16, 32, 64, 128):
    net = make_layered_dag(4, w, max_out=2, seed=40 + w)
    mu = make_meaning_map(net, w, "context", 40 + w)
    P = path_measure(net); Q = selection_law(viterbi_selection(P, mu), meaning_law(P, mu))
    cx.append(dict(task="memory_states_Nf", size_name="meanings", size=len(Q),
                   seconds=best_time(lambda: memory_states(Q))))


def write(fn, rows):
    with open(RES / fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})


write("e7_tasa.csv", rate)
write("e7_desalineacion.csv", mis)
write("e7_embebido.csv", emb)
write("e7_complejidad.csv", cx)
slopes = []
for task in dict.fromkeys(r["task"] for r in cx):
    rs = [r for r in cx if r["task"] == task]
    x = np.log([r["size"] for r in rs]); y = np.log([max(r["seconds"], 1e-9) for r in rs])
    slopes.append(dict(task=task, size_name=rs[0]["size_name"], slope=round(float(np.polyfit(x, y, 1)[0]), 1),
                       slope_last_pair=round(float((y[-1] - y[-2]) / (x[-1] - x[-2])), 1),
                       min_size=rs[0]["size"], max_size=rs[-1]["size"]))
write("e7_pendientes.csv", slopes)

# ---------------------------------------------------------------- resumen en consola
for src in ("synthetic-endpoint", "synthetic-context", "week", "wordnet"):
    rs = [r for r in rate if r["source"] == src]
    m = lambda k: np.mean([r[k] for r in rs])
    print(f"{src:20s} n={len(rs):3d} Huff={m('index_huffman'):.3f} SFE={m('index_sfe'):.3f} syn={m('syntactic'):.3f} "
          f"nat={m('native'):.3f} adapt0={m('adapted_memoryless'):.3f} N(f)max={max(r['memory_states'] for r in rs)} "
          f"params idx/path={m('params_index'):.1f}/{m('params_path'):.1f}")
for src in ("synthetic-endpoint", "synthetic-context", "wordnet"):
    rs = [r for r in mis if r["source"] == src]
    print(f"desalineación {src:20s} A={np.mean([r['A_path'] for r in rs]):.4f} índice={np.mean([r['index_or_semantic_arith'] for r in rs]):.4f}")
for src in ("synthetic-tree", "wordnet"):
    rs = [r for r in emb if r["source"] == src and 0 < r["level"]]
    print(f"embebido {src}: exceso Huffman/nivel máx = {max(r['embedded_huffman'] - r['H'] for r in rs):.3f}, "
          f"medio = {np.mean([r['embedded_huffman'] - r['H'] for r in rs]):.3f}")
for s in slopes:
    print(f"pendiente {s['task']:18s} vs {s['size_name']}: {s['slope']} (último par {s['slope_last_pair']})")
print("ok results/e7_*.csv")
