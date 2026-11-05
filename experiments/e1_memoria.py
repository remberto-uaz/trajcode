"""E1 (Parte 1): precio del ruteo nativo y de la falta de memoria. Salida: results/e1_*.csv, figures/e1_memoria.pdf."""
import csv, math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from style import plt, COL_W, INK, MID, LIGHT, SERIES, RES, save
from trajcode.graphs import make_random_dag, make_meaning_map
from trajcode.measures import path_measure, meaning_law, entropy, cond_entropy_walk_given_meaning
from trajcode.coding import (viterbi_selection, selection_law, native_length, adapted_length,
                             memoryless_adapted_length, bitsback_net_length)

SEED0, TRIALS = 2026, 600
rows = []
for t in range(TRIALS):
    seed = SEED0 + t
    n = 5 + t % 8
    kind = "endpoint" if (t // 8) % 2 == 0 else "context"   # independiente del tamaño
    k = 2 + (t // 16) % 3
    net = make_random_dag(n, max_out=3, seed=seed)
    mu = make_meaning_map(net, k, kind, seed)
    P = path_measure(net); PM = meaning_law(P, mu); HM = entropy(PM.values())
    f = viterbi_selection(P, mu); Q = selection_law(f, PM)
    rows.append(dict(seed=seed, n=n, k=k, kind=kind, HM=HM, HPiM=cond_entropy_walk_given_meaning(P, mu),
                     native=native_length(f, P, PM) - HM, adapted_mem=adapted_length(Q) - HM,
                     adapted_nomem=memoryless_adapted_length(Q) - HM, bitsback=bitsback_net_length(P, mu) - HM))

with open(RES / "e1_memoria.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# comprobaciones del teorema sobre los datos (constitution P4.4)
assert all(-1e-9 <= r["native"] <= r["HPiM"] + 1e-9 for r in rows)
assert all(abs(r["adapted_mem"]) < 1e-9 and abs(r["bitsback"]) < 1e-9 for r in rows)
assert all(abs(r["adapted_nomem"]) < 1e-9 for r in rows if r["kind"] == "endpoint")

fig, (a, b) = plt.subplots(1, 2, figsize=(2 * COL_W, 2.2))
x = np.array([r["HPiM"] for r in rows]); y = np.array([r["native"] for r in rows])
a.scatter(x, y, s=5, color=LIGHT, edgecolor="none", label="native code (one graph)")
edges = np.linspace(0, x.max(), 9); mids = (edges[1:] + edges[:-1]) / 2
means = [y[(x >= lo) & (x < hi)].mean() if ((x >= lo) & (x < hi)).any() else np.nan for lo, hi in zip(edges, edges[1:])]
a.plot(mids, means, **SERIES[0], label="native code, binned mean")
a.plot([0, x.max()], [0, x.max()], color=MID, ls="--", lw=1.0, label=r"bound $H(\Pi\mid M)$ (Thm. 1b)")
a.axhline(0, color=INK, lw=0.8, ls=":", label="adapted with memory, bits-back")
a.set_xlabel(r"synonymy entropy $H(\Pi\mid M)$ [bits]"); a.set_ylabel(r"excess over $H(M)$ [bits]")
a.set_title("(a) price of native routing"); a.legend(loc="upper left")

# (b) familia controlada: raíz -> contexto -> palabra compartida -> terminal, con tasa de polisemia rho
from trajcode.graphs import TrajectoryNetwork, walks
from trajcode.coding import memory_states

def polysemous_net(rho, seed, n_ctx=3, n_words=3, n_term=4):
    rng = np.random.default_rng(seed)
    C = [f"c{i}" for i in range(n_ctx)]; W = [f"w{j}" for j in range(n_words)]; T = [f"t{k}" for k in range(n_term)]
    succ = {"r": C, **{c: W for c in C}, **{w: T for w in W}, **{t: [] for t in T}}
    def row(keys):
        x = rng.dirichlet(np.ones(len(keys))); return dict(zip(keys, x))
    p = {"r": row(C), **{c: row(W) for c in C}, **{w: row(T) for w in W}}
    net = TrajectoryNetwork(succ, p, "r"); net.validate()
    flip = {(c, t): rng.random() < rho for c in C for t in T}        # el contexto cambia el significado de t
    mu = {w: (int(w[-1][1:]) + (1 + int(w[1][1:])) * flip[(w[1], w[-1])]) % n_term for w, _ in walks(net)}
    return net, mu

rhos = np.linspace(0, 1, 11); exc, need, inst = [], [], []
for rho in rhos:
    e, nd = [], []
    for s_ in range(60):
        net, mu = polysemous_net(rho, 9000 + s_)
        P = path_measure(net); PM = meaning_law(P, mu); Q = selection_law(viterbi_selection(P, mu), PM)
        e.append(memoryless_adapted_length(Q) - entropy(PM.values())); nd.append(memory_states(Q) > 1)
        inst.append((rho, 9000 + s_, e[-1], int(nd[-1])))
    exc.append(np.mean(e)); need.append(np.mean(nd))
with open(RES / "e1_polisemia_inst.csv", "w", newline="") as fh:
    w_ = csv.writer(fh); w_.writerow(["rho", "seed", "excess", "needs_memory"]); w_.writerows(inst)
with open(RES / "e1_polisemia.csv", "w", newline="") as fh:
    w_ = csv.writer(fh); w_.writerow(["rho", "excess_memoryless_viterbi", "frac_need_memory"])
    w_.writerows(zip(rhos, exc, need))
assert abs(exc[0]) < 1e-9 and need[0] == 0          # rho = 0: significado por nodo final (Thm. 2c)
b.plot(rhos, exc, **SERIES[0], label="memoryless adapted code (Viterbi selection)")
b.axhline(0, color=INK, lw=0.8, ls=":", label="adapted code with memory")
b.set_xlabel(r"polysemy rate $\rho$ (context flips a word's meaning)")
b.set_ylabel(r"excess over $H(M)$ [bits]")
b.set_title("(b) polysemy forces memory (Thm. 2c-d)"); b.legend(loc="lower center")
save(fig, "e1_memoria")

ctx = [r for r in rows if r["kind"] == "context"]
print(f"E1 polisemia: exceso sin memoria de {exc[0]:.3f} (rho=0) a {max(exc):.3f} bits; fraccion que necesita memoria hasta {max(need):.2f}")
print(f"E1: {len(rows)} grafos; contexto con exceso > 0 sin memoria: "
      f"{sum(r['adapted_nomem'] > 1e-9 for r in ctx)}/{len(ctx)}; "
      f"exceso máximo {max(r['adapted_nomem'] for r in ctx):.3f} bits; "
      f"exceso nativo medio {np.mean(y):.3f} bits")
