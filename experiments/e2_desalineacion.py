"""E2 (Parte 2): costo operacional de la desalineación y separación detección / error silencioso.
Salida: results/e2_*.csv, figures/e2_desalineacion.pdf."""
import csv, math, sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from style import plt, COL_W, INK, MID, LIGHT, SERIES, RES, save
from trajcode.graphs import TrajectoryNetwork, make_random_dag, make_meaning_map, walks
from trajcode.measures import path_measure, entropy
from trajcode.coding import ArithmeticPathCoder
from trajcode.alignment import (misalignment, local_bounds, drop_edges, detection_probability,
                                silent_error_probability)

K = 200                       # caminos concatenados por flujo aritmético (sobrecarga ≤ 2/K bits por camino)


def chain(net: TrajectoryNetwork, K: int) -> TrajectoryNetwork:
    """Red que recorre K copias seguidas de net: el terminal de la copia i enlaza con la raíz de la i+1."""
    succ, p = {}, {}
    for i in range(K):
        for v, s in net.succ.items():
            if s:
                succ[(i, v)] = [(i, u) for u in s]; p[(i, v)] = {(i, u): x for u, x in net.p[v].items()}
            elif i < K - 1:
                succ[(i, v)] = [(i + 1, net.root)]; p[(i, v)] = {(i + 1, net.root): 1.0}
            else:
                succ[(i, v)] = []
    return TrajectoryNetwork(succ, p, (0, net.root))


def chained_kernel(net, kernel_of):
    """Núcleo de la red encadenada: en (i, v) usa kernel_of[v]; en un terminal salta a la raíz de la copia i+1."""
    def k(prefix):
        i, v = prefix[-1]
        if net.succ[v]:
            return {(i, u): x for u, x in kernel_of[v].items()}
        return {(i + 1, net.root): 1.0}
    return k


def measured_bits_per_walk(net, qR, rng):
    """Codifica K caminos de P_S con el núcleo del receptor y con el del emisor (bits reales por camino).
    Devuelve también la diferencia ideal de la MISMA muestra, para separar sobrecarga del codificador
    (≤ 4/K) de ruido de muestreo."""
    P = path_measure(net); ws = list(P); pr = np.array([P[w] for w in ws])
    seq = [ws[i] for i in rng.choice(len(ws), size=K, p=pr / pr.sum())]
    big = chain(net, K)
    walk = tuple((i, v) for i, w in enumerate(seq) for v in w)
    coderR = ArithmeticPathCoder(big, chained_kernel(net, qR))
    bits = coderR.encode(walk)
    assert coderR.decode(bits) == walk
    bitsS = ArithmeticPathCoder(big, chained_kernel(net, net.p)).encode(walk)
    QR = lambda w: math.prod(qR[a][b] for a, b in zip(w, w[1:]))
    sample_ideal = np.mean([math.log2(P[w] / QR(w)) for w in seq])
    return len(bits) / K, len(bitsS) / K, sample_ideal


rng = np.random.default_rng(7)
net = make_random_dag(9, max_out=3, seed=11)
HP = entropy(path_measure(net).values())
rows = []
for strength in np.linspace(0, 2.5, 11):           # factor log-uniforme e^{±strength} por arista
    for rep in range(6):
        r = random.Random(1000 * rep + int(strength * 100))
        qR = {}
        for v, row in net.p.items():
            w = {u: x * math.exp(r.uniform(-strength, strength)) for u, x in row.items()}
            z = sum(w.values()); qR[v] = {u: x / z for u, x in w.items()}
        A = misalignment(net, qR); lo, hi = local_bounds(net, qR)
        bR, bS, ideal = measured_bits_per_walk(net, qR, rng)
        assert abs((bR - bS) - ideal) <= 4 / K + 1e-12      # sobrecarga del codificador (Lema 1)
        rows.append(dict(strength=strength, rep=rep, A=A, pinsker=lo, upper=hi, measured_excess=bR - bS, sample_ideal=ideal,
                         bits_receiver=bR, bits_aligned=bS, H=HP))
with open(RES / "e2_ruteo.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# (b) ediciones topológicas y de etiquetas sobre muchos grafos
rows_b = []
for frac in np.linspace(0, 1, 11):
    det_t, det_s, sil_t, sil_s = [], [], [], []
    for s_ in range(40):
        g = make_random_dag(9, seed=500 + s_); P = path_measure(g)
        qR, _ = drop_edges(g, frac, seed=700 + s_)
        det_t.append(detection_probability(g, qR))
        ws = list(P); pr = np.array([P[w] for w in ws]); idx = rng.choice(len(ws), size=400, p=pr / pr.sum())
        det_s.append(np.mean([any(qR.get(a, {}).get(b, 0) <= 0 for a, b in zip(ws[i], ws[i][1:])) for i in idx]))
        mu = make_meaning_map(g, 4, "endpoint", s_)
        rr = random.Random(s_); muR = {w: (m if rr.random() >= frac else (m + 1) % 4) for w, m in mu.items()}
        sil_t.append(silent_error_probability(g, mu, muR))
        sil_s.append(np.mean([mu[ws[i]] != muR[ws[i]] for i in idx]))
        assert misalignment(g, g.p) == 0
    rows_b.append(dict(fraction=frac, detection_theory=np.mean(det_t), detection_sim=np.mean(det_s),
                       silent_theory=np.mean(sil_t), silent_sim=np.mean(sil_s)))
with open(RES / "e2_ediciones.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_b[0])); w.writeheader(); w.writerows(rows_b)

fig, (a, b) = plt.subplots(1, 2, figsize=(2 * COL_W, 2.2))
xs = sorted({r["strength"] for r in rows})
mean = lambda key: [np.mean([r[key] for r in rows if r["strength"] == x]) for x in xs]
a.plot(xs, mean("pinsker"), color=MID, ls="--", lw=0.9, label="Pinsker lower bound (Thm. 1d)")
a.plot(xs, mean("A"), color=INK, ls="-", label=r"theory $A=\sum_v\nu_S(v)D_v$")
sd = [np.std([r["measured_excess"] for r in rows if r["strength"] == x]) for x in xs]
a.errorbar(xs, mean("measured_excess"), yerr=sd, color=MID, ls="none", marker="o", capsize=1.5, elinewidth=0.6,
           label="measured (arithmetic coder, ±1 sd)")
a.set_xlabel("routing-edit strength"); a.set_ylabel("extra bits per walk")
a.set_title("(a) routing edits cost bits"); a.legend(loc="upper left")
fr = [r["fraction"] for r in rows_b]
b.plot(fr, [r["detection_theory"] for r in rows_b], color=INK, ls="-", marker="x", markersize=4,
       label="topological edit: detected")
b.plot(fr, [r["detection_sim"] for r in rows_b], color=INK, ls="none", marker="o", mfc="none", markersize=5)
b.plot(fr, [r["silent_theory"] for r in rows_b], color=MID, ls="--", marker="+", markersize=5,
       label="label edit: silent error")
b.plot(fr, [r["silent_sim"] for r in rows_b], color=MID, ls="none", marker="s", mfc="none", markersize=4.5)
b.set_xlabel("fraction of edited nodes / labels"); b.set_ylabel("probability per message")
b.set_title("(b) detected vs. silent (Prop. 1)")
b.legend(loc="upper left", title="lines: theory; hollow markers: simulation", title_fontsize=6.5)
save(fig, "e2_desalineacion")

err = max(abs(r["measured_excess"] - r["sample_ideal"]) for r in rows)
noise = max(abs(np.mean([r["measured_excess"] for r in rows if r["strength"] == x]) - np.mean([r["A"] for r in rows if r["strength"] == x])) for x in xs)
print(f"E2: H(P_S)={HP:.3f}; |medido - ideal de la muestra| max = {err:.4f} bits/camino (cota 4/K = {4/K:.3f}); "
      f"|media medida - A| max = {noise:.3f} (ruido de muestreo, 6 repeticiones); "
      f"deteccion max {max(r['detection_theory'] for r in rows_b):.2f}")
