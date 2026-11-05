"""Estilo común de figuras (constitution P7): escala de grises, una columna IEEE, identidad por trazo y marcador."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

COL_W = 3.5          # ancho de columna IEEE en pulgadas
INK = "#1a1a1a"      # texto y trazo principal
MID = "#6e6e6e"      # segunda serie
LIGHT = "#b3b3b3"    # nube de puntos / bandas
GRID = "#e6e6e6"

SERIES = [  # orden fijo: identidad por estilo de línea y marcador, no por color
    dict(color=INK, ls="-", marker="o"),
    dict(color=MID, ls="--", marker="s"),
    dict(color=INK, ls=":", marker="^"),
    dict(color=MID, ls="-.", marker="D"),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MID, "axes.linewidth": 0.6, "xtick.color": MID, "ytick.color": MID,
    "axes.labelcolor": INK, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "mathtext.fontset": "stix",   # glifos con Unicode correcto (ρ, Π) al extraer texto
})


def save(fig, name: str):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
