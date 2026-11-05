# trajcode — knowledge graphs as codebooks for semantic communication

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23987514.svg)](https://doi.org/10.5281/zenodo.23987514)

Code, data and results for the paper

> R. Sandoval Aréchiga, J. R. Gómez Rodríguez, V. I. Rodríguez Abdalá, S. Ibarra Delgado and A. G. Castañeda Miranda,
> **"Knowledge Graphs as Codebooks for Semantic Communication: Rate, Memory, Misalignment, and Graceful Degradation,"**
> manuscript under review. A preprint of an earlier version is available in the
> [institutional repository of the Universidad Autónoma de Zacatecas](http://ricaxcan.uaz.edu.mx/jspui/retrieve/a8593222-a939-40ee-a956-22c209354ea6/preview.pdf).

In the paper, the knowledge shared by transmitter and receiver is a graph, and a meaning is conveyed by describing a
walk on it, one decision at a time. This repository implements the path codes (native, adapted and bits-back), the
misalignment between two graphs and the rules that decide which nodes to update, rate–distortion on concept
hierarchies, layered transmission over noisy channels with polar codes, and every experiment reported in the paper.
Every number in the paper is generated from the files in `results/` by these scripts, with fixed seeds.

## Install

Python 3.11 or later.

```bash
python3 -m pip install -e ".[test]"
```

## Reproduce

```bash
make test       # property and regression tests (pytest + hypothesis)
make data       # download WordNet 3.0 and verify its SHA-256 (needed by E4, E5 and E7)
make figures    # run E1–E7; writes results/ and figures/
make theory     # optional: compile the working notes in theory/ (needs a LaTeX installation)
```

`make figures` takes about 15 minutes on a laptop, dominated by E4 and E7. All CSV files in `results/` are
reproduced bit for bit, except the measured run times (`results/e7_complejidad.csv`, `results/e7_pendientes.csv`),
which depend on the machine.

## Experiments and where they appear in the paper

| Script | Paper | Outputs |
|---|---|---|
| `experiments/e1_memoria.py` | Sec. IV (rate and decoder memory), E1 | `results/e1_*.csv` |
| `experiments/e2_desalineacion.py` | Sec. V (misalignment, three kinds of edit), E2 | `results/e2_*.csv` |
| `experiments/e3_distorsion.py` | Sec. VI (geodesic rate–distortion, refinability), E3 | `results/e3_*.csv` |
| `experiments/fig_isit.py` | Fig. 3 (E1 polysemy, E3 gap) | `figures/fig_isit.pdf` |
| `experiments/e4_wordnet.py` | E4: WordNet noun hierarchy | `results/e4_*.csv`, `figures/e4_wordnet.pdf` |
| `experiments/e5_canal.py` | Sec. VII, E5: layered transmission over BI-AWGN, polar codes | `results/e5_*.csv`, `figures/e5_canal.pdf` |
| `experiments/e6_actualizacion.py` | Sec. V-A, E6: knowledge update on an RTLS site | `results/e6_*.csv`, `figures/e6_actualizacion.pdf` |
| `experiments/e7_referencias.py` | E7: reference schemes and measured complexity | `results/e7_*.csv` |
| `theory/check_t34.py` | Sec. VII numerical checks | `results/t34.json` |
| `theory/check_t09c.py` | Truncation count (Sec. I) | standard output |

## Layout

```
trajcode/      the library
  graphs.py        trajectory networks, generators, RTLS site
  measures.py      path measures, visit probabilities, divergences
  coding.py        native, adapted and bits-back path codes; exact rational arithmetic coder
  alignment.py     misalignment, edits, teach-or-talk rule
  kbupdate.py      knowledge update with known and unknown horizon
  distortion.py    geodesic distortion, quotient codes, Blahut–Arimoto
  progressive.py   class chains, expected distortion, exact allocation, cost of identifying classes
  channel.py       BEC/BSC/BI-AWGN, normal approximation, CRC, polar codes with SC decoding
  wordnet.py       WordNet 3.0 reader (standard library only)
  baselines.py     index, embedded and syntactic reference codes; linear-time Viterbi selection
experiments/   one script per experiment
theory/        working notes (Parts 0–6) with numerical checks
tests/         property-based and regression tests
results/       CSV and JSON files used in the paper
figures/       figures of the paper
```

## Data

WordNet 3.0 is not redistributed here: `make data` downloads the WordNet 3.0 package of the NLTK data repository and checks its SHA-256.
Its license is in `data/wordnet/LICENSE`.

## License

GPL-3.0-or-later (see `LICENSE`). WordNet keeps its own license.

## Citation

Archived on Zenodo: version 1.0.0, [10.5281/zenodo.23987514](https://doi.org/10.5281/zenodo.23987514); all versions, [10.5281/zenodo.23987513](https://doi.org/10.5281/zenodo.23987513). See `CITATION.cff`. Contact: Remberto Sandoval Aréchiga, Posgrado en Ingeniería para la Innovación Tecnológica, Universidad Autónoma de Zacatecas, rsandoval@uaz.edu.mx.
