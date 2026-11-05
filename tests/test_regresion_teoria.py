"""Regresión: los números citados en theory/*.tex deben salir de los scripts (constitution P4.1, P4.4)."""
import math, pathlib, runpy
THEORY = pathlib.Path(__file__).resolve().parents[1] / "theory"
L2 = math.log2

def _H(ps):
    return -sum(p * L2(p) for p in ps if p > 0)

def test_ejemplo_semana_numeros_del_texto():
    ns = runpy.run_path(str(THEORY / "ejemplo_semana.py"), run_name="not_main")
    out = ns["out"]
    assert round(out["H(M)"], 3) == 1.485          # Partes 0, 1, 4 (corregido en T09b)
    assert round(out["H(Pi)"], 3) == 2.456
    assert round(out["H(Pi|M)"], 3) == 0.971
    assert round(out["L0"], 3) == 2.222
    assert round(out["gap"], 3) == 0.737           # precio del ruteo nativo
    assert round(out["A1"], 3) == 0.060            # Parte 2, edición de ruteo
    assert round(out["A1 D_s"], 3) == 0.036 and round(out["A1 cond"], 3) == 0.024
    assert round(out["R cociente {lun,mar}|{mie}, D<=1"], 3) == 0.722

def test_parte4_cadena_weaver():
    HM = _H([0.5, 0.3, 0.2]); Ha = _H([0.5, 0.5])
    assert round(HM - Ha, 3) == 0.485

def test_pinsker_ejemplo():
    tv = abs(.5 - 1/3) + abs(.3 - 1/3) + abs(.2 - 1/3)
    assert round(0.6 * tv**2 / (2 * math.log(2)), 3) == 0.048

def test_t09c_truncamiento_y_contraste_huffman():
    import subprocess, sys
    out = subprocess.run([sys.executable, str(THEORY / "check_t09c.py")], capture_output=True, text=True, cwd=THEORY).stdout
    assert "violaciones de costo Shannon <= H(Q_l)+l: 0" in out
    assert "313 de 500" in out      # citado en Parte 3 (Obs. 2) y en paper/borradores/intro-motivacion.md


def test_parte3_coincide_con_experimento_e3():
    """Los números de theory/03-distorsion.tex deben venir de results/e3_*.csv (constitution P4.1)."""
    import csv, pytest
    res = THEORY.parent / "results"
    if not (res / "e3_refinabilidad.csv").exists():
        pytest.skip("correr `make figures` primero")
    tex = (THEORY / "03-distorsion.tex").read_text()
    for r in csv.DictReader(open(res / "e3_refinabilidad.csv")):
        assert f"${r['not_refinable']}/{r['trials']}$" in tex, r["case"]
    for r in csv.DictReader(open(res / "e3_semana.csv")):
        assert f"${float(r['R_BA']):.3f}$" in tex and f"${float(r['gap']):.3f}$" in tex


def test_parte5_programacion_dinamica_y_costo_de_clases():
    """Parte 5 (T34): la asignación por programación dinámica es exacta y Δ_j > 0 en niveles no triviales."""
    import runpy, pathlib
    g = runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1] / "theory/check_t34.py"), run_name="lib")
    g["check_A2"](cases=8)
    nontriv, gmin, _ = g["check_A3"](cases=6)
    assert nontriv > 0 and gmin > 0
