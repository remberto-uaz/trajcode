"""Prueba de mutación de trajcode.channel y trajcode.progressive (T35). Uso: python scripts/mutacion_t35.py

Cada mutante cambia una línea, corre las pruebas y restaura el archivo. Mutante equivalente excluido: Δ_h con
rd_at(0) en lugar de 0 (R(0) = H(M) hace Δ_h = 0 de todas formas)."""
import pathlib, subprocess, shutil
root = pathlib.Path(__file__).resolve().parents[1]
muts = [
 ("trajcode/channel.py", "np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))", "np.sign(a) * np.sign(b) * np.maximum(np.abs(a), np.abs(b))", "boxplus usa max"),
 ("trajcode/channel.py", "b + (1 - 2 * v1.astype(float)) * a", "b + (2 * v1.astype(float) - 1) * a", "signo en g"),
 ("trajcode/channel.py", "    return np.concatenate([v1 ^ v2, v2], axis=1)\n\n\ndef _boxplus", "    return np.concatenate([v1, v2], axis=1)\n\n\ndef _boxplus", "transformada sin XOR"),
 ("trajcode/channel.py", "return rec(n // 2, 2 * z - z * z) + rec(n // 2, z * z)", "return rec(n // 2, z * z) + rec(n // 2, 2 * z - z * z)", "orden de canales sintéticos BEC"),
 ("trajcode/channel.py", "reg[lead, i:i + c + 1] ^= poly", "reg[~lead, i:i + c + 1] ^= poly", "CRC divide mal"),
 ("trajcode/channel.py", "third = 0.0 if isinstance(ch, BEC) else 0.5 * math.log2(n)", "third = 0.0 if isinstance(ch, BEC) else -0.5 * math.log2(n)", "signo del término logarítmico"),
 ("trajcode/channel.py", "bad += int(np.sum(wrong | ~ok))", "bad += int(np.sum(~ok))", "BLER ignora errores no detectados"),
 ("trajcode/progressive.py", "        ok *= 1 - e", "        ok *= e", "Teorema 1: producto de fallos"),
 ("trajcode/progressive.py", "E[j][nj] * dbar[j] + (1 - E[j][nj]) * V[j + 1][r - nj]", "E[j][nj] * dbar[j + 1] + (1 - E[j][nj]) * V[j + 1][r - nj]", "DP con d̄ desplazada"),
 ("trajcode/progressive.py", "b = int(np.argmin(costs))", "b = int(np.argmax(costs))", "peor representante"),
]
tests = ["tests/test_channel.py", "tests/test_progressive.py"]
killed = 0
for f, a, b, name in muts:
    p = root / f; src = p.read_text()
    assert src.count(a) == 1, name
    p.write_text(src.replace(a, b))
    try:
        r = subprocess.run(["python3", "-m", "pytest", "-q", "-x", *tests], cwd=root, capture_output=True, text=True, timeout=600)
        dead = r.returncode != 0
    except subprocess.TimeoutExpired:
        dead = True
    finally:
        p.write_text(src)
    killed += dead
    print(("detectado " if dead else "SOBREVIVE ") + name)
print(f"{killed} de {len(muts)} mutantes detectados")
