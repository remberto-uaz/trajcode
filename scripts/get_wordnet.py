"""Descarga WordNet 3.0 (nltk_data) a data/wordnet/wordnet.zip y verifica su SHA-256 (ADR-T03, constitution E2)."""
import hashlib, pathlib, shutil, subprocess, sys

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
SHA256 = "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"
dst = pathlib.Path(__file__).resolve().parents[1] / "data/wordnet/wordnet.zip"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if dst.exists() and sha(dst) == SHA256:
    print(f"ok {dst.name} (ya presente, SHA-256 verificado)")
    sys.exit(0)
dst.parent.mkdir(parents=True, exist_ok=True)
tmp = dst.with_suffix(".part")
subprocess.run(["curl", "-sSL", "--fail", "-o", str(tmp), URL], check=True)
if sha(tmp) != SHA256:
    tmp.unlink()
    sys.exit("SHA-256 distinto del fijado: archivo rechazado")
shutil.move(tmp, dst)
print(f"ok {dst.name} descargado, SHA-256 verificado")
