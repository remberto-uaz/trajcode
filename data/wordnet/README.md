# WordNet 3.0 (sustantivos)

- Fuente: `wordnet.zip` de `nltk_data` (WordNet 3.0, Princeton University), descargado por `make data`
  (`scripts/get_wordnet.py`). El zip no se versiona; su SHA-256 está fijado en el script:
  `cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59`.
- Licencia: WordNet 3.0 License (copia en `LICENSE`): uso, copia y distribución permitidos con el aviso
  "WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved."
- Referencia citada en el artículo: G. A. Miller, "WordNet: A Lexical Database for English", Commun. ACM 38(11), 1995,
  doi:10.1145/219717.219748 (verificado en Crossref).
- Uso (ADR-T03): `data.noun` (punteros de hiperónimo `@` y de instancia `@i`) e `index.sense` (conteos de etiquetado
  de SemCor por sentido, para P_M). Lectura con la biblioteca estándar en `trajcode/wordnet.py`.
