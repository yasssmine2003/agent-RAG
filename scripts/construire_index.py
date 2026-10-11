"""
Indexation : charge les documents, les découpe, les vectorise et sauvegarde l'index.

À relancer chaque fois que les documents de data/documents/ changent.

Lancement, depuis la racine du projet :
    python scripts/construire_index.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chargement import charger_documents
from src.decoupage import decouper_document
from src.index import IndexVectoriel

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_DOCUMENTS = RACINE / "data" / "documents"
DOSSIER_INDEX = RACINE / "data" / "index"


def main() -> None:
    debut = time.perf_counter()

    documents = charger_documents(DOSSIER_DOCUMENTS)
    chunks = [c for doc in documents for c in decouper_document(doc)]
    print(f"{len(documents)} documents -> {len(chunks)} passages")

    index = IndexVectoriel()
    index.ajouter(chunks)
    index.sauvegarder(DOSSIER_INDEX)

    duree = time.perf_counter() - debut
    print(f"Matrice de vecteurs : {index.vecteurs.shape}")
    print(f"Index sauvegardé dans {DOSSIER_INDEX} en {duree:.1f} s")


if __name__ == "__main__":
    main()