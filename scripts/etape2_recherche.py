"""
Étape 2 : interroger l'index vectoriel.

Ce script :
0. vérifie TA méthode rechercher_par_vecteur sur un mini-index en 2 dimensions ;
1. recharge l'index construit par scripts/construire_index.py ;
2. pose plusieurs questions et affiche les 3 meilleurs passages pour chacune ;
3. mesure le temps d'une recherche.

Lancement, depuis la racine du projet (après construire_index.py) :
    python scripts/etape2_recherche.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.decoupage import Chunk
from src.index import IndexVectoriel

DOSSIER_INDEX = Path(__file__).resolve().parent.parent / "data" / "index"


def faux_chunk(identifiant: str) -> Chunk:
    return Chunk(id=identifiant, source="test.md", section="test", texte=identifiant, nb_mots=1)


def verifier_ma_fonction() -> None:
    """Mini-index de 3 passages en 2D, avec des vecteurs déjà normalisés (longueur 1)."""
    index = IndexVectoriel()
    index.chunks = [faux_chunk("droite"), faux_chunk("haut"), faux_chunk("diagonale")]
    index.vecteurs = np.array(
        [[1.0, 0.0],    # "droite"
         [0.0, 1.0],    # "haut"
         [0.6, 0.8]],   # "diagonale" (0.6² + 0.8² = 1 : bien normalisé)
        dtype=np.float32,
    )

    r = index.rechercher_par_vecteur(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert [x.chunk.id for x in r] == ["droite", "diagonale"], \
        "Requête vers la droite : attendu droite (1.0) puis diagonale (0.6)"
    assert abs(r[0].score - 1.0) < 1e-6 and abs(r[1].score - 0.6) < 1e-6, "Scores attendus : 1.0 et 0.6"
    assert isinstance(r[0].score, float), "Le score doit être un float Python"

    r = index.rechercher_par_vecteur(np.array([0.0, 1.0], dtype=np.float32), k=10)
    assert [x.chunk.id for x in r] == ["haut", "diagonale", "droite"], \
        "k plus grand que l'index : renvoyer tous les passages, du meilleur au moins bon"

    assert IndexVectoriel().rechercher_par_vecteur(np.array([1.0, 0.0]), k=3) == [], \
        "Index vide : liste vide"
    print("Ta méthode passe les 4 tests.\n")


def afficher(question: str, index: IndexVectoriel) -> None:
    print(f"Question : {question}")
    for rang, res in enumerate(index.rechercher(question, k=3), start=1):
        extrait = res.chunk.texte.split("\n", 1)[-1][:70]  # le passage sans le préfixe, tronqué
        print(f"  {rang}. {res.score:.3f}  [{res.chunk.source} > {res.chunk.section}]")
        print(f"            {extrait}...")
    print()


def main() -> None:
    verifier_ma_fonction()

    print("=" * 70)
    print("1. Chargement de l'index")
    print("=" * 70)
    index = IndexVectoriel.charger(DOSSIER_INDEX)
    print(f"{len(index)} passages, matrice {index.vecteurs.shape}\n")

    print("=" * 70)
    print("2. Recherches")
    print("=" * 70)
    questions = [
        "Quelle est la gratification du stage chez Nexalia ?",
        "Quels stages demandent du ML ?",
        "Quels stages demandent du Machine Learning ?",
        "Comment répondre à une question comportementale en entretien ?",
        "Combien de mots doit faire une lettre de motivation ?",
        "Kubernetes",
    ]
    for q in questions:
        afficher(q, index)

    print("=" * 70)
    print("3. Temps de recherche")
    print("=" * 70)
    debut = time.perf_counter()
    for _ in range(20):
        index.rechercher("stage en data science", k=3)
    duree_ms = (time.perf_counter() - debut) / 20 * 1000
    print(f"Une recherche (vectoriser la question + {len(index)} produits scalaires) : {duree_ms:.1f} ms")


if __name__ == "__main__":
    main()