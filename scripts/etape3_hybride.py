"""
Étape 3 : recherche hybride (sens + mots-clés), filtre et dédoublonnage.

Ce script :
0. vérifie TA fonction fusion_rrf ;
1. montre ce que voit BM25 (la tokenisation) ;
2. compare, sur les 6 questions de l'étape 2, la recherche par le sens seule
   et la recherche hybride ;
3. montre l'effet du filtre par type de document.

Lancement, depuis la racine du projet (l'index doit déjà exister) :
    python scripts/etape3_hybride.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import IndexVectoriel
from src.recherche import RechercheHybride, fusion_rrf, tokeniser

DOSSIER_INDEX = Path(__file__).resolve().parent.parent / "data" / "index"


def proche(x: float, y: float) -> bool:
    return abs(x - y) < 1e-9


def verifier_ma_fonction() -> None:
    r = fusion_rrf([["a", "b"], ["b"]])
    assert set(r) == {"a", "b"}, "Chaque identifiant présent dans au moins un classement doit avoir un score"
    assert proche(r["a"], 1 / 61), "a est 1er (rang 1) dans un seul classement : 1/61. Le rang commence-t-il bien à 1 ?"
    assert proche(r["b"], 1 / 62 + 1 / 61), "b : 1/62 (2e) + 1/61 (1er), les gains s'additionnent"

    r = fusion_rrf([["x", "y", "z"]], k_rrf=0)
    assert proche(r["x"], 1.0) and proche(r["z"], 1 / 3), "k_rrf doit être utilisé : avec k_rrf=0, le 1er vaut 1/1"

    assert fusion_rrf([]) == {}, "Aucun classement : dictionnaire vide"
    assert fusion_rrf([[], []]) == {}, "Classements vides : dictionnaire vide"

    r = fusion_rrf([["a", "b", "c"], ["c", "b", "a"]])
    assert r["b"] < r["a"] and proche(r["a"], r["c"]), \
        "Classements inversés : a et c à égalité, b (2e partout) un peu en dessous"
    print("Ta fonction passe les 7 tests.\n")


def afficher_hybride(question: str, moteur: RechercheHybride, **options) -> None:
    print(f"Question : {question}" + (f"   {options}" if options else ""))
    print(f"  Mots vus par BM25 : {tokeniser(question)}")
    ids_sens, scores_sens = moteur.classement_sens(question)
    top_sens = moteur.par_id[ids_sens[0]]
    print(f"  Sens seul, n° 1   : {scores_sens[top_sens.id]:.3f} [{top_sens.source} > {top_sens.section}]")
    print("  Hybride :")
    for rang, r in enumerate(moteur.rechercher(question, k=3, **options), start=1):
        rs = r.rang_sens if r.rang_sens is not None else "-"
        rm = r.rang_mots if r.rang_mots is not None else "-"
        print(f"    {rang}. [{r.chunk.source} > {r.chunk.section}]")
        print(f"       rang sens : {rs:>2}  | rang mots : {rm:>2}  | cosinus : {r.score_sens:.3f}")
    print()


def main() -> None:
    verifier_ma_fonction()

    index = IndexVectoriel.charger(DOSSIER_INDEX)
    moteur = RechercheHybride(index)

    print("=" * 70)
    print("1. Ce que voit BM25")
    print("=" * 70)
    exemple = index.chunks[0]
    print(f"Passage [{exemple.id}] :\n{exemple.texte}\n")
    print(f"Mots retenus : {tokeniser(exemple.texte)}\n")

    print("=" * 70)
    print("2. Sens seul vs hybride (mêmes questions qu'à l'étape 2)")
    print("=" * 70)
    for q in [
        "Quelle est la gratification du stage chez Nexalia ?",
        "Quels stages demandent du ML ?",
        "Quels stages demandent du Machine Learning ?",
        "Comment répondre à une question comportementale en entretien ?",
        "Combien de mots doit faire une lettre de motivation ?",
        "Kubernetes",
    ]:
        afficher_hybride(q, moteur)

    print("=" * 70)
    print("3. Avec filtre sur les offres de stage")
    print("=" * 70)
    afficher_hybride("Quels stages demandent du Machine Learning ?", moteur, type_document="offre")


if __name__ == "__main__":
    main()