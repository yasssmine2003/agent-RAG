"""
Étape 4 : le RAG classique (pipeline fixe, sans agent).

Ce script :
0. vérifie TA fonction formater_contexte (sans appeler le LLM) ;
1. montre le message exact envoyé au LLM pour une question ;
2. pose 6 questions et affiche réponse, sources, ancrage, temps et tokens.

Prérequis : l'index existe (construire_index.py) et le fichier .env contient GROQ_API_KEY.

Lancement, depuis la racine du projet :
    python scripts/etape4_rag_classique.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decoupage import Chunk
from src.index import IndexVectoriel
from src.rag_classique import construire_message, formater_contexte, repondre
from src.recherche import RechercheHybride, ResultatHybride

DOSSIER_INDEX = Path(__file__).resolve().parent.parent / "data" / "index"


def faux_resultat(source: str, section: str, texte: str) -> ResultatHybride:
    chunk = Chunk(id=f"{source}#0", source=source, section=section, texte=texte, nb_mots=1)
    return ResultatHybride(chunk=chunk, score_rrf=0.0, rang_sens=1, rang_mots=None, score_sens=0.5)


def verifier_ma_fonction() -> None:
    r1 = faux_resultat("a.md", "Intro", "Texte A")
    r2 = faux_resultat("b.md", "Profil", "Ligne 1\nLigne 2")

    attendu_un = '<passage source="a.md" section="Intro">\nTexte A\n</passage>'
    assert formater_contexte([r1]) == attendu_un, \
        f"Un passage : attendu exactement\n{attendu_un}\nobtenu\n{formater_contexte([r1])}"

    attendu_deux = attendu_un + "\n\n" + '<passage source="b.md" section="Profil">\nLigne 1\nLigne 2\n</passage>'
    assert formater_contexte([r1, r2]) == attendu_deux, "Deux passages : blocs séparés par une ligne vide (\\n\\n)"

    assert formater_contexte([]) == "", "Aucun passage : chaîne vide"
    print("Ta fonction passe les 3 tests.\n")


def afficher(resultat) -> None:
    rep = resultat.reponse
    print(f"Question : {resultat.question}")
    print("  Passages envoyés au LLM :")
    for p in resultat.passages:
        print(f"    - [{p.chunk.source} > {p.chunk.section}] cosinus {p.score_sens:.3f}")
    print(f"  Trouvé  : {rep.trouve}")
    print(f"  Réponse : {rep.reponse}")
    print(f"  Sources citées : {rep.sources}")
    if resultat.sources_inventees:
        print(f"  ALERTE, sources inventées : {resultat.sources_inventees}")
    print(f"  {resultat.duree_s:.1f} s | tokens entrée {resultat.tokens_entree} | sortie {resultat.tokens_sortie}")
    print()


def main() -> None:
    verifier_ma_fonction()

    moteur = RechercheHybride(IndexVectoriel.charger(DOSSIER_INDEX))

    print("=" * 70)
    print("1. Le message exact envoyé au LLM")
    print("=" * 70)
    question = "Combien de mots doit faire une lettre de motivation ?"
    print(construire_message(question, formater_contexte(moteur.rechercher(question, k=3))))
    print()

    print("=" * 70)
    print("2. Questions")
    print("=" * 70)
    for q in [
        "Combien de mots doit faire une lettre de motivation ?",
        "Comment répondre à une question comportementale en entretien ?",
        "Quels stages demandent du Machine Learning ?",
        "Quelle est la gratification du stage chez Nexalia ?",
        "Nexalia stipend",
        "Quel est le salaire moyen d'un data scientist en France ?",
    ]:
        afficher(repondre(q, moteur))


if __name__ == "__main__":
    main()