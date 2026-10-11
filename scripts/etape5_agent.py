"""
Étape 5 : le RAG agentique.

Ce script :
0. vérifie TON outil rechercher avec un faux moteur (sans LLM) ;
1. pose 7 questions à l'agent et affiche, pour chacune, la trace de ses recherches
   (ce qu'il a décidé de chercher), sa réponse, l'ancrage, les tours et les tokens.

Lancement, depuis la racine du projet :
    python scripts/etape5_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import executer_agent
from src.decoupage import Chunk
from src.index import IndexVectoriel
from src.outils import creer_outils
from src.recherche import RechercheHybride, ResultatHybride

DOSSIER_INDEX = Path(__file__).resolve().parent.parent / "data" / "index"


class FauxMoteur:
    """Remplace le vrai moteur pour tester l'outil sans index ni modèle."""

    def __init__(self):
        self.appels = []

    def rechercher(self, requete, k=3, type_document=None):
        self.appels.append((requete, type_document))
        if requete == "introuvable":
            return []
        chunk = Chunk(id="a.md#0", source="a.md", section="Intro", texte="Texte A", nb_mots=2)
        return [ResultatHybride(chunk=chunk, score_rrf=0.0, rang_sens=1, rang_mots=None, score_sens=0.5)]


def verifier_mon_outil() -> None:
    moteur = FauxMoteur()
    outils, journal = creer_outils(moteur)
    rechercher = outils["rechercher"]

    texte = rechercher("python", "offre")
    assert moteur.appels[-1] == ("python", "offre"), \
        "L'outil doit transmettre la requête ET le type_document au moteur"
    assert texte == '<passage source="a.md" section="Intro">\nTexte A\n</passage>', \
        "Des passages trouvés : renvoyer le contexte mis en forme par formater_contexte"

    rechercher("sql")
    assert moteur.appels[-1] == ("sql", None), "Sans filtre, type_document vaut None"

    assert rechercher("introuvable") == "Aucun passage trouvé pour cette requête.", \
        "Aucun passage : renvoyer exactement « Aucun passage trouvé pour cette requête. »"

    assert len(journal) == 3, "Chaque appel doit être ajouté au journal"
    assert journal[0].sources == ["a.md"] and journal[2].sources == [], "Le journal note les sources renvoyées"
    print("Ton outil passe les 5 tests.\n")


def afficher(r) -> None:
    print(f"Question : {r.question}")
    if not r.recherches:
        print("  Aucune recherche : l'agent a répondu directement.")
    for i, rech in enumerate(r.recherches, start=1):
        filtre = f"  (filtre : {rech.type_document})" if rech.type_document else ""
        print(f"  Recherche {i} : « {rech.requete} »{filtre}")
        print(f"      -> {sorted(set(rech.sources)) if rech.sources else 'rien'}")
    print(f"  Réponse : {r.reponse}")
    print(f"  Sources citées : {r.sources_citees}")
    if r.sources_inventees:
        print(f"  ALERTE, sources inventées : {r.sources_inventees}")
    print(f"  {r.nb_tours} tours | {r.duree_s:.1f} s | tokens entrée {r.tokens_entree} | sortie {r.tokens_sortie}")
    print()


def main() -> None:
    verifier_mon_outil()

    index = IndexVectoriel.charger(DOSSIER_INDEX)
    moteur = RechercheHybride(index)
    fichiers_connus = {c.source for c in index.chunks}

    print("=" * 70)
    print("Questions à l'agent")
    print("=" * 70)
    for q in [
        "Merci beaucoup !",
        "Combien de mots doit faire une lettre de motivation ?",
        "Comment répondre à une question comportementale en entretien ?",
        "Quels stages demandent du Machine Learning ?",
        "Quelle est la gratification du stage chez Nexalia ?",
        "Quel est le salaire moyen d'un data scientist en France ?",
        "Quel stage paie le mieux, et quelle est sa date limite ?",
    ]:
        afficher(executer_agent(q, moteur, fichiers_connus))


if __name__ == "__main__":
    main()