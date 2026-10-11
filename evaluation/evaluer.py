"""
Étape 6 : évaluer et comparer le RAG classique et le RAG agentique.

Le principe est celui d'un examen :
1. les questions et leurs réponses attendues sont dans evaluation/questions.json ;
2. chaque question est posée aux deux systèmes ;
3. chaque réponse est corrigée automatiquement : juste, faux négatif ou hallucination ;
4. un tableau compare les deux systèmes (qualité ET coût) ;
5. les résultats sont sauvegardés dans evaluation/resultats.json (pour comparer
   avec les prochaines versions : c'est le test de non-régression).

Lancement, depuis la racine du projet :
    python evaluation/evaluer.py
Durée : quelques minutes (environ 30 appels au LLM, avec une pause entre les questions
pour respecter les limites de débit de l'offre gratuite de Groq).
"""

import json
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from src.agent import executer_agent
from src.index import IndexVectoriel
from src.rag_classique import repondre
from src.recherche import RechercheHybride

RACINE = Path(__file__).resolve().parent.parent
FICHIER_QUESTIONS = RACINE / "evaluation" / "questions.json"
FICHIER_RESULTATS = RACINE / "evaluation" / "resultats.json"
PAUSE_S = 3  # pause entre deux questions, pour ne pas dépasser les limites de Groq


# ============================================================================
# 1. Le format d'une question d'examen (validé par Pydantic, comme au projet 1)
# ============================================================================
class QuestionEval(BaseModel):
    id: str
    famille: Literal["simple", "reformulation", "comparaison", "hors_corpus", "salutation"]
    question: str
    mots_attendus: list[str] = []      # chaque élément doit apparaître ; "a|b" = a OU b
    source_attendue: str | None = None  # fichier qui contient la réponse
    introuvable: bool = False           # True si la bonne réponse est « pas trouvé »


class Note(BaseModel):
    """Ce qu'on retient d'une question, pour un système."""
    id: str
    famille: str
    verdict: Literal["juste", "faux_negatif", "hallucination", "sans_reponse", "erreur"]
    bon_document_vu: bool | None    # None si aucune source attendue
    bonne_source_citee: bool | None
    nb_recherches: int
    tokens: int                     # entrée + sortie
    duree_s: float
    reponse: str


# ============================================================================
# 2. La correction
# ============================================================================
def normaliser(texte: str) -> str:
    """Minuscules, sans aucun espace : « 1 000 TND » et « 1000 tnd » deviennent identiques."""
    for espace in (" ", "\u00a0", "\u202f", "\n"):
        texte = texte.replace(espace, "")
    return texte.lower()


def dit_pas_trouve(reponse: str) -> bool:
    return "pastrouvé" in normaliser(reponse) or "pastrouve" in normaliser(reponse)


def contient_mots_attendus(reponse: str, mots_attendus: list[str]) -> bool:
    """Chaque mot attendu doit apparaître ; pour "a|b", une des alternatives suffit."""
    r = normaliser(reponse)
    return all(any(normaliser(alt) in r for alt in mot.split("|")) for mot in mots_attendus)


def corriger(q: QuestionEval, reponse: str) -> str:
    """Range une réponse dans une case : juste, faux_negatif, hallucination ou sans_reponse."""
    if "limite de tours atteinte" in reponse.lower():  # l'agent n'a jamais conclu :
        return "sans_reponse"                          # un échec, mais pas une invention
    if q.introuvable:                       # la bonne réponse est « pas trouvé »
        return "juste" if dit_pas_trouve(reponse) else "hallucination"
    if q.famille == "salutation":           # il suffit de ne pas dire « pas trouvé »
        return "faux_negatif" if dit_pas_trouve(reponse) else "juste"
    if dit_pas_trouve(reponse):             # l'info existe, mais le système ne l'a pas trouvée
        return "faux_negatif"
    if contient_mots_attendus(reponse, q.mots_attendus):
        return "juste"
    return "hallucination"                  # il a répondu, mais sans les éléments attendus


def verifier_correcteur() -> None:
    """On teste le correcteur avant de lui faire confiance (sans LLM)."""
    nexalia = QuestionEval(id="t", famille="reformulation", question="?", mots_attendus=["1000"])
    assert corriger(nexalia, "La gratification est de 1 000 TND.") == "juste"
    assert corriger(nexalia, "Elle est de 800 TND.") == "hallucination"
    assert corriger(nexalia, "Je n'ai pas trouvé cette information.") == "faux_negatif"
    hors = QuestionEval(id="t", famille="hors_corpus", question="?", introuvable=True)
    assert corriger(hors, "Je n'ai pas trouvé cette information dans les documents.") == "juste"
    assert corriger(hors, "Environ 4 500 euros par mois.") == "hallucination"
    star = QuestionEval(id="t", famille="simple", question="?", mots_attendus=["tâche|task"])
    assert corriger(star, "Situation, Task, Action, Result") == "juste"
    assert corriger(nexalia, "Limite de tours atteinte : l'agent n'a pas fourni de réponse finale.") == "sans_reponse"
    print("Le correcteur passe ses 7 tests.\n")


# ============================================================================
# 3. Faire passer l'examen aux deux systèmes
# ============================================================================
def passer_classique(q: QuestionEval, moteur) -> Note:
    r = repondre(q.question, moteur)
    vues = {p.chunk.source for p in r.passages}
    return Note(
        id=q.id, famille=q.famille, verdict=corriger(q, r.reponse.reponse),
        bon_document_vu=(q.source_attendue in vues) if q.source_attendue else None,
        bonne_source_citee=(q.source_attendue in r.reponse.sources) if q.source_attendue else None,
        nb_recherches=1, tokens=r.tokens_entree + r.tokens_sortie,
        duree_s=r.duree_s, reponse=r.reponse.reponse,
    )


def passer_agentique(q: QuestionEval, moteur, fichiers_connus) -> Note:
    r = executer_agent(q.question, moteur, fichiers_connus)
    vues = {s for rech in r.recherches for s in rech.sources}
    return Note(
        id=q.id, famille=q.famille, verdict=corriger(q, r.reponse),
        bon_document_vu=(q.source_attendue in vues) if q.source_attendue else None,
        bonne_source_citee=(q.source_attendue in r.sources_citees) if q.source_attendue else None,
        nb_recherches=len(r.recherches), tokens=r.tokens_entree + r.tokens_sortie,
        duree_s=r.duree_s, reponse=r.reponse,
    )


def passer_sans_planter(fonction, q: QuestionEval, *args) -> Note:
    """Une erreur (limite Groq, réseau) ne doit pas arrêter tout l'examen."""
    try:
        return fonction(q, *args)
    except Exception as erreur:
        return Note(id=q.id, famille=q.famille, verdict="erreur", bon_document_vu=None,
                    bonne_source_citee=None, nb_recherches=0, tokens=0, duree_s=0.0,
                    reponse=f"ERREUR : {erreur}")


# ============================================================================
# 4. L'affichage des résultats
# ============================================================================
SYMBOLES = {"juste": "OK", "faux_negatif": "FN", "hallucination": "HALLU", "sans_reponse": "LIMITE", "erreur": "ERR"}


def taux(valeurs: list[bool | None]) -> str:
    utiles = [v for v in valeurs if v is not None]
    return f"{sum(utiles)}/{len(utiles)}" if utiles else "-"


def afficher_bilan(notes: dict[str, list[Note]]) -> None:
    print("=" * 78)
    print("Détail par question  (OK = juste, FN = faux négatif, HALLU = hallucination, LIMITE = sans réponse)")
    print("=" * 78)
    print(f"{'id':<5}{'famille':<15}{'classique':<12}{'agentique':<12}{'recherches (agent)':>20}")
    for c, a in zip(notes["classique"], notes["agentique"]):
        print(f"{c.id:<5}{c.famille:<15}{SYMBOLES[c.verdict]:<12}{SYMBOLES[a.verdict]:<12}{a.nb_recherches:>20}")

    print("\n" + "=" * 78)
    print("Bilan")
    print("=" * 78)
    print(f"{'':<24}{'classique':>14}{'agentique':>14}")
    lignes = [
        ("Réponses justes", lambda n: f"{sum(x.verdict == 'juste' for x in n)}/{len(n)}"),
        ("Faux négatifs", lambda n: str(sum(x.verdict == "faux_negatif" for x in n))),
        ("Hallucinations", lambda n: str(sum(x.verdict == "hallucination" for x in n))),
        ("Sans réponse (limite)", lambda n: str(sum(x.verdict == "sans_reponse" for x in n))),
        ("Erreurs techniques", lambda n: str(sum(x.verdict == "erreur" for x in n))),
        ("Bon document vu", lambda n: taux([x.bon_document_vu for x in n])),
        ("Bonne source citée", lambda n: taux([x.bonne_source_citee for x in n])),
        ("Recherches (moyenne)", lambda n: f"{sum(x.nb_recherches for x in n) / len(n):.1f}"),
        ("Tokens (moyenne)", lambda n: f"{sum(x.tokens for x in n) / len(n):.0f}"),
        ("Temps (moyenne, s)", lambda n: f"{sum(x.duree_s for x in n) / len(n):.1f}"),
    ]
    for nom, calcul in lignes:
        print(f"{nom:<24}{calcul(notes['classique']):>14}{calcul(notes['agentique']):>14}")


def main() -> None:
    verifier_correcteur()

    donnees = json.loads(FICHIER_QUESTIONS.read_text(encoding="utf-8"))
    questions = [QuestionEval(**d) for d in donnees]   # Pydantic signale toute faute de format
    assert len(questions) >= 15, f"L'examen doit contenir au moins 15 questions ({len(questions)} pour l'instant)"
    assert len({q.id for q in questions}) == len(questions), "Deux questions ont le même id"
    print(f"{len(questions)} questions chargées.\n")

    index = IndexVectoriel.charger(RACINE / "data" / "index")
    moteur = RechercheHybride(index)
    fichiers_connus = {c.source for c in index.chunks}

    notes = {"classique": [], "agentique": []}
    for q in questions:
        print(f"[{q.id}] {q.question}")
        notes["classique"].append(passer_sans_planter(passer_classique, q, moteur))
        time.sleep(PAUSE_S)
        notes["agentique"].append(passer_sans_planter(passer_agentique, q, moteur, fichiers_connus))
        time.sleep(PAUSE_S)
        print(f"      classique : {notes['classique'][-1].verdict:<14} agentique : {notes['agentique'][-1].verdict}")
    print()

    afficher_bilan(notes)

    FICHIER_RESULTATS.write_text(
        json.dumps({s: [n.model_dump() for n in ns] for s, ns in notes.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDétail complet (avec les réponses) sauvegardé dans {FICHIER_RESULTATS}")


if __name__ == "__main__":
    main()