"""
Recherche hybride : combine la recherche par le sens (embeddings) et par mots-clés (BM25).

Pourquoi deux méthodes ?
- les embeddings comprennent les synonymes et les paraphrases, mais ratent les noms
  propres et les termes rares (Nexalia, Kubernetes) ;
- BM25 retrouve toujours un mot exact, mais ne comprend aucun synonyme.

Comment les combiner ? Leurs scores ont des échelles incomparables (cosinus entre -1 et 1,
BM25 entre 0 et plus de 10). On fusionne donc les RANGS, pas les scores : c'est la
Reciprocal Rank Fusion (RRF).
"""

import re

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from src.decoupage import Chunk
from src.index import IndexVectoriel

# Mots trop fréquents pour aider BM25 : on les retire de la question et des passages.
MOTS_VIDES = {
    # français
    "le", "la", "les", "l", "un", "une", "des", "de", "du", "d", "et", "ou", "en", "à", "au",
    "aux", "pour", "par", "sur", "dans", "avec", "sans", "est", "sont", "ce", "cette", "ces",
    "qui", "que", "qu", "quoi", "quel", "quelle", "quels", "quelles", "comment", "combien",
    "chez", "doit", "faire", "il", "elle", "on", "vous", "votre", "vos", "nous", "je", "j",
    "mon", "ma", "mes", "son", "sa", "ses", "leur", "leurs", "ne", "pas", "plus", "se", "s",
    # anglais
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "with", "is", "are",
    "be", "you", "your", "what", "which", "how",
}


def tokeniser(texte: str) -> list[str]:
    """Découpe un texte en mots en minuscules, sans ponctuation ni mots vides.

    \\w+ capture les suites de lettres, chiffres et _ (accents compris en Python).
    Limite assumée : pas de racinisation, donc « stage » et « stages » sont deux mots différents.
    """
    return [mot for mot in re.findall(r"\w+", texte.lower()) if mot not in MOTS_VIDES]


# ============================================================================
# À TOI DE JOUER : la fusion par rangs
# ============================================================================
def fusion_rrf(classements: list[list[str]], k_rrf: int = 60) -> dict[str, float]:
    """Fusionne plusieurs classements d'identifiants en un score par identifiant.

    Chaque classement est une liste d'identifiants, du meilleur au moins bon.
    Pour chaque classement, l'identifiant au rang r (r commence à 1, pas à 0 !)
    gagne 1 / (k_rrf + r). Les gains s'additionnent sur tous les classements.

    Exemple avec k_rrf=60 et classements = [["a", "b"], ["b"]] :
        a : 1/61                (1er dans le premier classement, absent du second)
        b : 1/62 + 1/61         (2e dans le premier, 1er dans le second)
        -> {"a": 0.01639..., "b": 0.03252...}

    Renvoie un dictionnaire {identifiant: score_rrf}. Aucun classement -> {}.
    """
    # Remplace la ligne ci-dessous par ton code (5 lignes environ)
    scores = {}
    for classement in classements:
        for rang, identifiant in enumerate(classement, start=1):
            scores[identifiant] =  scores.get(identifiant, 0.0)+ 1 / (k_rrf + rang)
    return scores

# ============================================================================
# Fourni : le moteur de recherche hybride
# ============================================================================
class ResultatHybride(BaseModel):
    chunk: Chunk
    score_rrf: float           # score de fusion, sert uniquement à classer
    rang_sens: int | None      # rang dans la recherche par embeddings (None si absent)
    rang_mots: int | None      # rang dans la recherche BM25 (None si aucun mot commun)
    score_sens: float          # cosinus : utile comme signal de confiance


class RechercheHybride:
    def __init__(self, index: IndexVectoriel) -> None:
        self.index = index
        self.par_id = {c.id: c for c in index.chunks}
        # BM25 a besoin de tous les passages découpés en mots, dans l'ordre de l'index
        self.bm25 = BM25Okapi([tokeniser(c.texte) for c in index.chunks])

    def classement_sens(self, requete: str) -> tuple[list[str], dict[str, float]]:
        """Tous les passages, triés par cosinus décroissant."""
        resultats = self.index.rechercher(requete, k=len(self.index))
        return [r.chunk.id for r in resultats], {r.chunk.id: r.score for r in resultats}

    def classement_mots(self, requete: str) -> list[str]:
        """Passages contenant au moins un mot de la question, triés par score BM25."""
        scores = self.bm25.get_scores(tokeniser(requete))
        paires = [(s, c.id) for s, c in zip(scores, self.index.chunks) if s > 0]
        paires.sort(reverse=True)
        return [identifiant for _, identifiant in paires]

    def rechercher(
        self,
        requete: str,
        k: int = 3,
        type_document: str | None = None,
        dedoublonner: bool = True,
    ) -> list[ResultatHybride]:
        """Recherche hybride.

        Args:
            type_document: "offre", "guide" ou "referentiel" pour filtrer selon le
                préfixe du nom de fichier ; None pour tout le corpus.
            dedoublonner: ne garder qu'un passage par (fichier, section).
        """
        ids_sens, scores_sens = self.classement_sens(requete)
        ids_mots = self.classement_mots(requete)
        scores_rrf = fusion_rrf([ids_sens, ids_mots])

        rang_sens = {i: r for r, i in enumerate(ids_sens, start=1)}
        rang_mots = {i: r for r, i in enumerate(ids_mots, start=1)}

        resultats = []
        deja_vus = set()
        for identifiant in sorted(scores_rrf, key=scores_rrf.get, reverse=True):
            chunk = self.par_id[identifiant]
            if type_document and not chunk.source.startswith(type_document + "_"):
                continue
            cle = (chunk.source, chunk.section)
            if dedoublonner and cle in deja_vus:
                continue
            deja_vus.add(cle)
            resultats.append(
                ResultatHybride(
                    chunk=chunk,
                    score_rrf=scores_rrf[identifiant],
                    rang_sens=rang_sens.get(identifiant),
                    rang_mots=rang_mots.get(identifiant),
                    score_sens=scores_sens[identifiant],
                )
            )
            if len(resultats) == k:
                break
        return resultats