"""
Sécurité du RAG : se protéger de l'injection INDIRECTE.

Au projet 1, l'attaque venait de l'utilisateur (un CV piégé envoyé directement).
Ici, elle vient d'un DOCUMENT de la base : quelqu'un a glissé des instructions
dans une offre de stage. L'agent les lit quand ses recherches font remonter ce passage.

Défense en profondeur (comme au projet 1) :
1. architecture : les passages sont encadrés par <passage> et déclarés comme des données ;
2. ancrage : la règle 6 du prompt interdit d'obéir à leur contenu ;
3. détection (ce fichier) : on retire les passages suspects AVANT qu'ils atteignent le LLM.
"""

import re

# Formulations typiques d'une injection (français et anglais). Liste volontairement simple :
# elle bloque les attaques grossières, pas un attaquant déterminé (d'où la défense en profondeur).
MOTIFS_INJECTION = [
    r"ignore[rz]? (toutes |tous )?(tes|vos|les) (instructions|règles|consignes)",
    r"ignore (all |any )?(previous|prior) instructions",
    r"tu dois (maintenant|désormais)",
    r"(nouvelles|new) instructions",
    r"pour l'assistant",
    r"you are now",
    r"(prompt système|system prompt)",
    r"mot de passe|password",
]


def detecter_injection(texte: str) -> list[str]:
    """Renvoie la liste des motifs suspects trouvés dans le texte (vide = rien de suspect)."""
    t = texte.lower()
    return [motif for motif in MOTIFS_INJECTION if re.search(motif, t)]


class MoteurProtege:
    """Enveloppe un moteur de recherche et retire les passages suspects.

    Il a la même méthode rechercher() que RechercheHybride : on peut donc le donner
    au RAG classique comme à l'agent SANS modifier leur code. Une seule protection,
    placée à la source, protège les deux systèmes.
    """

    def __init__(self, moteur) -> None:
        self.moteur = moteur
        self.bloques: list[str] = []   # identifiants des passages retirés (pour la trace)

    def rechercher(self, requete: str, k: int = 3, type_document: str | None = None):
        # On demande un peu plus de passages, pour en avoir encore k après le filtrage
        resultats = self.moteur.rechercher(requete, k=k + 3, type_document=type_document)
        surs = []
        for r in resultats:
            if detecter_injection(r.chunk.texte):
                self.bloques.append(r.chunk.id)
            else:
                surs.append(r)
        return surs[:k]