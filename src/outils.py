"""
L'outil de l'agent RAG : rechercher dans la base documentaire.

Même principe qu'au projet 1 :
- une fonction Python qui fait le travail (rechercher) ;
- une description pour le LLM (TOOLS), qui lui dit quand et comment l'utiliser ;
- une closure (creer_outils) qui enferme le moteur de recherche dans l'outil.

Nouveauté par rapport au projet 1 : l'outil prend des ARGUMENTS choisis par le LLM
(la requête et le filtre). C'est ce qui permet à l'agent de reformuler.
"""

from pydantic import BaseModel

from src.rag_classique import formater_contexte


class Recherche(BaseModel):
    """Trace d'un appel à l'outil : ce que l'agent a cherché et ce qu'il a obtenu."""

    requete: str
    type_document: str | None
    sources: list[str]   # fichiers des passages renvoyés


# ----------------------------------------------------------------------------
# 1. La description de l'outil : c'est TOUT ce que le LLM sait de lui
# ----------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rechercher",
            "description": (
                "Cherche dans la base documentaire (offres de stage, guides de candidature, "
                "référentiel de compétences IA). Renvoie les 3 passages les plus pertinents, "
                "chacun avec son fichier source. Tu peux l'appeler plusieurs fois avec des "
                "requêtes différentes si les premiers résultats ne contiennent pas la réponse."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requete": {
                        "type": "string",
                        "description": (
                            "Les mots à chercher. Les documents sont en français ou en anglais : "
                            "si besoin, reformule dans la langue du document visé, développe les "
                            "acronymes (ML -> machine learning, apprentissage automatique) et garde "
                            "les noms propres (entreprises, outils)."
                        ),
                    },
                    "type_document": {
                        "type": "string",
                        "enum": ["offre", "guide", "referentiel"],
                        "description": "Facultatif. Limite la recherche à un type de document.",
                    },
                },
                "required": ["requete"],
            },
        },
    }
]


# ----------------------------------------------------------------------------
# 2. La fonction de l'outil, fabriquée par une closure
# ----------------------------------------------------------------------------
def creer_outils(moteur):
    """Fabrique l'outil `rechercher` lié à UN moteur de recherche.

    Renvoie :
    - le registre {nom: fonction}, utilisé par la boucle pour exécuter l'outil demandé ;
    - le journal des recherches, rempli à chaque appel (pour la trace et l'ancrage).
    """
    journal: list[Recherche] = []

    def rechercher(requete: str, type_document: str | None = None) -> str:
        # ====================================================================
        # À TOI DE JOUER (voir le squelette dans le message de ton mentor)
        # ====================================================================
        resultats = moteur.rechercher(requete, k=3, type_document=type_document)
        journal.append(Recherche(requete=requete, type_document=type_document,
                                 sources=[r.chunk.source for r in resultats]))
        if not resultats:
            return "Aucun passage trouvé pour cette requête."
        return formater_contexte(resultats)

    return {"rechercher": rechercher}, journal