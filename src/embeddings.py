"""
Module d'embeddings : transforme du texte en vecteurs qui représentent son sens.

On utilise un modèle local (sentence-transformers) plutôt qu'une API :
- gratuit et sans clé, cohérent avec le choix de Groq pour le LLM ;
- multilingue, car nos documents mélangent français et anglais.

Ce module sera réutilisé à toutes les étapes du projet (indexation, recherche,
matching de compétences), d'où son placement dans src/.
"""

from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

# Modèle multilingue léger : produit des vecteurs de 384 nombres.
# Au premier lancement, il est téléchargé (~470 Mo) puis mis en cache sur le disque.
NOM_MODELE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def charger_modele() -> SentenceTransformer:
    """Charge le modèle une seule fois.

    lru_cache mémorise le résultat : les appels suivants renvoient
    le même objet au lieu de recharger 470 Mo en mémoire.
    """
    return SentenceTransformer(NOM_MODELE)


def encoder(textes: list[str], normaliser: bool = True) -> np.ndarray:
    """Transforme une liste de textes en matrice d'embeddings.

    Args:
        textes: les textes à encoder.
        normaliser: si True, chaque vecteur est ramené à une longueur (norme) de 1.
            On voit à l'étape 0 pourquoi c'est pratique pour la recherche.

    Returns:
        Une matrice NumPy de forme (nombre_de_textes, 384) :
        une ligne par texte, une colonne par dimension.
    """
    modele = charger_modele()
    vecteurs = modele.encode(
        textes,
        normalize_embeddings=normaliser,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # float32 : deux fois moins de mémoire que float64, précision largement suffisante
    return vecteurs.astype(np.float32)