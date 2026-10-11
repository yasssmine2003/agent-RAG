"""
Index vectoriel écrit à la main avec NumPy.

Un index, dans sa forme la plus simple, ce sont deux éléments rangés dans le même ordre :
- self.chunks   : la liste des passages (Chunk)
- self.vecteurs : une matrice (nombre_de_passages, 384), la ligne i = vecteur du passage i

Chercher revient à multiplier cette matrice par le vecteur de la question :
comme tous les vecteurs sont normalisés, on obtient directement les cosinus.

À l'étape suivante, on remplacera cette classe par ChromaDB, qui fait la même chose
avec en plus la gestion des filtres et de gros volumes. Écrire la version à la main
d'abord permet de voir qu'il n'y a pas de magie.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from src.decoupage import Chunk
from src.embeddings import NOM_MODELE, encoder


class ResultatRecherche(BaseModel):
    chunk: Chunk
    score: float  # cosinus entre la question et le passage


class IndexVectoriel:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        # Matrice vide au départ ; elle prend sa vraie forme au premier ajout
        self.vecteurs: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ indexation
    def ajouter(self, chunks: list[Chunk]) -> None:
        """Vectorise des passages et les ajoute à l'index (en gardant le même ordre)."""
        if not chunks:
            return
        # On vectorise le texte ENRICHI (contexte + passage), normalisé (défaut de encoder)
        nouveaux = encoder([c.texte for c in chunks])
        if len(self.chunks) == 0:
            self.vecteurs = nouveaux
        else:
            # vstack empile les nouvelles lignes sous les anciennes
            self.vecteurs = np.vstack([self.vecteurs, nouveaux])
        self.chunks.extend(chunks)

    # ------------------------------------------------------------------ recherche
    def rechercher_par_vecteur(self, vecteur_requete: np.ndarray, k: int = 3) -> list[ResultatRecherche]:
        """Renvoie les k passages les plus proches d'un vecteur de requête (déjà normalisé).

        À TOI DE JOUER : voir le squelette dans le message de ton mentor.
        """
        if len(self.chunks) == 0:
            return []
        # Remplace la ligne ci-dessous par ton code (3 lignes)
        scores =self.vecteurs@vecteur_requete   # un score par passage, en UNE opération
        ordre = np.argsort(-scores)    # indices des passages, du meilleur au moins bon
        return [
            ResultatRecherche(chunk=self.chunks[i], score=float(scores[i]))
            for i in ordre[:k]
                ]
    def rechercher(self, requete: str, k: int = 3) -> list[ResultatRecherche]:
        """Vectorise la question avec le MÊME modèle que l'index, puis cherche."""
        vecteur_requete = encoder([requete])[0]
        return self.rechercher_par_vecteur(vecteur_requete, k)

    # ------------------------------------------------------------------ persistance
    def sauvegarder(self, dossier: Path) -> None:
        """Écrit l'index sur le disque : 3 fichiers dans `dossier`."""
        dossier.mkdir(parents=True, exist_ok=True)
        np.save(dossier / "vecteurs.npy", self.vecteurs)
        (dossier / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # On note le modèle utilisé : un index ne doit être interrogé qu'avec ce modèle
        meta = {"modele": NOM_MODELE, "nb_passages": len(self.chunks)}
        (dossier / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def charger(cls, dossier: Path) -> "IndexVectoriel":
        """Recharge un index sauvegardé, en vérifiant qu'il a été fait avec le même modèle."""
        meta = json.loads((dossier / "meta.json").read_text(encoding="utf-8"))
        if meta["modele"] != NOM_MODELE:
            raise ValueError(
                f"Index construit avec {meta['modele']}, mais le modèle actuel est {NOM_MODELE}. "
                "Les vecteurs sont incompatibles : relance scripts/construire_index.py."
            )
        index = cls()
        index.vecteurs = np.load(dossier / "vecteurs.npy")
        donnees = json.loads((dossier / "chunks.json").read_text(encoding="utf-8"))
        index.chunks = [Chunk(**d) for d in donnees]  # Pydantic revalide chaque passage
        return index