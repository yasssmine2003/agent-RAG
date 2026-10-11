"""
Chargement des documents de la base documentaire.

Chaque fichier .md de data/documents/ devient un objet Document validé par Pydantic.
Le nom du fichier sert d'identifiant de source : c'est lui que l'agent citera.
"""

from pathlib import Path

from pydantic import BaseModel


class Document(BaseModel):
    source: str   # nom du fichier, ex. "offre_stage_devops.md" -> utilisé pour les citations
    titre: str    # contenu de la première ligne "# ..." du fichier
    contenu: str  # texte complet du fichier


def charger_documents(dossier: Path) -> list[Document]:
    """Lit tous les fichiers .md d'un dossier, triés par nom pour un ordre stable."""
    documents = []
    for chemin in sorted(dossier.glob("*.md")):
        # encoding="utf-8" est indispensable sous Windows : sans lui, Python utilise
        # l'encodage du système (cp1252) et les accents deviennent illisibles ou plantent.
        contenu = chemin.read_text(encoding="utf-8")
        premiere_ligne = contenu.strip().splitlines()[0]
        titre = premiere_ligne.lstrip("#").strip()  # "# Guide : ..." -> "Guide : ..."
        documents.append(Document(source=chemin.name, titre=titre, contenu=contenu))
    return documents