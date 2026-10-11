"""
Découpage des documents en passages (chunks) prêts à être vectorisés.

Stratégie à deux niveaux :
1. découpage par section Markdown (titres "## ...") : une section = un sujet ;
2. si une section est trop longue, découpage en fenêtres de mots avec chevauchement.

Chaque passage est préfixé par son contexte "titre du document > section"
pour lever les ambiguïtés (souviens-toi de « Python » / « serpent »).
"""

from pydantic import BaseModel

from src.chargement import Document


class Chunk(BaseModel):
    id: str        # identifiant unique, ex. "offre_stage_devops.md#2"
    source: str    # fichier d'origine, pour la citation
    section: str   # titre de la section d'origine
    texte: str     # texte ENRICHI (contexte + passage) : c'est lui qu'on vectorise
    nb_mots: int   # taille du passage sans le préfixe de contexte


# ============================================================================
# À TOI DE JOUER : le découpage en fenêtres glissantes
# ============================================================================
def decouper_texte(texte: str, taille: int, chevauchement: int) -> list[str]:
    """Découpe un texte en morceaux de `taille` mots, qui se chevauchent de `chevauchement` mots.

    Exemple avec 10 mots "m0 ... m9", taille=4, chevauchement=1 :
        ["m0 m1 m2 m3", "m3 m4 m5 m6", "m6 m7 m8 m9"]
    (m3 et m6 apparaissent deux fois : c'est le chevauchement)

    Règles :
    - si taille <= 0, ou chevauchement < 0, ou chevauchement >= taille :
      lever une ValueError (sinon la fenêtre n'avancerait jamais) ;
    - un texte vide (ou seulement des espaces) donne une liste vide ;
    - un texte plus court que `taille` donne un seul morceau ;
    - le dernier morceau ne doit pas être composé uniquement de mots déjà vus :
      dès qu'une fenêtre atteint la fin du texte, on s'arrête.

    Pistes :
    - texte.split() découpe en mots (et ignore les espaces multiples et sauts de ligne) ;
    - la fenêtre avance de pas = taille - chevauchement mots à chaque tour ;
    - range(debut, fin, pas) permet de parcourir les positions de départ ;
    - mots[debut:debut + taille] extrait une fenêtre (Python tolère de dépasser la fin) ;
    - " ".join(liste_de_mots) reconstruit une chaîne ;
    - break sort de la boucle.
    """
        # 1. Valider les paramètres (sinon la fenêtre n'avancerait jamais)
    if taille <= 0 or chevauchement < 0 or chevauchement >= taille:
        raise ValueError("taille et chevauchement invalides")

    # 2. Découper le texte en mots
    mots = texte.split()  # supprime les espaces multiples et sauts de ligne

    # 3. Faire glisser la fenêtre
    morceaux = []
    pas = taille - chevauchement
    for debut in range(0, len(mots), pas):
        fenetre = mots[debut:debut + taille]  # extrait la fenêtre de mots
        morceaux.append(" ".join(fenetre))        # attention : on veut une chaîne, pas une liste
        if debut+taille >= len(mots):        # la fenêtre a atteint la fin du texte
            break
    return morceaux


# ============================================================================
# Fourni : découpage par sections et assemblage
# ============================================================================
def decouper_en_sections(contenu: str) -> list[tuple[str, str]]:
    """Sépare un document Markdown en (titre_section, texte_section).

    Le texte situé avant le premier "##" (les informations d'en-tête des offres)
    forme une section nommée "Informations générales".
    """
    sections = []
    titre_courant = "Informations générales"
    lignes = []
    for ligne in contenu.splitlines():
        if ligne.startswith("# "):        # titre du document : déjà stocké dans Document.titre
            continue
        if ligne.startswith("## "):       # nouvelle section : on range la précédente
            if "\n".join(lignes).strip():
                sections.append((titre_courant, "\n".join(lignes).strip()))
            titre_courant = ligne[3:].strip()
            lignes = []
        else:
            lignes.append(ligne)
    if "\n".join(lignes).strip():         # ne pas oublier la dernière section
        sections.append((titre_courant, "\n".join(lignes).strip()))
    return sections


def decouper_document(document: Document, taille: int = 60, chevauchement: int = 10) -> list[Chunk]:
    """Transforme un Document en liste de Chunk enrichis de leur contexte.

    taille=80 mots : marge de sécurité sous la limite de 128 tokens du modèle
    (un mot fait souvent plus d'un token, surtout en français et avec les accents).
    """
    chunks = []
    for section, texte_section in decouper_en_sections(document.contenu):
        for morceau in decouper_texte(texte_section, taille, chevauchement):
            chunks.append(
                Chunk(
                    id=f"{document.source}#{len(chunks)}",
                    source=document.source,
                    section=section,
                    texte=f"{document.titre} > {section}\n{morceau}",
                    nb_mots=len(morceau.split()),
                )
            )
    return chunks