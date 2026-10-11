"""
Étape 0 : comprendre les embeddings et la similarité cosinus.

Ce script :
0. vérifie TA fonction de similarité cosinus sur des cas géométriques simples ;
1. montre à quoi ressemble un embedding ;
2. compare des paires de termes : comparaison exacte (projet 1) vs par le sens ;
3. fait une mini-recherche sémantique dans une liste de compétences ;
4. montre pourquoi on normalise les vecteurs.

Lancement, depuis la racine du projet :
    python scripts/etape0_similarite.py
"""

import sys
from pathlib import Path

# Permet d'importer src/ quand on lance le script depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.embeddings import encoder


# ============================================================================
# À TOI DE JOUER : la similarité cosinus
# ============================================================================
def similarite_cosinus(a: np.ndarray, b: np.ndarray) -> float:
    """Renvoie le cosinus de l'angle entre les vecteurs a et b.

    Formule : cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
      - a · b  : le produit scalaire      -> np.dot(a, b)
      - ‖a‖    : la norme (longueur) de a -> np.linalg.norm(a)

    Résultat attendu : un float entre -1 et 1.
      1  -> même direction (même sens)
      0  -> directions perpendiculaires (aucun rapport)
      -1 -> directions opposées

    Pense à convertir le résultat avec float(...) pour renvoyer
    un nombre Python et pas un scalaire NumPy.
    """
    # Remplace la ligne ci-dessous par ton code (2 à 3 lignes suffisent)
    p=np.dot(a, b)
    norme_a=np.linalg.norm(a)
    norme_b=np.linalg.norm(b)
    return float(p / (norme_a * norme_b))


def verifier_ma_fonction() -> None:
    """Tests géométriques en 2 dimensions, faciles à visualiser sur papier."""
    a = np.array([1.0, 0.0])   # flèche vers la droite
    b = np.array([0.0, 1.0])   # flèche vers le haut
    c = np.array([5.0, 0.0])   # vers la droite, mais 5 fois plus longue
    d = np.array([-1.0, 0.0])  # vers la gauche

    assert abs(similarite_cosinus(a, c) - 1.0) < 1e-6, "a et c ont la même direction : attendu 1"
    assert abs(similarite_cosinus(a, b)) < 1e-6, "a et b sont perpendiculaires : attendu 0"
    assert abs(similarite_cosinus(a, d) + 1.0) < 1e-6, "a et d sont opposés : attendu -1"
    print("Ta fonction passe les 3 tests géométriques.\n")


# ============================================================================
# Démonstrations
# ============================================================================
def afficher_un_embedding() -> None:
    print("=" * 70)
    print("1. À quoi ressemble un embedding ?")
    print("=" * 70)
    vecteur = encoder(["Machine Learning"], normaliser=False)[0]
    print(f"Forme du vecteur : {vecteur.shape}  -> 384 nombres")
    print(f"5 premières valeurs : {np.round(vecteur[:5], 4)}")
    print(f"Norme (longueur) : {np.linalg.norm(vecteur):.4f}")
    print("Chaque nombre seul ne veut rien dire : c'est la DIRECTION globale")
    print("du vecteur qui encode le sens.\n")


def comparer_des_paires() -> None:
    print("=" * 70)
    print("2. Comparaison exacte (projet 1) vs comparaison par le sens")
    print("=" * 70)
    paires = [
        ("ML", "Machine Learning"),
        ("ML", "apprentissage automatique"),
        ("Machine Learning", "apprentissage automatique"),
        ("NLP", "traitement du langage naturel"),
        ("Docker", "conteneurisation"),
        ("Java", "JavaScript"),
        ("Python", "serpent"),
        ("Machine Learning", "comptabilité"),
    ]
    # On encode tous les termes en un seul appel : beaucoup plus rapide
    termes = sorted({t for paire in paires for t in paire})
    vecteurs = dict(zip(termes, encoder(termes, normaliser=False)))

    print(f"{'Terme A':<18} {'Terme B':<32} {'exact':>6} {'cosinus':>8}")
    print("-" * 70)
    for t1, t2 in paires:
        exact = t1.lower().strip() == t2.lower().strip()  # la logique du projet 1
        score = similarite_cosinus(vecteurs[t1], vecteurs[t2])
        print(f"{t1:<18} {t2:<32} {str(exact):>6} {score:>8.3f}")
    print()


def mini_recherche() -> None:
    print("=" * 70)
    print("3. Mini-recherche sémantique : quelles compétences du CV")
    print("   correspondent à l'exigence « bonnes bases en ML » ?")
    print("=" * 70)
    competences_cv = [
        "apprentissage automatique avec scikit-learn",
        "réseaux de neurones avec PyTorch",
        "développement web avec React",
        "gestion de bases de données SQL",
        "classification de textes",
        "comptabilité analytique",
    ]
    exigence = "bonnes bases en Machine Learning"

    v_exigence = encoder([exigence], normaliser=False)[0]
    v_competences = encoder(competences_cv, normaliser=False)

    resultats = [
        (similarite_cosinus(v_exigence, v), texte)
        for v, texte in zip(v_competences, competences_cv)
    ]
    resultats.sort(reverse=True)  # du plus similaire au moins similaire
    for rang, (score, texte) in enumerate(resultats, start=1):
        print(f"{rang}. {score:.3f}  {texte}")
    print()


def pourquoi_normaliser() -> None:
    print("=" * 70)
    print("4. Pourquoi normaliser les vecteurs ?")
    print("=" * 70)
    brut = encoder(["ML", "Machine Learning"], normaliser=False)
    norme = encoder(["ML", "Machine Learning"], normaliser=True)

    cos = similarite_cosinus(brut[0], brut[1])
    produit_scalaire = float(np.dot(norme[0], norme[1]))
    print(f"Cosinus sur vecteurs bruts          : {cos:.6f}")
    print(f"Simple produit scalaire, normalisés : {produit_scalaire:.6f}")
    print(f"Normes après normalisation          : {np.linalg.norm(norme, axis=1)}")
    print("Si les vecteurs ont une norme de 1, le cosinus se réduit au produit")
    print("scalaire : plus besoin de diviser. C'est ce qui permettra de comparer")
    print("une question à des milliers de passages en une seule multiplication.\n")


if __name__ == "__main__":
    verifier_ma_fonction()
    print("Chargement du modèle (long au premier lancement : téléchargement)...\n")
    afficher_un_embedding()
    comparer_des_paires()
    mini_recherche()
    pourquoi_normaliser()