"""
Étape 1 : charger les documents et les découper en passages.

Ce script :
0. vérifie TA fonction decouper_texte sur des cas précis ;
1. charge les 6 documents ;
2. les découpe et affiche des statistiques ;
3. montre deux passages complets ;
4. vérifie qu'aucun passage ne dépasse la limite de tokens du modèle.

Lancement, depuis la racine du projet :
    python scripts/etape1_decoupage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chargement import charger_documents
from src.decoupage import decouper_document, decouper_texte
from src.embeddings import charger_modele

DOSSIER_DOCUMENTS = Path(__file__).resolve().parent.parent / "data" / "documents"


def verifier_ma_fonction() -> None:
    dix_mots = " ".join(f"m{i}" for i in range(10))     # "m0 m1 ... m9"
    onze_mots = " ".join(f"m{i}" for i in range(11))    # "m0 m1 ... m10"

    assert decouper_texte(dix_mots, 4, 1) == ["m0 m1 m2 m3", "m3 m4 m5 m6", "m6 m7 m8 m9"], \
        "Cas de base : 3 fenêtres de 4 mots qui se chevauchent d'1 mot"
    assert decouper_texte(onze_mots, 4, 1)[-1] == "m9 m10", \
        "Le dernier morceau peut être plus court s'il contient des mots nouveaux"
    assert decouper_texte("un deux trois", 10, 2) == ["un deux trois"], \
        "Texte plus court que la taille : un seul morceau"
    assert decouper_texte("   ", 4, 1) == [], "Texte vide : liste vide"
    assert decouper_texte("a  b\n\nc", 5, 0) == ["a b c"], \
        "Espaces multiples et sauts de ligne doivent être normalisés"

    for taille, chevauchement in [(3, 3), (3, 5), (0, 0), (4, -1)]:
        try:
            decouper_texte("a b c d e", taille, chevauchement)
        except ValueError:
            continue
        raise AssertionError(f"taille={taille}, chevauchement={chevauchement} devrait lever ValueError")

    print("Ta fonction passe les 6 tests.\n")


def main() -> None:
    verifier_ma_fonction()

    print("=" * 70)
    print("1. Chargement")
    print("=" * 70)
    documents = charger_documents(DOSSIER_DOCUMENTS)
    for doc in documents:
        print(f"- {doc.source:<34} {len(doc.contenu.split()):>4} mots  | {doc.titre}")
    print()

    print("=" * 70)
    print("2. Découpage (taille=60 mots, chevauchement=10)")
    print("=" * 70)
    tous_les_chunks = []
    for doc in documents:
        chunks = decouper_document(doc)
        tous_les_chunks.extend(chunks)
        sections = [c.section for c in chunks]
        print(f"- {doc.source:<34} {len(chunks):>2} passages : {sections}")
    print(f"\nTotal : {len(tous_les_chunks)} passages\n")

    print("=" * 70)
    print("3. Exemples de passages (ce qui sera réellement vectorisé)")
    print("=" * 70)
    for chunk in (tous_les_chunks[0], tous_les_chunks[-1]):
        print(f"[{chunk.id}]  ({chunk.nb_mots} mots)")
        print(chunk.texte)
        print("-" * 70)
    print()

    print("=" * 70)
    print("4. Contrôle de la limite de tokens du modèle")
    print("=" * 70)
    modele = charger_modele()
    limite = modele.max_seq_length
    nb_tokens = {c.id: len(modele.tokenizer(c.texte)["input_ids"]) for c in tous_les_chunks}
    plus_long = max(nb_tokens, key=nb_tokens.get)
    print(f"Limite du modèle : {limite} tokens")
    print(f"Passage le plus long : {plus_long} -> {nb_tokens[plus_long]} tokens")
    trop_longs = [cid for cid, n in nb_tokens.items() if n > limite]
    if trop_longs:
        print(f"ATTENTION : {len(trop_longs)} passage(s) seront tronqués : {trop_longs}")
    else:
        print("Aucun passage ne dépasse la limite : rien ne sera tronqué.")


if __name__ == "__main__":
    main()