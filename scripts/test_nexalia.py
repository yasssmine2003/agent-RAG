import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import IndexVectoriel
from src.recherche import RechercheHybride

moteur = RechercheHybride(IndexVectoriel.charger(Path("data/index")))
for question in ["Quelle est la gratification du stage chez Nexalia ?", "Nexalia stipend", "Nexalia"]:
    print(question)
    for r in moteur.rechercher(question, k=3):
        print(f"   [{r.chunk.source} > {r.chunk.section}] sens {r.rang_sens} | mots {r.rang_mots}")