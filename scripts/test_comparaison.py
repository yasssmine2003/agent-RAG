import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import executer_agent
from src.index import IndexVectoriel
from src.recherche import RechercheHybride

index = IndexVectoriel.charger(Path("data/index"))
moteur = RechercheHybride(index)
r = executer_agent("Quel stage paie le mieux, et quelle est sa date limite ?",
                   moteur, {c.source for c in index.chunks})
for rech in r.recherches:
    print(f"Recherche : « {rech.requete} » (filtre : {rech.type_document}) -> {sorted(set(rech.sources))}")
print("Réponse :", r.reponse)