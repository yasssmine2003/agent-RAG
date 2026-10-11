"""
API du RAG (même principe qu'au projet 1) : POST /question.

Lancement local, depuis la racine du projet :
    uvicorn src.api:app --reload
Puis ouvrir http://127.0.0.1:8000/docs pour tester.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent import executer_agent
from src.index import IndexVectoriel
from src.rag_classique import repondre
from src.recherche import RechercheHybride
from src.securite import MoteurProtege

DOSSIER_INDEX = Path(__file__).resolve().parent.parent / "data" / "index"
etat = {}  # rempli une seule fois au démarrage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Au démarrage : charger l'index une fois (pas à chaque question)."""
    index = IndexVectoriel.charger(DOSSIER_INDEX)
    etat["moteur"] = RechercheHybride(index)
    etat["fichiers"] = {c.source for c in index.chunks}
    yield


app = FastAPI(title="Agent RAG candidatures", lifespan=lifespan)


class Requete(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    mode: Literal["classique", "agentique"] = "agentique"


class ReponseAPI(BaseModel):
    reponse: str
    sources: list[str]
    nb_recherches: int
    passages_bloques: list[str]   # passages retirés par la protection anti-injection


@app.get("/sante")
def sante() -> dict:
    return {"statut": "ok"}


@app.post("/question", response_model=ReponseAPI)
def poser_question(req: Requete) -> ReponseAPI:
    moteur = MoteurProtege(etat["moteur"])  # protection activée pour chaque requête
    if req.mode == "classique":
        r = repondre(req.question, moteur)
        return ReponseAPI(reponse=r.reponse.reponse, sources=r.reponse.sources,
                          nb_recherches=1, passages_bloques=moteur.bloques)
    r = executer_agent(req.question, moteur, etat["fichiers"])
    return ReponseAPI(reponse=r.reponse, sources=r.sources_citees,
                      nb_recherches=len(r.recherches), passages_bloques=moteur.bloques)