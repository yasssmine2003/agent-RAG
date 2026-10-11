"""Connexion au LLM, partagée par tous les modules qui en ont besoin.

Repris tel quel du projet 1 : changer de fournisseur = modifier uniquement ce fichier (et .env).
"""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODELE = "openai/gpt-oss-20b"