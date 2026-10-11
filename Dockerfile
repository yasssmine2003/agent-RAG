FROM python:3.12-slim
WORKDIR /app

# PyTorch version CPU (beaucoup plus léger que la version GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY data/documents/ data/documents/

# Construit l'index pendant la création de l'image : télécharge aussi le modèle d'embeddings,
# donc le conteneur démarre vite et sans Internet pour cette partie.
RUN python scripts/construire_index.py

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]