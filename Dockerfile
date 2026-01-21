# Utiliser Python 3.14 comme image de base
FROM python:3.14

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de l'application
COPY . .

# Exposer le port Streamlit (par défaut 8501)
EXPOSE 8501

# Créer le répertoire pour ChromaDB s'il n'existe pas
RUN mkdir -p /app/chroma_db

# Commande pour lancer l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]