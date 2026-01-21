import os
import json
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

import chromadb

load_dotenv()


# Embeddings
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
)


# LLM
llm = AzureOpenAI(
    model="gpt-4.1",
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

Settings.embed_model = embed_model
Settings.llm = llm

# Load Dataset
df = pd.read_csv("data/data24.csv", encoding="latin1")

# Row to Texte RAG
def row_to_text(row: pd.Series) -> str:
    return f"""
Titre : {row['Track']}
Artiste : {row['Artist']}
Album : {row['Album Name']}
Date de sortie : {row['Release Date']}
ISRC : {row['ISRC']}

Spotify :
- Streams : {row['Spotify Streams']}
- Popularité : {row['Spotify Popularity']}
- Playlists : {row['Spotify Playlist Count']}

YouTube :
- Vues : {row['YouTube Views']}
- Likes : {row['YouTube Likes']}

TikTok :
- Posts : {row['TikTok Posts']}
- Vues : {row['TikTok Views']}

Classement :
- Rang historique : {row['All Time Rank']}
- Score global : {row['Track Score']}
"""


base_documents = [
    Document(
        text=row_to_text(row),
        metadata={
            "track": row["Track"],
            "artist": row["Artist"],
            "album": row["Album Name"],
            "release_date": row["Release Date"],
            "isrc": row["ISRC"],
        },
    )
    for _, row in df.iterrows()
]


# Chunking
splitter = SentenceSplitter(chunk_size=300, chunk_overlap=60)

nodes = splitter.get_nodes_from_documents(base_documents)

for i, n in enumerate(nodes):
    n.metadata["chunk_index"] = i

# Chroma Vector Store
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("spotify_2024")
vector_store = ChromaVectorStore(chroma_collection=collection)

index = VectorStoreIndex(
    nodes=nodes,
    vector_store=vector_store,
)

# Chat Engine
memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=index.as_retriever(similarity_top_k=5),
    memory=memory,
    llm=llm,
    context_prompt=(
        "Vous êtes un assistant expert en musique et streaming Spotify 2024. "
        "Vous répondez uniquement à partir des données du dataset Spotify 2024 "
        "(titres, artistes, albums, streams, popularité, classements, plateformes). "
        "Si la question est hors sujet, refusez poliment."
    ),
    verbose=False,
)


# Conversation Tracker
class ConversationTracker:
    def __init__(self, file="conversation_history.json"):
        self.file = file
        self.history = self._load()

    def _load(self):
        if os.path.exists(self.file):
            with open(self.file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def add(self, q, r, ok):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "question": q,
            "response": r,
            "is_relevant": ok,
        })
        self._save()

    def recent(self, n=10):
        return self.history[-n:]

    def stats(self):
        total = len(self.history)
        relevant = sum(h["is_relevant"] for h in self.history)
        return {
            "total": total,
            "relevant": relevant,
            "irrelevant": total - relevant,
            "rate": f"{relevant / total * 100:.1f}%" if total else "0%",
        }

tracker = ConversationTracker()


# Public API
def query_spotify(question: str) -> str:
    try:
        nodes_retrieved = index.as_retriever(similarity_top_k=4).retrieve(question)

        if not nodes_retrieved or nodes_retrieved[0].score < 0.3:
            msg = "Je peux répondre uniquement aux questions liées aux données Spotify 2024."
            tracker.add(question, msg, False)
            return msg

        response = str(chat_engine.chat(question))
        tracker.add(question, response, True)
        return response

    except Exception as e:
        err = f"Erreur : {e}"
        tracker.add(question, err, False)
        return err

def get_conversation_history(limit=10):
    return tracker.recent(limit)

def get_conversation_stats():
    return tracker.stats()
