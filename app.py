import streamlit as st
import pandas as pd

st.set_page_config(page_title="🎧 Spotify 2024 Assistant", layout="wide")
st.title("🎧 Spotify 2024 – RAG Assistant")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/data244.csv", encoding="latin1")

    df["Spotify Streams"] = (
        df["Spotify Streams"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(".", "", regex=False)
    )
    df["Spotify Streams"] = pd.to_numeric(df["Spotify Streams"], errors="coerce").fillna(0)

    # YouTube Views
    if "YouTube Views" in df.columns:
        df["YouTube Views"] = (
            df["YouTube Views"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(".", "", regex=False)
        )
        df["YouTube Views"] = pd.to_numeric(df["YouTube Views"], errors="coerce").fillna(0)

    # ikTok Views
    if "TikTok Views" in df.columns:
        df["TikTok Views"] = (
            df["TikTok Views"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(".", "", regex=False)
        )
        df["TikTok Views"] = pd.to_numeric(df["TikTok Views"], errors="coerce").fillna(0)

    return df

df = load_data()


# DATASET
st.header("🎵 Dataset Spotify 2024")

col1, col2, col3 = st.columns(3)
col1.metric("Titres", len(df))
col2.metric("Artistes uniques", df["Artist"].nunique())
col3.metric("Spotify Streams totaux", f"{int(df['Spotify Streams'].sum()):,}")

with st.expander("📄 Voir un aperçu (20 premières chansons)"):
    st.dataframe(df.head(20), use_container_width=True)


#  ANALYSES RAPIDES (SANS LLM)
st.divider()
st.header(" Analyses rapides ")

colA, colB = st.columns(2)

with colA:
    if st.button("🔥 Top 5 artistes (Spotify Streams)"):
        top_artists = (
            df.groupby("Artist")["Spotify Streams"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        st.subheader("Top 5 artistes – Spotify Streams")
        st.dataframe(top_artists.reset_index(), use_container_width=True)


# ASSISTANT RAG
st.divider()
st.header("💬 Assistant Spotify 2024")

question = st.text_input(
    "Pose une question sur les artistes, chansons, streams ou plateformes Spotify 2024 "
    "(ex: compare Drake vs Taylor Swift)"
)

if st.button("🎶 Demander"):
    if question:
        with st.spinner("Analyse des données Spotify..."):
            from src.rag import query_spotify
            answer = query_spotify(question)

        st.success("✅ Réponse")
        st.write(answer)
    else:
        st.warning("Veuillez entrer une question.")
