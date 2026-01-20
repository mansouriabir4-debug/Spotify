import streamlit as st
import pandas as pd
from src.rag import query_spotify, get_conversation_history

st.set_page_config(page_title="🎧 Spotify 2024 Assistant", layout="wide")
st.title("🎧 Spotify 2024 – RAG Assistant")


@st.cache_data
def load_data():
    df = pd.read_csv("data/data24.csv", encoding="latin1")

    df["Spotify Streams"] = (
        df["Spotify Streams"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(".", "", regex=False)
    )
    df["Spotify Streams"] = pd.to_numeric(
        df["Spotify Streams"], errors="coerce"
    ).fillna(0)

    return df

df = load_data()


st.header("🎵 Dataset Spotify 2024")

col1, col2, col3 = st.columns(3)
col1.metric("Titres", len(df))
col2.metric("Artistes uniques", df["Artist"].nunique())
col3.metric(
    "Spotify Streams totaux",
    f"{int(df['Spotify Streams'].sum()):,}"
)

with st.expander("les premières musiques"):
    st.dataframe(df.head(20), use_container_width=True)


st.divider()



st.header(" Assistant Spotify 2024")

question = st.text_input(
    "Pose une question sur les artistes, chansons, streams ou plateformes Spotify 2024"
)

if st.button("Demander"):
    if question:
        with st.spinner("Analyse des données Spotify..."):
            answer = query_spotify(question)

        st.success("Réponse")
        st.write(answer)
    else:
        st.warning("Veuillez entrer une question.")

