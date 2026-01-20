## Architecture
- **Frontend**: Streamlit app (`app.py`) for user interface, data visualization, and RAG queries
- **Backend Logic**: RAG implementation in `src/rag.py` module with data loading and QA chain
- **Data**: Movie information stored in `data/data.csv` with detailed movie details
- **AI Integration**: LangChain orchestrator with GPT-4.1 hosted on Azure for natural language queries

## Key Components
- `src/rag.py`: RAG logic module - loads CSV data, creates VectorStoreIndex, provides `query_movies()` function
- `app.py`: Streamlit interface with data display and query input calling `src.rag.query_movies()`
- `data/data.csv`: Movie data from Kaggle

## Development Workflow
- **Run Application**: `streamlit run app.py`
- **Environment**: Use `.env` file for Azure OpenAI credentials
- **Dependencies**: Install from `requirements.txt` (includes streamlit, llama-index packages, chromadb, pandas)

## Data Handling
- Movie data is CSV format with various columns about movies
- Use pandas or similar for data processing; avoid loading entire CSV into memory for large queries

## AI Integration Patterns
- Use LlamaIndex with Azure OpenAI: embeddings with `text-embedding-3-large`, LLM with `gpt-4.1`
- RAG implementation in `src/rag.py`: load CSV data, create VectorStoreIndex with ChromaDB, query engine
- Environment variables in `.env` for Azure credentials (endpoint, api_key, models, versions)
- Direct function call `query_movies(question)` for natural language queries about movies
- Example: User inputs "What are the best action movies?" → `query_movies()` retrieves docs and generates response

## Coding Conventions
- Use `os.getenv()` for environment variables (loaded via dotenv)
- RAG logic in `src/` module, UI in root `app.py`
- Streamlit components use `st.spinner()` for loading states
- Import backend functions: `from src.rag import query_movies`</content>
<parameter name="filePath">c:\Users\emac\Desktop\cloud\projet-lol\.github\copilot-instructions.md