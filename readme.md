# NY Policy Chatbot

This project creates a Streamlit-based chatbot that can answer questions about solar policy documents, using local vector search and LLM-based summarization.

## 🧠 What It Does

- Extracts and chunks text from documents in the `data/` folder
- Converts those text chunks into vector embeddings using a pretrained model (`intfloat/e5-base-v2`)
- Stores the vectorized chunks in Snowflake
- Uses a Streamlit chatbot UI to let users ask questions
- Matches queries to the most relevant text chunks via cosine similarity
- Sends the top matches as context to the [Mistral.ai](https://mistral.ai) API to generate a response

## 📂 Scripts

All code is in `src/`:

1. `process_upload_doc_embeddings.py` – extracts, chunks, embeds, and uploads document data to Snowflake
2. `streamlit_app.py` – launches the chatbot interface

## 🚀 How to Run

1. **Get Snowflake access** (ask Dan or Jake)
2. **Get a [Mistral API key](https://mistral.ai)** and save it in your local `.env` file along with Snowflake credentials:
    ```
    MISTRAL_API_KEY=your_key_here
    GEMMA_API_KEY = your_key_here
    SNOWFLAKE_USER=...
    SNOWFLAKE_PASSWORD=...
    SNOWFLAKE_ACCOUNT=...
    SNOWFLAKE_DATABASE=NY_POLICY_CHATBOT_DB
    SNOWFLAKE_SCHEMA=EMBEDDINGS
    SNOWFLAKE_WAREHOUSE=...
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Preprocess and upload documents:
    ```bash
    python process_upload_doc_embeddings.py
    ```
5. Launch the chatbot:
    ```bash
    streamlit run streamlit_app.py
    ```

---

Let your friendly neighborhood data scientist know if you have any questions! 