import streamlit as st
import os
import numpy as np
import pandas as pd
import requests
import ast
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import generativeai as genai
import io
import pickle

load_dotenv()

# === API Keys from Secrets ===
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
GEMMA_API_KEY = st.secrets["GEMMA_API_KEY"]

# === Helper functions ===
def get_mistral_response(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistral-small",  # Or mistral-medium / mistral-large if you have access
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://api.mistral.ai/v1/chat/completions", json=body, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

genai.configure(api_key=GEMMA_API_KEY)

def get_gemma_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")  
    response = model.generate_content(prompt)
    return response.text

@st.cache_data
def load_pickle_from_drive(url):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_pickle(io.BytesIO(response.content))

# === Load model for embedding user query ===
model = SentenceTransformer("intfloat/e5-base-v2")

# === Streamlit UI ===
st.title("📘 NY Policy Chatbot")

# === Load pickled data instead of Snowflake ===
POLICY_DOCUMENTS_URL = "https://drive.google.com/uc?id=1fcg0TXzahnsOk2AruWKjkfQvY6gz06Wf"
CENHUD_WEB_URL = "https://drive.google.com/uc?id=17tl5JPnky3ifSqaSf1D4LfcdfhoNtkuu"


with st.spinner("Loading documents..."):
    policy_documents = load_pickle_from_drive(POLICY_DOCUMENTS_URL)
    cenhud_documents = load_pickle_from_drive(CENHUD_WEB_URL)

all_documents = pd.concat([policy_documents, cenhud_documents], ignore_index=True)

# === User Inputs ===
query = st.text_input("Ask a question about either NY Sun Run or MA SMART programs:")
model_choice = st.selectbox("Choose a model:", ["Gemma", "Mistral"])
compare_mode = st.checkbox("Compare NY vs MA?", value=True)

if query:
    with st.spinner("Thinking..."):
        query_embedding = model.encode(query).reshape(1, -1)

        doc_names = all_documents["DOC_NAME"].tolist()
        contents = all_documents["CONTENT"].tolist()
        embeddings = np.vstack(all_documents["EMBEDDING"].tolist())

        sims = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:5]

        top_chunks = [contents[i] for i in top_indices]
        top_sources = [doc_names[i] for i in top_indices]

        if compare_mode:
            ny_chunks = [contents[i] for i in top_indices if "NY_" in doc_names[i]]
            ma_chunks = [contents[i] for i in top_indices if "MA_SMART" in doc_names[i]]

            ny_context = "\n\n---\n\n".join(ny_chunks)
            ma_context = "\n\n---\n\n".join(ma_chunks)

            prompt = f"""
Compare the NY Sun and MA SMART programs based on the following information.

NY Sun Program Context:
{ny_context}

MA SMART Program Context:
{ma_context}

Question: {query}

Answer:
            """
        else:
            context = "\n\n---\n\n".join(top_chunks)
            prompt = f"""
Answer the question based on the following policy content:

Context:
{context}

Question: {query}
Answer:
            """

        if model_choice == "Gemma":
            answer = get_gemma_response(prompt)
        else:
            answer = get_mistral_response(prompt)

        st.markdown("### 💬 Answer")
        st.write(answer)

        st.markdown("---")
        st.markdown("##### 📚 Top Sources")
        for source in top_sources:
            st.write(f"- {source}")
