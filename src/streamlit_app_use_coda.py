import streamlit as st

# === Set page config immediately ===
st.set_page_config(page_title="Policy Chatbot", page_icon="📘", layout="wide")

import pickle
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

# === API keys ===
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
GEMMA_API_KEY = st.secrets.get("GEMMA_API_KEY") or os.getenv("GEMMA_API_KEY")

# === LLM Functions ===
def get_mistral_response(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistral-small",  # Change if you have medium/large access
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

# === Load embedding model ===
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/e5-base-v2")

model = load_model()

# === Helper to load local pickle ===
@st.cache_data
def load_local_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# === Load document embeddings ===
policy_documents = load_local_pickle("data/outputs/policy_documents.pkl")

# === Extract available State-Programs from CODA documents ===
coda_programs = sorted(
    policy_documents[
        policy_documents["DOC_NAME"].str.startswith("CODA_")
    ]["DOC_NAME"]
    .str.replace("CODA_", "")
    .dropna()
    .unique()
)

# === Streamlit UI ===
st.title("📘 Policy AI Chatbot")

selected_program = st.selectbox(
    "Select a State - Program (optional to narrow your search):",
    options=["All Programs"] + coda_programs
)

model_choice = st.selectbox("Choose a model:", ["Gemma", "Mistral"])

query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        query_embedding = model.encode(query).reshape(1, -1)

        # === Apply Program Filter ===
        if selected_program != "All Programs":
            filtered_docs = policy_documents[
                policy_documents["DOC_NAME"].str.contains(selected_program)
            ]
            if filtered_docs.empty:
                st.warning("No documents found for that program. Searching all programs instead.")
                filtered_docs = policy_documents
        else:
            filtered_docs = policy_documents

        doc_names = filtered_docs["DOC_NAME"].tolist()
        contents = filtered_docs["CONTENT"].tolist()
        embeddings = np.vstack(filtered_docs["EMBEDDING"].tolist())

        # === Compute Similarities ===
        sims = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:5]

        top_chunks = [contents[i] for i in top_indices]
        top_sources = [doc_names[i] for i in top_indices]

        # === Build Context for the LLM ===
        context = "\n\n---\n\n".join(top_chunks)
        prompt = f"""
Answer the question based on the following policy content:

Context:
{context}

Question: {query}
Answer:
        """

        # === Send to selected LLM ===
        if model_choice == "Gemma":
            answer = get_gemma_response(prompt)
        else:
            answer = get_mistral_response(prompt)

        # === Display Answer ===
        st.markdown("### 💬 Answer")
        st.write(answer)

        # === Display Sources ===
        st.markdown("---")
        st.markdown("##### 📚 Top Sources")
        for src in top_sources:
            label = ""
            if src.startswith("CODA_"):
                label = "📋 [Coda Notes]"
            elif src.startswith("NY_") or src.startswith("MA_SMART"):
                label = "📄 [Document]"
            elif src.startswith("CENHUD_Web"):
                label = "🌐 [Website]"
            st.write(f"- {label} {src}")
