import streamlit as st

st.set_page_config(page_title="Policy Chatbot", page_icon="📘", layout="wide")

import pandas as pd
import pickle
import numpy as np
import io
import requests
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# === Load environment variables ===
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

# === Load policy documents from Google Drive ===
POLICY_DOCUMENTS_URL = "https://drive.google.com/file/d/1q6TjIiTOVVkwwIEJMOg-wAklYwSnDTWO/view?usp=sharing"

@st.cache_data
def load_pickle_from_drive(drive_url):
    file_id = drive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    response = requests.get(download_url)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "html" in content_type:
        raise ValueError("⚠️ Received HTML instead of a binary file. Check if the Google Drive file is publicly shareable and not restricted.")

    return pd.read_pickle(io.BytesIO(response.content))

with st.spinner("Loading documents..."):
    policy_documents = load_pickle_from_drive(POLICY_DOCUMENTS_URL)

# === LLM functions ===
def get_mistral_response(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistral-small",
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

# === Robust structured answer extractor ===
def extract_structured_answers(text):
    fields = {
        "LMI Savings Rate": ["What is the LMI Savings Rate?", "LMI Savings Rate"],
        "Utilities Participating": ["Utilities Participating?", "Utilities Participating"],
        "LMI Verification Types": ["What are the LMI Verification Types?", "LMI Verification Types"]
    }
    results = {}
    for label, variants in fields.items():
        for v in variants:
            pattern = rf"{re.escape(v)}[:\-–]?\s*(.*?)(?=\n\S|$)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                results[label] = match.group(1).strip()
                break
        if label not in results:
            results[label] = ""
    return results

# === UI ===
st.title("📘 NM Renewables Policy Chatbot")

areas = sorted(policy_documents["AREA"].dropna().unique())
area_choice = st.selectbox("🔍 Filter by renewable energy type:", ["All Areas"] + areas)
model_choice = st.selectbox("Choose a model:", ["Gemma", "Mistral"])
query = st.text_input("Ask your question:")

# === Query logic ===
if query:
    with st.spinner("Thinking..."):
        query_embedding = model.encode(query).reshape(1, -1)

        # Filter docs
        if area_choice != "All Areas":
            filtered_docs = policy_documents[policy_documents["AREA"] == area_choice]
            if filtered_docs.empty:
                st.warning("No documents found for that area. Searching all instead.")
                filtered_docs = policy_documents
        else:
            filtered_docs = policy_documents

        doc_names = filtered_docs["DOC_NAME"].tolist()
        contents = filtered_docs["CONTENT"].tolist()
        embeddings = np.vstack(filtered_docs["EMBEDDING"].tolist())

        # Similarity
        sims = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:5]
        top_chunks = [contents[i] for i in top_indices]
        top_sources = [doc_names[i] for i in top_indices]
        context = "\n\n---\n\n".join(top_chunks)

        # Prompt
        if area_choice == "community_solar":
            structured_insights = (
                "Also provide clear and labeled answers to the following:\n"
                "1. What is the LMI Savings Rate?\n"
                "2. Utilities Participating?\n"
                "3. What are the LMI Verification Types?\n\n"
            )
        else:
            structured_insights = ""

        prompt = f"""
Answer the user's question below based on the following renewable energy policy documents.

{structured_insights}
Context:
{context}

User's Question: {query}
Answer:
        """

        # Get LLM answer
        if model_choice == "Gemma":
            answer = get_gemma_response(prompt)
        else:
            answer = get_mistral_response(prompt)

        # === Structured display
        if area_choice == "community_solar":
            st.markdown("### 📊 Community Solar Key Facts")
            structured = extract_structured_answers(answer)

            with st.expander("💸 LMI Savings Rate"):
                st.markdown(structured["LMI Savings Rate"] or "_Not found_")

            with st.expander("🔌 Utilities Participating"):
                st.markdown(structured["Utilities Participating"] or "_Not found_")

            with st.expander("📋 LMI Verification Types"):
                st.markdown(structured["LMI Verification Types"] or "_Not found_")

        # === General LLM Answer
        st.markdown("---")
        st.markdown("### 💬 Answer")
        st.write(answer)

        # === Sources
        st.markdown("##### 📚 Top Sources")
        for src in top_sources:
            st.write(f"- 📄 {src}")
