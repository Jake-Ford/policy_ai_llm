import streamlit as st
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv
from google import generativeai as genai
#import google.generativeai as genai


# --- Load environment variables ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

# --- Load vector store ---
with open("../plastic_doc_embeddings.pkl", "rb") as f:
    vectors, metadata = pickle.load(f)

# --- Load embedding model ---
embed_model = SentenceTransformer("intfloat/e5-base-v2")

# --- LLM API setup ---
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

# --- Streamlit UI ---
st.title("🌍 Plastic Policy Chatbot")

query = st.text_input("Ask a question about global plastic regulations:")
llm_choice = st.selectbox("Choose an LLM", ["Mistral", "Gemma"])

if query:
    with st.spinner("Thinking..."):
        query_embedding = embed_model.encode(query).reshape(1, -1)
        embedding_matrix = np.vstack(vectors)
        sims = cosine_similarity(query_embedding, embedding_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:5]

        top_chunks = [metadata[i]["chunk"] for i in top_indices]
        top_sources = [metadata[i]["source"] for i in top_indices]

        context = "\n\n---\n\n".join(top_chunks)

        prompt = f"""
You are a policy expert. Answer the following question using the provided plastic policy documents.

Context:
{context}

Question:
{query}

Answer:
        """

        if llm_choice == "Gemma":
            answer = get_gemma_response(prompt)
        else:
            answer = get_mistral_response(prompt)

        st.markdown("### 💬 Answer")
        st.write(answer)

        st.markdown("---")
        st.markdown("### 📄 Top Sources")
        for src in set(top_sources):
            st.write(f"- {src}")
