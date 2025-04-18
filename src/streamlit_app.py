import streamlit as st
import os
import numpy as np
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from openai import OpenAI
import requests
load_dotenv()
from google import generativeai as genai

import streamlit as st
import snowflake.connector
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import requests
import ast




#Connect to Snowflake
conn = snowflake.connector.connect(
    user=st.secrets["SNOWFLAKE_USER"],
    password=st.secrets["SNOWFLAKE_PASSWORD"],
    account=st.secrets["SNOWFLAKE_ACCOUNT"],
    warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
    database=st.secrets["SNOWFLAKE_DATABASE"],
    schema=st.secrets["SNOWFLAKE_SCHEMA"]
)

MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
GEMMA_API_KEY = st.secrets["GEMMA_API_KEY"]

# Load environment variables
#MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
#GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

#openai.api_key = os.getenv("OPENAI_API_KEY")

def get_mistral_response(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "mistral-small",  # Or mistral-medium / mistral-large if you have access
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }



    response = requests.post("https://api.mistral.ai/v1/chat/completions", json=body, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


genai.configure(api_key=os.getenv("GEMMA_API_KEY"))

def get_gemma_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")  
    response = model.generate_content(prompt)
    return response.text


# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
#model = SentenceTransformer('BAAI/bge-base-en-v1.5')
#model = SentenceTransformer('all-mpnet-base-v2')

model = SentenceTransformer("intfloat/e5-base-v2")



st.title("📘 NY Policy Chatbot")

query = st.text_input("Ask a question about either NY Sun Run or MA SMART programs:")
#program_filter = st.selectbox("Optional: Filter by program", options=["All", "NEM", "RC", "VDER"])

model_choice = st.selectbox("Choose a model:", ["Gemma", "Mistral"])
compare_mode = st.checkbox("Compare NY vs MA?", value=True)

if query:
    with st.spinner("Thinking..."):
        query_embedding = model.encode(query).reshape(1, -1)

        sql = "SELECT DOC_NAME, CONTENT, EMBEDDING FROM EMBEDDINGS.POLICY_DOCUMENTS"
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

        doc_names, contents, embeddings = [], [], []
        for doc_name, content, embedding in rows:
            try:
                embedding_list = ast.literal_eval(embedding)
                if isinstance(embedding_list, list) and len(embedding_list) == len(query_embedding[0]):
                    doc_names.append(doc_name)
                    contents.append(content)
                    embeddings.append(np.array(embedding_list))
            except Exception as e:
                print(f"❌ Skipping row due to error: {e}")

        if not embeddings:
            st.error("No valid embeddings found.")
        else:
            embedding_matrix = np.vstack(embeddings)
            sims = cosine_similarity(query_embedding, embedding_matrix)[0]
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
            for i in range(len(top_sources)):
                st.write(f"- {top_sources[i]}")



# if query:
#     with st.spinner("Thinking..."):
#         query_embedding = model.encode(query).reshape(1, -1)

#         sql = "SELECT DOC_NAME, CONTENT, EMBEDDING FROM EMBEDDINGS.POLICY_DOCUMENTS"

#         # Build SQL query
#         # # if program_filter == "All":
#         # #     sql = "SELECT DOC_NAME, CONTENT, EMBEDDING FROM PUBLIC.POLICY_DOCUMENTS"
#         # else:
#         #     sql = f"""
#         #         SELECT DOC_NAME, CONTENT, EMBEDDING
#         #         FROM PUBLIC.POLICY_DOCUMENTS
#         #         WHERE PROGRAM_AREA = '{program_filter}'
#         #     """

#         cursor = conn.cursor()
#         cursor.execute(sql)
#         rows = cursor.fetchall()

#         doc_names, contents, embeddings = [], [], []
#         for doc_name, content, embedding in rows:
#             try:
#                 embedding_list = ast.literal_eval(embedding)  # Safely convert from string to list
#                 if isinstance(embedding_list, list) and len(embedding_list) == len(query_embedding[0]):
#                     doc_names.append(doc_name)
#                     contents.append(content)
#                     embeddings.append(np.array(embedding_list))
#             except Exception as e:
#                 print(f"❌ Skipping row due to error: {e}")

#         if not embeddings:
#             st.error("No valid embeddings found.")
#         else:
#             # Calculate similarity
#             embedding_matrix = np.vstack(embeddings)
#             sims = cosine_similarity(query_embedding, embedding_matrix)[0]
#             top_indices = np.argsort(sims)[::-1][:3] #grabs the top three most relevant docs

#             top_chunks = [contents[i] for i in top_indices]
#             top_sources = [doc_names[i] for i in top_indices]
#             context = "\n\n---\n\n".join(top_chunks)

#             # Ask OpenAI
#             prompt = f"""
# Answer the question based on the following policy content:

# Context:
# {context}

# Question: {query}
# Answer:
#             """
#             # client = openai.OpenAI()
#             # response = client.chat.completions.create(
#             #     model="gpt-3.5-turbo",
#             #     messages=[{"role": "user", "content": prompt}]
#             # )
#            # answer = get_mistral_response(prompt)
#             if model_choice == "Gemma":
#                 answer = get_gemma_response(prompt)
#             else: 
#                 answer = get_mistral_response(prompt)
#             st.write(answer)

#             st.markdown("### 💬 Answer")
#            # st.write(response.choices[0].message["content"])
#             #st.write(response.choices[0].message.content)


#             st.markdown("---")
#             st.markdown("##### 📚 Top Sources")
#             for i in range(len(top_sources)):
#                 st.write(f"- {top_sources[i]}")


# # Now includes web content from CENHUD site automatically via process_upload_doc_embeddings.py