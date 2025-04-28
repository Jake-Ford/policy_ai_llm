import os
import uuid
import fitz
import docx
import pandas as pd
import tiktoken
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from textwrap import wrap
import pickle
from bs4 import BeautifulSoup

# === Load model ===
model = SentenceTransformer("intfloat/e5-base-v2")

# === Helper functions ===
def extract_text(filepath):
    if filepath.endswith(".pdf"):
        return "\n".join([page.get_text() for page in fitz.open(filepath)])
    elif filepath.endswith(".docx"):
        return "\n".join([p.text for p in docx.Document(filepath).paragraphs])
    return ""

def chunk_text(text, max_tokens=800):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    while tokens:
        chunk_tokens = tokens[:max_tokens]
        chunk = enc.decode(chunk_tokens)
        chunks.append(chunk)
        tokens = tokens[len(chunk_tokens):]
    return chunks

def get_embedding(text):
    return model.encode(f"passage: {text.strip()}").tolist()

def process_folder(folder_path, tag):
    rows = []
    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.pdf', '.docx')):
            continue
        full_path = os.path.join(folder_path, file)
        try:
            raw_text = extract_text(full_path)
            chunks = chunk_text(raw_text)
            for chunk in chunks:
                embedding = get_embedding(chunk)
                rows.append({
                    "DOC_ID": str(uuid.uuid4()),
                    "DOC_NAME": f"{tag}_{file}",
                    "FILE_TYPE": file.split('.')[-1],
                    "CONTENT": chunk,
                    "EMBEDDING": embedding
                })
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
    return rows

def clean_embeddings(df):
    good_rows = []
    bad_rows = []
    for idx, emb in enumerate(df["EMBEDDING"]):
        if isinstance(emb, (list, np.ndarray)) and len(emb) == 768:
            good_rows.append(idx)
        else:
            bad_rows.append(idx)
    print(f"✅ Good embeddings: {len(good_rows)}")
    print(f"⚠️ Bad embeddings removed: {len(bad_rows)}")
    return df.iloc[good_rows].reset_index(drop=True)

def get_webpage_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# === Main Script ===
if __name__ == "__main__":
    # === Process local PDFs and DOCXs ===
    nem_rows = process_folder("../data/NEM", "NY_NEM")
    ma_rows = process_folder("../data/MA_SMART", "MA_SMART")
    all_rows = nem_rows + ma_rows
    df_docs = pd.DataFrame(all_rows)
    df_docs = clean_embeddings(df_docs)

    # === Process CenHud webpage ===
    web_url = "https://www.cenhud.com/en/account-resources/rates/gas--electric-supply-prices/"
    web_text = get_webpage_text(web_url)
    web_chunks = wrap(web_text, 512)
    web_embeddings = model.encode(web_chunks, show_progress_bar=True)

    df_web = pd.DataFrame({
        "DOC_NAME": ["CENHUD_Web"] * len(web_chunks),
        "CONTENT": web_chunks,
        "EMBEDDING": web_embeddings.tolist()
    })
    df_web = clean_embeddings(df_web)

    # === Process Coda Program Matrix ===
    coda_df = pd.read_csv("../data/coda/program_matrix_data.csv")

    coda_rows = []
    for idx, row in coda_df.iterrows():
        content_parts = []
        for col_name in coda_df.columns:
            col_value = row.get(col_name, '')
            if pd.notna(col_value):
                content_parts.append(f"{col_name}: {col_value}")
        
        content = "\n".join(content_parts)

        embedding = get_embedding(content)
        coda_rows.append({
            "DOC_ID": str(uuid.uuid4()),
            "DOC_NAME": f"CODA_{row.get('State - Program (LMI Adder)', 'Unknown')}",
            "CONTENT": content,
            "EMBEDDING": embedding
        })

    df_coda = pd.DataFrame(coda_rows)
    df_coda = clean_embeddings(df_coda)

    # === Merge everything ===
    all_df = pd.concat([df_docs, df_web, df_coda], ignore_index=True)
    all_df = clean_embeddings(all_df)

    # === Save Pickles ===
    os.makedirs("../data/outputs", exist_ok=True)

    with open("../data/outputs/policy_documents.pkl", "wb") as f:
        pickle.dump(all_df, f)

    print("✅ Saved combined policy_documents.pkl!")
