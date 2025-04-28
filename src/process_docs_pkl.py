import os
import uuid
import fitz  
import docx
import pandas as pd
import tiktoken
import numpy as np
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from textwrap import wrap
import pickle

# Load env vars
load_dotenv()

# === Load embedding model ===
model = SentenceTransformer("intfloat/e5-base-v2")

# === Connect to Snowflake ===
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

# === Helper Functions ===
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
                    "EMBEDDING": embedding  # Save as raw list, not string
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

# === Main Workflow ===
if __name__ == "__main__":
    # Process local documents
    nem_rows = process_folder("../data/NEM", "NY_NEM")
    ma_rows = process_folder("../data/MA_SMART", "MA_SMART")
    all_rows = nem_rows + ma_rows
    df = pd.DataFrame(all_rows)

    # Clean dataframe
    df = clean_embeddings(df)

    # Upload cleaned documents to Snowflake
    result = write_pandas(conn, df, "POLICY_DOCUMENTS")
    print(f"✅ Upload result: {result}")

    # Process CenHud webpage
    web_url = "https://www.cenhud.com/en/account-resources/rates/gas--electric-supply-prices/"
    web_text = get_webpage_text(web_url)
    web_chunks = wrap(web_text, 512)
    web_embeddings = model.encode(web_chunks, show_progress_bar=True)

    web_df = pd.DataFrame({
        "DOC_NAME": ["CENHUD_Web"] * len(web_chunks),
        "CONTENT": web_chunks,
        "EMBEDDING": web_embeddings.tolist()
    })

    # Clean webpage dataframe too
    web_df = clean_embeddings(web_df)

    # Upload CenHud content to Snowflake
    write_pandas(
        conn,
        web_df,
        table_name="POLICY_DOCUMENTS",
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        quote_identifiers=False
    )

    # Save clean local pickle versions
    with open("policy_documents.pkl", "wb") as f:
        pickle.dump(df, f)

    with open("cenhud_web.pkl", "wb") as f:
        pickle.dump(web_df, f)

    print("✅ Local pickles saved successfully!")
