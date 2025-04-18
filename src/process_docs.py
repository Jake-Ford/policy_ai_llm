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

# Load env vars
load_dotenv()

model = SentenceTransformer("intfloat/e5-base-v2")

# Snowflake connection
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

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
                    "EMBEDDING": f"[{','.join(map(str, embedding))}]"
                })
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
    return rows

# === Process NY NEM documents ===
nem_rows = process_folder("../data/NEM", "NY_NEM")

# === Process MA SMART documents ===
ma_rows = process_folder("../data/MA_SMART", "MA_SMART")

# === Combine and upload ===
all_rows = nem_rows + ma_rows
df = pd.DataFrame(all_rows)
result = write_pandas(conn, df, "POLICY_DOCUMENTS")
print(f"✅ Upload result: {result}")

# === Web content (CENHUD) ===
def get_webpage_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

web_url = "https://www.cenhud.com/en/account-resources/rates/gas--electric-supply-prices/"
web_text = get_webpage_text(web_url)
web_chunks = wrap(web_text, 512)
web_embeddings = model.encode(web_chunks, show_progress_bar=True)

web_df = pd.DataFrame({
    "DOC_NAME": ["CENHUD_Web"] * len(web_chunks),
    "CONTENT": web_chunks,
    "EMBEDDING": web_embeddings.tolist()
})

write_pandas(
    conn,
    web_df,
    table_name="POLICY_DOCUMENTS",
    schema=os.getenv("SNOWFLAKE_SCHEMA"),
    quote_identifiers=False
)
