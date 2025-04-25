import os
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Config
CSV_PATH = "../data/plastic/plastic-source.csv"
DOC_FOLDER = "../data/plastic/plastics-policies"
#MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = "intfloat/e5-base-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
OUTFILE = "../plastic_doc_embeddings.pkl"

# Load filenames from CSV
df = pd.read_csv(CSV_PATH, nrows=100)

# Load model and chunker
embedder = SentenceTransformer(MODEL_NAME)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

embeddings = []
metadata = []

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

print(f"\nProcessing {len(df)} documents...\n")

for filename in df["filename"]:
    pdf_path = os.path.join(DOC_FOLDER, filename)
    if os.path.isfile(pdf_path) and pdf_path.endswith(".pdf"):
        try:
            print(f"📄 Processing: {filename}")
            raw_text = extract_pdf_text(pdf_path)
            chunks = splitter.split_text(raw_text)
            print(f"   ↳ Extracted {len(chunks)} chunks")

            if len(chunks) == 0:
                print(f"   ⚠️  No chunks found in: {filename}")

            chunk_embeddings = embedder.encode(chunks)
            embeddings.extend(chunk_embeddings)
            metadata.extend([{"source": filename, "chunk": chunk} for chunk in chunks])

        except Exception as e:
            print(f"   ❌ Failed to process {filename}: {e}")
    else:
        print(f"   ⚠️  Skipping (not found or not a PDF): {filename}")

# Save embeddings + metadata
print(f"\n✅ Saving {len(embeddings)} embeddings and {len(metadata)} metadata entries to {OUTFILE}")
with open(OUTFILE, "wb") as f:
    pickle.dump((embeddings, metadata), f)

print("✅ Done!")
