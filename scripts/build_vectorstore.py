# scripts/build_vectorstore.py
# =============================
# Builds a FAISS vector index for each agent from data/bubbles/.
# Each agent gets its own index in assets/vectorstores/<slug>/
#
# HOW TO USE:
#   python scripts/build_vectorstore.py
#
# Prerequisites:
#   - data/bubbles/<slug>.jsonl exists for each agent defined in roles.yaml
#   - Each line in the JSONL has a "text" field
#
# Output:
#   assets/vectorstores/<slug>/index.faiss
#   assets/vectorstores/<slug>/index.pkl

# ── Imports ────────────────────────────────────────────────────────────────
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Folder paths and model name ────────────────────────────────────────────
BUBBLES_DIR     = Path("data/bubbles")          # cleaned agent bubble files
VECTORSTORE_DIR = Path("assets/vectorstores")   # output: one folder per agent
MODEL_NAME      = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Load the embedding model ───────────────────────────────────────────────
# Loaded once and reused for all agents to avoid repeated downloads
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ── Process each bubble file ───────────────────────────────────────────────
# Each .jsonl file in data/bubbles/ corresponds to one agent slug
bubble_files = sorted(BUBBLES_DIR.glob("*.jsonl"))

if not bubble_files:
    print("No bubble files found in data/bubbles/. Run Tema 2 first.")
else:
    for bubble_path in bubble_files:
        slug = bubble_path.stem  # filename without .jsonl = agent slug
        print(f"\nProcessing agent: {slug}")

        # ── Read bubble file ───────────────────────────────────────────────
        # Each line is a JSON object with at least a "text" field
        records = []
        with open(bubble_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            print(f"  No records found — skipping {slug}")
            continue

        print(f"  Records loaded: {len(records)}")

        # ── Generate embeddings ────────────────────────────────────────────
        # Each text is encoded into a dense vector using the multilingual model
        texts = [r["text"] for r in records]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        # ── Create the FAISS index ─────────────────────────────────────────
        # IndexFlatIP = inner product (cosine similarity with normalized vectors)
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"  Vectors in index: {index.ntotal}")

        # ── Save index.faiss and index.pkl ─────────────────────────────────
        out_dir = VECTORSTORE_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(out_dir / "index.faiss"))

        with open(out_dir / "index.pkl", "wb") as f:
            pickle.dump(records, f)

        # ── Print progress ─────────────────────────────────────────────────
        print(f"  Saved: {out_dir / 'index.faiss'}")
        print(f"  Saved: {out_dir / 'index.pkl'}")

print("\nDone. All vector stores built.")
