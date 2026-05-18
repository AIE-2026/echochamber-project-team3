# core/retriever.py
# ==================
# Loads a FAISS vector index for one agent and performs semantic search.
# This is the RAG (Retrieval-Augmented Generation) component.
#
# HOW RAG WORKS HERE:
#   When an agent responds, it first searches its own corpus for the
#   most similar comments to the stimulus. These are injected into the
#   prompt, grounding the agent's response in authentic community language.
#
# Students: you don't need to modify this file.
# To build the indexes, run: python scripts/build_vectorstore.py
# core/retriever.py
# =================
# Semantic retrieval over a FAISS vector store.
#
# This file does NOT use a generative LLM.
# It only does retrieval:
#
#   query text → query embedding → FAISS search → top-k fragments
#
# In C6, these fragments will be inserted into an LLM prompt.

from pathlib import Path
import argparse
import pickle

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path("assets/vectorstores")


class Retriever:
    """
    Retriever for one agent / one bubble.

    Expected files:
        assets/vectorstores/<agent_slug>/index.faiss
        assets/vectorstores/<agent_slug>/index.pkl
    """

    def __init__(self, agent_slug: str):
        self.agent_slug = agent_slug
        self.path = VECTORSTORE_DIR / agent_slug

        index_path = self.path / "index.faiss"
        metadata_path = self.path / "index.pkl"

        # TODO 1: Check if index_path exists.
        if not index_path.exists():
            raise FileNotFoundError(f"Nu s-a găsit indexul FAISS la calea: {index_path}")

        # TODO 2: Check if metadata_path exists.
        if not metadata_path.exists():
            raise FileNotFoundError(f"Nu s-au găsit metadatele la calea: {metadata_path}")

        # TODO 3: Load the FAISS index from index_path.
        self.index = faiss.read_index(str(index_path))

        # TODO 4: Load metadata from metadata_path using pickle.
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # TODO 5: Load the same SentenceTransformer model used in C5.
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for the top-k most similar fragments.

        Returns a list of dictionaries.
        Each result should contain:
            - original metadata fields
            - score
            - position
        """

        # TODO 6: Encode the query as an embedding.
        query_vector = self.model.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        ).astype("float32")

        # TODO 7: Search in the FAISS index.
        scores, positions = self.index.search(query_vector, k)

        results = []

        # TODO 8: Loop through scores[0] and positions[0].
        for rank, pos in enumerate(positions[0]):
            if pos == -1:
                continue
            
            item = dict(self.metadata[pos])
            item["score"] = float(scores[0][rank])
            item["position"] = int(pos)
            results.append(item)

        return results

    def format_for_prompt(self, chunks: list[dict]) -> str:
        """
        Format retrieved fragments as context.
        This will be useful in C6.
        """

        if not chunks:
            return "(Nu au fost găsite fragmente relevante.)"

        lines = []

        # TODO 9: Format each chunk
        for i, chunk in enumerate(chunks, start=1):
            score_formatted = round(chunk["score"], 3)
            lines.append(f"[Fragment {i} | score={score_formatted}]")
            lines.append(chunk["text"].strip())

        return "\n\n".join(lines)


def main():
    """
    Terminal test.

    Example:
        python -m core.retriever --agent anti_sistem --query "CCR a decis anularea alegerilor după suspiciuni privind influențe externe." --k 5
    """

    parser = argparse.ArgumentParser(
        description="Test semantic retrieval for one agent bubble."
    )

    parser.add_argument(
        "--agent",
        required=True,
        help="Agent slug, for example: anti_sistem"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Text used as semantic search query"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved fragments"
    )

    args = parser.parse_args()

    retriever = Retriever(args.agent)
    chunks = retriever.search(args.query, k=args.k)

    print("Agent:", args.agent)
    print("Interogare:", args.query)
    print("Vectori în index:", retriever.index.ntotal)
    print("Rezultate recuperate:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nRezultat {i}")
        print("Poziție:", chunk["position"])
        print("Scor:", round(chunk["score"], 3))

        if "agent" in chunk:
            print("Agent text:", chunk["agent"])

        if "source_channel" in chunk:
            print("Sursă:", chunk["source_channel"])

        if "video_title" in chunk:
            print("Video:", chunk["video_title"])

        print("Text:", chunk["text"][:500])


if __name__ == "__main__":
    main()