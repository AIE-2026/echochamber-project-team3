# core/agent.py
# ==============
# The Agent class. Each agent:
#   1. Reads its persona from roles.yaml (via config.py)
#   2. Retrieves similar comments from its corpus (RAG via retriever.py)
#   3. Calls the LLM with its system prompt + retrieved context + stimulus
#
# Students: you don't need to modify this file.
# Your work is in assets/roles/roles.yaml — that's where you define the persona.

from pathlib import Path
import os
import pickle

import faiss
import yaml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

load_dotenv()

# ── constante ──────────────────────────────────────────────────────────────
MODEL_NAME      = "paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path("assets/vectorstores")
ROLES_FILE      = Path("assets/roles/roles.yaml")
ROLES_DIR       = Path("assets/roles")

# cache model embeddings (încărcat o singură dată)
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


# ── încărcare rol ───────────────────────────────────────────────────────────
def load_role(agent_slug: str) -> dict:
    """
    Caută rolul agentului în:
      1. assets/roles/roles.yaml  (key: agents.<slug>)
      2. assets/roles/role_XX.yaml (key: <slug>)
    """
    # 1. roles.yaml
    if ROLES_FILE.exists():
        with open(ROLES_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and isinstance(data.get("agents"), dict):
            if agent_slug in data["agents"]:
                role = dict(data["agents"][agent_slug])
                role.setdefault("slug", agent_slug)
                return role

    # 2. fișiere individuale role_XX.yaml
    for role_file in sorted(ROLES_DIR.glob("role_*.yaml")):
        with open(role_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and agent_slug in data:
            role = dict(data[agent_slug])
            role.setdefault("slug", agent_slug)
            return role

    raise ValueError(
        f"Rolul agentului '{agent_slug}' nu a fost găsit în roles.yaml "
        f"sau în fișierele individuale din assets/roles/."
    )


# ── retrieval ───────────────────────────────────────────────────────────────
def retrieve_context(agent_slug: str, query: str, k: int = 5) -> str:
    """
    Caută cele mai similare k fragmente din vectorstore-ul agentului.
    Returnează un string formatat pentru prompt.
    """
    index_path    = VECTORSTORE_DIR / agent_slug / "index.faiss"
    metadata_path = VECTORSTORE_DIR / agent_slug / "index.pkl"

    if not index_path.exists():
        return "(Nu există vectorstore pentru acest agent.)"

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    model        = _get_embedding_model()
    query_vector = model.encode(
        [query], normalize_embeddings=True
    ).astype("float32")

    scores, positions = index.search(query_vector, k)

    parts = []
    for i, (score, pos) in enumerate(zip(scores[0], positions[0]), start=1):
        if pos == -1:
            continue
        item  = metadata[pos]
        parts.append(
            f"[Fragment {i} | score={round(float(score), 3)} | "
            f"source={item.get('source_channel', '')}]\n"
            f"{item.get('text', '').strip()}"
        )

    return "\n\n".join(parts) if parts else "(Nu au fost găsite fragmente relevante.)"


# ── LLM factory ─────────────────────────────────────────────────────────────
def make_llm(provider: str = "gemini", temperature: float = 0.3) -> ChatOpenAI:
    """
    Returnează un ChatOpenAI configurat pentru provider-ul ales.
    Provideri suportați: 'gemini', 'deepseek'.
    """
    if provider == "gemini":
        return ChatOpenAI(
            model      = "gemini-2.5-flash",
            api_key    = os.getenv("GEMINI_API_KEY"),
            base_url   = "https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature= temperature,
        )
    elif provider == "deepseek":
        return ChatOpenAI(
            model      = "deepseek-chat",
            api_key    = os.getenv("DEEPSEEK_API_KEY"),
            base_url   = "https://api.deepseek.com/v1",
            temperature= temperature,
        )
    else:
        raise ValueError(f"Provider necunoscut: {provider}. Folosește 'gemini' sau 'deepseek'.")


# ── funcția principală ───────────────────────────────────────────────────────
def generate_agent_response(
    agent_slug: str,
    stimulus:   str,
    provider:   str = "gemini",
    k:          int = 5,
) -> dict:
    """
    Generează un răspuns RAG pentru agentul indicat.

    Returnează un dicționar cu:
        agent_name  — numele afișabil al agentului
        slug        — slug-ul agentului
        stimulus    — inputul politic primit
        response    — răspunsul generat de LLM
        rag_text    — fragmentele recuperate din FAISS
    """
    role     = load_role(agent_slug)
    rag_text = retrieve_context(agent_slug, stimulus, k=k)

    prompt = (
        f"{role.get('system', '')}\n\n"
        f"[STIMULUS]\n{stimulus}\n\n"
        f"[COMENTARII SIMILARE]\n{rag_text}"
    )

    llm      = make_llm(provider=provider, temperature=0.3)
    response = llm.invoke(prompt)

    return {
        "agent_name": role.get("name", agent_slug),
        "slug":       agent_slug,
        "stimulus":   stimulus,
        "response":   response.content,
        "rag_text":   rag_text,
    }


# ── terminal test ────────────────────────────────────────────────────────────
# Run: python -m core.agent --agent <slug> --text "<stimulus>" --provider gemini --k 5
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG agent from terminal.")
    parser.add_argument("--agent",    required=True, help="Agent slug (ex: intelectual_critic)")
    parser.add_argument("--text",     required=True, help="Stimulus text")
    parser.add_argument("--provider", default="gemini", help="LLM provider: gemini / deepseek")
    parser.add_argument("--k",        type=int, default=5, help="Number of retrieved fragments")
    args = parser.parse_args()

    result = generate_agent_response(
        agent_slug=args.agent,
        stimulus=args.text,
        provider=args.provider,
        k=args.k,
    )

    print(f"\nAgent:    {result['agent_name']}")
    print(f"Stimulus: {result['stimulus']}")
    print(f"\n--- Context RAG ---\n{result['rag_text']}")
    print(f"\n--- Răspuns agent ---\n{result['response']}")
