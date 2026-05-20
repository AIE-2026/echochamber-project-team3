"""
EchoChamber Studio — app.py
===========================
Structure:
  1. IMPORTS & SETUP
  2. PROVIDERS AND API KEYS
  3. MODEL CALL
  4. APP LOGIC
  5. AGENT RAG LOGIC
  6. GRADIO UI
  7. LAUNCH
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS & SETUP
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import yaml
import html
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, AuthenticationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # asigură că căile relative din core/ funcționează corect

from core.config import (
    PROVIDER_PRINCIPAL,
    MODEL_PRINCIPAL,
    PROVIDER_FALLBACK,
    MODEL_FALLBACK,
    TEMPERATURE,
)

from core.agent import generate_agent_response
from core.graph import run_thread

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROVIDERS AND API KEYS
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv(PROJECT_ROOT / ".env")

BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1"
}

API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY")
}


def make_client(provider):
    """Creează clientul API pentru providerul ales."""
    return OpenAI(
        api_key=API_KEYS[provider],
        base_url=BASE_URLS[provider]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL CALL
# ─────────────────────────────────────────────────────────────────────────────

def ask(provider, model, prompt, system=None, temperature=0.7, json_schema=None):
    """Trimite un prompt la model. Poate returna text simplu sau JSON structurat."""

    client = make_client(provider)
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    extra_args = {}

    if json_schema:
        extra_args["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema
        }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **extra_args
        )

        text = response.choices[0].message.content.strip()

        if json_schema:
            return json.loads(text)

        return text

    except RateLimitError:
        return f"[Eroare: quota/rate limit pentru modelul {model}.]"

    except AuthenticationError:
        return "[Eroare: API key invalidă sau lipsă. Verifică .env.]"

    except APIError as e:
        return f"[Eroare API: {e}]"

    except Exception as e:
        return f"[Eroare: {type(e).__name__} — {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# 4. APP LOGIC (Chat simplu)
# ─────────────────────────────────────────────────────────────────────────────

def chat(prompt):
    """Trimite promptul la modelul principal. Dacă apare eroare, încearcă fallback-ul."""

    if not prompt.strip():
        return "Scrie un prompt mai întâi."

    answer = ask(
        provider=PROVIDER_PRINCIPAL,
        model=MODEL_PRINCIPAL,
        prompt=prompt,
        temperature=TEMPERATURE
    )

    if isinstance(answer, str) and answer.startswith("[Eroare"):
        answer = ask(
            provider=PROVIDER_FALLBACK,
            model=MODEL_FALLBACK,
            prompt=prompt,
            temperature=TEMPERATURE
        )

    return answer


# ─────────────────────────────────────────────────────────────────────────────
# 5. AGENT RAG LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def load_agent_choices():
    """Citește agenții disponibili din assets/roles/roles.yaml."""
    roles_path = PROJECT_ROOT / "assets" / "roles" / "roles.yaml"

    if not roles_path.exists():
        return []

    with open(roles_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    roles = data["agents"] if "agents" in data else data

    return list(roles.keys())


def rag_agent_response(agent_slug, stimulus, provider, k):
    """Apelează generate_agent_response() și returnează răspunsul și contextul RAG."""

    if not agent_slug:
        return "Nu există agenți în assets/roles/roles.yaml.", ""

    if not stimulus.strip():
        return "Scrie un text politic pentru agent.", ""

    try:
        result = generate_agent_response(
            agent_slug=agent_slug,
            stimulus=stimulus,
            provider=provider,
            k=int(k),
            temperature=0.3,
            roles_path="assets/roles/roles.yaml",
        )

        return result["response"], result["rag_text"]

    except Exception as e:
        return f"[Eroare Agent RAG: {type(e).__name__} — {e}]", ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. MULTI-AGENT THREAD LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def render_thread_html(messages):
    """Transformă lista de mesaje în carduri HTML pentru afișare."""
    cards = []
    for msg in messages:
        agent  = html.escape(str(msg.get("agent", "")))
        handle = html.escape(str(msg.get("handle", msg.get("slug", ""))))
        text   = html.escape(str(msg.get("text", "")))
        turn   = msg.get("turn", "")
        cards.append(f"""
        <div style='border-left:3px solid #e05a35; padding:.7rem 1rem; margin:.3rem 0; background:#16161a'>
            <div style='font-size:.75rem; color:#e05a35; text-transform:uppercase'>{agent}</div>
            <div style='font-size:.7rem; color:#888'>{handle} · #{turn}</div>
            <p style='color:#c0bcb6; margin:.4rem 0 0'>{text}</p>
        </div>
        """)
    return "\n".join(cards)


def run_multi_agent_thread(stimulus, provider, total_turns,
                           use_conspirationist, use_intelectual_critic, use_pro_european,
                           use_anti_sistem, use_anti_suveranist, use_personalist_salvator):
    """Rulează un thread multi-agent și returnează HTML."""
    active_slugs = []
    if use_conspirationist:
        active_slugs.append("conspirationist")
    if use_intelectual_critic:
        active_slugs.append("intelectual_critic")
    if use_pro_european:
        active_slugs.append("pro_european")
    if use_anti_sistem:
        active_slugs.append("anti_sistem")
    if use_anti_suveranist:
        active_slugs.append("anti_suveranist")
    if use_personalist_salvator:
        active_slugs.append("personalist_salvator")

    if not stimulus.strip():
        return "Scrie un text politic mai întâi."
    if not active_slugs:
        return "Selectează cel puțin un agent."

    try:
        messages = run_thread(
            stimulus=stimulus,
            active_slugs=active_slugs,
            total_turns=int(total_turns),
            provider=provider,
            k=3,
        )
        return render_thread_html(messages)
    except Exception as e:
        return f"[Eroare Multi-agent Thread: {type(e).__name__} — {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

agent_choices = load_agent_choices()

with gr.Blocks(title="EchoChamber") as demo:
    gr.Markdown("# EchoChamber")
    gr.Markdown("Aplicație minimă pentru testarea modelelor și a agenților RAG.")

    with gr.Tab("Chat simplu"):
        prompt_box = gr.Textbox(
            label="Prompt",
            value="Explică în 2 propoziții ce este un LLM.",
            lines=4
        )

        chat_button = gr.Button("Trimite")

        chat_output = gr.Textbox(
            label="Răspuns",
            lines=8
        )

        chat_button.click(
            fn=chat,
            inputs=prompt_box,
            outputs=chat_output
        )

    with gr.Tab("Agent RAG"):
        agent_dropdown = gr.Dropdown(
            choices=agent_choices,
            value=agent_choices[0] if agent_choices else None,
            label="Agent"
        )

        provider_dropdown = gr.Dropdown(
            choices=["gemini", "deepseek"],
            value="gemini",
            label="Provider"
        )

        stimulus_box = gr.Textbox(
            label="Text politic nou",
            value="CCR a decis anularea alegerilor după suspiciuni privind influențe externe.",
            lines=4
        )

        k_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Număr fragmente recuperate"
        )

        agent_button = gr.Button("Generează răspuns RAG")

        agent_response_box = gr.Textbox(
            label="Răspuns agent",
            lines=8
        )

        context_box = gr.Textbox(
            label="Context recuperat",
            lines=12
        )

        agent_button.click(
            fn=rag_agent_response,
            inputs=[agent_dropdown, stimulus_box, provider_dropdown, k_slider],
            outputs=[agent_response_box, context_box]
        )


    with gr.Tab("Multi-agent thread"):
        thread_stimulus = gr.Textbox(
            label="Text politic",
            value="CCR a decis anularea alegerilor după suspiciuni privind influențe externe.",
            lines=4
        )
        thread_provider = gr.Dropdown(
            choices=["gemini", "deepseek"],
            value="gemini",
            label="Provider"
        )
        thread_turns = gr.Slider(
            minimum=2, maximum=8, value=4, step=1,
            label="Număr intervenții"
        )
        use_conspirationist      = gr.Checkbox(value=True,  label="Conspiraționist")
        use_intelectual_critic   = gr.Checkbox(value=True,  label="Intelectual-critic")
        use_pro_european         = gr.Checkbox(value=True,  label="Pro-european")
        use_anti_sistem          = gr.Checkbox(value=False, label="Anti-sistem")
        use_anti_suveranist      = gr.Checkbox(value=False, label="Anti-suveranist")
        use_personalist_salvator = gr.Checkbox(value=False, label="Personalist-salvator")

        thread_button = gr.Button("Pornește thread")
        thread_output = gr.HTML(label="Thread generat")

        thread_button.click(
            fn=run_multi_agent_thread,
            inputs=[thread_stimulus, thread_provider, thread_turns,
                    use_conspirationist, use_intelectual_critic, use_pro_european,
                    use_anti_sistem, use_anti_suveranist, use_personalist_salvator],
            outputs=thread_output
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. LAUNCH
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
 

 