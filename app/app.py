"""
EchoChamber Studio — app.py
===========================
Structure:
  1. IMPORTS & SETUP
  2. PROVIDERS AND API KEYS
  3. MODEL CALL
  4. NEWS LOADING
  5. CHAT & SUMMARY LOGIC
  6. AGENT RAG LOGIC
  7. ALL AGENTS LOGIC
  8. DEBATE LOGIC
  9. GRADIO UI
 10. LAUNCH
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS & SETUP
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import html
import yaml
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, AuthenticationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from core.agent import generate_agent_response
from core.graph import run_thread

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROVIDERS AND API KEYS
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv(PROJECT_ROOT / ".env")

BASE_URLS = {
    "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter":  "https://openrouter.ai/api/v1",
    "deepseek":    "https://api.deepseek.com/v1",
}

API_KEYS = {
    "gemini":     os.getenv("GEMINI_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "deepseek":   os.getenv("DEEPSEEK_API_KEY"),
}

DEFAULT_MODELS = {
    "gemini":     "gemini-2.5-flash-lite",
    "openrouter": "openrouter/free",
    "deepseek":   "deepseek-chat",
}

# Retrieval constant — never exposed in the UI
DEFAULT_K = 5


def make_client(provider: str) -> OpenAI:
    return OpenAI(
        api_key=API_KEYS.get(provider),
        base_url=BASE_URLS.get(provider),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL CALL
# ─────────────────────────────────────────────────────────────────────────────

def ask(provider: str, model: str, prompt: str,
        system: str = None, temperature: float = 0.7) -> str:
    """Call the LLM. Returns text or an error string."""
    client = make_client(provider)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
        timeout=60,
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError:
        return f"[Eroare: quota/rate limit pentru {model}.]"
    except AuthenticationError:
        return "[Eroare: API key invalidă sau lipsă. Verifică .env.]"
    except APIError as e:
        return f"[Eroare API: {e}]"
    except Exception as e:
        return f"[Eroare: {type(e).__name__} — {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# 4. NEWS LOADING
# ─────────────────────────────────────────────────────────────────────────────
def smart_preview(text: str, limit: int = 650) -> str:
    """Taie preview-ul fără să rupă cuvintele."""
    text = " ".join(str(text).split())

    if len(text) <= limit:
        return text

    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "..."

def load_news(url: str):
    """
    Fetch and extract the title + body of a news article from a URL.
    Returns (preview_text, full_article_text, status_message).
    """
    if not url or not url.strip():
        return "Introdu un URL mai întâi.", "", "Nicio știre încărcată."

    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url.strip(), headers=headers, timeout=6)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Fără titlu"

        # Body — try article tag first, then paragraphs
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        body = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)

        if not body:
            return (
                "Nu s-a putut extrage textul automat. Poți lipi textul manual în tab.",
                "",
                "⚠️ Extragere eșuată — introdu textul manual.",
            )

        full_text = f"{title}\n\n{body}"
        preview = f"**{title}**\n\n{smart_preview(body, 650)}"
        status = f"✅ Știre încărcată: {title[:60]}{'…' if len(title) > 60 else ''}"

        return preview, full_text, status

    except Exception as e:
        return (
            f"Eroare la încărcare: {type(e).__name__} — {e}",
            "",
            "⚠️ Eroare la încărcare URL.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHAT & SUMMARY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def chat_respond(history, user_message: str, article_text: str,
                 provider: str, model: str, temperature: float):
    """Respond to the user in the chatbot, optionally using the loaded article."""

    history = history or []

    if not user_message or not user_message.strip():
        return history, ""

    # Curățăm istoricul vechi, în caz că browserul are tuple-uri din rulări anterioare
    clean_history = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            clean_history.append(msg)

    if article_text and article_text.strip():
        system = (
            "Ești un asistent care ajută utilizatorul să analizeze un articol de știri. "
            "Răspunde concis și la obiect, referindu-te la articolul de mai jos când este relevant.\n\n"
            f"ARTICOL:\n{article_text[:3000]}"
        )
    else:
        system = "Ești un asistent util. Răspunde concis și clar."

    reply = ask(provider, model, user_message, system=system, temperature=temperature)

    clean_history.append({
        "role": "user",
        "content": user_message
    })

    clean_history.append({
        "role": "assistant",
        "content": reply
    })

    return clean_history, ""

def summarize_article(article_text: str, provider: str, model: str, temperature: float):
    """Summarize the loaded article in 4 concise points."""
    if not article_text or not article_text.strip():
        return "Nicio știre încărcată. Folosește sidebar-ul stâng pentru a încărca un articol."

    prompt = (
        "Rezumă articolul de mai jos în 4 idei principale, scurt și clar, în română. "
        "Nu scrie introducere.\n\n"
        f"ARTICOL:\n{article_text[:2500]}"
    )

    return ask(provider, model, prompt, temperature=0.2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. AGENT RAG LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def load_roles_data():
    """Read available agents with metadata from assets/roles/roles.yaml."""
    roles_path = PROJECT_ROOT / "assets" / "roles" / "roles.yaml"
    if not roles_path.exists():
        return {}

    with open(roles_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("agents", data)


def load_agent_choices():
    """Read available agent slugs from assets/roles/roles.yaml."""
    return list(load_roles_data().keys())


def agent_respond(agent_slug: str, source: str, manual_text: str,
                  article_text: str, provider: str, model: str, temperature: float):
    """Generate one agent response using RAG."""
    if not agent_slug:
        return "Niciun agent selectat."

    stimulus = article_text if source == "Știre încărcată" else manual_text
    if not stimulus or not stimulus.strip():
        return "Textul sursă este gol. Încarcă o știre sau introdu text manual."

    try:
        result = generate_agent_response(
            agent_slug=agent_slug,
            stimulus=stimulus,
            provider=provider,
            k=DEFAULT_K,
            temperature=temperature,
            roles_path="assets/roles/roles.yaml",
        )
        return result["response"]
    except Exception as e:
        return f"[Eroare Agent: {type(e).__name__} — {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# 7. ALL AGENTS LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def all_agents_respond(source: str, manual_text: str, article_text: str,
                       provider: str, model: str, temperature: float):
    """Run all agents from roles.yaml and return their responses as HTML cards."""
    agent_choices = load_agent_choices()
    if not agent_choices:
        return "<p style='color:#e05a35'>Nu există agenți în assets/roles/roles.yaml.</p>"

    stimulus = article_text if source == "Știre încărcată" else manual_text
    if not stimulus or not stimulus.strip():
        return "<p style='color:#e05a35'>Textul sursă este gol. Încarcă o știre sau introdu text manual.</p>"

    cards = []
    for slug in agent_choices:
        try:
            result = generate_agent_response(
                agent_slug=slug,
                stimulus=stimulus,
                provider=provider,
                k=DEFAULT_K,
                temperature=temperature,
                roles_path="assets/roles/roles.yaml",
            )
            response_text = html.escape(result["response"])

            role = roles_data.get(slug, {})
            label = html.escape(role.get("name", slug.replace("_", " ").title()))
            emoji = html.escape(role.get("emoji", ""))
            color = html.escape(role.get("color", "#e05a35"))
        except Exception as e:
            response_text = html.escape(f"[Eroare: {type(e).__name__} — {e}]")
            label = html.escape(slug)

        cards.append(f"""
        <div style="border-left:5px solid {color}; padding:1rem 1.2rem; margin:.7rem 0;
                    background:#16161a; border-radius:10px;">
            <div style="font-size:.8rem; color:{color}; text-transform:uppercase;
                        letter-spacing:.08em; font-weight:800; margin-bottom:.5rem">
                {emoji} {label}
            </div>
            <p style="color:#d6d3d1; margin:0; line-height:1.65">{response_text}</p>
        </div>
        """)
    return "\n".join(cards)


# ─────────────────────────────────────────────────────────────────────────────
# 8. DEBATE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def clean_agent_text(text: str) -> str:
    """Remove direct @mentions from the beginning of generated comments."""
    text = str(text).strip()
    text = re.sub(r"^@\S+[:,]?\s*", "", text)
    return text

def render_debate_html(messages):
    """Render debate messages as styled HTML cards."""
    if not messages:
        return "<p style='color:#888'>Nicio intervenție generată.</p>"

    cards = []

    for msg in messages:
        slug = str(msg.get("slug", msg.get("handle", "")))
        text = html.escape(clean_agent_text(msg.get("text", "")))
        turn = msg.get("turn", "")

        role = roles_data.get(slug, {})
        color = html.escape(role.get("color", "#e05a35"))
        emoji = html.escape(role.get("emoji", ""))
        name = html.escape(role.get("name", msg.get("agent", slug)))

        cards.append(f"""
        <div style="border-left:5px solid {color}; padding:.9rem 1.1rem; margin:.7rem 0;
                    background:#16161a; border-radius:10px;">
            <div style="display:flex; gap:.5rem; align-items:center; margin-bottom:.45rem">
                <span style="font-size:.85rem; color:{color}; text-transform:uppercase;
                             letter-spacing:.06em; font-weight:800">
                    {emoji} {name}
                </span>
                <span style="font-size:.75rem; color:#777">· tura {turn}</span>
            </div>
            <p style="color:#d6d3d1; margin:0; line-height:1.65">{text}</p>
        </div>
        """)

    return "\n".join(cards)


def run_debate(selected_agents, turns: int, source: str, manual_text: str,
               article_text: str, provider: str, model: str, temperature: float):
    """Run a multi-agent debate thread."""
    if not selected_agents or len(selected_agents) < 2:
        return "<p style='color:#e05a35'>Selectează cel puțin 2 agenți pentru dezbatere.</p>"

    stimulus = article_text if source == "Știre încărcată" else manual_text
    if not stimulus or not stimulus.strip():
        return "<p style='color:#e05a35'>Textul sursă este gol. Încarcă o știre sau introdu text manual.</p>"

    try:
        messages = run_thread(
            stimulus=stimulus,
            active_slugs=selected_agents,
            total_turns=int(turns),
            provider=provider,
            k=DEFAULT_K,
        )
        return render_debate_html(messages)
    except Exception as e:
        return f"<p style='color:#e05a35'>[Eroare Dezbatere: {type(e).__name__} — {e}]</p>"


# ─────────────────────────────────────────────────────────────────────────────
# 9. GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

agent_choices = load_agent_choices()
roles_data = load_roles_data()

DARK_CSS = """
.gradio-container { background: #0e0e12 !important; }
footer { display: none !important; }
.sidebar-col { border-right: 1px solid #1e1e26; padding-right: 1rem; }
"""

with gr.Blocks(
    title="EchoChamber Studio",
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
    ),
    css=DARK_CSS,
) as demo:

    # ── Shared state ─────────────────────────────────────────────────────────
    article_state = gr.State("")

    # ── Header ───────────────────────────────────────────────────────────────
    gr.Markdown(
        "# 📡 EchoChamber Studio\n"
        "*Analizează știri. Simulează voci discursive. Explorează dezbateri.*"
    )

    with gr.Row():
        # ── LEFT SIDEBAR ─────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="sidebar-col"):
            gr.Markdown("### ⚙️ Configurare")

            provider_dd = gr.Dropdown(
                choices=["gemini", "openrouter", "deepseek"],
                value="gemini",
                label="Provider",
            )
            model_txt = gr.Textbox(
                label="Model",
                value="gemini-2.5-flash-lite",
                placeholder="ex: gemini-2.5-flash",
            )
            temperature_sl = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.3, step=0.05,
                label="Temperatură",
            )

            gr.Markdown("---")
            gr.Markdown("### 📰 Încarcă știre")

            news_url = gr.Textbox(
                label="URL știre",
                placeholder="https://…",
            )
            load_btn = gr.Button("🔗 Încarcă știrea", variant="primary")
            news_status = gr.Markdown("*Nicio știre încărcată.*")
            news_preview = gr.Markdown(label="Previzualizare", visible=False)

            # Update model textbox when provider changes
            def update_model(provider):
                return DEFAULT_MODELS.get(provider, "")

            provider_dd.change(fn=update_model, inputs=provider_dd, outputs=model_txt)

            # Load news handler
            def on_load_news(url):
                preview, full_text, status = load_news(url)

                if full_text:
                    return full_text, status, gr.update(value=preview, visible=True)

                return "", status, gr.update(value=preview, visible=True)
        
            load_btn.click(
                fn=on_load_news,
                inputs=news_url,
                outputs=[article_state, news_status, news_preview],
            )

        # ── MAIN TABS ─────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():

                # ── TAB: CHAT ─────────────────────────────────────────────────
                with gr.Tab("💬 Chat"):
                    gr.Markdown(
                        "Pune întrebări sau discută pe marginea știrii încărcate. "
                        "Dacă nu e nicio știre, modelul răspunde liber."
                    )
                    chatbot = gr.Chatbot(
                        height=400, 
                        label="Conversație"
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Scrie o întrebare sau un prompt…",
                            label="",
                            scale=4,
                            container=False,
                        )
                        chat_send_btn = gr.Button("Trimite", scale=1)

                    summarize_btn = gr.Button("📋 Rezumă știrea", variant="secondary")
                    summary_out = gr.Textbox(
                        label="Rezumat",
                        lines=6,
                        interactive=False,
                        placeholder="Rezumatul știrii va apărea aici…",
                    )

                    # Chat
                    chat_send_btn.click(
                        fn=chat_respond,
                        inputs=[chatbot, chat_input, article_state,
                                provider_dd, model_txt, temperature_sl],
                        outputs=[chatbot, chat_input],
                    )
                    chat_input.submit(
                        fn=chat_respond,
                        inputs=[chatbot, chat_input, article_state,
                                provider_dd, model_txt, temperature_sl],
                        outputs=[chatbot, chat_input],
                    )

                    # Summary
                    summarize_btn.click(
                        fn=summarize_article,
                        inputs=[article_state, provider_dd, model_txt, temperature_sl],
                        outputs=summary_out,
                    )

                # ── TAB: AGENT ────────────────────────────────────────────────
                with gr.Tab("🎭 Agent"):
                    gr.Markdown(
                        "Generează răspunsul unui singur agent la știrea sau textul ales."
                    )
                    with gr.Row():
                        agent_dd = gr.Dropdown(
                            choices=agent_choices,
                            value=agent_choices[0] if agent_choices else None,
                            label="Agent",
                            scale=2,
                        )
                        agent_source = gr.Radio(
                            choices=["Știre încărcată", "Text manual"],
                            value="Știre încărcată",
                            label="Sursă",
                            scale=2,
                        )

                    agent_manual = gr.Textbox(
                        label="Text manual (dacă nu e știre încărcată)",
                        lines=4,
                        placeholder="Sau introdu textul direct aici…",
                        visible=False,
                    )

                    def toggle_manual_agent(source):
                        return gr.update(visible=(source == "Text manual"))

                    agent_source.change(
                        fn=toggle_manual_agent,
                        inputs=agent_source,
                        outputs=agent_manual,
                    )

                    agent_run_btn = gr.Button("▶ Generează răspuns", variant="primary")
                    agent_out = gr.Textbox(
                        label="Răspuns agent",
                        lines=10,
                        interactive=False,
                    )

                    agent_run_btn.click(
                        fn=agent_respond,
                        inputs=[agent_dd, agent_source, agent_manual,
                                article_state, provider_dd, model_txt, temperature_sl],
                        outputs=agent_out,
                    )

                # ── TAB: ALL AGENTS ───────────────────────────────────────────
                with gr.Tab("👥 Toți agenții"):
                    gr.Markdown(
                        "Rulează automat toți agenții disponibili și compară răspunsurile lor."
                    )
                    all_source = gr.Radio(
                        choices=["Știre încărcată", "Text manual"],
                        value="Știre încărcată",
                        label="Sursă",
                    )
                    all_manual = gr.Textbox(
                        label="Text manual",
                        lines=4,
                        placeholder="Sau introdu textul direct aici…",
                        visible=False,
                    )

                    def toggle_manual_all(source):
                        return gr.update(visible=(source == "Text manual"))

                    all_source.change(
                        fn=toggle_manual_all,
                        inputs=all_source,
                        outputs=all_manual,
                    )

                    all_run_btn = gr.Button("▶ Rulează toți agenții", variant="primary")
                    all_out = gr.HTML(label="Răspunsuri agenți")

                    all_run_btn.click(
                        fn=all_agents_respond,
                        inputs=[all_source, all_manual, article_state,
                                provider_dd, model_txt, temperature_sl],
                        outputs=all_out,
                    )

                # ── TAB: DEBATE ───────────────────────────────────────────────
                with gr.Tab("⚔️ Dezbatere"):
                    gr.Markdown(
                        "Selectează agenți participanți și pornește o dezbatere pe runde."
                    )
                    with gr.Row():
                        debate_source = gr.Radio(
                            choices=["Știre încărcată", "Text manual"],
                            value="Știre încărcată",
                            label="Sursă",
                            scale=2,
                        )
                        debate_turns = gr.Slider(
                            minimum=2, maximum=10, value=4, step=1,
                            label="Număr de intervenții",
                            scale=2,
                        )

                    debate_manual = gr.Textbox(
                        label="Text manual",
                        lines=4,
                        placeholder="Sau introdu textul direct aici…",
                        visible=False,
                    )

                    def toggle_manual_debate(source):
                        return gr.update(visible=(source == "Text manual"))

                    debate_source.change(
                        fn=toggle_manual_debate,
                        inputs=debate_source,
                        outputs=debate_manual,
                    )

                    debate_agents = gr.Dropdown(
                        choices=agent_choices,
                        value=agent_choices[:2] if len(agent_choices) >= 2 else agent_choices,
                        multiselect=True,
                        label="Agenți participanți (minim 2)",
                    )

                    debate_run_btn = gr.Button("⚔️ Pornește dezbaterea", variant="primary")
                    debate_out = gr.HTML(label="Dezbatere")

                    debate_run_btn.click(
                        fn=run_debate,
                        inputs=[debate_agents, debate_turns, debate_source, debate_manual,
                                article_state, provider_dd, model_txt, temperature_sl],
                        outputs=debate_out,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. LAUNCH
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()