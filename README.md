# EchoChamber Studio

Simulation of discursive bubbles using political comments.
Each agent responds from the perspective of its own political community.

---

## Project Overview

EchoChamber Studio is an environment designed to simulate and analyze different tones and nuances used in digital communication. The model tries to impersonate diverse ideological views to interpret news, comments, to summarize articles and to generate responses in a selected ideological voice.
To achieve this, the platform integrates an automated pipeline that can load or use a political or social news article and immediately summarize the article to extract its core structural data. Once the context is established, users can generate a response from one selected agent to observe an individual ideological reaction. Furthermore, the studio provides features to compare responses from all agents side-by-side or run a short multi-agent debate, where different discursive personas dynamically challenge and respond to each other's arguments in a closed conversational loop.

---

## Why this matters

This project explores how AI can be used in political communication. The prototype tries to deconstruct how specific language patterns and ideological biases propagate and to identify the subtle boundaries between each output from each agent voice. In an era dominated by algorithmic distribution and fragmented media landscapes, understanding the mechanics of digital echo chambers is critical for researchers, policy analysts, and communication experts.

Rather than attempting to forecast or measure real-world public opinion, this prototype provides a controlled sandbox environment. It isolates the rhetorical structures behind political messaging, helping stakeholders audit the behavior of language models and evaluate the structural alignment of simulated discursive personas under specialized vector conditions.

---

## Main Workflow

The system coordinates data flow paths depending on whether it is evaluating a standalone perspective or an interactive debate thread. The architecture supports two primary execution workflows depending on whether the interaction involves a single-perspective analysis or a multi-agent discussion thread:

### Single-Agent Execution Flow

news/text → selected agent role → retrieved similar comments → LLM → simulated response

### Multi-agent Debate Flow

news/text → selected agents → conversation state → multi-agent thread

The input text or news article serves as the primary stimulus and the main object of the response, anchoring the generation pipeline to a specific topic. Rather than acting as factual evidence or source truth, the retrieved comments from the vector database function strictly as discursive context to shape the unique style, vocabulary, and ideological lens of the output. Finally, the behavioral boundaries, rhetorical tone, and operational constraints of this generation are governed by the role configuration file, which explicitly defines each agent's distinct voice and rules.

---

## Repository Structure

```text
echochamber/
├── notebooks/              # Weekly course notebooks (added during the semester)
├── collector/              # Scripts for collecting comments from YouTube / RSS
├── data/
│   ├── raw/                # Raw collected comments (CSV or JSONL)
│   ├── cleaned/            # Cleaned and standardized corpus
│   └── bubbles/            # One JSONL file per agent after annotation
├── assets/
│   └── roles/              # Agent role cards (roles.yaml) — written by students
├── scripts/
│   ├── clean_corpus.py     # Cleans and standardizes raw data
│   └── build_vectorstore.py # Builds FAISS vector index from data/bubbles/
├── core/                   # Core infrastructure — do not modify
│   ├── agent.py            # Agent class: reads roles.yaml + retrieves from corpus
│   ├── retriever.py        # Semantic search over FAISS index
│   ├── graph.py            # LangGraph agentic debate orchestration
│   └── metrics.py          # Dissimilarity, sentiment, and visualization
├── app/
│   └── app.py              # Gradio application (built incrementally during course)
└── reports/                # Final report and ethics checklist templates
```

---

## Setup

By default, EchoChamber Studio is a local application. It operates entirely within an isolated sandboxed interface on your machine and does not deploy or broadcast data to external hosting services.

These are the steps within **Windows PowerShell** to clone, configure, and execute the prototype:

```bash
1. Clone the repository and navigate to the project root
git clone <your-repo-url>
cd echochamber-project-team3

# 2. Synchronize repository data
git pull

# 3. Create an isolated Python virtual environment
python -m venv .venv

# 4. Activate the virtual environment
.venv\Scripts\Activate

# 5. Install the required project dependencies
pip install -r requirements.txt

# 6. Initialize your local configuration file
cp .env.example .env

# 7. Launch the local web application interface
python -m app.app
```

---

## Environment variables

A local `.env` file is strictly required to handle model provider authorizations. This file must reside only on your local machine to keep credentials secure. **Do not commit the `.env` file or any raw API keys to public repositories.**

To set it up, create your local `.env` file based on the distributed `.env.example` template:

```bash
# Duplicate the template to create your secure environment file
cp .env.example .env
```
Open the newly created .env file and populate only the variables used by our active pipeline:

```bash
OPENAI_API_KEY=your-key-here
DEEPSEEK_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
```

---

## Application features

The system exposes its capabilities through a professional layout divided into a persistent configuration sidebar and a dynamic multi-tab workspace:

* ** Persistant Configuration Sidebar (Configurare & Încarcă știre):** Located on the left, this panel allows users to select the LLM provider (e.g., Gemini) and model version, adjust generation temperature via a slider, and parse external news content directly by inserting a URL.
* ** Chat Tab:** A standalone conversational workspace to ask questions or discuss the loaded article directly with the base language model, serving as a clean baseline checkpoint before any narrative styling.
* ** Agent Tab:** Generates an ideological reaction from a single selected discursive persona, injecting specific historical context and narrative constraints.
* ** Toți agenții (All Agents) Tab:** Computes and displays side-by-side responses from all available ideological personas simultaneously, highlighting divergent framings of the exact same event.
* ** Dezbatere (Debate) Tab:** Orchestrates a multi-agent discussion loop where different personas dynamically respond to and challenge each other's arguments based on the loaded news.
* ** Rezumă știrea (Article Summary Area):** Positioned globally beneath the core workspace tabs, this dedicated module extracts and displays a structural summary of the ingested article at the user's request.

---

## Agents

The prototype initializes specific ideological personas based on the configurations defined within `assets/roles/roles.yaml`. The active lineup available in the system includes:

* `personalist_salvator` — Devoted leader-centric perspective focusing on patriotism, faith, and perceived systemic persecution.
* `anti_sistem` — Radical institutional distrust targetting mainstream politicians, media, and established structures with sharp sarcasm.
* `pro_european` — Democratic, reformist, and pro-Western stance advocating for civic responsibility and Euro-Atlantic integration.
* `conspirationist` — Hyper-suspicious, alarmist framing that connects events to globalist elites or hidden actors.
* `intelectual_critic` — Analytical, logic-driven observer prioritizing factual evidence, structural cause-and-effect, and pragmatic metrics over emotion.
* `anti_suveranist` — Rationalist, defensive lens focused on exposing populist demagoguery, disinformation networks, and the political manipulation of religious themes.

Each agent is defined by a role, voice, worldview and response rules. The agents are simulated discursive roles, not real people or real social groups.

---

## Technical components

The repository separates data scraping, semantic retrieval, state orchestration, and presentation into isolated layers:

* `core/retriever.py` — Manages the semantic query encoding step and searches the local FAISS vectorstores for stylistically relevant comment anchors.
* `core/agent.py` — Merges YAML role instructions, contextual snippets from the vector store, and the parsed news input to compile target prompts for the LLM.
* `core/graph.py` — Orchestrates the multi-agent debate sequence, routing tokens, managing turn-taking, and updating the conversation state using LangGraph logic.
* `app/app.py` — Builds, styles, and exposes the entire dual-column processing pipeline through a Gradio user interface.

---

## Ethics and limitations

EchoChamber Studio is a teaching and research prototype. Its agents are simulated discursive roles, not real people or representatives of real social groups. Generated outputs may contain bias, unsupported claims, or amplified conflict and must be interpreted critically.

Before analyzing or re-using the system outputs, the following technical and ethical boundaries must be considered:

* **Simulation vs. Reality:** The agents are computed roles designed for narrative analysis; they do not represent real individuals, specific demographic cohorts, or actual social groups.
* **Non-Factual Outputs:** Generated responses are produced by language models optimized for stylistic mimicry and do not constitute factual evidence or historical truth.
* **Discursive Contextualization:** Retrieved comments extracted from the database function strictly as lexical and stylistic context to guide the tone, not as proof or endorsement of widespread public beliefs.
* **Model Vulnerabilities:** The system may occasionally produce biased, generic, inaccurate, or harmful outputs due to underlying LLM limitations or training data anomalies.
* **Technical Constraints:** Automated news extraction from URLs depends on standard HTML parsing and may fail on websites that implement strict paywalls, anti-scraping mechanisms, or CAPTCHAs.
* **No Public Opinion Mapping:** The system is a closed computational playground and should never be used to infer, measure, or extrapolate real public opinion or sentiment dynamics.
* **Security Guardrails:** API keys, endpoints, and private credentials must remain strictly confidential in a local `.env` environment and must never be committed to repository history.

See also: [`docs/ethics_checklist.md`](docs/ethics_checklist.md) for the full ethics note, limitations, and final checklist.

---

## Team contributors

Ianitchi Valeria -> personalist-salvator agent

Minciuna Catalina -> anti-sistem agent

Lostun Flaviu -> pro-european

Ignat Carmina -> conspirationist

Hutanu Diana -> anti-suveranist agent

Havrisciuc George -> intelectual-critic agent


---

## Known issues

As a developing prototype, the current version of EchoChamber Studio operates with several structural and technical constraints that are actively monitored:

* **Scraping and Extraction Blocks:** Automated article extraction via URL parsing may fail or return null strings on high-security news websites that implement anti-bot frameworks, strict cookie consent layers, or cloud firewalls.
* **Response Genericity:** Depending on the specificity of the input topic, certain agents may occasionally produce generic responses or exhibit safe alignment behaviors inherent to the underlying base LLM.
* **Basic Debate Routing:** The multi-agent debate loop utilizes a straightforward round-robin routing logic; it does not currently feature dynamic conversational turn-taking or intent-based interruption flags.
* **Prototype Deployment Status:** The application is architected strictly as a localized web prototype for evaluation and sandbox testing. It is not deployed on public cloud infrastructure and lacks concurrent multi-tenant support or enterprise scaling guards.
* **Corpus Dependencies:** The nuance and linguistic accuracy of simulated agent outputs are directly bounded by the size and alignment of the local historical comment corpus, as well as the inference capabilities of the selected API model provider.

---

## License

This project is a research and educational prototype. Outputs should be reviewed by humans before interpretation or reuse.

---
