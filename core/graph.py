# core/graph.py
# ==============
# LangGraph multi-agent workflow pentru EchoChamber.
#
# Workflow:
#   START → router → agent_node → router → agent_node → ... → END
#
# Funcția principală expusă către app/app.py:
#   run_thread(stimulus, active_slugs, total_turns=4, provider="gemini", k=3)

from typing import TypedDict
import argparse

from langgraph.graph import StateGraph, START, END

from core.agent import generate_agent_response


# ── Handles pentru afișarea thread-ului ────────────────────────────────────
HANDLES = {
    "anti_sistem":          "@LibertateRO99",
    "conspirationist":      "@AdevarulViu",
    "pro_european":         "@EuroOptimistRO",
    "anti_suveranist":      "@CetateanEU",
    "personalist_salvator": "@Marian_GS_Fan",
    "intelectual_critic":   "@CriticRational",
}


# ── Structura stării conversației ───────────────────────────────────────────
class ThreadState(TypedDict):
    stimulus:     str    # textul politic inițial
    messages:     list   # lista mesajelor produse până acum
    active_slugs: list   # agenții care participă
    total_turns:  int    # numărul total de intervenții
    current_turn: int    # câte intervenții au fost produse
    next_slug:    str    # agentul ales de router
    provider:     str    # gemini / deepseek
    k:            int    # numărul de fragmente recuperate din FAISS


# ── Formatare thread pentru prompt ─────────────────────────────────────────
def thread_to_text(messages):
    """
    Transformă lista de mesaje într-un text citibil.
    Trimis agentului ca THREAD ANTERIOR în prompt.
    """
    if not messages:
        return "(nu există mesaje anterioare)"

    lines = []
    for msg in messages:
        handle = msg.get("handle", msg.get("slug", "?"))
        turn   = msg.get("turn", "?")
        text   = msg.get("text", "").strip()
        lines.append(f"Turn {turn} — {handle}: {text}")

    return "\n".join(lines)


# ── Router round-robin ──────────────────────────────────────────────────────
def pick_next_agent(active_slugs, current_turn):
    """
    Selectează următorul agent prin rotație (round-robin).
    Exemplu: anti_sistem → conspirationist → pro_european → anti_sistem ...
    """
    return active_slugs[current_turn % len(active_slugs)]


def router_node(state: ThreadState):
    """
    Nodul router decide cine vorbește următor
    sau oprește conversația la total_turns.
    """
    if state["current_turn"] >= state["total_turns"]:
        return {"next_slug": "__end__"}

    next_slug = pick_next_agent(state["active_slugs"], state["current_turn"])
    return {"next_slug": next_slug}


def route_decision(state: ThreadState):
    """
    Funcție pentru conditional edge — returnează slug-ul următor.
    """
    return state["next_slug"]


# ── Nod agent ───────────────────────────────────────────────────────────────
def make_agent_node(slug):
    """
    Creează un nod pentru un agent specific.
    Agentul citește stimulusul + thread-ul anterior și generează un răspuns.
    """

    def agent_node(state: ThreadState):
        # Transformă mesajele anterioare în text pentru prompt
        thread_text = thread_to_text(state["messages"])

        # Handle-ul agentului curent
        my_handle = HANDLES.get(slug, f"@{slug}")

        # Instrucțiune pentru agent: răspunde ultimului vorbitor dacă există
        if state["messages"]:
            last_msg    = state["messages"][-1]
            last_handle = last_msg.get("handle", last_msg.get("slug", ""))
            task = (
                f"Scrie ca {my_handle}. "
                f"Răspunde direct ultimului vorbitor ({last_handle})."
            )
        else:
            task = (
                f"Scrie ca {my_handle}. "
                f"Ești primul care reacționează la știre."
            )

        # Construiește inputul complet pentru agent
        agent_input = (
            f"[STIMULUS]\n{state['stimulus']}\n\n"
            f"[THREAD ANTERIOR]\n{thread_text}\n\n"
            f"[SARCINĂ]\n{task}"
        )

        # Apelează agentul RAG
        result = generate_agent_response(
            agent_slug=slug,
            stimulus=agent_input,
            provider=state["provider"],
            k=state["k"],
        )

        # Construiește mesajul nou
        new_message = {
            "agent":  result["agent_name"],
            "slug":   slug,
            "handle": my_handle,
            "text":   result["response"],
            "turn":   state["current_turn"] + 1,
        }

        return {
            "messages":     state["messages"] + [new_message],
            "current_turn": state["current_turn"] + 1,
        }

    return agent_node


# ── Construire graf ─────────────────────────────────────────────────────────
def build_graph(active_slugs):
    """
    Asamblează graful LangGraph:
    START → router → agent_node → router → ... → END
    """
    workflow = StateGraph(ThreadState)

    # Adaugă nodul router
    workflow.add_node("router", router_node)

    # Adaugă câte un nod pentru fiecare agent activ
    for slug in active_slugs:
        workflow.add_node(slug, make_agent_node(slug))

    # START → router
    workflow.add_edge(START, "router")

    # Router → agent selectat sau END
    route_map = {slug: slug for slug in active_slugs}
    route_map["__end__"] = END

    workflow.add_conditional_edges("router", route_decision, route_map)

    # Fiecare agent → înapoi la router
    for slug in active_slugs:
        workflow.add_edge(slug, "router")

    return workflow.compile()


# ── Funcția principală ──────────────────────────────────────────────────────
def run_thread(
    stimulus,
    active_slugs,
    total_turns=4,
    provider="gemini",
    k=3,
):
    """
    Rulează un thread multi-agent complet.
    Returnează lista finală de mesaje.
    Apelată de notebook și de app/app.py.
    """
    graph = build_graph(active_slugs)

    initial_state = {
        "stimulus":     stimulus,
        "messages":     [],
        "active_slugs": active_slugs,
        "total_turns":  total_turns,
        "current_turn": 0,
        "next_slug":    "",
        "provider":     provider,
        "k":            k,
    }

    final_state = graph.invoke(initial_state)
    return final_state["messages"]


# ── Terminal test ───────────────────────────────────────────────────────────
# python -m core.graph --agents conspirationist intelectual_critic pro_european
#        --text "CCR a decis anularea alegerilor după suspiciuni privind influențe externe."
#        --turns 4 --provider gemini
def main():
    parser = argparse.ArgumentParser(description="Test multi-agent LangGraph thread.")
    parser.add_argument("--agents",   nargs="+", required=True, help="Agent slugs")
    parser.add_argument("--text",     required=True,            help="Stimulus text")
    parser.add_argument("--turns",    type=int, default=4,      help="Total turns")
    parser.add_argument("--provider", default="gemini",         help="gemini / deepseek")
    parser.add_argument("--k",        type=int, default=3,      help="FAISS fragments")
    args = parser.parse_args()

    messages = run_thread(
        stimulus=args.text,
        active_slugs=args.agents,
        total_turns=args.turns,
        provider=args.provider,
        k=args.k,
    )

    print(thread_to_text(messages))


if __name__ == "__main__":
    main()
