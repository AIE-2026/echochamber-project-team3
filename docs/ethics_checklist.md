# Ethics and limitations

EchoChamber is a teaching and research prototype for simulating discursive responses with AI agents.
The goal is educational: to understand RAG, role prompts, LangGraph workflows, multi-agent interaction, and the risks of generated political discourse.

EchoChamber must not be presented as a system that measures real public opinion, predicts political behavior, or represents real social groups.

---

## 1. What the agents are

The agents in this project are simulated discursive roles.
They are not real people.
They are not voters.
They are not representatives of parties, communities, demographic groups, or social categories.

Each agent is created from:
- a YAML role prompt;
- retrieved fragments from a corpus;
- an LLM-generated response;
- workflow rules defined in the application.

**Correct wording:**
> "The anti-system agent generated a simulated anti-system framing."

**Incorrect wording:**
> "Anti-system voters think this."

---

## 2. What the outputs mean

Generated responses are not factual claims.
They are synthetic comments produced by a model under a role constraint.

Even when RAG is used, retrieved context is only supporting material.
Retrieved context does not automatically make the generated answer true.

All outputs must be interpreted critically by students, researchers, or instructors.

---

## 3. Main risks

| Risk | What it means | Minimum control |
|------|---------------|-----------------|
| Anthropomorphism | Users may treat agents as real people | Always label them as simulated agents |
| False factuality | Generated text may sound like verified information | Separate retrieved context from generated response |
| Corpus bias | The corpus may contain biased, hostile, repetitive, or unbalanced content | Document corpus source and known limits |
| Amplification | Multi-agent threads may intensify conflict or polarization | Limit number of turns and review outputs |
| Misrepresentation | A role may be confused with a real social group | Use "constructed discursive position," not "group opinion" |
| Privacy | Public comments may still contain personal data or identifiers | Avoid unnecessary personal data and do not profile individuals |
| Political misuse | Outputs could be reused as persuasion material | Use only for education, prototyping, and critical analysis |

---

## 4. Data protection rules

- Use only public or classroom-approved data.
- Do not commit API keys, `.env` files, private documents, or personal data to GitHub.
- Avoid storing unnecessary identifiers such as usernames, names, contact details, or links to individual profiles.
- If real comments are used, treat them as research material, not as free text that can be copied without care.
- Use short excerpts only when needed for analysis.
- Do not build profiles of real users.
- Do not infer personal traits, political identity, ethnicity, religion, health, or other sensitive attributes from individual comments.

---

## 5. Responsible use

EchoChamber **may** be used to:
- test how different role prompts shape generated responses;
- compare discursive framings;
- inspect how RAG affects generated text;
- understand how multi-agent workflows can amplify or moderate discourse;
- support classroom discussion about AI, discourse, bias, and governance.

EchoChamber **must not** be used to:
- generate political persuasion material;
- imitate real citizens or groups;
- publish synthetic comments as authentic public opinion;
- target individuals or social groups;
- make factual claims about public events without verification;
- replace empirical social research.

---

## 6. Human oversight

Human interpretation is required.

Students must inspect:
- whether the retrieved context is relevant;
- whether the generated response follows the role;
- whether the output introduces unsupported claims;
- whether the interaction escalates conflict;
- whether the result could be misleading if shown publicly.

No generated output should be used outside the classroom without human review.

---

## 7. Transparency in the application

The application should make clear:
- which agent generated the response;
- which input was used;
- which provider/model was used;
- whether RAG context was retrieved;
- how many fragments were retrieved;
- that the output is synthetic.

**Recommended app disclaimer:**

> EchoChamber generates simulated political-discourse responses. The agents are fictional analytical constructs, not real people and not representatives of real social groups. Generated outputs may contain bias, exaggeration, unsupported claims, or problematic language inherited from the corpus and the model. Use the results only for education, prototyping, and critical discourse analysis.

---

## 8. How to present results

Use careful language when writing reports or presenting the demo.

**Recommended:**
- "The agent generated a simulated framing."
- "The thread shows possible escalation under this prompt structure."
- "The response is based on retrieved corpus fragments, but it is not factual verification."
- "The result requires human interpretation."

**Avoid:**
- "This group believes..."
- "The public thinks..."
- "The model proves..."
- "This is what voters would say..."

---

## 9. Minimal logging for reproducibility

For each important run, document:
- input text;
- agent slug;
- provider/model;
- retrieval parameter k;
- number of turns, if multi-agent;
- whether the router was round-robin or LLM-based;
- short note on problems observed.

**Example:**
```
input: CCR a decis anularea alegerilor după suspiciuni privind influențe externe.
agents: anti_sistem, conspirationist, pro_european
provider: gemini
k: 3
turns: 4
router: round-robin
observed problem: mild repetition in turns 3–4
```

---

## 10. Final note

EchoChamber is a learning tool.
It helps us study how AI systems can simulate, structure, and amplify discourse.
It does not replace empirical research, factual verification, or human judgment.

The responsibility for interpreting and presenting results belongs to the researcher, not to the system.

---

## 11. Ethics checklist

- [ ] Agents are clearly labeled as simulated roles, not real people.
- [ ] Generated outputs are not presented as facts or public opinion.
- [ ] Retrieved context is shown separately from generated responses.
- [ ] No API keys or `.env` files are committed.
- [ ] No unnecessary personal data is stored.
- [ ] The app or README includes a visible disclaimer.
- [ ] The team reviewed outputs before presentation.
