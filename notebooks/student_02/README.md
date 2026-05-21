# student_02 Catalina Minciuna

This is my individual notebook workspace for the AI Engineering course.

## Model ales pentru proiect

Model principal ales: Gemini 2.5 Flash Lite
  
Model de rezervă: OpenRouter Free  
Temperature recomandată: 0.1

Echipa a ales Gemini 2.5 Flash Lite deoarece a oferit răspunsuri mai clare și mai ușor de înțeles în comparație cu celelalte modele testate. La temperature = 0.1, modelul a fost mai stabil și a produs răspunsuri consecvente, potrivite pentru adnotarea comentariilor politice. OpenRouter Free poate fi folosit ca model de rezervă, în cazul în care modelul principal nu este disponibil.


# Contribuția mea la EchoChamber Studio

**Student:** `student_02`
**GitHub:** `@catalinaminciuna`
**Agent principal:** `anti_sistem`
**Rol în proiect:** construcția și testarea bulei discursive anti-sistem, integrarea rolurilor agenților, validarea fluxurilor RAG/LangGraph și contribuție la finalizarea aplicației Gradio.

---

## 1. Obiectivul contribuției

Am construit și integrat un agent conversațional bazat pe o bulă discursivă politică. Agentul principal este **Anti-sistem** — o voce definită prin neîncredere față de instituții, partide, presă mainstream, justiție, BOR și clasa politică.

Activitatea mea a acoperit toate etapele proiectului: primul apel către LLM, compararea modelelor, colectarea și curățarea comentariilor YouTube, construirea unui prompt de adnotare, selecția corpusului pentru agent, construirea vectorstore-ului FAISS, testarea agentului RAG, rularea conversațiilor multi-agent cu LangGraph și integrarea finală în aplicația Gradio.

---

## 2. Activitate pe notebook-uri

### C1 — Primul apel către un LLM

Am configurat cheia API în `.env`, am testat primul apel către Gemini prin endpoint compatibil OpenAI și am înțeles structura minimă a unui apel LLM: `model`, `messages`, `temperature`, output. Am generat un rezumat neutru al unei știri și am rulat un experiment cu valori diferite de `temperature` (0.0, 0.7, 1.5), observând că temperaturile mici sunt mai potrivite pentru sarcini stabile și factuale. Am salvat experimentul în `outputs/c1_first_summary.json` împreună cu parametrii folosiți, pentru reproductibilitate.

### Tema 1 — Explorarea corpusului și primul prompt

Am explorat corpusul curățat de 30.451 comentarii YouTube din `data/cleaned/corpus_youtube_large_clean.jsonl` și am construit un prompt exploratoriu pe 7 axe: `target`, `stance`, `sentiment`, `tone`, `topic`, `interpretation_problem`, `rhetoric_type`, plus o justificare scurtă. Am testat promptul pe 10 comentarii și am salvat rezultatele în `outputs/student_02_prompt_outputs.csv`.

**Probleme identificate în reflecție:**
- la rândul 0, "Multă sănătate dl. Președinte Călin Georgescu" a fost marcat `stance: contra`, deși comentariul e clar pro-CG — modelul a interpretat politețea ca ironie;
- pe 8 din 10 cazuri `sentiment` și `stance` au aceeași polaritate, semn că modelul tratează cele două axe ca sinonime, deși am specificat explicit că sunt diferite;
- la rândul 3, ținta a devenit o frază lungă în loc de o entitate concretă.

**Îmbunătățiri propuse pentru versiunea următoare:** few-shot examples pentru distincția sentiment vs stance, constrângere pe lungimea target-ului (maxim 5 cuvinte), eșantionare stratificată pentru testare, testare pe un set mai mare (100-200 comentarii) pentru a calcula rate de confuzie.

### C2 — Ecosistemul de modele

Am comparat trei modele: `gemini-2.5-flash-lite`, `gemini-2.5-flash` și `openrouter/free`. Am testat calitatea răspunsurilor în limba română, respectarea instrucțiunilor din system prompt, outputul structurat în JSON (schema cu enum-uri pentru adnotare politică) și stabilitatea la temperaturi diferite (0.1, 0.7, 1.2).

**Decizia echipei:** model principal `gemini-2.5-flash-lite` la `temperature = 0.1`, model de rezervă `openrouter/free`. Gemini Flash Lite oferă răspunsuri clare, stabile și consecvente pentru adnotare, iar OpenRouter Free rămâne backup când Gemini are limite de quota.

### C3 — Colectare și curățare comentarii YouTube

Am folosit YouTube Data API pentru colectarea comentariilor publice de pe canalul `CălinGeorgescu-CanalulOficial`. Am extras 100 de comentarii brute din 10 videoclipuri recente și le-am salvat în:

```
data/raw/student_02_youtube_raw.jsonl
```

**Pași de curățare aplicați** (toți cuprinși în funcția `clean_comments`):
- eliminarea linkurilor cu regex `r"http\S+"`;
- normalizarea spațiilor cu `r"\s+"`;
- filtrarea comentariilor sub 60 de caractere;
- filtrarea textelor cu sub 50% litere (eliminăm comentariile dominate de emoji);
- eliminarea duplicatelor (case-insensitive).

După curățare au rămas **51 de comentarii** în:

```
data/cleaned/student_02_youtube_clean.jsonl
```

### C4 — Prompt de adnotare

Am construit un mini-prompt de adnotare pentru comentarii politice. Am ales două axe relevante pentru direcția anti-sistem:

- **`anti_elite`** — atac asupra elitelor politice, economice, mediatice sau culturale (0/1/2);
- **`system_rejection`** — respingerea sistemului sau instituțiilor ca ilegitime (0/1/2).

Distincția contează: poți ataca o elită fără să respingi sistemul (critică reformistă) sau poți respinge sistemul fără să numești o elită anume (cinism difuz).

**Mini-tipologie 2x2 rezultată:**
- `anti-sistem dur` (ae≥1 și sr≥1)
- `critica elitelor` (ae≥1 și sr=0)
- `cinism difuz` (ae=0 și sr≥1)
- `apolitic / loial` (ae=0 și sr=0)

Promptul a fost testat pe 5 comentarii. **Probleme identificate:**
- Gemini 2.5 Flash Lite a prefixat răspunsurile cu ` ```json `, chiar dacă promptul cerea „doar JSON valid, fără markdown" — am rezolvat cu un parser regex care extrage primul obiect `{...}`;
- distincția între „instituția e ostilă" și „instituția e compromisă din exterior" rămâne ambiguă în prompt (soft system rejection).

**Ce aș schimba:** few-shot examples pentru `system_rejection`, un câmp `evidence` cu fragmentul concret din text care a justificat scorul, instrucțiune mai dură împotriva fence-urilor markdown.

### C5.01 — Explorarea corpusului tipologizat

Am analizat corpusul tipologizat din `data/typed/corpus_typed.jsonl` (17.886 comentarii). Tipul relevant pentru agentul meu era `T2_grievance_anti_sistem`, cel mai numeros (7.583 exemple), mapat la agentul `anti_sistem`.

Am pregătit un subset stratificat cu maxim 150 texte per bulă, sortat după `type_confidence`, salvat în `data/typed/corpus_c5_sample.jsonl` (896 texte unice). Apoi am inspectat exemplele din bula mea (149 texte disponibile), am verificat manual dacă exprimă clar vocea anti-sistem și am selectat **50 de texte** pentru vectorizare, salvate în `data/bubbles/anti_sistem.jsonl`.

**Descrierea agentului meu** (cum o văd pe baza corpusului):
- vede instituțiile statului român (parlament, guvern, justiție, biserică, presă mainstream) ca pe o structură unitară de putere, coruptă și ostilă cetățeanului obișnuit;
- ton acuzator, indignat, deseori cu majuscule și exclamări în lanț;
- furie morală directă, uneori cu sarcasm amar;
- lexic dur: „hoți", „escroci", „mafia", „securiști", „paraziți", „pupincuriști", „șarlatani";
- acuzații recurente: corupție generalizată, trădare națională, salarii mici și prețuri mari, bani trimiși în Ucraina, finanțări la BOR, dosare îngropate;
- **delegitimare globală a sistemului, fără propunere reformistă clară** — asta îl diferențiază de `personalist_salvator` (care propune un lider-mântuitor), de `conspirationist` (care construiește scheme cu forțe ascunse), de `anti_suveranist` (care apără ordinea europeană) și de `pro_european` (care apără procedurile).

### C5.02 — Construirea vectorstore-ului

Am construit vectorstore-ul FAISS pentru bula `anti_sistem`. Am pornit de la `data/bubbles/anti_sistem.jsonl` (50 texte), am generat embeddings cu modelul multilingv `paraphrase-multilingual-MiniLM-L12-v2` și am obținut o matrice `(50, 384)` — 50 de reprezentări vectoriale, fiecare cu 384 de dimensiuni.

Am normalizat vectorii la lungime 1, am construit un `IndexFlatIP` (produsul scalar = similaritate cosinus pe vectori normalizați) și am salvat:

```
assets/vectorstores/anti_sistem/index.faiss
assets/vectorstores/anti_sistem/index.pkl
```

**Testare retrieval** cu inputul „Partidele vechi și instituțiile statului își apără privilegiile, în timp ce oamenii obișnuiți plătesc pentru corupția și nepăsarea lor": 5 din 5 rezultate au fost relevante pentru bula anti_sistem, acoperind teme precum corupția politică, banii publici, privilegiile, instituțiile ineficiente și opoziția dintre oamenii obișnuiți și clasa politică.

### C6 — Agent RAG

Am testat agentul RAG simplu prin trei variante crescânde de complexitate:

**(1) RAG manual** — am construit promptul cu un f-string care combină `agent_system` (din `role_02.yaml`), `[STIMULUS]` (inputul nou) și `[COMENTARII SIMILARE]` (fragmentele FAISS).

**(2) RAG cu LangChain** — am rescris același flux folosind `PromptTemplate.from_template()`. LangChain nu face modelul mai inteligent, dar standardizează promptul și pregătește trecerea la LangGraph.

**(3) Agentic RAG cu tool de regăsire** — am definit `retrieve_similar_comments` ca tool și am folosit `create_agent` din LangChain. Agentul decide singur când să apeleze tool-ul. Pe inputul „Universitatea ar trebui să fie gratuită pentru toată lumea", agentul a făcut un `tool_call`, a primit fragmentele și apoi a generat răspunsul.

**(4) Mini-agent RSS** — am extins cu un al doilea tool, `get_latest_news_from_rss`, care citește feed-ul `g4media.ro`. Agentul ia o știre recentă, caută în propria bulă comentarii similare și generează răspunsul. Fluxul devine semi-autonom.

**Două teste de validare:**

*Test 1 — anularea alegerilor* (`CCR a decis anularea alegerilor după suspiciuni privind influențe externe`): răspunsul preia furia din corpus și atacă CCR și magistratura ca parte din „sistem" — context_used: yes, voice_coherent: yes.

*Test 2 — proteste economice* (`Guvernul a anunțat noi măsuri economice care au provocat proteste`): răspunsul folosește contextul recuperat și păstrează vocea anti-sistem, dar rămâne abstract pe alocuri pentru că corpusul nu conține multe exemple specifice despre proteste economice recente.

**Reflecție etică:** înainte ca un astfel de răspuns să ajungă într-o aplicație publică, un om ar trebui să verifice (1) că tool-urile au fost efectiv folosite, (2) că răspunsul nu inventează fapte care nu apar în știre sau în fragmentele recuperate, (3) că nu derapează spre limbaj abuziv. Amplificarea automată a unei bule are nevoie de moderare umană, altfel doar redistribuim furie scalată.

### C7 — LangGraph și thread multi-agent

Am extins agentul RAG într-un workflow multi-agent cu LangGraph. Am definit `ThreadState` (un `TypedDict`) care păstrează:

- `stimulus` — inputul politic inițial;
- `messages` — conversația produsă până acum;
- `active_slugs` — agenții care participă;
- `total_turns`, `current_turn` — controlul rundelor;
- `next_slug` — agentul ales pentru următoarea intervenție;
- `provider`, `k` — config LLM și FAISS.

**De ce `messages` trebuie să fie în state:** fiecare nod din graf are nevoie să vadă conversația actualizată. Dacă mesajele ar fi într-o variabilă globală, fluxul ar fi mai greu de testat, de reluat și de controlat.

Am construit un **router round-robin** (agenții vorbesc pe rând), apoi noduri de agent care apelează `generate_agent_response` din `core/agent.py`, și am conectat:

```
START → router → agent_node → router → agent_node → ... → END
```

**Fluxul C6** (`input politic → agent RAG → răspuns`) a devenit în C7 un thread controlat. Am rulat un thread de 4 ture cu `anti_sistem`, `conspirationist`, `pro_european` pe stimulusul anulării alegerilor de către CCR. Diferențierea vocilor a fost clară:
- `anti_sistem` — acuzator și mobilizator;
- `conspirationist` — amplifică suspiciunea cu „păpușari din umbră";
- `pro_european` — readuce discuția spre proceduri, dovezi și reguli democratice.

Am implementat și o **extensie router LLM** (modelul citește conversația și decide cine vorbește), cu regula că niciun agent nu vorbește de două ori la rând, plus fallback pe primul candidat permis dacă răspunsul LLM-ului nu e valid.

**Etică și limite recunoscute:**
- agenții nu sunt persoane reale, ci simulări discursive;
- modelul poate halucina afirmații nesusținute de context;
- interacțiunea multi-agent poate amplifica polarizarea;
- corpusul conține bias din selecția stilistică (am ales comentariile cele mai marcate, am eliminat vocile moderate);
- simularea nu înlocuiește analiza empirică a actorilor reali.

Disclaimer obligatoriu: **EchoChamber este un instrument de simulare și analiză discursivă, nu un sistem de predicție politică și nu o reprezentare fidelă a unor persoane reale.**

### C8 — Aplicația Gradio

Am contribuit la integrarea componentelor într-o aplicație Gradio. Principiul urmărit: `app/app.py` este un strat subțire de interfață peste funcțiile din `core/` (`generate_agent_response`, `run_thread`), nu o rescriere a logicii deja construite.

**Aplicația finală permite:**
- încărcarea unei știri din URL;
- afișarea unui preview al știrii;
- rezumarea articolului;
- adresarea unei întrebări despre articol;
- generarea unui răspuns de la un singur agent;
- rularea tuturor agenților pe același text;
- pornirea unei dezbateri multi-agent.

**Tema 3 — Extensii individuale peste schelet:**

1. **Tab nou „Despre & Etică"** — disclaimer AI, limite ale simulării, descrierea bulei `anti_sistem`;
2. **Funcție nouă `count_words_and_chars()`** — numără cuvintele, caracterele și propozițiile dintr-un răspuns, verifică dacă agentul respectă limita de „maxim 3 propoziții" din `role_02.yaml`;
3. **Design schimbat** — temă `Monochrome(primary_hue="red")` (culoarea bulei anti-sistem), titlu cu emoji 🗣️, subtitlu cu data ultimei rulări, disclaimer vizibil pe toate tab-urile.

---

## 3. Agentul individual: `anti_sistem`

Agentul `anti_sistem` este o voce politică revoltată, suspicioasă și dezamăgită. Critică instituțiile, partidele, presa mainstream, justiția, BOR și modul în care sunt folosiți banii publici.

**Stilul agentului:** direct, acuzator, moralizator, mobilizator, neîncrezător față de instituții, critic față de partide și elite.

**Construcție pe bază de corpus:** rolul a fost construit pe baza celor 50 de comentarii din `data/bubbles/anti_sistem.jsonl`, nu inventat liber. Am urmărit formule recurente: acuzații de corupție, neîncredere în instituții, opoziția „oameni simpli vs sistem", critica banilor publici, apeluri la revoltă civică.

**Ajustare pentru context academic:** unele răspunsuri puteau deveni prea vulgare pentru un demo academic. Rolul a fost ajustat să păstreze tonul dur și revoltat, dar să fie prezentabil în aplicația finală.

---

## 4. Issue-uri de echipă rezolvate

### 4.1. Issue C4.1 — Add C4 data, script, and reference prompt

**Obiectiv:** adăugarea materialelor comune pentru etapa C4 (fișiere de date, script de adnotare, prompt de referință al instructorului).

**Fișiere adăugate:**
```
data/cleaned/corpus_youtube_sample.jsonl
data/cleaned/corpus_youtube_sample_annotated.jsonl
scripts/annotate_axis.py
prompts/annotation_prompt.md
```

Am avut grijă ca promptul din `prompts/annotation_prompt.md` să rămână promptul de referință al instructorului, nu o variantă modificată.

**Commit:**
```bash
git add data/cleaned/ scripts/ prompts/annotation_prompt.md
git commit -m "Add C4 annotation materials"
git push
```

**Comentariu postat pe issue:**
> DONE - @catalinaminciuna
> I added the C4 data files: yes
> I added scripts/annotate_axis.py: yes
> I added prompts/annotation_prompt.md: yes
> The files are visible on GitHub: yes

---

### 4.2. Issue C6.2 — Integrate individual agent roles into roles.yaml

**Obiectiv:** integrarea tuturor rolurilor individuale într-un singur fișier comun `assets/roles/roles.yaml`, folosit de notebook-ul C6 RAG și de backend-ul aplicației.

**Surse:**
```
assets/roles/role_01.yaml ... role_06.yaml
```

**Output:** un singur bloc `agents:` în `assets/roles/roles.yaml` cu cinci agenți: `personalist_salvator`, `anti_sistem`, `anti_suveranist`, `conspirationist`, `pro_european`.

**Verificări:**
- un singur bloc `agents:`;
- fiecare agent apare o singură dată;
- fiecare agent are `slug`, `name`, `emoji`, `color`, `system`;
- indentare YAML validă;
- fișierul se încarcă prin `yaml.safe_load()`.

**Test de încărcare:**
```python
from pathlib import Path
import yaml

path = Path("assets/roles/roles.yaml")
with open(path, "r", encoding="utf-8") as f:
    roles = yaml.safe_load(f)

print(roles.keys())
print(roles["agents"].keys())
```

**Comentariu postat pe issue:**
> DONE C6.2 roles integration
> Final file: assets/roles/roles.yaml
> Agents included: personalist_salvator, anti_sistem, anti_suveranist, conspirationist, pro_european
> YAML load test: passed
> Integrated by: @catalinaminciuna
> Notes: roles.yaml was integrated under a single agents key and loaded successfully.

---

### 4.3. Issue C8.2 — Finalize EchoChamber Studio team app

**Obiectiv:** finalizarea aplicației Gradio `app/app.py` pentru demo în clasă.

**Cerințe principale:** încărcare știre din URL, rezumare articol, întrebări despre articol, răspuns de la un agent, rulare toți agenții, dezbatere multi-agent, layout pe două coloane, fără fragmente FAISS în UI, `DEFAULT_K = 5` în backend, `generate_agent_response(..., k=DEFAULT_K)` și `run_thread(..., k=DEFAULT_K)`.

**Structura finală:**

*Sidebar:* Provider · Model · Temperature · News URL · Load news button · News preview.

*Main tabs:* Chat · Agent · Toți agenții · Dezbatere.

**Îmbunătățiri aduse:**
- `DEFAULT_K = 5` păstrat în backend;
- eliminarea sliderului pentru fragmente FAISS;
- ascunderea contextului RAG brut din UI;
- agenții afișați în carduri clare;
- culorile din `roles.yaml` folosite consistent;
- preview de știre mai curat;
- temperatură implicită setată la 0.3;
- curățarea mențiunilor `@username` din mesajele afișate în dezbatere;
- testare cu providerele `gemini` și `deepseek`;
- model demo: `gemini-2.5-flash-lite` (stabil și rapid).

**Verificare:**
```bash
python -m py_compile app/app.py
python -m app.app
```

**Comentariu postat pe issue:**
> DONE C8.2 Final app - @catalinaminciuna
> Updated file: app/app.py, assets/roles/roles.yaml
> App starts with python -m app.app: yes
> News loading works: yes
> News summary works: yes
> Agent with article works: yes
> All agents with article works: yes
> Debate works: yes
> Agents tested: anti_sistem, conspirationist, pro_european
> Provider tested: gemini, deepseek
> One issue noticed: minor Gradio/Safari processing behavior during testing, handled by simplifying the UI flow and preview updates.

---

### 4.4. Issue C8.2.1 — Polish final app for demo (issue creat de mine)

**Obiectiv:** îmbunătățirea aplicației finale înainte de demo, prin rezolvarea problemelor de UI/UX și stabilizarea comportamentului în browser.

**Context:** după C8.2, în testarea în browser au apărut probleme mici dar deranjante:
- unele evenimente Gradio rămâneau blocate pe `processing`;
- tab-ul Chat avea probleme de compatibilitate cu `gr.Chatbot`;
- preview-ul știrii era prea lung și uneori tăia cuvinte;
- culorile agenților nu erau suficient de vizibile pe fundal închis;
- mesajele din dezbatere începeau uneori cu `@username`;
- temperatura implicită era prea mare pentru output stabil de demo.

**Tasks rezolvate:**
- înlocuirea fluxului instabil `Chatbot` cu un flux mai simplu de textbox întrebare/răspuns;
- adăugare `timeout` și `max_tokens` la apelul LLM;
- setarea temperaturii implicite la `0.3`;
- îmbunătățirea preview-ului articolului (scurt, lizibil, fără tăieri de cuvinte);
- menținerea preview-ului întotdeauna vizibil pentru a evita problemele `processing` din Gradio/Safari;
- folosirea culorilor agenților din `assets/roles/roles.yaml` în cardurile aplicației;
- îmbunătățirea contrastului culorilor pe fundal închis;
- eliminarea mențiunilor `@username` de la începutul mesajelor din dezbatere înainte de afișare;
- păstrarea `DEFAULT_K = 5` în backend și ascunderea fragmentelor FAISS din UI.

**Fișiere modificate:** `app/app.py`, opțional `assets/roles/roles.yaml` (pentru ajustări la culori sau reguli de rol).

**Verificare:**
```bash
python -m py_compile app/app.py
python -m app.app
```

**Comentariu postat pe issue:**
> DONE C8.2.1 app polish - @catalinaminciuna
> Updated files: app/app.py, assets/roles/roles.yaml
> App starts with python -m app.app: yes
> News loading works: yes
> News summary works: yes
> Chat works: yes
> Agent cards/colors improved: yes
> Debate works: yes
> Notes: fixed preview, processing issues, chat compatibility, default temperature, agent colors and debate display.

---

## 5. Fișiere relevante la care am contribuit

```
notebooks/student_02/
data/raw/student_02_youtube_raw.jsonl
data/cleaned/student_02_youtube_clean.jsonl
data/bubbles/anti_sistem.jsonl
assets/vectorstores/anti_sistem/index.faiss
assets/vectorstores/anti_sistem/index.pkl
assets/roles/role_02.yaml
assets/roles/roles.yaml
app/app.py
prompts/annotation_prompt.md
scripts/annotate_axis.py
outputs/c1_first_summary.json
outputs/student_02_prompt_outputs.csv
```

---

## 6. Probleme întâlnite și soluții

### 6.1. Adaptarea notebook-urilor pentru Mac

Unele notebook-uri aveau inițial căi hardcodate pentru Windows (`Path(r"C:\PROJECTS\echochamber-app")`). Le-am adaptat pentru macOS folosind detectarea automată a rădăcinii proiectului:

```python
ROOT = Path.cwd()
while not (ROOT / ".env").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
```

Notebook-urile pot fi rulate din `notebooks/student_02`, dar lucrează corect cu fișierele din root.

### 6.2. Validarea YAML

La integrarea `roles.yaml`, problema principală a fost poziționarea corectă a notebook-ului în root și evitarea duplicării blocului `agents:`. Am rezolvat printr-un test Python care caută fișierul și îl încarcă cu `yaml.safe_load()`.

### 6.3. Stabilitatea aplicației Gradio

În testare, Safari păstra uneori starea de `processing`. Am simplificat update-urile UI, am evitat schimbările dinamice inutile de vizibilitate și am preferat componente mai simple pentru demo. Soluția definitivă a venit prin issue-ul C8.2.1.

### 6.4. Controlul tonului agenților

Unele răspunsuri ale `anti_sistem` erau prea agresive pentru un context academic. Am ajustat regulile rolului pentru a păstra vocea anti-sistem, dar fără formulări excesiv de vulgare.

### 6.5. JSON cu fence-uri Markdown

Gemini 2.5 Flash Lite prefixa răspunsurile JSON cu ` ```json ` chiar dacă promptul cerea „fără markdown". Soluția: parser regex care extrage primul obiect `{...}` din răspuns.

```python
def parse_json_safe(raw):
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)
```

---

## 7. Concluzie

Contribuția mea a acoperit întregul traseu al unui agent în EchoChamber Studio: de la colectarea și curățarea datelor, la definirea axelor de adnotare, selecția corpusului, construirea vectorstore-ului FAISS, testarea agentului RAG (manual, cu LangChain, ca agentic RAG cu tools, și cu RSS), integrarea într-un workflow LangGraph și finalizarea aplicației Gradio.

Agentul `anti_sistem` este ancorat într-un corpus real de 50 de comentarii, iar răspunsurile sale sunt generate prin combinarea rolului discursiv cu fragmente recuperate semantic din FAISS. Prin issue-urile de echipă C4.1, C6.2, C8.2 și C8.2.1, am contribuit la integrarea materialelor comune, la consolidarea rolurilor agenților și la finalizarea + polisarea aplicației demonstrabile.

Prin această activitate am înțeles cum se leagă toate etapele proiectului: LLM-uri, API-uri, colectare de date, curățare, adnotare, embeddings, FAISS, RAG, LangGraph și Gradio. EchoChamber Studio nu este o aplicație separată, ci rezultatul integrării tuturor acestor pași într-un sistem funcțional și demonstrabil — și, la fel de important, un sistem care recunoaște explicit limitele simulării și nevoia de moderare umană.
