# Contribuția mea la proiectul EchoChamber

> **Valeria Ianițchi** · `@valeriaianitchi` · Echipa 3
> Curs: *AI Avansat / Analiza Datelor Complexe* · Facultatea de Sociologie și Asistență Socială, UBB
> Repository: [`AIE-2026/echochamber-project-team3`](https://github.com/AIE-2026/echochamber-project-team3)

---

## Agentul meu: Personalist-salvator

În cadrul proiectului de echipă **EchoChamber** — un simulator de bule discursive politice românești bazat pe comentarii YouTube reale — am construit agentul **Personalist-salvator**.

**Vocea agentului:** devotat, admirativ, sigur. Vede liderul ca soluție excepțională pentru un sistem perceput ca în criză. Tonul este afectiv, emoțional, încrezător, cu accent pe „trezirea poporului" și nedreptățile sistemului.

**De ce această bulă?** Combină limbaj afectiv pozitiv (admirație, loialitate) cu suspiciune față de instituții — o tensiune contraintuitivă pentru un model de clasificare obișnuit. Vroiam să testez dacă RAG poate captura exact această ambivalență, nu doar un stereotip.

---

## Ce am livrat pentru echipă

### C3 — Colectare și curățare corpus

**Ce am făcut:** Am descărcat și integrat în repo corpusul mare provided de instructor — **30.753 înregistrări** de comentarii YouTube brute.

- Fișier salvat: `data/raw/corpus_youtube_large_raw.jsonl`
- Am verificat manual că fișierul conține comentarii YouTube valide
- Am actualizat `.gitignore` ca să nu trimit datele pe GitHub
- Am documentat divergența de structură (am salvat în `data/raw/` în loc de `data/provided/` cum cerea issue-ul) și am notificat echipa

**De ce a contat:** Acest fișier e dataset-ul principal pentru toți pașii următori — adnotare, tipologie, vectorstore. Fără el, restul echipei nu putea avansa.

**Ce am făcut:** Am rulat scriptul `scripts/clean_youtube.py` pe corpusul brut și am livrat versiunea curată pentru întreaga echipă.

**Reguli de curățare aplicate:**
- Eliminare linkuri (regex `http\S+`)
- Normalizare spații (multiple whitespace → spațiu single)
- Filtru lungime minimă: **≥60 caractere** (comentariile mai scurte sunt prea ambigue pentru analiză discursivă)
- Filtru proporție litere: **≥50%** (eliminăm comentariile dominate de emoji / simboluri)
- Deduplicare pe text lowercase
- Păstrare text original în `text_raw` + text curățat în `text`

**Rezultat:**
- Input: `data/raw/corpus_youtube_large_raw.jsonl` (30.753 comentarii brute)
- Output: `data/cleaned/corpus_youtube_large_clean.jsonl`
- Toți colegii au lucrat ulterior pe această versiune curată — adnotare, tipologie, selecție bule

**Ce am făcut:** Am implementat scriptul reutilizabil `scripts/collect_youtube.py` pentru întreaga echipă.

**Funcționalități:**
- Citește `YOUTUBE_API_KEY` din `.env` (nu hardcoded — securitate)
- Acceptă argumente CLI: `--handle`, `--max-videos`, `--max-comments`, `--output`
- Convertește handle YouTube → channel_id prin endpoint `/channels`
- Colectează cele mai recente N videoclipuri din canal (ordonate după dată)
- Pentru fiecare videoclip, colectează comentariile publice ordonate după relevanță
- Salvează rezultatul ca JSONL cu schema standardizată: `id`, `source_platform`, `source_channel`, `text_raw`, `video_id`, `video_title`, `video_date`, `comment_date`, `likes`, `collected_at`

**Comenzi:**
```bash
python scripts/collect_youtube.py \
  --handle digi24hd56 \
  --max-videos 2 \
  --max-comments 50 \
  --output data/raw/student_01_youtube_raw.jsonl
```

**De ce a contat:** Este scriptul **shared** pe care îl folosesc toți colegii când vor să colecteze date noi. Am actualizat și `.env.example` cu cheia `YOUTUBE_API_KEY`, fără să expun chei reale.

---

## Contribuția mea individuală (Teme 1+2+3)

### Tema 1 — Primul prompt exploratoriu (C3)

Pe corpusul curat livrat de mine la Issue #26, am construit și testat **primul prompt** de analiză discursivă pe 10 comentarii eșantionate.

**Promptul cerea identificarea pe 7 dimensiuni:**
- `target` — ținta principală a comentariului
- `stance` — poziționarea față de țintă (pro / anti / neutru / ambiguu)
- `sentiment` — polaritatea generală
- `tone` — modul de formulare
- `topic` — tema
- `interpretation_problem` — semnale de ambiguitate
- `reason` — justificarea adnotării

**Reflecție critică documentată în notebook:**

> Promptul a identificat corect ținta pe **9 din 10 comentarii**. Comentariile cu o singură țintă clară au fost clasificate consistent.
>
> **Unde a eșuat:**
> - *„Multă sănătate dl. Președinte..."* — marcat **stance pro**, deși comentariul e sarcastic. Modelul a fost păcălit de politețea de suprafață.
> - *„Dumnezeu să-i apere pe cei care stau acolo"* — `target unclear`, deși ținta era Primarul Negoiță, deductibilă din titlul video-ului. Modelul nu folosește contextul video.
> - Un denunț implicit de evaziune a fost marcat **stance neutru** pentru că nu conținea cuvinte negative explicite.
>
> Pe 8 din 10 cazuri, sentiment și stance au aceeași polaritate. La cazul sarcastic, modelul a marcat `interpretation_problem=sentiment_vs_stance` — adică **sesizează tensiunea, dar nu știe să o rezolve**.
>
> **Ce aș schimba la versiunea următoare:**
> 1. Few-shot examples pentru sarcasm românesc învelit în politețe
> 2. Includerea `video_title` în prompt pentru ținte implicite
> 3. Sample stratificat în loc de `df.head(10)` — a prins comentarii prea ușoare

---

### Tema 2 — Construirea agentului Personalist-salvator (C5 + C6)

#### Pasul 1 — Selecția corpusului pentru bula mea (C5)
- Am pornit de la corpusul tipologizat al echipei (rezultatul adnotării LLM + tipologiei rule-based)
- Am filtrat după `discourse_type == "T1_suport_personalist"`
- Am sortat după `type_confidence` descrescător
- Am inspectat manual primele 70 candidate
- Am eliminat textele slabe (ambigue, prea scurte, off-topic) și am păstrat **50 comentarii curate**
- Output: `data/bubbles/personalist_salvator.jsonl`

#### Pasul 2 — Vectorstore FAISS (C5_02)
- Model embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (suport pentru română)
- 50 comentarii → 50 vectori de 384 dimensiuni
- Normalizare la lungime 1 → similaritate cosinus prin produs scalar
- Index: `IndexFlatIP` (exact search, suficient pentru 50 de vectori)
- Output salvat: `assets/vectorstores/personalist_salvator/index.faiss` + `index.pkl`

#### Pasul 3 — Definirea rolului discursiv

Am scris în `assets/roles/role_01.yaml` descrierea agentului meu:

> Agentul *Personalist-salvator* vede liderul ca pe o figură excepțională, capabilă să salveze țara și să repare un sistem corupt. Instituțiile sunt privite cu neîncredere, iar liderul este prezentat ca fiind mai apropiat de oameni decât clasa politică tradițională.
>
> Tonul folosit este afectiv, admirativ și emoțional. Apar frecvent idei despre „trezirea poporului", nedreptăți făcute de sistem și nevoia unui conducător puternic care să spună adevărul. Discursul pune accent pe patriotism, credință, onoare, curaj.
>
> Față de alte bule, această voce este mai puțin argumentativă și mai mult emoțională.

#### Pasul 4 — Agent RAG funcțional (C6)

Am testat agentul pe stimuli politici diverși și am validat că:
- Folosește contextul recuperat din FAISS (verificat: `role["system"][:50] in prompt`)
- Păstrează vocea agentului (verificat manual pe 10 stimuli)
- Nu inventează informații care nu apar în input sau context (limitări notate la cazurile excepționale)

#### Pasul 5 — Mini-agent cu tool-uri (C6.10)

Am integrat **două tool-uri** în agent:
1. `retrieve_similar_comments` — căutare semantică în FAISS
2. `get_latest_news_from_rss` — citește o știre recentă din feed-ul G4Media

Agentul ia singur o știre actuală și o comentează în vocea bulei, fără ca eu să dau manual stimulusul. Am verificat că ambele tool-uri sunt apelate.

---

### Tema 3 — Extensia mea în Gradio (C8)

În aplicația finală `app/app.py`, am adăugat **3 modificări vizibile**:

#### 1. Tab nou: „Markeri discursivi"
Detectează prezența cuvintelor-cheie specifice unei bule într-un text dat și calculează un scor de intensitate retorică.

**Funcția mea:** `detector_markeri(text, bula)`
- Numără markerii bulei (ex: pentru `anti_sistem`: "sistem", "mafia", "corupți", "hoți", "trădare")
- Calculează `scor = 100 × nr_markeri / total_cuvinte`
- Returnează verdict pe 3 niveluri:
  - 🟢 < 3% — intensitate mică (text aproape neutru)
  - 🟡 3–8% — intensitate medie (câteva semnale clare)
  - 🔴 ≥ 8% — intensitate mare (vocabular puternic marcat)

#### 2. Element de design
- Schimbat tema pe `gr.themes.Citrus()`
- Emoji la fiecare tab
- Header refăcut cu `gr.Row()` + două `gr.Column()` (titlu + subtitlu stânga, data + bula activă dreapta)
- Disclaimer etic vizibil sub header

#### 3. Reflecție pentru iterație viitoare
Lista de markeri e acum hardcodată după intuiție. Pentru o versiune mai bună, aș extrage automat markerii din corpus cu **TF-IDF pe comentariile fiecărei bule** — ar elimina bias-ul meu și ar produce markeri obiectivi, statistici. În plus, aș conecta tab-ul Markeri direct la output-ul tab-ului Agent, ca să poți inspecta automat ce generează modelul fără copy-paste.

---

## Sumar contribuție individuală

| Categorie | Livrat | Locație |
|---|---|---|
| Issues de echipă închise cu commit | 3 issues (#24, #25, #26) | GitHub team3 |
| Script colectare YouTube (shared) | `scripts/collect_youtube.py` | repo team3 |
| Script curățare corpus (shared) | `scripts/clean_youtube.py` | repo team3 |
| Corpus brut integrat (shared) | 30.753 comentarii | `data/raw/` |
| Corpus curat integrat (shared) | corpus_youtube_large_clean | `data/cleaned/` |
| Agent individual: Personalist-salvator | 50 comentarii + FAISS + rol YAML | `data/bubbles/`, `assets/` |
| Tema 1 — primul prompt | notebook + 10 cazuri analizate | `notebooks/student_01/` |
| Tema 2 — agent RAG complet | vectorstore + role card + agent funcțional | `core/` + `assets/` |
| Tema 3 — extensie Gradio | tab „Markeri" + design tematic | `app/app.py` |

--