# MySearchEngine — SANTIAGO AGUAS DIAZ
## Technological Entrepreneurship Domain

> **Course:** Big Data  
> **Student:** *Santiago Aguas Diaz*  
> **Institution:** *UAG*  
> **Date:** 18/03/2026

---

## Domain & Justification

**Chosen Domain:** Technological Entrepreneurship

*I chose this topic because, while I love the software field, I wouldn't want to dedicate my whole life to development. Lately, I've been focusing a lot on the business side—companies, startups, etc.—which I find much more interesting than development itself.*

---

## Project Purpose

Build a functional search engine **from scratch**, without any external IR
libraries (no Whoosh, Elasticsearch, Lucene, or equivalent). The system
demonstrates core Information Retrieval concepts through direct implementation:

- Corpus loading from a structured JSON file
- Text preprocessing pipeline (tokenization, stopwords, stemming)
- Inverted index with posting lists
- BM25 probabilistic ranking model
- Web interface served with Flask

---

## Features Implemented

| Feature | Description |
|---|---|
| **BM25 Search** | Robertson et al. (1994) probabilistic model with configurable k₁ and b |
| **Inverted Index** | `term → {doc_id: tf}` posting lists built at startup |
| **Porter Stemmer** | Reduces terms to their root form (e.g., *startup* → *startup*, *innovating* → *innov*) |
| **Stopword Filtering** | Local English stopword list — no external data download required |
| **Snippet Generation** | Context window around first query term hit in each document |
| **Query Highlighting** | Query terms highlighted in result snippets |
| **Search Timing** | Per-query latency reported in milliseconds |
| **Corpus Validation** | Startup check for JSON integrity, required fields, and non-empty corpus |
| **REST API** | JSON endpoints at `/api/search` and `/api/stats` |
| **Web Interface** | Responsive HTML/CSS/JS interface with stats panel and "How it works" section |
| **Enhancement F** | Index Visualization — interactive inverted index browser with bar charts, posting lists, IDF values, and term filter |

---

## Enhancement F — Index Visualization

Accessible at **`/index-visualization`** from the top navigation bar.

The page shows:

| Section | Description |
|---|---|
| **Corpus Statistics** | Documents, vocabulary size, average token length, BM25 k₁ and b parameters, index status |
| **Top Terms Bar Chart** | Horizontal CSS bar chart of the top 15 terms by document frequency (DF) — shows both DF and total TF |
| **Index Browser** | Paginated table (50 terms/page) of every term in the vocabulary |
| **Term filter** | Type a substring to instantly filter the vocabulary (e.g. `start` → shows `startup`, `startups`, …) |
| **Posting list** | Per-row pills showing every `doc N : tf=X` pair for each term |
| **IDF column** | Pre-computed BM25 IDF value per term |
| **Corpus Documents** | Reference grid showing all documents with their IDs and token counts |

### How to use

1. Run the app (`python app.py`) and open **http://localhost:5000**
2. Click **Índice** in the top navigation bar, or navigate directly to `/index-visualization`
3. The bar chart shows the top 15 most common terms at a glance
4. To inspect a specific term, type it (or a substring) in the **Filter by term stem** box and click **Filter**
5. Use the **← Prev / Next →** pagination buttons to browse all vocabulary terms
6. Hover over a posting pill to see the document title in a tooltip

---

## BM25 Scoring Model

BM25 (Best Match 25) is a probabilistic relevance model that scores documents
for a query by summing contributions from each query term:

```
BM25(D, Q) = Σ  IDF(t) × [tf(t,D) × (k₁ + 1)] / [tf(t,D) + k₁ × (1 - b + b × |D|/avgdl)]

IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

| Parameter | Value | Meaning |
|---|---|---|
| `k₁` | 1.5 | Term frequency saturation (higher = less saturation) |
| `b` | 0.75 | Document length normalization (1.0 = full, 0.0 = none) |
| `N` | varies | Total number of documents in the corpus |
| `df(t)` | varies | Number of documents containing term t |
| `avgdl` | varies | Average document length in tokens |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+ · Flask 3.x |
| IR Engine | Pure Python — no external IR libraries |
| Preprocessing | NLTK PorterStemmer (no data download required) |
| Frontend | HTML5 · CSS3 · JavaScript (vanilla) |
| Data | JSON (`corpus.json`) |

---

## Project Structure

```
my-search-engine/
│
├── app.py              ← Flask entry point: routes, corpus validation, error handling
├── search_engine.py    ← IR engine: preprocessing, inverted index, BM25 scoring
├── corpus.json         ← Document corpus (to be filled with real documents)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
│
├── templates/
│   ├── layout.html     ← Base HTML template (header, footer, nav)
│   └── index.html      ← Main view: search bar, stats panel, results, how-it-works
│
└── static/
    ├── style.css       ← All styles (academic palette, result cards, responsive)
    └── app.js          ← Query highlighting, loading indicator, card animations
```

---

## Installation & Running Locally

### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Fill the corpus (see section below)

### 4. Run the application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Available Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Main search interface |
| GET | `/search?q=<query>` | Web search (returns HTML) |
| GET | `/search?q=<query>&top_k=20` | Web search with custom result count |
| GET | `/index-visualization` | Enhancement F — inverted index browser |
| GET | `/index-visualization?term=start` | Filter vocabulary by substring |
| GET | `/index-visualization?page=2` | Paginate through terms (50/page) |
| GET | `/api/search?q=<query>` | Search — returns JSON |
| GET | `/api/stats` | Engine statistics — returns JSON |

---

## Corpus Format

The file `corpus.json` must be a **JSON array** of document objects.
Each document requires at minimum: `id`, `title`, and `text`.

```json
[
  {
    "id": 1,
    "title": "Title of the document",
    "text": "Full body text of the document. The more text, the better the BM25 index.",
    "source": "URL, book, article reference, etc.",
    "category": "Optional category (e.g., Startups, Fintech, Innovation)"
  }
]
```

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | int | **Yes** | Unique document identifier |
| `title` | string | **Yes** | Document title (indexed and displayed) |
| `text` | string | **Yes** | Full document content (used for indexing) |
| `source` | string | No | Bibliographic reference or URL |
| `category` | string | No | Thematic category for filtering display |

> After editing `corpus.json`, restart the server with `python app.py` to
> rebuild the index.

