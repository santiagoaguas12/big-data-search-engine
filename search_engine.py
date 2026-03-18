"""
search_engine.py
----------------
Motor de búsqueda académico implementado desde cero.

Pipeline:
    load_corpus() → preprocess() → build_index() → search()

Modelo de ranking : BM25 (Best Match 25, Robertson et al. 1994)
Índice            : Índice invertido con posting lists
Preprocesamiento  : normalización Unicode, limpieza, tokenización,
                    stopwords, Snowball stemming (inglés)
Highlighting      : server-side, stem-aware — mapea stems a formas
                    originales del corpus para resaltar correctamente
"""

import html as _html
import json
import math
import os
import re
import unicodedata
from collections import defaultdict

from nltk.stem import SnowballStemmer   # pip install nltk


# ---------------------------------------------------------------------------
# Ruta al corpus
# ---------------------------------------------------------------------------
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.json")


# ---------------------------------------------------------------------------
# Parámetros BM25 (valores canónicos de la literatura)
# ---------------------------------------------------------------------------
DEFAULT_K1 = 1.5    # Saturación de tf. Rango típico: [1.2, 2.0]
DEFAULT_B  = 0.75   # Penalización por longitud. 0 = ninguna, 1 = total


# ---------------------------------------------------------------------------
# Stopwords en inglés — lista local, sin descarga de datos NLTK
# ---------------------------------------------------------------------------
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "not", "no", "nor",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "so", "yet", "both", "either", "neither", "each",
    "than", "too", "very", "just", "as", "if", "then",
    "because", "while", "although", "however", "therefore", "thus",
    "also", "between",
    "s", "t", "re", "ve", "ll", "d", "m",   # contracciones residuales
})


# ===========================================================================
# Clase principal
# ===========================================================================
class SearchEngine:
    """
    Motor de búsqueda basado en BM25 con índice invertido.

    Uso típico:
        engine = SearchEngine()
        engine.load_corpus()
        engine.build_index()
        results = engine.search("startup funding")
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        self.k1 = k1
        self.b  = b

        # Snowball (English) — más consistente que el Porter original de NLTK
        self._stemmer = SnowballStemmer("english")

        # ---- Corpus --------------------------------------------------------
        self.corpus: list[dict] = []
        self.doc_count: int = 0

        # ---- Índice invertido ----------------------------------------------
        # index[stem][doc_id] = frecuencia raw del stem en ese documento
        self.index: dict[str, dict[int, int]] = defaultdict(dict)

        # ---- Estadísticas --------------------------------------------------
        self.doc_lengths: dict[int, int] = {}
        self.avg_doc_length: float = 0.0
        self.df: dict[str, int] = {}          # document frequency por stem
        self.vocab_size: int = 0
        self._index_built: bool = False

        # ---- Mapa stem → formas originales del corpus (para highlighting) --
        # Ejemplo: "fund" → {"fund", "funds", "funded", "funding", "funder"}
        self.stem_to_forms: dict[str, set[str]] = defaultdict(set)

    # =======================================================================
    # 1. CARGA DEL CORPUS
    # =======================================================================
    def load_corpus(self) -> bool:
        """Lee corpus.json y carga los documentos en memoria."""
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
            self.doc_count = len(self.corpus)
            print(f"[INFO] Corpus cargado: {self.doc_count} documentos.")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Archivo no encontrado: {CORPUS_PATH}")
            return False
        except json.JSONDecodeError as e:
            print(f"[ERROR] corpus.json inválido: {e}")
            return False

    # =======================================================================
    # 2. PREPROCESAMIENTO — pipeline unificado (indexación + queries)
    # =======================================================================
    def preprocess(self, text: str) -> list[str]:
        """
        Pipeline de normalización y tokenización:
          1. Normalización Unicode NFKD → ASCII  (é→e, ü→u, …)
          2. Lowercase
          3. Separar palabras con guion en tokens independientes
          4. Conservar solo letras, dígitos y espacios
          5. Tokenizar por whitespace
          6. Eliminar stopwords y tokens de longitud < 2
          7. Snowball stemming

        Idéntico para documentos y para queries, garantizando
        que los términos indexados y consultados sean comparables.
        """
        # 1. Unicode → ASCII
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")

        # 2. Lowercase
        text = text.lower()

        # 3. Separar en guiones (e-commerce → e commerce)
        text = text.replace("-", " ")

        # 4. Solo letras, dígitos y espacios
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # 5. Tokenizar
        tokens = text.split()

        # 6. Filtrar stopwords y tokens muy cortos
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]

        # 7. Snowball stemming
        tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens

    def _tokenize_with_originals(self, text: str) -> list[tuple[str, str]]:
        """
        Igual que preprocess(), pero devuelve pares (forma_limpia, stem).
        Usado en build_index() para construir stem_to_forms.

        La 'forma_limpia' es el token después de Unicode+lower+clean,
        pero antes del stemming — es lo que aparece en el texto visible
        en su variante lowercase.
        """
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = text.lower().replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        pairs = []
        for t in text.split():
            if t not in STOPWORDS and len(t) >= 2:
                pairs.append((t, self._stemmer.stem(t)))
        return pairs

    # =======================================================================
    # 3. CONSTRUCCIÓN DEL ÍNDICE INVERTIDO
    # =======================================================================
    def build_index(self) -> None:
        """
        Construye el índice invertido y las estadísticas del corpus.

        Produce:
          - index[stem][doc_id]  : frecuencia raw del stem en el documento
          - doc_lengths[doc_id]  : longitud del documento en tokens
          - df[stem]             : número de documentos que contienen el stem
          - avg_doc_length       : longitud media del corpus
          - vocab_size           : número de stems únicos
          - stem_to_forms[stem]  : formas originales (lowercase) del stem
        """
        if not self.corpus:
            print("[WARN] Corpus vacío — llena corpus.json antes de indexar.")
            return

        # Resetear estructuras para permitir re-indexación
        self.index.clear()
        self.doc_lengths.clear()
        self.stem_to_forms.clear()
        total_tokens = 0

        for doc in self.corpus:
            doc_id   = doc["id"]
            raw_text = f"{doc.get('title', '')} {doc.get('text', '')}"
            pairs    = self._tokenize_with_originals(raw_text)

            # Actualizar mapa stem → formas originales
            for original, stem in pairs:
                self.stem_to_forms[stem].add(original)

            # Contar frecuencias de stem en este documento
            term_freq: dict[str, int] = defaultdict(int)
            for _, stem in pairs:
                term_freq[stem] += 1

            for stem, freq in term_freq.items():
                self.index[stem][doc_id] = freq

            self.doc_lengths[doc_id] = len(pairs)
            total_tokens += len(pairs)

        self.df             = {t: len(p) for t, p in self.index.items()}
        self.avg_doc_length = total_tokens / self.doc_count if self.doc_count else 1.0
        self.vocab_size     = len(self.index)
        self._index_built   = True

        print(
            f"[INFO] Índice construido — "
            f"Vocab: {self.vocab_size} stems | "
            f"Docs: {self.doc_count} | "
            f"Avg. doc length: {self.avg_doc_length:.1f} tokens"
        )

    # =======================================================================
    # 4. SCORING BM25
    # =======================================================================
    def _bm25_score(self, query_tokens: list[str], doc_id: int) -> float:
        """
        Score BM25 para un par (query, doc).

        BM25(Q,D) = Σ_t  IDF(t) · [tf(t,D)·(k1+1)] / [tf(t,D) + k1·(1-b+b·|D|/avgdl)]
        IDF(t)    = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
        """
        score   = 0.0
        doc_len = self.doc_lengths.get(doc_id, 0)

        for term in query_tokens:
            if term not in self.index:
                continue
            tf = self.index[term].get(doc_id, 0)
            if tf == 0:
                continue

            df_t       = self.df[term]
            idf        = math.log((self.doc_count - df_t + 0.5) / (df_t + 0.5) + 1.0)
            normalizer = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length)
            tf_norm    = tf * (self.k1 + 1.0) / normalizer
            score     += idf * tf_norm

        return score

    # =======================================================================
    # 5. BÚSQUEDA
    # =======================================================================
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Ejecuta una consulta BM25 y devuelve los top_k documentos más relevantes.

        Cada resultado incluye:
          - id, title, text, source, category  (campos del corpus)
          - score          : BM25 redondeado a 4 decimales
          - snippet        : fragmento relevante en texto plano
          - snippet_html   : snippet con términos resaltados en HTML (<mark>)
          - title_html     : título con términos resaltados en HTML (<mark>)
        """
        if not self._index_built:
            return []

        query_tokens = self.preprocess(query)
        if not query_tokens:
            return []

        # Recuperar candidatos (union de posting lists)
        candidate_ids: set[int] = set()
        for token in query_tokens:
            if token in self.index:
                candidate_ids.update(self.index[token].keys())

        if not candidate_ids:
            return []

        # Calcular y ordenar por BM25
        scored = sorted(
            ((self._bm25_score(query_tokens, did), did) for did in candidate_ids),
            key=lambda x: x[0],
            reverse=True,
        )

        doc_lookup = {doc["id"]: doc for doc in self.corpus}
        results    = []

        for bm25_score, doc_id in scored[:top_k]:
            doc     = dict(doc_lookup[doc_id])
            snippet = self._generate_snippet(doc.get("text", ""), query_tokens)

            doc["score"]        = round(bm25_score, 4)
            doc["snippet"]      = snippet
            doc["snippet_html"] = self._highlight_text(snippet,           query_tokens)
            doc["title_html"]   = self._highlight_text(doc.get("title", ""), query_tokens)
            results.append(doc)

        return results

    # =======================================================================
    # 6. UTILIDADES
    # =======================================================================
    def _generate_snippet(self, text: str, query_tokens: list[str],
                          window: int = 30) -> str:
        """
        Extrae un fragmento del texto centrado en la primera ocurrencia
        de cualquier stem de la query.
        """
        words         = text.split()
        stemmed_words = [self._stemmer.stem(
                             unicodedata.normalize("NFKD", w.lower())
                             .encode("ascii", "ignore").decode("ascii")
                         ) for w in words]

        for i, sw in enumerate(stemmed_words):
            if sw in query_tokens:
                start    = max(0, i - window)
                end      = min(len(words), i + window + 1)
                fragment = " ".join(words[start:end])
                prefix   = "…" if start > 0 else ""
                suffix   = "…" if end < len(words) else ""
                return f"{prefix}{fragment}{suffix}"

        return text[:280] + ("…" if len(text) > 280 else "")

    def _highlight_text(self, text: str, query_tokens: list[str]) -> str:
        """
        Devuelve HTML con los términos relevantes envueltos en <mark>.

        Estrategia (stem-aware):
          1. Para cada stem en query_tokens, recupera todas las formas
             originales (lowercase) vistas en el corpus desde stem_to_forms.
          2. También incluye el propio stem (forma base).
          3. HTML-escapa el texto original para evitar XSS.
          4. Aplica regex word-boundary case-insensitive para envolver
             cada forma en <mark>…</mark>.

        Resultado: "funding rounds" con query "fund" → "<mark>funding</mark> rounds"
        """
        forms: set[str] = set()
        for stem in query_tokens:
            forms.update(self.stem_to_forms.get(stem, set()))
            forms.add(stem)

        # HTML-escapar primero (seguro: solo añadimos <mark> nosotros)
        escaped = _html.escape(text)

        if not forms:
            return escaped

        # Ordenar por longitud desc para evitar solapamientos parciales
        pattern = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True) if f)
        if not pattern:
            return escaped

        highlighted = re.sub(
            rf"\b({pattern})\b",
            r"<mark>\1</mark>",
            escaped,
            flags=re.IGNORECASE,
        )
        return highlighted

    def debug_query(self, query: str, top_k: int = 10) -> dict:
        """
        Inspecciona el pipeline completo para una query.
        Útil para debugging y defensa del proyecto.

        Retorna:
          - query_original     : texto tal como lo escribió el usuario
          - query_tokens       : stems resultantes del preprocesamiento
          - unmatched_tokens   : stems que no están en el índice
          - matched_terms      : stem → lista de doc_ids que lo contienen
          - surface_forms      : stem → formas originales del corpus
          - candidate_count    : número de documentos candidatos
          - top_scores         : lista de {doc_id, title, bm25_score}
        """
        if not self._index_built:
            return {"error": "El índice no está construido."}

        query_tokens = self.preprocess(query)
        doc_lookup   = {doc["id"]: doc for doc in self.corpus}

        matched_terms: dict[str, list[int]] = {}
        candidate_ids: set[int] = set()

        for token in query_tokens:
            if token in self.index:
                ids = sorted(self.index[token].keys())
                matched_terms[token] = ids
                candidate_ids.update(ids)

        scored = sorted(
            ((self._bm25_score(query_tokens, did), did) for did in candidate_ids),
            key=lambda x: x[0],
            reverse=True,
        )

        return {
            "query_original":   query,
            "query_tokens":     query_tokens,
            "unmatched_tokens": [t for t in query_tokens if t not in self.index],
            "matched_terms":    matched_terms,
            "surface_forms":    {
                t: sorted(self.stem_to_forms.get(t, set()))
                for t in query_tokens
            },
            "candidate_count":  len(candidate_ids),
            "top_scores":       [
                {
                    "doc_id":    did,
                    "title":     doc_lookup[did].get("title", "?"),
                    "bm25":      round(sc, 4),
                }
                for sc, did in scored[:top_k]
            ],
        }

    def get_stats(self) -> dict:
        """Métricas del corpus e índice para la UI y la API /api/stats."""
        return {
            "total_documents": self.doc_count,
            "vocab_size":      self.vocab_size,
            "avg_doc_length":  round(self.avg_doc_length, 1),
            "index_built":     self._index_built,
            "bm25_k1":         self.k1,
            "bm25_b":          self.b,
            "corpus_path":     CORPUS_PATH,
        }

    def get_index_data(self, top_n: int = 300,
                       term_filter: str = "",
                       page: int = 1,
                       per_page: int = 50) -> dict:
        """
        Datos del índice invertido estructurados para la página de visualización.
        """
        doc_titles = {
            doc["id"]: doc.get("title", f"Doc {doc['id']}")
            for doc in self.corpus
        }

        all_sorted = sorted(self.df.items(), key=lambda x: x[1], reverse=True)

        # Top 15 para el gráfico (sin filtro)
        top_chart = [
            {"term": t, "df": d, "total_tf": sum(self.index[t].values())}
            for t, d in all_sorted[:15]
        ]

        if term_filter:
            all_sorted = [(t, d) for t, d in all_sorted if term_filter in t]

        all_sorted  = all_sorted[:top_n]
        total_terms = len(all_sorted)

        start      = (page - 1) * per_page
        page_terms = all_sorted[start: start + per_page]

        entries = []
        for rank, (term, df_val) in enumerate(page_terms, start=start + 1):
            posting = self.index[term]
            idf     = math.log((self.doc_count - df_val + 0.5) / (df_val + 0.5) + 1.0)
            entries.append({
                "rank":     rank,
                "term":     term,
                "df":       df_val,
                "idf":      round(idf, 4),
                "total_tf": sum(posting.values()),
                "posting_list": [
                    {"doc_id": did, "tf": tf,
                     "title": doc_titles.get(did, f"Doc {did}")}
                    for did, tf in sorted(posting.items())
                ],
            })

        return {
            "entries":         entries,
            "total_terms":     total_terms,
            "page":            page,
            "per_page":        per_page,
            "total_pages":     max(1, math.ceil(total_terms / per_page)),
            "top_chart_terms": top_chart,
            "doc_lengths":     self.doc_lengths,
            "doc_titles":      doc_titles,
        }


# ---------------------------------------------------------------------------
# Instancia global — importada por app.py
# ---------------------------------------------------------------------------
engine = SearchEngine()

