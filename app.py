"""
app.py
------
Punto de entrada de la aplicación Flask.
Define las rutas HTTP y conecta la interfaz web con el motor de búsqueda.

Ejecutar con:
    python app.py
"""

import json
import os
import time

from flask import Flask, render_template, request, jsonify
from search_engine import engine, CORPUS_PATH

# ---------------------------------------------------------------------------
# Inicialización de Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key-change-in-production"

# Error de corpus persistente: se setea al arrancar y se pasa a cada vista
_corpus_error: str | None = None


# ---------------------------------------------------------------------------
# Validación del corpus
# ---------------------------------------------------------------------------
def validate_corpus() -> tuple[bool, str | None]:
    """
    Verifica que corpus.json exista, sea JSON válido y tenga la
    estructura mínima correcta.

    Returns:
        (True, None)            si todo está bien.
        (False, "mensaje")      si hay un problema.
    """
    if not os.path.exists(CORPUS_PATH):
        return False, f"No se encontró corpus.json en: {CORPUS_PATH}"

    try:
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"corpus.json tiene formato JSON inválido: {e}"

    if not isinstance(data, list):
        return False, "corpus.json debe ser un array JSON (lista de documentos)."

    if len(data) == 0:
        return False, ("corpus.json está vacío. "
                       "Agrega al menos un documento con los campos: id, title, text.")

    required_fields = {"id", "title", "text"}
    for i, doc in enumerate(data):
        missing = required_fields - set(doc.keys())
        if missing:
            return False, (f"Documento #{i+1} (id={doc.get('id', '?')}) "
                           f"le falta los campos requeridos: {', '.join(missing)}.")

    return True, None


# ---------------------------------------------------------------------------
# Inicialización del motor — se ejecuta una sola vez al arrancar
# ---------------------------------------------------------------------------
def initialize_engine() -> None:
    """Valida el corpus, lo carga y construye el índice."""
    global _corpus_error
    print("[INFO] Inicializando motor de búsqueda...")

    valid, error_msg = validate_corpus()
    if not valid:
        _corpus_error = error_msg
        print(f"[ERROR] Corpus inválido: {error_msg}")
        return

    loaded = engine.load_corpus()
    if loaded:
        engine.build_index()
        print(f"[INFO] Motor listo — {engine.get_stats()}")
    else:
        _corpus_error = "No se pudo cargar corpus.json. Revisa el archivo y reinicia."
        print(f"[ERROR] {_corpus_error}")


# ---------------------------------------------------------------------------
# Deshabilitar caché del navegador en todas las respuestas HTML
# ---------------------------------------------------------------------------
@app.after_request
def no_browser_cache(response):
    if "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Helper: parámetros comunes a todas las vistas
# ---------------------------------------------------------------------------
def _base_ctx() -> dict:
    """Devuelve el contexto base que toda vista necesita."""
    return {
        "stats": engine.get_stats(),
        "corpus_error": _corpus_error,
    }


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Ruta principal — renderiza la pantalla de bienvenida."""
    return render_template("index.html", **_base_ctx())


@app.route("/search", methods=["GET"])
def search():
    """
    Ruta de búsqueda.

    Query params:
        q     (str)  – Texto de la consulta.
        top_k (int)  – Número máximo de resultados (default: 10, max: 50).
    """
    ctx = _base_ctx()
    query = request.args.get("q", "").strip()

    # Bounds-check de top_k para evitar abusos
    try:
        top_k = max(1, min(int(request.args.get("top_k", 10)), 50))
    except (ValueError, TypeError):
        top_k = 10

    debug_mode = request.args.get("debug", "0") == "1"

    if not query:
        ctx["error"] = "Por favor escribe una consulta antes de buscar."
        return render_template("index.html", **ctx)

    if _corpus_error or not ctx["stats"]["index_built"]:
        ctx["error"] = "El índice no está disponible. Revisa corpus.json y reinicia el servidor."
        ctx["query"] = query
        return render_template("index.html", **ctx)

    # Detectar queries que producen cero tokens tras el preprocesamiento
    # (p. ej. solo stopwords o solo caracteres especiales)
    if not engine.preprocess(query):
        ctx["query"] = query
        ctx["error"] = (
            "La consulta contiene únicamente stopwords o caracteres no indexables. "
            "Prueba con términos más específicos."
        )
        return render_template("index.html", **ctx)

    t0 = time.perf_counter()
    results = engine.search(query, top_k=top_k)
    search_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    debug_info = engine.debug_query(query, top_k=top_k) if debug_mode else None

    ctx.update({
        "query":          query,
        "results":        results,
        "result_count":   len(results),
        "search_time_ms": search_time_ms,
        "debug_info":     debug_info,
        "debug_mode":     debug_mode,
    })
    return render_template("index.html", **ctx)


@app.route("/api/search", methods=["GET"])
def api_search():
    """
    Endpoint JSON para consumo programático.

    Query params:
        q     (str)  – Texto de la consulta.
        top_k (int)  – Número máximo de resultados (default: 10, max: 50).

    Returns:
        JSON con lista de resultados y metadatos de la búsqueda.
    """
    query = request.args.get("q", "").strip()

    try:
        top_k = max(1, min(int(request.args.get("top_k", 10)), 50))
    except (ValueError, TypeError):
        top_k = 10

    if not query:
        return jsonify({"error": "Parámetro 'q' requerido.", "results": []}), 400

    if _corpus_error or not engine.get_stats()["index_built"]:
        return jsonify({"error": "Índice no disponible.", "results": []}), 503

    t0 = time.perf_counter()
    results = engine.search(query, top_k=top_k)
    search_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Excluir campos HTML del JSON (snippet_html / title_html son solo para la UI)
    clean_results = [
        {k: v for k, v in r.items() if k not in ("snippet_html", "title_html")}
        for r in results
    ]

    return jsonify({
        "query":          query,
        "total_results":  len(clean_results),
        "search_time_ms": search_time_ms,
        "results":        clean_results,
    })


@app.route("/index-visualization", methods=["GET"])
def index_visualization():
    """
    Enhancement F — Index Visualization.
    Muestra el índice invertido completo con estadísticas, gráfico de
    frecuencias, posting lists y referencia de documentos del corpus.

    Query params:
        term  (str)  – Filtro de subcadena para buscar en el vocabulario.
        page  (int)  – Página de resultados (default: 1).
    """
    ctx = _base_ctx()

    if _corpus_error or not ctx["stats"]["index_built"]:
        ctx["error"] = "El índice no está disponible. Revisa corpus.json y reinicia el servidor."
        return render_template("index_viz.html", **ctx)

    term_filter = request.args.get("term", "").strip().lower()

    try:
        page = max(1, int(request.args.get("page", 1)))
    except (ValueError, TypeError):
        page = 1

    per_page = 50
    index_data = engine.get_index_data(
        top_n=500,
        term_filter=term_filter,
        page=page,
        per_page=per_page,
    )

    ctx.update(index_data)
    ctx["term_filter"] = term_filter
    return render_template("index_viz.html", **ctx)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Devuelve las estadísticas del motor en formato JSON."""
    stats = engine.get_stats()
    stats["corpus_error"] = _corpus_error
    return jsonify(stats)


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    initialize_engine()
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
