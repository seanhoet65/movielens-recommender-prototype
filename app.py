import sys
import json
import time
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_data, temporal_split, get_movie_display
from src.recommenders.nonpersonalised import (
    TopNRecommender,
    BayesianRecommender,
    AssociationRecommender,
)
from src.recommenders.user_cf import UserCFRecommender
from src.recommenders.item_cf import ItemCFRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.matrix_factorisation import MatrixFactorisationRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.baselines import (
    RandomRecommender,
    UserAverageRecommender,
    ItemAverageRecommender,
)
from src.evaluation.metrics import evaluate_recommender

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="ESADE Recommender Systems — Sean Hoet",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system — CSS extracted verbatim from the Claude Design export
# (Reel — Recommender Explorer). Injected once at page load. Class names are
# used exactly as exported; do not rename.
# ---------------------------------------------------------------------------

DESIGN_CSS = """
<style>
:root {
  --bg: #0E1117;
  --panel: #161B22;
  --text: #E8E8E8;
  --muted: rgba(255,255,255,.4);
  --muted2: rgba(255,255,255,.6);
  --border: rgba(255,255,255,.07);
  --blue: #40BCF4;
  --green: #1DB954;
  --amber: #F59E0B;
  --red: #E45756;
  --mono: ui-monospace, 'SF Mono', Menlo, Monaco, Consolas, monospace;
}

/* ---------- PAGE HEADER ---------- */
.page-head { padding: 0 0 4px 0; }
.page-head h1 { margin: 0; font-size: 27px; font-weight: 800; letter-spacing: -.5px; color: var(--text); }
.page-head .tagline { margin-top: 5px; font-size: 13px; color: var(--muted); font-family: var(--mono); }

/* ---------- SIDEBAR BRAND ---------- */
.brand { display: flex; align-items: center; gap: 11px; margin-bottom: 4px; }
.brand-mark {
  width: 32px; height: 32px; border-radius: 8px; background: var(--blue);
  display: flex; align-items: center; justify-content: center; flex: 0 0 auto;
}
.brand-mark .play {
  width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent;
  border-left: 10px solid var(--bg); margin-left: 3px;
}
.brand-name { font-size: 16px; font-weight: 700; letter-spacing: .3px; line-height: 1.15; color: var(--text); }
.brand-sub { font-size: 11px; color: var(--muted); }

/* ---------- ALGORITHM DESCRIPTION CARD ---------- */
.desc-card {
  margin-top: 4px; padding: 12px 13px; background: var(--panel);
  border: 1px solid var(--border); border-left: 3px solid var(--blue); border-radius: 8px;
}
.desc-card p { margin: 0; font-size: 12.5px; line-height: 1.5; color: var(--muted2); }

/* ---------- GENERIC CARD ---------- */
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 6px; }
.panel-pad { padding: 16px 18px; }

/* ---------- USER PROFILE CARD ---------- */
.profile { display: flex; align-items: center; gap: 16px; padding: 18px; }
.avatar {
  position: relative; width: 54px; height: 54px; border-radius: 14px; overflow: hidden;
  flex: 0 0 auto; display: flex; align-items: center; justify-content: center;
  background: hsl(174 45% 22%);
}
.avatar span { font-weight: 800; font-size: 18px; color: hsl(174 55% 70%); font-family: var(--mono); }
.profile-name { font-size: 18px; font-weight: 700; color: var(--text); }
.profile-stats { font-size: 13px; color: rgba(255,255,255,.45); margin-top: 2px; }
.method-badge-wrap { margin-left: auto; flex: 0 0 auto; text-align: right; }
.method-badge-label { font-size: 10px; letter-spacing: .5px; text-transform: uppercase; color: rgba(255,255,255,.35); margin-bottom: 4px; }
.method-badge {
  font-size: 12.5px; font-weight: 600; color: var(--blue); background: rgba(64,188,244,.1);
  border: 1px solid rgba(64,188,244,.25); padding: 5px 11px; border-radius: 7px; white-space: nowrap; display: inline-block;
}

/* ---------- RECS HEADER ---------- */
.recs-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 8px 0 2px; }
.recs-head .h { font-size: 16px; font-weight: 700; color: var(--text); }
.recs-head .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }

/* ---------- POSTER (shared) ---------- */
.poster { position: relative; overflow: hidden; flex: 0 0 auto; background: var(--ph, hsl(210 26% 15%)); }
.poster img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; display: block; }
.poster .ph-init {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
  font-weight: 700; letter-spacing: 1px; color: var(--pht, hsl(210 38% 60%)); z-index: 0; font-family: var(--mono);
}

/* ---------- RECOMMENDATION LIST (Direction A) ---------- */
.rec-list { display: flex; flex-direction: column; gap: 10px; }
.rec-row {
  display: flex; gap: 15px; align-items: center; padding: 13px 15px;
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.rec-row:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.4); border-color: rgba(255,255,255,.14); }
.rec-rank { font-family: var(--mono); font-size: 26px; font-weight: 600; color: rgba(255,255,255,.22); width: 32px; text-align: center; flex: 0 0 auto; }
.rec-row .poster { width: 52px; height: 78px; border-radius: 6px; }
.rec-row .poster .ph-init { font-size: 15px; }
.rec-body { flex: 1; min-width: 0; }
.rec-titleline { display: flex; align-items: baseline; gap: 8px; }
.rec-title { font-size: 14.5px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text); }
.rec-year { font-size: 12.5px; color: rgba(255,255,255,.38); flex: 0 0 auto; }
.chips { display: flex; gap: 6px; margin-top: 7px; flex-wrap: wrap; }
.chip { font-size: 10.5px; color: var(--muted2); background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08); padding: 2px 7px; border-radius: 5px; white-space: nowrap; }
.reason { font-size: 11.5px; font-style: italic; color: var(--muted); margin-top: 8px; }
.score-line { display: flex; align-items: center; gap: 10px; margin-top: 9px; }
.score-track { flex: 1; height: 5px; background: rgba(255,255,255,.08); border-radius: 3px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--blue), var(--green)); }
.score-text { font-size: 12px; font-weight: 600; color: var(--text); font-family: var(--mono); flex: 0 0 auto; }

/* ---------- METRICS PANEL ---------- */
.metrics-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 4px; }
.metrics-head .t { font-size: 15px; font-weight: 700; flex: 0 0 auto; color: var(--text); }
.metrics-head .k { font-size: 10.5px; color: rgba(255,255,255,.35); font-family: var(--mono); white-space: nowrap; }
.metric-group { font-size: 10.5px; font-weight: 600; letter-spacing: .6px; text-transform: uppercase; color: rgba(255,255,255,.35); margin: 16px 0 9px; }
.metric { display: flex; align-items: center; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.05); }
.metric .ml { font-size: 13px; color: var(--muted2); }
.metric .mv { font-size: 13.5px; font-weight: 600; font-family: var(--mono); }
.mv.good { color: var(--green); }
.mv.mid { color: var(--amber); }
.mv.weak { color: rgba(255,255,255,.45); }
.legend { display: flex; gap: 14px; margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,.06); }
.legend > div { display: flex; align-items: center; gap: 6px; }
.legend .dot { width: 8px; height: 8px; border-radius: 2px; }
.legend span.lbl { font-size: 10.5px; color: var(--muted); }

/* ---------- VIEW TOGGLE ---------- */
.view-toggle { display: flex; gap: 6px; }
.vbtn {
  padding: 5px 14px; font-size: 12px; font-weight: 600; border-radius: 7px; cursor: pointer;
  border: 1px solid rgba(255,255,255,.12); background: transparent; color: var(--muted2);
  transition: background .15s, border-color .15s, color .15s;
}
.vbtn.active { background: var(--blue); border-color: var(--blue); color: var(--bg); }

/* ---------- RECOMMENDATION GRID ---------- */
.rec-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
}
.rec-card {
  position: relative; border-radius: 10px; overflow: hidden;
  background: var(--panel); border: 1px solid var(--border);
  transition: transform .15s ease, box-shadow .15s ease;
}
.rec-card:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,.55); }
.rec-card .card-poster {
  width: 100%; padding-top: 148%; position: relative; overflow: hidden;
  background: var(--ph, hsl(210 26% 15%));
}
.rec-card .card-poster img {
  position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; display: block;
}
.rec-card .card-poster .ph-init {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
  font-weight: 700; font-size: 18px; letter-spacing: 1px; color: var(--pht, hsl(210 38% 60%));
  font-family: var(--mono); pointer-events: none;
}
.rec-card .card-rank {
  position: absolute; top: 7px; left: 7px;
  background: rgba(0,0,0,.72); color: var(--text);
  font-size: 12px; font-weight: 700; font-family: var(--mono);
  padding: 2px 6px; border-radius: 5px; z-index: 2;
}
.rec-card .card-body { padding: 8px 9px 10px; }
.rec-card .card-title {
  font-size: 12px; font-weight: 700; color: var(--text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.3;
}
.rec-card .card-year { font-size: 11px; color: var(--muted); margin-top: 1px; }
.rec-card .card-reason { font-size: 10.5px; font-style: italic; color: var(--muted); margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rec-card .score-line { margin-top: 6px; display: flex; align-items: center; gap: 7px; }
.rec-card .score-track { flex: 1; height: 3px; background: rgba(255,255,255,.08); border-radius: 2px; overflow: hidden; }
.rec-card .score-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--blue), var(--green)); }
.rec-card .score-text { font-size: 10.5px; font-weight: 600; color: var(--text); font-family: var(--mono); flex: 0 0 auto; }
</style>
"""

st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data & fit models (cached across sessions)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading data and fitting models — please wait...")
def setup():
    ratings, movies, tags = load_data()
    train, test = temporal_split(ratings)
    models = {
        "Random": RandomRecommender(),
        "User Average": UserAverageRecommender(),
        "Item Average": ItemAverageRecommender(),
        "Top-N (Most Popular)": TopNRecommender(),
        "Bayesian Weighted Rating": BayesianRecommender(),
        "Product Associations": AssociationRecommender(),
        "User-Based CF": UserCFRecommender(k=50),
        "Item-Based CF": ItemCFRecommender(k=50),
        "Content-Based": ContentBasedRecommender(),
        "Matrix Factorisation": MatrixFactorisationRecommender(k=50),
        "Hybrid": HybridRecommender(alpha=0.6),
    }
    status = st.status("Fitting recommenders…", expanded=True)
    for name, model in models.items():
        status.write(f"  Fitting {name}…")
        model.fit(train, movies, tags)
    status.update(label="All models ready!", state="complete", expanded=False)
    return ratings, movies, tags, train, test, models


ratings, movies, tags, train, test, models = setup()


@st.cache_resource(show_spinner=False)
def build_primitive_recs(_topn, _test, k: int = 10):
    """Map user_id -> list of movie_ids the naive popularity baseline would recommend.
    Used as the 'expected' set for the serendipity metric."""
    cache = {}
    for uid in _test["userId"].unique():
        recs = _topn.recommend(int(uid), k)
        cache[int(uid)] = [mid for mid, _ in recs]
    return cache


primitive_recs_cache = build_primitive_recs(models["Top-N (Most Popular)"], test, 10)

# ---------------------------------------------------------------------------
# Design palette — consistent across all charts
# ---------------------------------------------------------------------------

PALETTE = {
    "blue":   "#40BCF4",  # primary / most charts
    "green":  "#1DB954",  # secondary
    "red":    "#E45756",  # warning / bubble
    "orange": "#F59E0B",  # tertiary
    "purple": "#A78BFA",  # quaternary
    "grey":   "#64748B",  # muted
}
CHART_BG = "rgba(0,0,0,0)"   # transparent — inherits Streamlit dark bg
PLOTLY_THEME = "plotly_dark"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STAR_BAR_MAX = 5.0


def stars(score: float, max_val: float = STAR_BAR_MAX) -> str:
    filled = int(round(score / max_val * 10))
    return "█" * filled + "░" * (10 - filled)


def fmt(val, decimals=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


def method_key_of(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
    )


def load_results(method_key: str) -> dict | None:
    path = RESULTS_DIR / f"{method_key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_results(method_key: str, results: dict):
    path = RESULTS_DIR / f"{method_key}.json"
    path.write_text(json.dumps(results, indent=2))


@st.cache_data(show_spinner=False)
def movie_genres_map(_movies: pd.DataFrame) -> dict:
    """movieId -> list of genres."""
    out = {}
    for _, row in _movies.iterrows():
        g = row["genres"]
        out[row["movieId"]] = [] if g == "(no genres listed)" else g.split("|")
    return out


GENRES_BY_MOVIE = movie_genres_map(movies)


@st.cache_data(show_spinner=False)
def load_tmdb_id_map() -> dict:
    """movieId -> tmdbId (int), loaded once from links.csv."""
    links_path = Path(__file__).parent / "DATA/raw/links.csv"
    try:
        ldf = pd.read_csv(links_path)
        return {
            int(row["movieId"]): int(row["tmdbId"])
            for _, row in ldf.iterrows()
            if pd.notna(row.get("tmdbId")) and int(row.get("tmdbId", 0)) > 0
        }
    except Exception:
        return {}


TMDB_ID_MAP = load_tmdb_id_map()

# Hardcoded poster URLs for the 10 prototype movies (TMDB, no API key needed for images)
POSTER_FALLBACKS = {
    "The Shawshank Redemption": "https://image.tmdb.org/t/p/w185/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
    "Schindler's List": "https://image.tmdb.org/t/p/w185/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg",
    "Pulp Fiction": "https://image.tmdb.org/t/p/w185/vQWk5YBFWF4bZaofAbv0tShwBvQ.jpg",
    "The Dark Knight": "https://image.tmdb.org/t/p/w185/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "Forrest Gump": "https://image.tmdb.org/t/p/w185/Cw4hIUIAmSYfK9QfaUW5igp9La.jpg",
    "The Matrix": "https://image.tmdb.org/t/p/w185/aOIuZAjPaRIE6CMzbazvcHuHXDc.jpg",
    "Goodfellas": "https://image.tmdb.org/t/p/w185/9OkCLM73MIU2CrKZbqiT8Ln1wY2.jpg",
    "Fight Club": "https://image.tmdb.org/t/p/w185/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg",
    "Inception": "https://image.tmdb.org/t/p/w185/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg",
    "The Silence of the Lambs": "https://image.tmdb.org/t/p/w185/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
}

# ---------------------------------------------------------------------------
# Design component renderers — build the HTML from the Claude Design export.
# Each returns a single-line HTML string so Streamlit's markdown never treats
# indentation as a code block. Class names match the export exactly.
# ---------------------------------------------------------------------------

import hashlib
import html as _html

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

_INITIAL_STOPWORDS = {"of", "the", "a", "an", "and", "to", "in", "on", "for"}


def _poster_hue(title: str) -> int:
    return int(hashlib.md5(title.encode("utf-8")).hexdigest(), 16) % 360


def _poster_initials(title: str) -> str:
    words = [w for w in title.replace(":", " ").replace("-", " ").split() if w]
    letters = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in _INITIAL_STOPWORDS:
            continue
        if w[0].isalnum():
            letters.append(w[0].upper())
    return "".join(letters[:3]) if letters else "?"


def _quality(key: str, v) -> str:
    """Classify a metric value as good / mid / weak for colour coding."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "weak"
    lower_better = key in ("mae", "rmse", "popularity_bias")
    thresholds = {
        "mae": (1.0, 1.5), "rmse": (1.1, 1.6), "popularity_bias": (0.3, 0.6),
        "precision_at_k": (0.1, 0.03), "recall_at_k": (0.1, 0.03),
        "ndcg_at_k": (0.1, 0.03), "mrr": (0.1, 0.03),
        "coverage": (0.3, 0.05), "novelty": (4.0, 2.0),
        "diversity": (0.5, 0.3), "serendipity": (0.1, 0.03),
    }
    good, mid = thresholds.get(key, (None, None))
    if good is None:
        return "mid"
    if lower_better:
        return "good" if v < good else ("mid" if v < mid else "weak")
    return "good" if v >= good else ("mid" if v >= mid else "weak")


@st.cache_data(show_spinner=False, ttl=86400)
def get_poster_url(title: str, year: int | None = None, movie_id: int | None = None) -> str | None:
    """Return a TMDB poster URL for a movie.

    Priority:
    1. Hardcoded fallback dict (10 prototype movies — instant)
    2. TMDB /movie/{tmdbId} lookup via links.csv (fast, no search needed)
    3. TMDB search by title+year (slower, for movies missing from links.csv)
    """
    if title in POSTER_FALLBACKS:
        return POSTER_FALLBACKS[title]
    if not TMDB_API_KEY:
        return None
    import urllib.request, urllib.parse, json as _json
    # Path 2: direct lookup via tmdbId
    if movie_id:
        tmdb_id = TMDB_ID_MAP.get(int(movie_id), 0)
        if tmdb_id:
            try:
                url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
                with urllib.request.urlopen(url, timeout=3) as r:
                    data = _json.loads(r.read())
                if data.get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w185{data['poster_path']}"
            except Exception:
                pass
    # Path 3: search by title
    try:
        query = urllib.parse.urlencode({
            "api_key": TMDB_API_KEY, "query": title, "year": year or "",
        })
        url = f"https://api.themoviedb.org/3/search/movie?{query}"
        with urllib.request.urlopen(url, timeout=3) as r:
            data = _json.loads(r.read())
        res = data.get("results", [])
        if res and res[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w185{res[0]['poster_path']}"
    except Exception:
        pass
    return None


def render_page_header() -> str:
    return (
        '<div class="page-head">'
        '<h1>Recommender Explorer</h1>'
        '<div class="tagline">11 algorithms · MovieLens Latest Small · ESADE MiBA</div>'
        '</div>'
    )


def render_sidebar_brand() -> str:
    return (
        '<div class="brand">'
        '<div class="brand-mark"><div class="play"></div></div>'
        '<div><div class="brand-name">Reel</div>'
        '<div class="brand-sub">Recommender Explorer</div></div>'
        '</div>'
    )


def algo_description_card(text: str) -> str:
    return f'<div class="desc-card"><p>{_html.escape(text)}</p></div>'


def render_profile_card(user_id, rated_count, avg_rating, method_label) -> str:
    return (
        '<div class="card profile">'
        f'<div class="avatar"><span>{user_id}</span></div>'
        f'<div><div class="profile-name">User {user_id}</div>'
        f'<div class="profile-stats">{rated_count} movies rated · Avg rating {avg_rating:.1f}★</div></div>'
        '<div class="method-badge-wrap"><div class="method-badge-label">Method</div>'
        f'<div class="method-badge">{_html.escape(method_label)}</div></div>'
        '</div>'
    )


def render_recs_head(n_recs: int, method_label: str) -> str:
    return (
        '<div class="recs-head"><div>'
        f'<div class="h">Top {n_recs} recommendations</div>'
        f'<div class="sub">via {_html.escape(method_label)}</div>'
        '</div></div>'
    )


def render_rec_card(rank, title, year, genres, score_pct, score, reason, poster_url=None) -> str:
    hue = _poster_hue(title)
    init = _poster_initials(title)
    poster_img = (
        f'<img loading="lazy" src="{poster_url}" alt="{_html.escape(title)} poster" onerror="this.remove()">'
        if poster_url else ""
    )
    chips = "".join(f'<span class="chip">{_html.escape(g)}</span>' for g in (genres or [])[:4])
    year_html = f'<span class="rec-year">{year}</span>' if year else ""
    reason_html = f'<div class="reason">{_html.escape(reason)}</div>' if reason else ""
    pct = max(0.0, min(100.0, float(score_pct)))
    return (
        '<div class="rec-row">'
        f'<div class="rec-rank">{rank}</div>'
        f'<div class="poster" style="--ph:hsl({hue} 26% 15%);--pht:hsl({hue} 38% 60%)">'
        f'<span class="ph-init">{init}</span>{poster_img}</div>'
        '<div class="rec-body">'
        f'<div class="rec-titleline"><span class="rec-title">{_html.escape(title)}</span>{year_html}</div>'
        f'<div class="chips">{chips}</div>'
        f'{reason_html}'
        f'<div class="score-line"><div class="score-track"><div class="score-fill" style="width:{pct:.1f}%"></div></div>'
        f'<span class="score-text">{score:.3f}</span></div>'
        '</div></div>'
    )


def render_rec_grid_card(rank, title, year, score_pct, score, reason, poster_url=None) -> str:
    hue = _poster_hue(title)
    init = _poster_initials(title)
    poster_img = (
        f'<img loading="lazy" src="{poster_url}" alt="{_html.escape(title)}" onerror="this.style.display=\'none\'">'
        if poster_url else ""
    )
    year_html = f'<div class="card-year">{year}</div>' if year else ""
    reason_html = f'<div class="card-reason">{_html.escape(reason)}</div>' if reason else ""
    pct = max(0.0, min(100.0, float(score_pct)))
    return (
        '<div class="rec-card">'
        f'<div class="card-poster" style="--ph:hsl({hue} 26% 15%);--pht:hsl({hue} 38% 60%)">'
        f'<span class="ph-init">{init}</span>{poster_img}'
        f'<div class="card-rank">{rank}</div>'
        '</div>'
        '<div class="card-body">'
        f'<div class="card-title">{_html.escape(title)}</div>'
        f'{year_html}'
        f'{reason_html}'
        f'<div class="score-line"><div class="score-track"><div class="score-fill" style="width:{pct:.1f}%"></div></div>'
        f'<span class="score-text">{score:.3f}</span></div>'
        '</div></div>'
    )


def render_metrics_panel(results: dict, n_users=None, k: int = 10) -> str:
    groups = [
        ("Rating prediction", [("MAE", "mae"), ("RMSE", "rmse")]),
        ("Ranking quality", [
            ("Precision@10", "precision_at_k"), ("Recall@10", "recall_at_k"),
            ("NDCG@10", "ndcg_at_k"), ("MRR", "mrr"),
        ]),
        ("Beyond accuracy", [
            ("Coverage", "coverage"), ("Novelty", "novelty"),
            ("Diversity", "diversity"), ("Popularity Bias", "popularity_bias"),
            ("Serendipity", "serendipity"),
        ]),
    ]
    users_txt = f"{n_users} users" if n_users not in (None, "?") else ""
    head_k = f"k={k}" + (f" · {users_txt}" if users_txt else "")
    body = (
        '<div class="metrics-head"><div class="t">Evaluation</div>'
        f'<div class="k">{head_k}</div></div>'
    )
    for group_title, items in groups:
        body += f'<div class="metric-group">{group_title}</div>'
        for label, key in items:
            val = results.get(key)
            q = _quality(key, val)
            body += (
                f'<div class="metric"><span class="ml">{label}</span>'
                f'<span class="mv {q}">{fmt(val)}</span></div>'
            )
    body += (
        '<div class="legend">'
        '<div><span class="dot" style="background:var(--green)"></span><span class="lbl">strong</span></div>'
        '<div><span class="dot" style="background:var(--amber)"></span><span class="lbl">moderate</span></div>'
        '<div><span class="dot" style="background:rgba(255,255,255,.45)"></span><span class="lbl">weak</span></div>'
        '</div>'
    )
    return f'<div class="card panel-pad">{body}</div>'


def user_genre_profile(user_id: int) -> pd.Series:
    """Average rating per genre for one user, from the training set.

    Returns a Series indexed by genre, sorted descending by mean rating.
    """
    ut = train[train["userId"] == user_id]
    rows = []
    for _, r in ut.iterrows():
        for g in GENRES_BY_MOVIE.get(r["movieId"], []):
            rows.append((g, r["rating"]))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["genre", "rating"])
    agg = df.groupby("genre")["rating"].agg(["mean", "count"])
    # require at least 2 rated movies in a genre to be meaningful
    agg = agg[agg["count"] >= 2] if (agg["count"] >= 2).any() else agg
    return agg["mean"].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Explanation generation (Task 3c)
# ---------------------------------------------------------------------------

def explain_recommendation(method_name: str, mid: int, user_id: int, model,
                           top_genres: list, score: float) -> str:
    """Return a one-line, method-specific reason for a recommendation."""
    rec_genres = GENRES_BY_MOVIE.get(mid, [])

    if method_name in ("Content-Based", "Hybrid"):
        matched = [g for g in top_genres if g in rec_genres][:2]
        if matched:
            return "Matched: " + " · ".join(matched)
        return "Matches your taste profile"

    if method_name == "User-Based CF":
        # Count this user's similar neighbours who rated the movie >= 4.
        n_liked = _count_similar_likers(model, user_id, mid)
        return f"Liked by {n_liked} similar users"

    if method_name == "Item-Based CF":
        src = _best_source_movie(model, user_id, mid)
        if src is not None:
            title = movies.loc[movies["movieId"] == src, "title"].values
            if len(title):
                clean = title[0].split(" (")[0]
                return f"Similar to {clean}"
        return "Similar to movies you rated highly"

    if method_name in ("Top-N (Most Popular)", "Bayesian Weighted Rating"):
        n_ratings = int((train["movieId"] == mid).sum())
        return f"Popular pick · rated by {n_ratings} users"

    if method_name == "Matrix Factorisation":
        try:
            pr = model.predict(user_id, mid)
        except Exception:
            pr = score
        return f"Predicted rating: {pr:.1f}★"

    if method_name == "Product Associations":
        return "Frequently liked together with your favourites"

    return ""


def _count_similar_likers(model, user_id: int, mid: int) -> int:
    """For User-CF: number of the user's top-k neighbours who rated mid >= 4."""
    try:
        i = model.user_idx[user_id]
        neigh_idx = model._top_k_idx[i]
        if mid not in model.matrix.columns:
            return 0
        col = model.matrix[mid]
        count = 0
        for j in neigh_idx:
            uid = model.users[j]
            val = col.get(uid, np.nan)
            if not np.isnan(val) and val >= 4.0:
                count += 1
        return count
    except Exception:
        return 0


def _best_source_movie(model, user_id: int, mid: int):
    """For Item-CF: the user's highest-rated movie that lists mid as a neighbour."""
    try:
        ut = train[train["userId"] == user_id].sort_values("rating", ascending=False)
        for _, r in ut.iterrows():
            src = r["movieId"]
            neigh = model.item_neighbours.get(src, [])
            if any(t == mid for t, _ in neigh):
                return src
        # fallback: simply the single highest-rated movie
        if len(ut):
            return ut.iloc[0]["movieId"]
    except Exception:
        pass
    return None


def genre_entropy(rec_ids: list) -> float:
    """Shannon entropy (base 2) over the genre distribution of a rec list."""
    counts = {}
    for mid in rec_ids:
        for g in GENRES_BY_MOVIE.get(mid, []):
            counts[g] = counts.get(g, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * log2(c / total) for c in counts.values())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(render_sidebar_brand(), unsafe_allow_html=True)
    st.divider()

    all_users = sorted(train["userId"].unique())
    user_id = st.selectbox(
        "Select user",
        all_users,
        format_func=lambda u: f"User {u}",
    )

    METHOD_DESCRIPTIONS = {
        "Random": "Uniform random from unseen catalog — the accuracy floor.",
        "User Average": "Predicts your own mean rating for every item — best MAE, zero ranking power.",
        "Item Average": "Predicts each item's mean rating — strong cold-start baseline.",
        "Top-N (Most Popular)": "Ranks items by total rating count — no personalisation.",
        "Bayesian Weighted Rating": "Shrinks small-sample items toward the global mean — reduces popularity bias.",
        "Product Associations": "Recommends items frequently co-rated with your favourites using lift.",
        "User-Based CF": "Finds similar users via Pearson correlation, predicts from their ratings.",
        "Item-Based CF": "Finds similar items via adjusted cosine similarity — more stable than user-based.",
        "Content-Based": "Builds a TF-IDF profile from genres, tags and year — high coverage, low diversity.",
        "Matrix Factorisation": "Truncated SVD with k=50 latent factors — best accuracy, highest popularity bias.",
        "Hybrid": "Weighted blend of SVD and Content-Based — balances accuracy and discovery.",
    }
    method_name = st.selectbox("Recommender method", list(models.keys()))
    st.markdown(
        algo_description_card(METHOD_DESCRIPTIONS.get(method_name, "")),
        unsafe_allow_html=True,
    )
    n_recs = st.slider("Number of recommendations", 5, 20, 10)

    st.divider()
    st.caption("ESADE · Recommender Systems · Sean Hoet\n\nDataset: MovieLens Latest Small (GroupLens)")

model = models[method_name]
method_key = method_key_of(method_name)

# Display names that match the academic terminology in the deck
DISPLAY_NAME = {
    "Matrix Factorisation": "Matrix Factorisation (SVD)",
    "Hybrid": "Hybrid (SVD + Content)",
}
method_label = DISPLAY_NAME.get(method_name, method_name)

# ---------------------------------------------------------------------------
# Main layout — tabs
# ---------------------------------------------------------------------------

st.markdown(render_page_header(), unsafe_allow_html=True)

tab_explore, tab_eda, tab_bubble, tab_genre, tab_coldstart = st.tabs([
    "🔍 Explorer", "📊 Dataset & EDA", "🫧 Filter Bubble Simulation", "🎬 By Genre", "❄️ Cold Start"
])

# ===========================================================================
# TAB 0 — DATASET & EDA
# ===========================================================================
with tab_eda:
    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total ratings", "100,836")
    c2.metric("Unique movies", "9,742")
    c3.metric("Unique users", "610")
    c4.metric("Matrix sparsity", "98.3%")

    @st.cache_data(show_spinner=False)
    def eda_data(_ratings, _movies):
        import pandas as pd
        rating_counts = _ratings["rating"].value_counts().sort_index()
        ratings_per_user = _ratings.groupby("userId").size()
        ratings_per_movie = _ratings.groupby("movieId").size()
        genre_counts = {}
        for g_str in _movies["genres"]:
            if g_str == "(no genres listed)":
                continue
            for g in g_str.split("|"):
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_series = pd.Series(genre_counts).sort_values(ascending=False).head(15)
        years = pd.to_datetime(_ratings["timestamp"], unit="s").dt.year
        ratings_by_year = years.value_counts().sort_index()
        return rating_counts, ratings_per_user, ratings_per_movie, genre_series, ratings_by_year

    rating_counts, ratings_per_user, ratings_per_movie, genre_series, ratings_by_year = eda_data(ratings, movies)

    # Rating distribution
    fig_rdist = go.Figure(go.Bar(
        x=[str(v) for v in rating_counts.index],
        y=rating_counts.values,
        marker_color=PALETTE["blue"],
    ))
    fig_rdist.update_layout(
        title="Rating Distribution",
        xaxis_title="Rating",
        yaxis_title="Count",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_THEME,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    st.plotly_chart(fig_rdist, use_container_width=True)

    # User activity and item popularity side by side
    col_u, col_m = st.columns(2)
    with col_u:
        fig_uact = go.Figure(go.Histogram(
            x=ratings_per_user.values,
            nbinsx=50,
            marker_color=PALETTE["blue"],
        ))
        fig_uact.update_layout(
            title="User Activity — how many movies each user rated",
            xaxis_title="Ratings per user",
            yaxis_title="Count (log)",
            yaxis_type="log",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            template=PLOTLY_THEME,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
        )
        st.plotly_chart(fig_uact, use_container_width=True)

    with col_m:
        fig_mpop = go.Figure(go.Histogram(
            x=ratings_per_movie.values,
            nbinsx=50,
            marker_color=PALETTE["green"],
        ))
        fig_mpop.update_layout(
            title="Item Popularity — how many ratings each movie received",
            xaxis_title="Ratings per movie",
            yaxis_title="Count (log)",
            yaxis_type="log",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            template=PLOTLY_THEME,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
        )
        st.plotly_chart(fig_mpop, use_container_width=True)

    # Genre frequency (horizontal bar)
    fig_genre = go.Figure(go.Bar(
        x=genre_series.values[::-1],
        y=genre_series.index[::-1],
        orientation="h",
        marker_color=PALETTE["blue"],
    ))
    fig_genre.update_layout(
        title="Genre Distribution",
        xaxis_title="Number of movies",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_THEME,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    st.plotly_chart(fig_genre, use_container_width=True)

    # Rating volume over time
    fig_time = go.Figure(go.Scatter(
        x=ratings_by_year.index.tolist(),
        y=ratings_by_year.values.tolist(),
        mode="lines+markers",
        line=dict(color=PALETTE["blue"], width=2.5),
        marker=dict(size=7),
    ))
    fig_time.update_layout(
        title="Rating Volume Over Time",
        xaxis_title="Year",
        yaxis_title="Number of ratings",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        template=PLOTLY_THEME,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.caption(
        "Dataset: MovieLens Latest Small — F. Maxwell Harper and Joseph A. Konstan. "
        "2015. The MovieLens Datasets. ACM TIIS."
    )

# ===========================================================================
# TAB 1 — EXPLORER
# ===========================================================================
with tab_explore:
    col_recs, col_metrics = st.columns([3, 2], gap="large")

    # ── Left column: recommendations ──────────────────────────────────────
    with col_recs:
        user_train = train[train["userId"] == user_id]
        rated_count = len(user_train)
        avg_rating = user_train["rating"].mean()

        st.markdown(
            render_profile_card(user_id, rated_count, avg_rating, method_label),
            unsafe_allow_html=True,
        )

        # ── Task 3b: taste profile bar chart ──────────────────────────────
        profile = user_genre_profile(user_id)
        if not profile.empty:
            top_profile = profile.head(8)[::-1]  # reverse for horizontal bar order
            fig_taste = go.Figure(
                go.Bar(
                    x=top_profile.values,
                    y=top_profile.index,
                    orientation="h",
                    marker=dict(color=top_profile.values, colorscale="Blues",
                                cmin=top_profile.values.min() - 0.3),
                    text=[f"{v:.1f}★" for v in top_profile.values],
                    textposition="auto",
                )
            )
            fig_taste.update_layout(
                title="Your taste profile (avg rating by genre)",
                height=260,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(range=[0, 5], title="Avg rating"),
                showlegend=False,
                template=PLOTLY_THEME,
                paper_bgcolor=CHART_BG,
                plot_bgcolor=CHART_BG,
            )
            st.plotly_chart(fig_taste, use_container_width=True)

        top_genres = list(profile.head(5).index)  # used by explanations

        with st.expander("User's top-rated movies", expanded=False):
            top_rated = user_train.nlargest(8, "rating")
            info = get_movie_display(top_rated["movieId"].tolist(), movies)
            for _, row in top_rated.iterrows():
                mid = row["movieId"]
                if mid in info.index:
                    title = info.at[mid, "clean_title"]
                    year = info.at[mid, "year"]
                    genre = info.at[mid, "genres"].replace("|", " · ")
                    st.markdown(
                        f"**{title}** ({int(year) if pd.notna(year) else '?'})  "
                        f"{'★' * int(row['rating'])}{'☆' * (5 - int(row['rating']))}  \n"
                        f"<small>{genre}</small>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ── List / Grid toggle ─────────────────────────────────────────────
        if "rec_view_mode" not in st.session_state:
            st.session_state["rec_view_mode"] = "List"

        toggle_col, _ = st.columns([1, 4])
        with toggle_col:
            view_mode = st.radio(
                "View",
                ["List", "Grid"],
                index=0 if st.session_state["rec_view_mode"] == "List" else 1,
                horizontal=True,
                label_visibility="collapsed",
                key="rec_view_toggle",
            )
            st.session_state["rec_view_mode"] = view_mode

        st.markdown(render_recs_head(n_recs, method_label), unsafe_allow_html=True)

        if method_name == "Hybrid":
            alpha = st.slider(
                "Accuracy ↔ Discovery",
                min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                help="Left = more novel/diverse (content-based). Right = more accurate (SVD).",
                key="hybrid_alpha",
            )
            st.caption(f"SVD weight: **{alpha:.0%}** · Content weight: **{1-alpha:.0%}**")

            if alpha < 0.3:
                st.info("Maximising novelty and serendipity — expect less popular, more surprising picks.")
            elif alpha > 0.7:
                st.info("Maximising accuracy — expect popular, well-predicted picks.")
            else:
                st.info("Balanced trade-off between accuracy and discovery.")

            # Re-mix scores live using already-fitted sub-models
            svd_model = models["Matrix Factorisation"]
            content_model = models["Content-Based"]
            seen = set(train[train["userId"] == user_id]["movieId"])

            pool = 200
            svd_raw = dict(svd_model.recommend(user_id, pool))
            content_raw = dict(content_model.recommend(user_id, pool))

            def _minmax(d):
                if not d:
                    return {}
                vals = list(d.values())
                lo, hi = min(vals), max(vals)
                rng = hi - lo if hi != lo else 1.0
                return {k: (v - lo) / rng for k, v in d.items()}

            svd_n = _minmax(svd_raw)
            content_n = _minmax(content_raw)
            candidates = set(svd_n) | set(content_n)
            combined = {
                mid: alpha * svd_n.get(mid, 0.0) + (1 - alpha) * content_n.get(mid, 0.0)
                for mid in candidates if mid not in seen
            }
            recs = sorted(combined.items(), key=lambda x: -x[1])[:n_recs]
        else:
            recs = model.recommend(user_id, n_recs)

        if not recs:
            st.warning("No recommendations available for this user with this method.")
        else:
            rec_movie_ids = [mid for mid, _ in recs]
            rec_scores = [score for _, score in recs]
            info = get_movie_display(rec_movie_ids, movies)

            max_score = max(rec_scores) if rec_scores else 1.0
            min_score = min(rec_scores)
            score_range = max_score - min_score if max_score != min_score else 1.0

            list_cards = []
            grid_cards = []
            for rank, (mid, score) in enumerate(zip(rec_movie_ids, rec_scores), start=1):
                if mid not in info.index:
                    continue
                title = info.at[mid, "clean_title"]
                year_raw = info.at[mid, "year"]
                year_int = int(year_raw) if pd.notna(year_raw) else None
                genres_raw = info.at[mid, "genres"]
                genres_list = genres_raw.split("|") if pd.notna(genres_raw) else []
                norm = (score - min_score) / score_range
                score_pct = norm * 100
                reason = explain_recommendation(
                    method_name, mid, user_id, model, top_genres, score
                )
                poster_url = get_poster_url(title, year_int, movie_id=mid)
                list_cards.append(render_rec_card(
                    rank, title, year_int, genres_list, score_pct, score, reason, poster_url
                ))
                grid_cards.append(render_rec_grid_card(
                    rank, title, year_int, score_pct, score, reason, poster_url
                ))

            if view_mode == "Grid":
                st.markdown(
                    '<div class="rec-grid">' + "".join(grid_cards) + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="rec-list">' + "".join(list_cards) + "</div>",
                    unsafe_allow_html=True,
                )

    # ── Right column: evaluation metrics ──────────────────────────────────
    with col_metrics:
        cached = load_results(method_key)

        if cached:
            results = cached
        else:
            st.info("Evaluation not yet run for this method.")
            results = None

        if st.button("Run evaluation (≈30s)", type="primary"):
            with st.spinner("Evaluating…"):
                cb_model = models.get("Content-Based")
                item_matrix = getattr(cb_model, "item_matrix", None)
                feature_idx = getattr(cb_model, "feature_idx", None)
                results = evaluate_recommender(
                    model, train, test,
                    k=10,
                    sample_users=100,
                    item_matrix=item_matrix,
                    feature_idx=feature_idx,
                    primitive_recs=primitive_recs_cache,
                )
                save_results(method_key, results)
            st.success("Done!")

        if results:
            st.markdown(
                render_metrics_panel(
                    results,
                    n_users=results.get("n_users"),
                    k=results.get("k", 10),
                ),
                unsafe_allow_html=True,
            )

        # ── Trade-off radar ────────────────────────────────────────────────
        st.divider()

        all_results = {}
        for mname in models.keys():
            mkey = method_key_of(mname)
            r = load_results(mkey)
            if r:
                label = DISPLAY_NAME.get(mname, mname)
                all_results[label] = r

        if not all_results:
            st.caption("Run evaluation on each method to populate the trade-off radar.")
        else:
            # ── Task 3a: radar chart ───────────────────────────────────────
            axes = [
                ("Accuracy", "ndcg_at_k", False),
                ("Novelty", "novelty", False),
                ("Diversity", "diversity", False),
                ("Coverage", "coverage", False),
                ("Low bias", "popularity_bias", True),  # invert: 1 - bias
            ]
            # Normalise each axis across all methods
            raw = {label: [] for label in all_results}
            for _, key, invert in axes:
                vals = {}
                for label, r in all_results.items():
                    v = r.get(key)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        v = 0.0
                    if invert:
                        v = 1.0 - v
                    vals[label] = v
                vmin, vmax = min(vals.values()), max(vals.values())
                rng = vmax - vmin if vmax != vmin else 1.0
                for label in all_results:
                    raw[label].append((vals[label] - vmin) / rng)

            axis_labels = [a[0] for a in axes]
            current_label = DISPLAY_NAME.get(method_name, method_name)
            n_others = len(all_results) - 1

            fig_radar = go.Figure()
            # Draw other methods first as faint background lines
            for label, vals in raw.items():
                if label == current_label:
                    continue
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=axis_labels + [axis_labels[0]],
                        fill="none",
                        name=label,
                        line=dict(color="rgba(255,255,255,0.12)", width=1),
                        showlegend=False,
                    )
                )
            # Draw current method on top, filled and opaque
            if current_label in raw:
                cv = raw[current_label]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=cv + [cv[0]],
                        theta=axis_labels + [axis_labels[0]],
                        fill="toself",
                        name=current_label,
                        fillcolor="rgba(64,188,244,0.18)",
                        line=dict(color="#40BCF4", width=2),
                        showlegend=False,
                    )
                )

            fig_radar.update_layout(
                title=dict(
                    text="Trade-off radar",
                    font=dict(size=14, color="#E8E8E8"),
                    x=0,
                ),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(
                        tickfont=dict(size=11, color="rgba(255,255,255,0.55)"),
                        linecolor="rgba(255,255,255,0.1)",
                        gridcolor="rgba(255,255,255,0.07)",
                    ),
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=9, color="rgba(255,255,255,0.25)"),
                        gridcolor="rgba(255,255,255,0.07)",
                        linecolor="rgba(255,255,255,0.1)",
                        showticklabels=False,
                    ),
                ),
                height=340,
                margin=dict(l=20, r=20, t=50, b=10),
                template=PLOTLY_THEME,
                paper_bgcolor=CHART_BG,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            others_txt = f"{n_others} other{'s' if n_others != 1 else ''}" if n_others > 0 else "no others yet"
            st.caption(f"— {current_label} vs {others_txt} · normalised across all 11 methods")

            with st.expander("Full comparison table", expanded=True):
                metrics_order = [
                    ("Precision@10", "precision_at_k"),
                    ("Recall@10", "recall_at_k"),
                    ("NDCG@10", "ndcg_at_k"),
                    ("MRR", "mrr"),
                    ("MAE", "mae"),
                    ("RMSE", "rmse"),
                    ("Coverage", "coverage"),
                    ("Novelty", "novelty"),
                    ("Diversity", "diversity"),
                    ("Pop. Bias", "popularity_bias"),
                    ("Serendipity", "serendipity"),
                ]
                rows = {}
                for label, key in metrics_order:
                    row = {}
                    for mname, r in all_results.items():
                        val = r.get(key)
                        row[mname] = fmt(val) if val is not None else "—"
                    rows[label] = row
                df_compare = pd.DataFrame(rows).T
                st.dataframe(df_compare, use_container_width=True)

        # ── Task 6: K-sensitivity analysis (only show for CF methods) ──────
        if method_name in ("User-Based CF", "Item-Based CF"):
            with st.expander("K-sensitivity: how neighbourhood size affects precision", expanded=False):
                st.caption("Precision@10 as k varies — shows the bias-variance trade-off in collaborative filtering.")

                @st.cache_data(show_spinner=False)
                def k_sensitivity_data(method: str):
                    from src.recommenders.user_cf import UserCFRecommender
                    from src.recommenders.item_cf import ItemCFRecommender
                    from src.evaluation.metrics import evaluate_recommender

                    k_vals = [5, 10, 20, 50, 100, 200]
                    prec_vals = []
                    ndcg_vals = []
                    for k_val in k_vals:
                        if method == "User-Based CF":
                            m = UserCFRecommender(k=k_val)
                        else:
                            m = ItemCFRecommender(k=k_val)
                        m.fit(train, movies, tags)
                        res = evaluate_recommender(m, train, test, k=10, sample_users=50)
                        prec_vals.append(round(res.get("precision_at_k") or 0, 4))
                        ndcg_vals.append(round(res.get("ndcg_at_k") or 0, 4))
                    return k_vals, prec_vals, ndcg_vals

                with st.spinner(f"Computing k-sensitivity for {method_name}... (~30s)"):
                    k_vals, prec_vals, ndcg_vals = k_sensitivity_data(method_name)

                fig_k = go.Figure()
                fig_k.add_trace(go.Scatter(x=k_vals, y=prec_vals, mode="lines+markers",
                                           name="Precision@10", line=dict(color=PALETTE["green"], width=2.5)))
                fig_k.add_trace(go.Scatter(x=k_vals, y=ndcg_vals, mode="lines+markers",
                                           name="NDCG@10", line=dict(color=PALETTE["blue"], width=2.5, dash="dash")))
                fig_k.update_layout(
                    xaxis=dict(title="Neighbourhood size k"),
                    yaxis=dict(title="Score"),
                    height=280,
                    margin=dict(l=10, r=10, t=20, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35),
                    template=PLOTLY_THEME,
                    paper_bgcolor=CHART_BG,
                    plot_bgcolor=CHART_BG,
                )
                st.plotly_chart(fig_k, use_container_width=True)

        with st.expander("⚡ Scalability notes", expanded=False):
            scalability_data = {
                "Method": [
                    "Random / Averages",
                    "Top-N / Bayesian / Associations",
                    "User-Based CF",
                    "Item-Based CF",
                    "Content-Based",
                    "Matrix Factorisation",
                    "Hybrid",
                ],
                "Train complexity": [
                    "O(n)", "O(n log n)", "O(u²·i)", "O(i²·u)", "O(i·f)", "O(u·i·k)", "SVD + CB"
                ],
                "Inference": [
                    "O(1)", "O(1)", "O(k)", "O(k)", "O(i)", "O(k)", "O(k)"
                ],
                "Scales to millions?": [
                    "✅ Yes", "✅ Yes", "❌ No", "⚠️ Better", "✅ Yes", "⚠️ Periodic retraining", "⚠️ Inherits SVD limits"
                ],
            }
            df_scale = pd.DataFrame(scalability_data)
            st.dataframe(df_scale, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2 — FILTER BUBBLE SIMULATION  (Task 4)
# ===========================================================================
with tab_bubble:
    st.subheader("🫧 Filter Bubble Simulation")
    st.markdown(
        "Inspired by Eli Pariser's *The Filter Bubble* (2011). We simulate a user who "
        "**watches everything the system recommends**, round after round, and measure how "
        "the **genre diversity** of their recommendations evolves."
    )

    N_ROUNDS = 5
    sim_methods = {
        "Top-N (Most Popular)": models["Top-N (Most Popular)"],
        "User-Based CF": models["User-Based CF"],
        "Content-Based": models["Content-Based"],
    }

    st.caption(f"Simulating {N_ROUNDS} rounds for user {user_id} …")

    @st.cache_data(show_spinner=False)
    def run_simulation(_models_keys: tuple, user_id: int, n_rounds: int):
        """Returns {method: [entropy_round1, ..., entropy_roundN]}."""
        out = {}
        for mname in _models_keys:
            mdl = models[mname]
            newly_watched: set = set()
            entropies = []
            for rnd in range(n_rounds):
                # Pull enough candidates to still have 10 after removing newly watched
                pull = 10 + len(newly_watched) + 20
                raw = mdl.recommend(user_id, pull)
                filtered = [mid for mid, _ in raw if mid not in newly_watched][:10]
                entropies.append(genre_entropy(filtered))
                newly_watched.update(filtered)  # "watch" them -> excluded next round
            out[mname] = entropies
        return out

    sim_results = run_simulation(tuple(sim_methods.keys()), user_id, N_ROUNDS)

    fig_bubble = go.Figure()
    palette = {
        "Top-N (Most Popular)": PALETTE["red"],
        "User-Based CF": PALETTE["orange"],
        "Content-Based": PALETTE["blue"],
    }
    for mname, ents in sim_results.items():
        fig_bubble.add_trace(
            go.Scatter(
                x=list(range(1, N_ROUNDS + 1)),
                y=ents,
                mode="lines+markers",
                name=mname,
                line=dict(width=3, color=palette.get(mname)),
                marker=dict(size=9),
            )
        )
    fig_bubble.update_layout(
        title="Genre entropy of recommendations over repeated rounds",
        xaxis=dict(title="Round", dtick=1),
        yaxis=dict(title="Genre entropy (bits) — higher = more diverse"),
        height=460,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        template=PLOTLY_THEME,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.info(
        "**Methods with low / collapsing entropy fall into an echo chamber.** "
        "Higher and steadier entropy means the system keeps surfacing diverse genres "
        "over time. Popularity-driven methods tend to narrow the user's world; "
        "content and discovery-oriented methods resist the bubble."
    )

    with st.expander("Entropy values per round", expanded=False):
        df_sim = pd.DataFrame(
            sim_results, index=[f"Round {i}" for i in range(1, N_ROUNDS + 1)]
        ).T.round(3)
        st.dataframe(df_sim, use_container_width=True)

# ===========================================================================
# TAB 3 — BY GENRE (Netflix-style rows)  (Task 4)
# ===========================================================================
with tab_genre:
    GENRE_EMOJI = {
        "Action": "💥", "Adventure": "🗺️", "Animation": "🎨", "Children": "🧸",
        "Comedy": "😄", "Crime": "🔍", "Documentary": "🎥", "Drama": "🎭",
        "Fantasy": "🧙", "Film-Noir": "🕵️", "Horror": "👻", "Musical": "🎵",
        "Mystery": "🔮", "Romance": "❤️", "Sci-Fi": "🚀", "Thriller": "😰",
        "War": "⚔️", "Western": "🤠",
    }
    st.subheader(f"Personalised by genre — User {user_id}")
    st.caption("Netflix-style rows: your top genres, filled with recommendations from the selected method.")

    # Get the user's top genres from their taste profile
    profile = user_genre_profile(user_id)
    top_genres = list(profile.head(6).index)  # top 6 genres

    if not top_genres:
        st.warning("Not enough ratings to build a taste profile for this user.")
    else:
        # Get a large candidate pool from the selected method
        pool_recs = model.recommend(user_id, 200)
        if not pool_recs:
            st.warning("No recommendations available for this user with this method.")
        else:
            pool_ids = [mid for mid, _ in pool_recs]
            pool_scores = {mid: score for mid, score in pool_recs}
            info = get_movie_display(pool_ids, movies)

            for genre in top_genres:
                # Filter recs that include this genre
                genre_matches = [
                    mid for mid in pool_ids
                    if mid in info.index and genre in info.at[mid, "genres"].split("|")
                ][:5]  # max 5 per row

                if not genre_matches:
                    continue

                st.markdown(f"### {GENRE_EMOJI.get(genre, '🎬')} {genre}")
                cols = st.columns(len(genre_matches))
                for col, mid in zip(cols, genre_matches):
                    with col:
                        title = info.at[mid, "clean_title"] if mid in info.index else str(mid)
                        year = info.at[mid, "year"] if mid in info.index else ""
                        score = pool_scores.get(mid, 0.0)
                        col.markdown(
                            f"**{title}**  \n"
                            f"<small>({int(year) if pd.notna(year) else '?'})</small>  \n"
                            f"<small>Score: {score:.2f}</small>",
                            unsafe_allow_html=True,
                        )
                st.divider()

        st.caption(
            "Rows are filled from the currently selected recommender method — "
            "switch methods in the sidebar to see how genre coverage changes."
        )

# ===========================================================================
# TAB 4 — COLD START SIMULATION  (Task 5)
# ===========================================================================
with tab_coldstart:
    st.subheader("❄️ Cold Start Simulation")
    st.markdown(
        "What happens to your recommendations when you've only rated a handful of movies? "
        "We simulate a new user with limited history and show how each method responds."
    )

    user_train_full = train[train["userId"] == user_id].copy()

    if len(user_train_full) < 10:
        st.warning("This user doesn't have enough ratings for a meaningful cold-start simulation. Try a different user.")
    else:
        st.caption(f"User {user_id} has {len(user_train_full)} ratings in training data.")

        @st.cache_data(show_spinner=False)
        def run_cold_start(_train_full, _movies, _tags, uid, n_list):
            import warnings
            from src.recommenders.baselines import RandomRecommender, ItemAverageRecommender
            from src.recommenders.content_based import ContentBasedRecommender

            cold_models = {
                "Random": RandomRecommender,
                "Item Average": ItemAverageRecommender,
                "Content-Based": ContentBasedRecommender,
            }
            results = {mname: [] for mname in cold_models}
            actual_ns = []

            for n_val in n_list:
                if n_val == "all":
                    subset = _train_full
                    actual_n = len(_train_full)
                else:
                    actual_n = min(int(n_val), len(_train_full))
                    subset = _train_full.sample(n=actual_n, random_state=42)

                actual_ns.append(actual_n)
                # Build a minimal train set: the rest of the data minus this user's subset
                other_users = train[train["userId"] != uid]
                mini_train = pd.concat([other_users, subset])

                for mname, Cls in cold_models.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = Cls()
                        try:
                            m.fit(mini_train, _movies, _tags)
                            recs = m.recommend(int(uid), 10)
                            rec_ids = [mid for mid, _ in recs]
                            # Compute genre diversity (entropy)
                            counts = {}
                            for mid in rec_ids:
                                for g in GENRES_BY_MOVIE.get(mid, []):
                                    counts[g] = counts.get(g, 0) + 1
                            total = sum(counts.values())
                            entropy = -sum((c/total)*log2(c/total) for c in counts.values()) if total > 0 else 0.0
                            results[mname].append(round(entropy, 3))
                        except Exception:
                            results[mname].append(None)

            return results, actual_ns

        with st.spinner("Running cold start simulation..."):
            cs_results, actual_ns = run_cold_start(
                user_train_full, movies, tags, user_id,
                [3, 5, 10, 20, 50, "all"]
            )

        # Plot: genre diversity vs number of ratings
        fig_cs = go.Figure()
        palette_cs = {"Random": PALETTE["grey"], "Item Average": PALETTE["orange"], "Content-Based": PALETTE["blue"]}
        for mname, vals in cs_results.items():
            valid = [(n, v) for n, v in zip(actual_ns, vals) if v is not None]
            if valid:
                xs, ys = zip(*valid)
                fig_cs.add_trace(go.Scatter(
                    x=list(xs), y=list(ys),
                    mode="lines+markers",
                    name=mname,
                    line=dict(width=2.5, color=palette_cs.get(mname)),
                    marker=dict(size=8),
                ))
        fig_cs.update_layout(
            title="Genre diversity of recommendations vs. number of ratings available",
            xaxis=dict(title="Ratings available (log scale)", type="log"),
            yaxis=dict(title="Genre entropy (bits)"),
            height=400,
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            template=PLOTLY_THEME,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
        )
        st.plotly_chart(fig_cs, use_container_width=True)

        st.info(
            "Content-Based filtering maintains diversity even with very few ratings because it "
            "relies on item features (genres, tags), not collaborative signal. "
            "Random is diverse by definition. Item Average collapses to popular genres quickly."
        )

        with st.expander("Raw diversity values", expanded=False):
            df_cs = pd.DataFrame(cs_results, index=[f"{n} ratings" for n in actual_ns])
            st.dataframe(df_cs.round(3), use_container_width=True)
