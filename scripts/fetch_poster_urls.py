"""
fetch_poster_urls.py
--------------------
Fetches TMDB poster URLs for the 10 movies used in the Claude Design prototype.

Run: python scripts/fetch_poster_urls.py

OUTPUT:
  Prints a Python dict you can paste directly into the Claude Design brief
  and into app.py as a hardcoded fallback poster map.
"""

import urllib.request
import urllib.parse
import json

TMDB_API_KEY = "c6c9fc467087559ddbf510668d56b03d"

BASE_IMAGE_URL = "https://image.tmdb.org/t/p/w185"
SEARCH_URL     = "https://api.themoviedb.org/3/search/movie"

MOVIES = [
    ("The Shawshank Redemption", 1994),
    ("Schindler's List",         1993),
    ("Pulp Fiction",             1994),
    ("The Dark Knight",          2008),
    ("Forrest Gump",             1994),
    ("The Matrix",               1999),
    ("Goodfellas",               1990),
    ("Fight Club",               1999),
    ("Inception",                2010),
    ("The Silence of the Lambs", 1991),
]


def fetch_poster(title: str, year: int) -> str | None:
    params = urllib.parse.urlencode({
        "api_key": TMDB_API_KEY,
        "query":   title,
        "year":    year,
    })
    try:
        with urllib.request.urlopen(f"{SEARCH_URL}?{params}", timeout=5) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if results and results[0].get("poster_path"):
            return f"{BASE_IMAGE_URL}{results[0]['poster_path']}"
    except Exception as e:
        print(f"  Error fetching '{title}': {e}")
    return None


if __name__ == "__main__":
    if not TMDB_API_KEY:
        print("ERROR: Paste your TMDB API key into TMDB_API_KEY at the top of this script.")
        print("Get one free at https://www.themoviedb.org/settings/api")
        exit(1)

    print("Fetching poster URLs from TMDB...\n")
    results = {}
    for title, year in MOVIES:
        url = fetch_poster(title, year)
        results[title] = url
        status = "✓" if url else "✗ (not found)"
        print(f"  {status}  {title} ({year})")
        if url:
            print(f"       {url}")

    # ── Output 1: Python dict for hardcoding in app.py ──────────────────────
    print("\n\n# ── Paste into app.py as POSTER_FALLBACKS ──")
    print("POSTER_FALLBACKS = {")
    for title, url in results.items():
        if url:
            print(f'    "{title}": "{url}",')
    print("}")

    # ── Output 2: Markdown image tags for Claude Design brief ───────────────
    print("\n\n# ── Poster URLs for Claude Design brief ──")
    for title, url in results.items():
        if url:
            print(f"- {title}: {url}")
        else:
            print(f"- {title}: (not found — use placeholder)")

    # ── Output 3: Placeholder fallbacks for any that failed ─────────────────
    missing = [t for t, u in results.items() if not u]
    if missing:
        print(f"\n# Placeholders for {len(missing)} missing poster(s):")
        for title in missing:
            slug = title.replace(" ", "+")
            print(f"- {title}: https://placehold.co/185x278/1a1c2e/40bcf4?text={slug}")
