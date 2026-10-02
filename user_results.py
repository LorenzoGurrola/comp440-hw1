"""User viewer.

Builds one self-contained HTML page with ten people: me (userId 999999) and the nine others
picked by `pick_others()` in `part3_users.py`. For each person it shows

  * their top 10 tags by my `score(user, tag)`, highest first;
  * their 3 highest-rated movies. Ties in rating are broken by how many tags that person
    applied to the movie, more first, and a tie in that too is broken at random under SEED.

    uv run python user_results.py            # writes user_results.html
    uv run python user_results.py --text     # the same content as plain text
"""

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import load_all
from part3_users import ME, SEED, add_me, pick_others, read_my_ratings, score

REPO = Path(__file__).resolve().parent


def top_movies(ratings: pd.DataFrame, tags: pd.DataFrame, titles: pd.Series, n: int = 3):
    """Each user's n highest-rated movies: rating first, then how many tags the user
    applied to that movie, then a random draw under SEED."""
    own = tags.groupby(["userId", "movieId"]).size().rename("my_tags")
    r = ratings[["userId", "movieId", "rating"]].join(own, on=["userId", "movieId"])
    r["my_tags"] = r["my_tags"].fillna(0).astype(int)
    r["draw"] = np.random.default_rng(SEED).random(len(r))
    r = r.sort_values(["userId", "rating", "my_tags", "draw"], ascending=[True, False, False, True])
    r = r.groupby("userId").head(n)
    return r.assign(title=r["movieId"].map(titles))


def build():
    ratings, tags, movies, _ = load_all()
    mine, _ = read_my_ratings()
    ratings = add_me(ratings, mine)
    my_scores = score(ratings[ratings["userId"] == ME], tags, movies)
    others = pick_others(ratings, tags, my_scores)
    ids = [ME] + [p["userId"] for p in others]
    sub = ratings[ratings["userId"].isin(ids)]
    scored = score(sub, tags, movies)
    best = top_movies(sub, tags, movies.set_index("movieId")["title"])
    people = []
    for i in ids:
        people.append({
            "userId": i,
            "label": "me" if i == ME else f"user {i}",
            "n_ratings": int((sub["userId"] == i).sum()),
            "tags": scored[scored["userId"] == i].head(10)[["tag", "score"]],
            "movies": best[best["userId"] == i][["title", "rating", "my_tags"]],
        })
    return people


def render_text(people) -> str:
    out = []
    for p in people:
        out.append(f"== {p['label']} ({p['n_ratings']:,} ratings) ==")
        out.append("top 10 tags by score:")
        out.append(p["tags"].to_string(index=False))
        out.append("top 3 movies by rating:")
        out.append(p["movies"].rename(columns={"my_tags": "tags they applied"}).to_string(index=False))
        out.append("")
    return "\n".join(out)


def table_html(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(f'{v:.3f}' if isinstance(v, float) and c == 'score' else str(v))}</td>"
                                    for c, v in zip(df.columns, row)) + "</tr>"
                   for row in df.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def render_html(people) -> str:
    css = """
:root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --head:#f4f4f4; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --fg:#eee; --muted:#aaa; --line:#333; --head:#222; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:900px; padding:16px; }
h2 { margin-top:2em; border-bottom:1px solid var(--line); }
.meta { color:var(--muted); }
.pair { display:flex; flex-wrap:wrap; gap:24px; }
.pair > div { flex:1 1 320px; min-width:0; }
table { border-collapse:collapse; width:100%; }
th, td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--line); }
th { background:var(--head); }
"""
    body = ["<h1>User viewer</h1>",
            "<p class='meta'>Top 10 tags by my score(user, tag), and each person's 3 highest-rated "
            "movies (ties: more tags they applied first, then random).</p>"]
    for p in people:
        movies = p["movies"].rename(columns={"title": "movie", "my_tags": "tags they applied"})
        body.append(f"<h2>{html.escape(p['label'])}</h2>"
                    f"<p class='meta'>{p['n_ratings']:,} ratings</p><div class='pair'>"
                    f"<div><h3>Top 10 tags</h3>{table_html(p['tags'])}</div>"
                    f"<div><h3>Top 3 movies</h3>{table_html(movies)}</div></div>")
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>User viewer</title>\n<style>{css}</style>\n</head>\n<body>\n"
            + "\n".join(body) + "\n</body>\n</html>\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", action="store_true", help="print instead of writing HTML")
    parser.add_argument("--out", default=str(REPO / "user_results.html"))
    args = parser.parse_args()
    people = build()
    if args.text:
        print(render_text(people))
    else:
        Path(args.out).write_text(render_html(people), encoding="utf-8")
        print(f"wrote {args.out}")
