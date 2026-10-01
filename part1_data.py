"""
Part 1: whose data is this?

    uv run python part1_data.py

Write your own cut rule and your two checks before you run anything here. Doing it in that
order is what Part 1 is asking for. What this script must print, under the labels shown:

    == (a) how much ==
        Rows in each of the four files, distinct users, distinct movies, and the share of
        all 32,000,204 MovieLens ratings this set holds.

    == (b) spread ==
        Ratings per user and ratings per movie: median, minimum and maximum of each. Tag
        applications per user and per movie: the same three. How many of the users who
        rated anything ever applied a tag, as a count and as a share.

    == (c) top tags, two ways ==
        The 20 most-used tags by number of applications, and the 20 most-used tags by number
        of distinct users who applied them. Print the two lists one after the other, with
        both numbers on every row, so you can see where a tag's two ranks differ.

    == (d) two checks ==
        Two claims from (a) to (c) re-derived by a route that does not reuse the code that
        produced them, printed with both numbers side by side and the word MATCH or DIFFER.
        Targets that exist in this data: the share of all 32M ratings the set holds
        (`data/README.md` says 15.6 percent); the number of distinct users who applied a
        tag (14,019); the rating count of the least-rated kept movie (83); the 6 tag
        rows whose text is literally `NA`, which vanish if a reader is built without
        `keep_default_na=False`.

No figures are required in Part 1. `WRITEUP.md` takes one interesting thing from
`data/README.md`, your own cut rule and the rule you rejected, how `data/make_compact.py`'s
rule differs from yours, and your two checks.
"""

import numpy as np

from load_data import DATA, load_all

FULL_RATINGS = 32_000_204   # all ML-32M ratings, from data/README.md


def spread(label, counts):
    print(f"{label}: median {counts.median():g}, min {counts.min():,}, max {counts.max():,}")


def part1_data(ratings, tags, movies, links):
    print("== (a) how much ==")
    for name, df in (("ratings", ratings), ("tags", tags), ("movies", movies), ("links", links)):
        print(f"{name}: {len(df):,} rows")
    print(f"distinct users in ratings: {ratings.userId.nunique():,}")
    print(f"distinct movies in ratings: {ratings.movieId.nunique():,}")
    share = len(ratings) / FULL_RATINGS
    print(f"share of all {FULL_RATINGS:,} ML-32M ratings: {share:.2%}")

    print("== (b) spread ==")
    per_movie = ratings.groupby("movieId").size()
    spread("ratings per user", ratings.groupby("userId").size())
    spread("ratings per movie", per_movie)
    spread("tag applications per user", tags.groupby("userId").size())
    spread("tag applications per movie", tags.groupby("movieId").size())
    raters = ratings.userId.unique()
    taggers = np.isin(raters, tags.userId.unique()).sum()
    print(f"raters who ever applied a tag: {taggers:,} of {len(raters):,} ({taggers / len(raters):.2%})")

    print("== (c) top tags, two ways ==")
    # Raw tag strings, exactly as typed: no lowercasing or trimming.
    by_tag = tags.groupby("tag").agg(applications=("userId", "size"), users=("userId", "nunique"))
    print("-- by applications --")
    print(by_tag.sort_values("applications", ascending=False).head(20).to_string())
    print("-- by distinct users --")
    print(by_tag.sort_values("users", ascending=False).head(20).to_string())

    print("== (d) two checks ==")
    # Student's route for both checks: numpy reads the raw file, no pandas, no load_data.py.
    ids = np.loadtxt(DATA / "ratings.csv.gz", delimiter=",", skiprows=1, usecols=1, dtype=np.int64)
    _, n_per_movie = np.unique(ids, return_counts=True)

    def check(label, script, theirs):
        verdict = "MATCH" if script == theirs else "DIFFER"
        print(f"{label}: script {script}  numpy {theirs}  {verdict}")

    check("least-rated movie's rating count", int(per_movie.min()), int(n_per_movie.min()))
    check("share of all ML-32M ratings", f"{share:.4%}", f"{len(ids) / FULL_RATINGS:.4%}")


if __name__ == "__main__":
    ratings, tags, movies, links = load_all()
    part1_data(ratings, tags, movies, links)
