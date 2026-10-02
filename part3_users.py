"""
Part 3: what tags best describe a user?

    uv run python part3_users.py

The handout's Part 3 is the spec. One piece is written for you, the piece that has to agree
with `WRITEUP.md` line for line: reading the 20 ratings out of your "My 20 ratings" slot and
adding you to the ratings table as a user of your own. Everything after that is yours.

You are added under userId 999999. Real userIds in `data/ratings.csv.gz` stop at 200,935, so
that number cannot be a real person's, and it is easy to pick out of a printout.

What this script must print, under the labels shown:

    == (1) my ratings ==
        How many ratings were read out of your slot, how many lines it could not read a
        rating from, and how many rows the ratings table has with yours in it. Twenty
        ratings is what the handout asks for; the script reports what it found and leaves
        the count to you.

    == (2) score(user, tag) ==
        Your `score(user, tag)` over the users you are looking at, your own row included.
        Write it in this file as

            score(ratings_df, tags_df, movies_df) -> DataFrame[userId, tag, score]

        one row per user-tag pair, higher score meaning the tag describes the user better.
        Print your own ten best tags, and the number of rows and distinct users it returned.
        What the score is, and why you started there, is yours and goes in `WRITEUP.md`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import load_all

REPO = Path(__file__).resolve().parent
WRITEUP = REPO / "WRITEUP.md"

ME = 999999                 # your userId: above every real one, so it collides with nobody
SLOT = "My 20 ratings"      # the WRITEUP.md slot your ratings are read from


def read_my_ratings(writeup: Path = WRITEUP) -> tuple[pd.DataFrame, int]:
    """Your ratings from the "My 20 ratings" slot in WRITEUP.md, as movieId and rating.

    The same rule the judge uses for "My ten movies": every line in that slot starts with a
    movieId. The rating is the last number on the line, so the title between them is for
    people and may hold anything, the year included. Bare lines only: a bulleted or a
    numbered list reads as no ratings at all, or reads the list numbers as movieIds.

        296, Pulp Fiction (1994), 4.5

    A line whose last number is not a rating between 0.5 and 5.0 is left out and counted,
    because the year in a title is a number too: `296, Pulp Fiction (1994)` with the rating
    forgotten would otherwise be read as a rating of 1994. So is the `XXXX` an unfilled slot
    holds, which is why this is safe to run before you have written anything.

    Returns the ratings and how many lines were left out."""
    rows, skipped, inside = [], 0, False
    for line in writeup.read_text(encoding="utf-8").splitlines():
        if line.startswith("**"):            # a bold label opens the next slot
            inside = SLOT in line
            continue
        if not inside or not re.match(r"\s*\d", line):
            continue
        numbers = re.findall(r"\d+(?:\.\d+)?", line)
        rating = float(numbers[-1]) if len(numbers) > 1 else 0.0
        if not 0.5 <= rating <= 5.0:
            skipped += 1
            continue
        rows.append({"movieId": int(numbers[0].split(".")[0]), "rating": rating})
    return pd.DataFrame(rows, columns=["movieId", "rating"]), skipped


def add_me(ratings: pd.DataFrame, mine: pd.DataFrame) -> pd.DataFrame:
    """Your ratings appended to everybody else's, under userId ME.

    The timestamp is the newest one in the data: you rated these after everyone else did."""
    if mine.empty:
        return ratings
    mine = mine.assign(userId=ME, timestamp=int(ratings["timestamp"].max()))
    return pd.concat([ratings, mine[ratings.columns]], ignore_index=True)


# ------------------------------------------------------------------- yours to write ---

SEED = 440                  # breaks ties at the 25% cut at random, the same way every run


def top_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Each movie's top 25% of tags, rounded up, by applications, with each one's dominance.

    Tags are cleaned by the Part 2 rule (case and spaces at either end do not make a
    different tag). Ties at the cut are broken at random under SEED. Dominance is a tag's
    applications over the applications of all the chosen tags on that movie."""
    from part2_tags import clean
    counts = clean(tags).groupby(["movieId", "tag"]).size().rename("applications").reset_index()
    counts = counts.sample(frac=1, random_state=SEED)                  # random order first,
    counts = counts.sort_values(["movieId", "applications"],            # then a stable sort,
                                ascending=[True, False], kind="stable")  # so ties stay random
    rank = counts.groupby("movieId").cumcount() + 1
    keep = -(-counts.groupby("movieId")["tag"].transform("size") // 4)  # ceil(n / 4)
    chosen = counts[rank <= keep].copy()
    chosen["dominance"] = (chosen["applications"]
                           / chosen.groupby("movieId")["applications"].transform("sum"))
    return chosen[["movieId", "tag", "dominance"]]


def score(ratings: pd.DataFrame, tags: pd.DataFrame, movies: pd.DataFrame):
    """The student's score: for every movie a user rated, each of its top-25% tags gets the
    user's rating times its dominance on that movie; a tag's score is the sum over the
    user's movies. Vectorized: one merge of ratings onto each movie's chosen tags."""
    pairs = ratings[["userId", "movieId", "rating"]].merge(top_tags(tags), on="movieId")
    pairs["score"] = pairs["rating"] * pairs["dominance"]
    return (pairs.groupby(["userId", "tag"], as_index=False)["score"].sum()
            .sort_values(["userId", "score"], ascending=[True, False], ignore_index=True))


def pick_others(ratings: pd.DataFrame, tags: pd.DataFrame, mine: pd.DataFrame,
                n: int = 9, draw: int = 10) -> list[dict]:
    """The student's rule for the other users in the viewer. The pool is the judge's
    vocabulary minus my own top ten. For each pick: draw `draw` tags from the pool at
    random, total every user's score over those tags (0 where a user has none), and take
    the highest-scoring user not already picked. I am never a candidate.

    Same score as score(), computed for every user at once as a sparse product, user by
    movie ratings times movie by tag dominance, so 5 million ratings need no loop."""
    from scipy import sparse
    vocab = [t.strip() for t in (REPO / "judge" / "vocabulary.txt").read_text().splitlines() if t.strip()]
    pool = sorted(set(vocab) - set(mine["tag"].head(10)))
    chosen = top_tags(tags)
    chosen = chosen[chosen["tag"].isin(pool)]
    others = ratings[ratings["userId"] != ME]
    users, u = np.unique(others["userId"], return_inverse=True)
    movies = np.unique(np.concatenate([others["movieId"].to_numpy(), chosen["movieId"].to_numpy()]))
    col = {t: j for j, t in enumerate(pool)}
    R = sparse.csr_matrix((others["rating"], (u, np.searchsorted(movies, others["movieId"]))),
                          shape=(len(users), len(movies)))
    D = sparse.csr_matrix((chosen["dominance"], (np.searchsorted(movies, chosen["movieId"]),
                           chosen["tag"].map(col))), shape=(len(movies), len(pool)))
    S = (R @ D).tocsc()                                  # every user's score on every pool tag
    rng, taken, picks = np.random.default_rng(SEED), set(), []
    for _ in range(n):
        drawn = list(rng.choice(pool, size=draw, replace=False))
        total = np.asarray(S[:, [col[t] for t in drawn]].sum(axis=1)).ravel()
        for i in np.argsort(-total, kind="stable"):
            if users[i] not in taken:
                break
        taken.add(users[i])
        picks.append({"userId": int(users[i]), "total": float(total[i]), "tags": drawn})
    return picks


def part3_users(ratings, tags, movies, links):
    print("== (1) my ratings ==")
    mine, skipped = read_my_ratings()
    print(f'{len(mine)} rating(s) read from the "{SLOT}" slot in WRITEUP.md.')
    if not len(mine):
        print(f'Nothing was read out of the "{SLOT}" slot. It is read one rating to a line, '
              f"with no bullets and no numbering: the movieId first, then the title, then "
              f"your rating, as in `296, Pulp Fiction (1994), 4.5`.")
    if skipped:
        print(f"{skipped} line(s) in that slot had no rating between 0.5 and 5.0 at the "
              f"end and were left out.")
    ratings = add_me(ratings, mine)
    if len(mine):
        print(f"{len(ratings):,} ratings with yours in, as userId {ME}.")
    else:
        print(f"{len(ratings):,} ratings, none of them yours yet.")

    print("== (2) score(user, tag) ==")
    users = [ME]                # the users scored so far: you alone, until you name others
    scored = score(ratings[ratings["userId"].isin(users)], tags, movies)
    print(scored[scored["userId"] == ME].head(10)[["tag", "score"]].to_string(index=False))
    print(f"{len(scored):,} user-tag rows over {scored['userId'].nunique()} user(s)")

    print("== (3) the nine others ==")
    picks = pick_others(ratings, tags, scored[scored["userId"] == ME])
    for p in picks:
        print(f"user {p['userId']}: total {p['total']:.3f} on {', '.join(p['tags'])}")


if __name__ == "__main__":
    ratings, tags, movies, links = load_all()
    part3_users(ratings, tags, movies, links)
