"""
Part 2: what tags best describe a movie?

    uv run python part2_tags.py

Steps 1 to 4 of the handout's Part 2 live here, plus the scores and the rankings that steps 5
and 6 need. The judge itself runs through `/judge`, and its answer is read through
`agreement.py` and `results_viewer.py`. What this script must print, under the labels shown,
and what it must write:

    == (1) the obvious answer ==
        Your chosen movie's title, its rating count and its tag-application count, then
        every tag applied to it with how many times it was applied, most-applied first.
        Pick a movie with at least 500 ratings and 30 tag applications. The most misleading
        entry in that list is your sentence in `WRITEUP.md`, not this script's.

    == (2) up close ==
        The numbers behind the one required figure and the two tables, so that everything
        shown here has printed output a reader can check it against. Write, to `figures/`:

            figures/part2_when.png          when the tags arrived: tag applications over
                                            time, with the movie's ratings over time behind
                                            them.

        The figure has labeled axes and a caption naming the question it answers. Claude
        may draw and label it; the sentence in `WRITEUP.md` about what it shows is yours.

        Then two tables, each printed under its own label:

            who added each tag              the movie's heaviest taggers, how many tag
                                            applications each made, and what share of the
                                            movie's applications that is.
            how the taggers rated it        for each of the movie's top tags, how the
                                            people who applied it rated the movie, beside
                                            how everyone else rated it.

        Claude prints the tables and says what the columns are. What they show is your two
        interesting details in `WRITEUP.md`, not this script's.

    == (3) my definition ==
        Your `score` over the whole set. Write it in this file as

            score(tags_df, ratings_df, movies_df) -> DataFrame[movieId, tag, score]

        one row per movie-tag pair, higher score meaning the tag describes the movie better.
        Print its top 15 rows for your chosen movie, and the number of rows and distinct
        movies it returned over the whole set. Families you could use, none of them
        preferred: distinct users who applied the tag; a rarity weight, the count times how
        few movies carry the tag; a damped version of either; something of your own. Whatever
        you choose, `WRITEUP.md` gets what you chose, what you rejected, and why.

    == (4) cleaning ==
        Whatever cleaning your `score()` does, and its size: how many raw tag strings went
        in, how many distinct tags came out, and the five mergers that absorbed the most
        applications. If you clean nothing, print that and say why in `WRITEUP.md`.
        Merging `Sci-Fi`, `sci-fi` and `scifi` is a decision, and so is not merging them.

    == (5) scores.csv ==
        `scores.csv` in the repo root, columns `movieId,tag,score`, holding a score for every
        movie and tag the judge will be asked about. That is two sets put together:

            every movie and tag in `judge/movies.csv`, which has one row per movie and a
            `tags` column of tags joined by `|`;
            plus, for each of the ten movies in your "My ten movies" slot, every tag from
            `judge/vocabulary.txt` that appears on it, matched after stripping and
            lowercasing, which is the same rule `judge/movies.csv` used.

        The second set matters because the judge adds your ten movies to its list, and
        `agreement.py` compares exactly what the two files share: a tag you never scored is
        dropped without a number. Print how many were asked for and how many you wrote.

    == (6) the four rankings ==
        For each of the ten movies in your "My ten movies" slot, four rankings of the same tags,
        printed one after another and never in one table:

            the counts: the ten most-used tags, by how many times each was applied;
            your own order, from the `WRITEUP.md` slot you filled before seeing any data;
            the judge's order, from `judge/ratings_movies.csv`;
            your `score()`'s order.

        Print each list under its own heading, best first. `results_viewer.py` builds the same
        four lists as a page you can read. Which tag is the artifact, and what the
        disagreements mean, is your paragraph in `WRITEUP.md`.
"""

from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_all

MY_MOVIE = 58559   # Dark Knight, The (2008), the student's step 1 movie
FIGURES = Path(__file__).resolve().parent / "figures"


def per_month(df):
    months = pd.to_datetime(df["timestamp"], unit="s").dt.to_period("M")
    return months.value_counts().sort_index()


def when_figure(my_ratings, my_tags, title):
    """figures/part2_when.png: tag applications and ratings per month, one panel each."""
    r, t = per_month(my_ratings), per_month(my_tags)
    span = pd.period_range(min(r.index.min(), t.index.min()), max(r.index.max(), t.index.max()), freq="M")
    r, t = r.reindex(span, fill_value=0), t.reindex(span, fill_value=0)
    x = span.to_timestamp()

    # Two panels on a shared time axis, not one chart with two y-scales.
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), facecolor="#fcfcfb")
    for ax, series, color, label in ((top, t, "#2a78d6", "tag applications per month"),
                                     (bottom, r, "#eb6834", "ratings per month")):
        ax.set_facecolor("#fcfcfb")
        ax.plot(x, series.to_numpy(), color=color, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#e4e3dc", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    bottom.set_xlabel("year (one point per month)")
    # One labelled tick at the start of every year.
    bottom.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    bottom.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    bottom.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"When did the tags and the ratings on {title} arrive?")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "part2_when.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # The numbers behind the figure, summed by year so they fit on screen.
    by_year = pd.DataFrame({"tag applications": t, "ratings": r})
    by_year = by_year.groupby(by_year.index.year).sum()
    by_year.index.name = "year"
    print("-- figures/part2_when.png, by year --")
    print(by_year.to_string())


def part2_tags(ratings, tags, movies, links):
    print("== (1) the obvious answer ==")
    title = movies.set_index("movieId").loc[MY_MOVIE, "title"]
    mine = tags[tags.movieId == MY_MOVIE]
    print(f"{title}: {(ratings.movieId == MY_MOVIE).sum():,} ratings, {len(mine):,} tag applications")
    # Raw tag strings, exactly as typed: no lowercasing or trimming.
    with pd.option_context("display.max_rows", None):
        print(mine["tag"].value_counts().rename("applications").to_string())

    print("== (2) up close ==")
    when_figure(ratings[ratings.movieId == MY_MOVIE], mine, title)

    print("-- who added each tag --")
    who = mine.groupby("userId").size().rename("tag applications").sort_values(ascending=False)
    who = who.to_frame().assign(share=lambda d: (d["tag applications"] / len(mine)).map("{:.1%}".format))
    print(who.head(10).to_string())

    print("-- how the taggers rated it --")
    # Top ten raw tag strings by applications. A tagger who never rated the movie has no
    # rating to count, so n_taggers can be smaller than the tag's application count.
    my_ratings = ratings[ratings.movieId == MY_MOVIE].set_index("userId")["rating"]
    rows = []
    for tag in mine["tag"].value_counts().head(10).index:
        applied = my_ratings.index.isin(mine.loc[mine["tag"] == tag, "userId"])
        rows.append({"tag": tag,
                     "n_taggers": int(applied.sum()), "taggers_mean": my_ratings[applied].mean(),
                     "n_others": int((~applied).sum()), "others_mean": my_ratings[~applied].mean()})
    print(pd.DataFrame(rows).set_index("tag").round(2).to_string())

    print("== (3) my definition ==")

    print("== (4) cleaning ==")

    print("== (5) scores.csv ==")

    print("== (6) the four rankings ==")


if __name__ == "__main__":
    ratings, tags, movies, links = load_all()
    part2_tags(ratings, tags, movies, links)
