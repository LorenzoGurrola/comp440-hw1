# HW1 writeup

**Name:** Lorenzo Gurrola
**Date:** 2026-09-30

Every placeholder below gets your answer, told to Claude or typed in here yourself. Every number
you give comes from a script in this repo; say which one. Claude may format tables and figures
here; the words are yours.

## Part 0. Predictions

Give these to Claude before any analysis runs. One sentence each, plus one sentence on why you
think so.

**(1) A movie you know well, and what its three most-used tags will be:** Dark Knight: Epic, Action, Dark

**(1) Why you think so:** Because the movie is those things to me.

**(2) Out of every 100 people who rated movies here, how many ever added a tag?** 20

**(2) Why you think so:** I think many people would be content to just rate a movie.

**(3) Can one person's tags take over a movie's tag list? Yes or no:** yes

**(3) Why you think so:** If the movie has no other tags, the only one to tag it would be 100% of the data

## Part 1. Whose data is this?

Code: `part1_data.py`.

**My rule for cutting 32 million ratings to 5 million** (written before reading `data/make_compact.py`)**:** I would do a completely random selection

**One rule I considered and rejected, and why:** By only selecting movies with a certain number of ratings or tags. However, I don't want to bias the sample

**One interesting thing from `data/README.md`:** Oh, what's interesting is that their rule is to keep the top 4000 movies with the most ratings, so their data ends up being a lot denser and I suppose, easier to work with

**How the script's rule differs from mine, and what each keeps that the other drops:** The script keeps only the top x movies, and users that are eligible, while mine has no extra screening in this way. Their rule loses the randomness, my rule loses the density.

**First check. Which of Claude's numbers, the different route you took, and whether it matched** (one good target: 6 tags are the literal text `NA`, which pandas drops unless told not to)**:** I checked the least-rated movie's rating count (83), and I used numpy.loadtxt instead of pandas, and it matched

**Second check. Which of Claude's numbers, the different route you took, and whether it matched:** I checked the share of all ML-32M ratings (15.625%), again using numpy, and it matched

## Part 2. What tags best describe a movie?

Code: `part2_tags.py`.

**My movie, and why I picked it:** The Dark Knight. This has been my favorite movie since I watched it a few years ago. I've seen it many times, and I think it touched on something significant in life that I want to explore more.

**Its most misleading tag in the count-ordered list, and why it misleads:** I don't find any entry particularly misleading, but if I had to choose, I would pick "Morgan Freeman", because I guess he was less of a significant character for me than the rest

**What I learned about how MovieLens collects ratings and tags, from rating and tagging my movie myself (about 100 words):** I noticed that before tagging a movie, you can see popular tags, which may bias you to agree with the consensus

### Up close

One sentence on the figure written before you saw it and one after. The two tables are where the
details below come from. Say which script made them.

**The figure, when the tags and the ratings arrived. What I expected:** I think there will be a surge in the tag activity when the movie was first released in 2008, and slowly trend downwards and plateau with some activity still remaining. I think the ratings will follow a similar trend.
**The figure, what it shows:** The figure shows the number of tag applications per month in blue, and the number of ratings per month in orange. The tag data is quite noisy, with frequent spikes. The ratings data has two obvious spikes, upon release in 2008, and in 2015

**Two interesting details I learned up close that the counts did not show:** The top tagger made 51 tag applications, just under 2% of the total tags on Dark Knight, which is quite a lot. Second, the average tagger rating was indeed higher than the average rating by others, at least for the top 10 tags on this movie.

**Anything up close that contradicted something I had already written down. Which one, what the data showed, and what you now think. Or "nothing yet":** The slot named: "The figure, when the tags and the ratings arrived. What I expected". 2015 had another surge of ratings per month for the Dark Knight, which I did not predict. I did some research on a different Claude, and it couldn't find anything notable that happened with the Dark Knight movie in 2015. However, it did find that the MovieLens website got a redesign the year before, and likely had many new users onboarding and partaking in the ratings process.

### My definition

**My `score(movie, tag)`** (one or two sentences, precise enough that a classmate could code it)**:** My score is (this tag popularity on this movie)/(average of this tag popularity on other movies + 0.00025). Popularity means the share of this movie's distinct taggers, and that movies without the tag do count as 0 in the average.

**One definition I considered and rejected, and why:** I rejected the possibility of not including the constant in the denominator, because of dividing by 0 errors

**Which tags I merged as the same tag, which I kept apart, and why:** I merged tags that had case differences, and trimmed spaces from start and end, because the meaning is the same in both these situations. I kept spelling variants, plurals, and spaces inside a tag separate because I didn't want the rules to get too complicated

**Why my definition, in about 150 words. Name one thing it gains and one thing it loses:**

I chose this score because I wanted a "distinct" tag that described the movie well against other movies. I thought that many people would agree on this tag for this movie, and that it would be a somewhat rare tag across the whole dataset, so it really described this specific movie well. One thing it gains is that a good tag here is rare, like I said. One thing it loses is that it penalizes common tags such as "action" that could describe many movies well.

### The judge

The two slots below are read by scripts, so write them as bare lines: one item to a line, the
movieId first, no bullets and no numbering. A movie line looks like `296, Pulp Fiction (1994)`.
An order line looks like `296: nonlinear, hit men, dark comedy, ...`, the tags best first.

**My ten movies:**

58559, Dark Knight, The (2008)
5810, 8 Mile (2002)
56782, There Will Be Blood (2007)
428, Bronx Tale, A (1993)
105504, Captain Phillips (2013)
1270, Back to the Future (1985)
1907, Mulan (1998)
2762, Sixth Sense, The (1999)
136562, Steve Jobs (2015)
4878, Donnie Darko (2001)

**My own order of the ten most-used tags, written before looking at any data: my movie from step 1, then my nine others from step 4:**

58559: action, dark, superhero, thriller, psychology, Batman, Christian Bale, Heath Ledger, Christopher Nolan, Morgan Freeman
5810: inspiring, eminem, 1990s, music, Eminem, rap, true story, hip hop, based on a true story, Detroit
56782: gritty, greed, morality, intense, religion, father-son relationship, Daniel Day-Lewis, cerebral, atmospheric, visually appealing
428: mafia, coming of age, gangsters, 1960s, organized crime, new york, Robert De Niro, parent child relationship, peer presssure, father-son relationship
105504: suspense, great acting, tense, true story, believable, based on a true story, hijacking, SEAL, Tom Hanks, tom hanks
1270: adventure, 1980s, time travel, classic, future, sci-fi, comedy, alternate reality, quirky, Michael J. Fox
1907: musical, great soundtrack, China, Disney, animation, feminism, strong female lead, Chinese culture, Eddie Murphy, cross dressing
2762: mindfuck, twist ending, suspense, great ending, excellent script, psychological, unpredictable, psychology, ghosts, Bruce Willis
136562: biographical drama, biography, Steve Jobs, technology, apple, computers, dialogue, Kate Winslet, Michael Fassbender, Aaron Sorkin
4878: mindfuck, mental illness, dreamlike, surreal, twist ending, thought-provoking, psychology, time travel, original, atmospheric

**One criterion I considered for the judge and rejected, and why** (the one I used is in `judge/criterion.md`)**:** My initial idea was to give it a mathematical formula, similar to how we calculated score(). If the judge had used something comparable to score(), it would have made the succeeding section difficult because there would be no contrast in ranking.

**Agreement. The number `agreement.py` gives for your `score()`, for popularity and for your own order, and which of the three came closest to the judge:** These numbers are the average of how many of the top 5 tags the judge rated a 4 or 5: 3.6 on my top 10 list, 2.71 for the score method, and 2.75 for the popularity. 3.6 is much higher, which is what I predicted, since I gave instructions to the judge that I used myself

**How the judge skill is built: the files it is made of and what each one does (about 150 words):**

SKILL.md just alerts claude that it has a judge skill, and to look at judge/README.md for a more detailed explanation. This README file breaks down each file in the judge/ directory, and how specifically claude is to interact with each one. It also provides in-depth instructions for how Claude is to execute this "judge skill" at a high level. The system.md file is a short prompt, and shown to each copy of claude code with concise instructions on how to give the correct one-line output, and what input to expect. judge.py is the script for grabbing the data, passing it to each of 5 claude judges, and checking for edge cases and process success. criterion.md was my paragraph for part 2 on how the judge should operate, and criterion_users.md will be my paragraph for part 3. The movies.csv and vocabulary.txt are the inputs to the judge.py script, and ratings_movies.csv and ratings_movies.log are the outputs.

**What happens when I run `/judge`, from the first check to the CSV (about 150 words):**

First, the script checks to make sure it can open the file properly. When the file is named users.csv, it checks the criterion_users.md file for my instructions, otherwise it checks criterion.md. (So ensuring proper spelling of users.csv is important). If the file is still the template, the run stops. If the file is movies.csv, it also reads my top 10 movies out of WRITEUP. It then states which criterion it read, and how many items and tag ratings it wants. For each item, it runs an isolated claude session that sees only the criterion, 1 movie, and its tags. It gives back its rating (1 to 5) for each tag. Then the script asks once more about any item that came back with fewer ratings than it asked for. It then writes the id, tag, and rating into ratings_movies.csv, and the entire summary into ratings_movies.log

**Why a skill: what a skill like this gives you that a script or a prompt alone does not, and where you would use one next (about 100 words):**

A skill lets you define a unique set of behaviors that are useful in certain contexts, but not all. I would build a skill for testing my own knowledge of a certain subject. I could give specific instructions on how I want to be quizzed, and how I want feedback for optimal learning.

### The viewer and the disagreements

**One thing `movie_results.html` showed me that was useful, and one thing about it that got in my way:** One thing is it showed me the top 10 tag outputs of everything we've done so far, which was helpful. What was not so helpful is that it printed out all the tags from the movies, which was hard to scroll through

Then three improvements. For each: what the page would not let you see, what you had Claude
change, and what the changed page shows that the first draft did not.

**Improvement 1:** The page was rough to scroll through, as each tag was listed as a row, so we changed it into each tag listed once with the count shown, organized by count descending

**Improvement 2:** The page was not good at displaying how the actual judge and score() rankings compared, so we changed that by adding a table and delta column

**Improvement 3:** The tables were quite long, and I noticed there were a lot of tags with only 1 application, so I removed these. The changed page shows much shorter tables, which lends itself to easier reading

Then the three disagreements. A disagreement is a movie and a tag where your `score()` and the
judge are furthest apart. For each: the movie and the tag, where your `score()` put it and where
the judge put it, and what you think accounts for the gap.

**Disagreement 1:** Dark Knight, Morgan Freeman. The score() ranked it 4 while the judge ranked it 42. I believe this is because I created the rule to penalize actor names

**Disagreement 2:** There will be blood: dark. Judge ranked 3 while score ranked 18. I believe this is because I told the judge to rate overall themes and moods highly

**Disagreement 3:** The bronx tale: coming of age. Judge rank 1 while score ranked 8. I believe this is because the judge is trying to find the overall theme of the movie.

**One other high-level pattern in the results, and what you think is behind it:** Once again I'm seeing the judge rate overall themes and emotions very highly, and penalizing names. I think this is because I explicitly told the judge to behave this way, and it's nice to see that preference reflected in the data

## Predictions revisited

**Which of my three predictions were wrong, and what I make of each miss:** For prediction 1, I missed the obvious one (Batman). I was thinking more along the line of themes, not characters. I think for prediction 2, I was thinking about the large dataset, not the subset of data that was picked for its tagging and rating density. For prediction 3, I still agree with it, but it is obsolete given the way we picked the subset of data

## Part 3. What tags best describe a user?

Code: `part3_users.py`.

The slot below is read by a script, so write it as bare lines: one rating to a line, no bullets
and no numbering, the movieId first and the rating last, as in `296, Pulp Fiction (1994), 4.5`.

**My 20 ratings:**

34048, War of the Worlds (2005), 4
337, What's Eating Gilbert Grape (1993), 5
5389, Spirit: Stallion of the Cimarron (2002), 5
364, Lion King, The (1994), 5
1393, Jerry Maguire (1996), 5
2288, Thing, The (1982), 4.5
2712, Eyes Wide Shut (1999), 3
4388, Scary Movie 2 (2001), 1.5
79592, Other Guys, The (2010), 3.5
5419, Scooby-Doo (2002), 4.5
121231, It Follows (2014), 3.5
36529, Lord of War (2005), 4
111, Taxi Driver (1976), 4.5
59369, Taken (2008), 4.5
457, Fugitive, The (1993), 3
5502, Signs (2002), 5
46578, Little Miss Sunshine (2006), 4
60074, Hancock (2008), 3.5
4878, Donnie Darko (2001), 4.5
106920, Her (2013), 4.5

**My `score(user, tag)`, in a sentence, and why I started there (about 100 words):**

On each movie I rate, take the top 25% of tags (rounded up to the nearest whole number), ("top" being defined as the tags with the most applications on that movie), and calculate each of those tags "dominance", which is defined as its share of the chosen tags, going off of application numbers. Each remaining tag on this movie should now have a number between 0 and 1. Multiply this number by my rating, and you get, for each tag, a value of how much I like that tag. If these are new tags that have not been calculated yet, those numbers remain. If I have already rated a movie that shares the tags, add these new values to the old values to get the new score. I noticed that a lot of lower-popularity tags on movies were nonsensical, so if we're curating recommendations, I only wanted the highest quality tags to influence that.

**What my score says about me: my top ten tags, and whether they describe my taste (about 100 words):**

My top ten, from `part3_users.py`: animation, aliens, sci-fi, tom cruise, coming of age, soundtrack, mental illness, action, atmospheric, bittersweet.

I'm happy to say they roughly do describe my tastes! I think if I were to sit down and curate this top 10 tag list, though, I could come up with a better one. I'm not particularly partial to tom cruise as an actor

**What my user viewer shows and why I chose that (about 100 words):**

It shows each user's top 10 tags, along with the score. I also showed their top 3 movies in an attempt to be able to quickly see if the tags match the movies

**What I put in the description column for a person, and why (about 150 words):**

I put that persons top five movies, each of those movie's title and year, and the vocabulary tags other people applied to that movie in alphabetical order. I figured this would give the agent enough of description to judge tags accurately

**My criterion for people: what it asks the judge to do that the movie criterion did not (about 60 words):**

It ranks tags higher if they are applied to movies that show up in the top 5 list.

**The user-tag pairs I chose to judge, how many, and why those (about 100 words):**

I chose the top 50 tags by popularity for each person, due to budgeting restraints. I chose those based off of which are in the vocabulary.txt file, because the judge only looks at tags in the vocabulary.txt file

**Improvement 1: what I changed in the scoring function, what the judge and the viewer showed before and after (about 150 words):**

I wanted to penalize very common tags that weren't very specific, so I gathered the top 1% of tags and applyed a 50% penalty to their application counts before making the top 25% cut. For both the judge and the viewer, the rankings shifted places slightly, but not as much as I expected

**Improvement 2: the same (about 150 words):**

For me, the scores maxxed out at 1, and for others, maxxed out around 50. So we min-max scaled them in the viewer to the range 1-5, making for easier interpretation and comparision. It is interesting to see how far the bottom of the top 10 list is from 1. They are all still quite far above 1. This change didn't affect the judge.

## Part 4. Working with Claude

Give these to Claude the way you gave it the rest. Graded on the catch and the candor, not on
making Claude look good or bad.

**A moment where Claude was wrong or overconfident, how you caught it, and where it
happened. Name the part and the step, so the moment can be found:** XXXX

**One call where you overrode Claude, and why:** XXXX

**What you would hand to Claude sooner next time:** XXXX

**Did Claude name the misleading tag in Part 2 step 1 before you did? What happened:** XXXX

**The figure. Would asking Claude "what does this show?" have produced your sentence, and what
would have been missing from it:** XXXX

**Hours spent:** XXXX

**Anyone who helped you, or "no one":** XXXX
