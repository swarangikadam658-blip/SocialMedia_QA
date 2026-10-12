"""
qa_system.py - analysis layer for Forum Answers.

Everything the app does *around* retrieval lives here:
  - trust score (explains how much to believe an answer)
  - Reddit-vs-Wikipedia comparison
  - keyword highlighting and key-point splitting
  - query logging + feedback reading (for the Insights tab)
  - helpers for the Explore tab

Only needs pandas + scikit-learn, which the project already uses.
"""

import csv
import html
import math
import os
import re
from collections import Counter
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOG_PATH = os.path.join("data", "query_log.csv")
FEEDBACK_PATH = os.path.join("data", "feedback.csv")
LOG_FIELDS = ["time", "question", "community", "source",
              "trust", "trust_label", "seconds"]

_EXTRA_STOP = {
    "just", "like", "really", "think", "know", "people", "thing", "things",
    "one", "get", "would", "also", "even", "much", "going", "say", "said",
    "don", "ll", "ve", "re", "didn", "doesn", "https", "http", "www", "com",
    "gt", "amp", "quot", "nbsp", "yes", "https", "edit",
}
STOP = set(ENGLISH_STOP_WORDS) | _EXTRA_STOP
_HTML_WORDS = {"amp", "quot", "nbsp", "gt", "lt"}


# ============================================================
# TEXT HELPERS
# ============================================================

def keywords(text, min_len=3):
    """Lower-case content words (no stop words)."""
    return [t for t in re.findall(r"[a-z]+", str(text).lower())
            if len(t) >= min_len and t not in STOP]


def strip_citations(text):
    """Remove [1] [2] markers."""
    return re.sub(r"\s*\[\d+\]", "", str(text)).strip()


def shorten(text, limit=450):
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def text_similarity(a, b):
    """TF-IDF cosine similarity between two texts (0..1)."""
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([a, b])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:  # empty vocabulary
        return 0.0


def highlight_html(text, words):
    """HTML-escape `text`, then wrap the query's keywords in <mark>. Safe to render."""
    safe = html.escape(str(text), quote=True)
    words = sorted({w for w in words if w.isalpha() and w not in _HTML_WORDS},
                   key=len, reverse=True)
    if not words:
        return safe
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in words) + r")(?:s|es|ing|ed)?\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)


def key_points(answer):
    """Split a synthesized answer into one bullet per cited source."""
    parts = re.split(r"(?<=\[\d\])\s+", str(answer).strip())
    return [p.strip() for p in parts if p.strip()]


# ============================================================
# TRUST SCORE
# ============================================================

def _clip(x):
    return max(0.0, min(1.0, float(x)))


def trust_label(score):
    if score >= 70:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def source_agreement(snippets):
    """Mean pairwise similarity between the snippets (None if fewer than 2)."""
    if len(snippets) < 2:
        return None
    sims = []
    for i in range(len(snippets)):
        for j in range(i + 1, len(snippets)):
            sims.append(text_similarity(snippets[i], snippets[j]))
    return sum(sims) / len(sims)


def compute_trust(relevance, snippets, upvotes, max_upvotes, wiki_similarity):
    """
    Explainable 0-100 trust score for a Reddit-based answer.

    Four signals (a signal that is unavailable is dropped and the rest are
    re-weighted, so the score is never silently penalised for missing data):
      question match     40%  how relevant the thread is to the question
      sources agree      20%  do the cited threads say similar things
      community support  20%  average upvotes of the cited answers
      Wikipedia backs it 20%  similarity between the answer and Wikipedia
    """
    parts = [{
        "name": "Question match", "value": _clip(relevance), "weight": 0.40,
        "note": "How closely the forum thread matches what you asked.",
    }]

    agree = source_agreement(snippets)
    if agree is not None:
        parts.append({
            "name": "Sources agree", "value": _clip(agree / 0.30), "weight": 0.20,
            "note": "Do the different Reddit threads say similar things?",
        })

    if max_upvotes and max_upvotes > 0 and upvotes:
        support = math.log1p(sum(upvotes) / len(upvotes)) / math.log1p(max_upvotes)
        parts.append({
            "name": "Community support", "value": _clip(support), "weight": 0.20,
            "note": "Average upvotes of the cited answers (log scale).",
        })

    if wiki_similarity is not None:
        parts.append({
            "name": "Wikipedia backs it up", "value": _clip(wiki_similarity / 0.30),
            "weight": 0.20,
            "note": "Does the encyclopedia say something similar?",
        })

    total_w = sum(p["weight"] for p in parts)
    for p in parts:
        p["share"] = p["weight"] / total_w
    score = round(100 * sum(p["value"] * p["share"] for p in parts))
    return {"score": score, "label": trust_label(score), "parts": parts}


def wiki_only_trust(relevance):
    """Trust for a Wikipedia-only answer (one signal, capped at 80)."""
    value = _clip(relevance / 0.35)
    score = min(80, round(100 * value))
    return {
        "score": score,
        "label": trust_label(score),
        "parts": [{
            "name": "Question match", "value": value, "weight": 1.0, "share": 1.0,
            "note": "How closely the Wikipedia article matches your question. "
                    "Capped at 80 because only one source was used.",
        }],
    }


# ============================================================
# REDDIT vs WIKIPEDIA
# ============================================================

def _stem(word):
    """Very light stemmer so 'vaccine' and 'vaccines' count as the same idea."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def compare_sources(reddit_text, wiki_text, n=8):
    """Which ideas do the community and the encyclopedia share, and where do they differ?"""
    r_words, w_words = keywords(reddit_text), keywords(wiki_text)
    rc = Counter(_stem(w) for w in r_words)
    wc = Counter(_stem(w) for w in w_words)

    # show real words, not stems: the most common spelling of each stem
    forms = {}
    for w in r_words + w_words:
        forms.setdefault(_stem(w), Counter())[w] += 1
    show = lambda stem: forms[stem].most_common(1)[0][0]

    shared = sorted(set(rc) & set(wc), key=lambda k: -(rc[k] + wc[k]))[:n]
    only_reddit = [k for k, _ in rc.most_common() if k not in wc][:n]
    only_wiki = [k for k, _ in wc.most_common() if k not in rc][:n]
    shared, only_reddit, only_wiki = (
        [show(k) for k in group] for group in (shared, only_reddit, only_wiki)
    )

    sim = text_similarity(reddit_text, wiki_text)
    if sim >= 0.30:
        verdict = "Strong overlap - the community and Wikipedia tell a similar story."
    elif sim >= 0.12:
        verdict = "Partial overlap - they cover the topic from different angles."
    else:
        verdict = "Little overlap - treat this answer with extra care."

    return {"similarity": sim, "verdict": verdict, "shared": shared,
            "only_reddit": only_reddit, "only_wiki": only_wiki}


# ============================================================
# LOGGING (feeds the Insights tab)
# ============================================================

def _csv_safe(text):
    text = str(text)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def log_query(question, community, source, trust, trust_label_, seconds):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LOG_FIELDS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"), _csv_safe(question),
            community, source, trust, trust_label_, round(seconds, 2),
        ])


def read_log():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_FIELDS)
    try:
        return pd.read_csv(LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=LOG_FIELDS)


def read_feedback():
    cols = ["time", "question", "source", "vote"]
    if not os.path.exists(FEEDBACK_PATH):
        return pd.DataFrame(columns=cols)
    try:
        return pd.read_csv(FEEDBACK_PATH)
    except Exception:
        return pd.DataFrame(columns=cols)


# ============================================================
# EXPLORE TAB
# ============================================================

def community_terms(df, community, n=15):
    """Most characteristic words in a community's answers (mean TF-IDF)."""
    sub = df if community == "All" else df[df["subreddit"] == community]
    texts = sub["body"].astype(str).tolist()
    if len(texts) < 2:
        return pd.DataFrame(columns=["term", "weight"])
    try:
        vec = TfidfVectorizer(
            stop_words=list(STOP), max_features=5000,
            min_df=2 if len(texts) > 20 else 1, max_df=0.6,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
        )
        matrix = vec.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["term", "weight"])
    weights = matrix.mean(axis=0).A1
    top = weights.argsort()[::-1][:n]
    terms = vec.get_feature_names_out()
    return pd.DataFrame({"term": [terms[i] for i in top],
                         "weight": [float(weights[i]) for i in top]})


def length_distribution(df):
    """How many answers fall in each word-count bucket."""
    words = df["body"].astype(str).str.split().str.len()
    bins = [0, 25, 50, 100, 200, 400, 10 ** 7]
    labels = ["<25", "25-50", "50-100", "100-200", "200-400", "400+"]
    cut = pd.cut(words, bins=bins, labels=labels, right=False)
    return cut.value_counts().reindex(labels).fillna(0).astype(int).rename("answers")
