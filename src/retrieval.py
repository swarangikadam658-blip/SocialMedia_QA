import os
import re
import html
import math

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception as _exc:  # ImportError, or a broken torch/transformers install
    print(f"sentence-transformers unavailable ({type(_exc).__name__}: {_exc}). "
          "Falling back to TF-IDF/BM25. See the README note on torch versions.")
    SentenceTransformer = None
    CrossEncoder = None


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = os.environ.get("QA_DATA_PATH", "data/processed/reddit_qa_nlp.csv")
EMBED_CACHE = os.environ.get(
    "QA_EMBED_CACHE", "data/processed/question_embeddings.npy"
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

USE_EMBEDDINGS = True      # set False to force the lightweight TF-IDF/BM25 mode
USE_RERANKER = True        # cross-encoder rerank of the candidate pool

CANDIDATE_POOL = 30        # how many candidates go into reranking
DEFAULT_UPVOTE_WEIGHT = 0.25

# hybrid weights (re-normalized over whichever signals are available)
HYBRID_WEIGHTS = {"embedding": 0.50, "bm25": 0.25, "tfidf": 0.25}

# thresholds: (min semantic cosine, min relevance)
THRESHOLDS_EMBED = {"semantic": 0.55, "relevance": 0.50}
# TF-IDF cosine on short titles is naturally lower than embedding cosine,
# so the lightweight-mode semantic bar is lower (was 0.50).
THRESHOLDS_TFIDF = {"semantic": 0.40, "relevance": 0.55}

# "Reddit-first": below the strict thresholds above we still return the closest
# Reddit thread (flagged as `loose`, low trust) instead of sending the user to
# Wikipedia, as long as it shares at least one content word with the question.
LOOSE_THRESHOLDS = {"semantic": 0.15, "relevance": 0.20}

SCORE_COLUMNS = ["score", "upvotes", "ups", "answer_score",
                 "comment_score", "num_upvotes"]
LINK_COLUMNS = ["permalink", "url", "link", "answer_url"]


# ============================================================
# NLP SETUP
# ============================================================

def _ensure_nltk():
    needed = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for name, path in needed.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


_ensure_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ============================================================
# TEXT HELPERS
# ============================================================

def preprocess_query(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stop_words
    ]
    return " ".join(tokens)


def get_content_words(text):
    processed = preprocess_query(text)
    return set(w for w in processed.split() if len(w) > 2)


def clean_answer(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# QUESTION INTENT
# ============================================================

_WHY = [r"^why\b", r"^how come\b", r"^what caused\b", r"^what causes\b",
        r"^what was the reason\b", r"^what were the reasons\b",
        r"^what is the reason\b", r"^what are the reasons\b",
        r"^what led to\b", r"^what led\b", r"^for what reason\b"]
_WHEN = [r"^when\b", r"^what year\b", r"^what date\b", r"^what time\b"]


def get_question_intent(text):
    text = str(text).lower().strip()
    if any(re.search(p, text) for p in _WHY):
        return "why"
    if any(re.search(p, text) for p in _WHEN):
        return "when"
    for word in ("where", "who", "how", "what", "which"):
        if re.match(rf"^{word}\b", text):
            return word
    if re.match(r"^(is|are|can|could|does|do|did|will|would|should)\b", text):
        return "yes_no"
    return "general"


# ============================================================
# ANSWER QUALITY
# ============================================================

_BAD_PATTERNS = [
    "your submission has been removed", "your post has been removed",
    "your comment has been removed", "i am a bot", "i'm a bot",
    "this action was performed automatically", "please read our rules",
    "read our rules", "contact the moderators", "contact the mods",
    "submission rules", "post has been removed",
    "removed because it violates", "violates our submission rules",
    "[deleted]", "[removed]",
]


MIN_ANSWER_WORDS = 10      # answers shorter than this are dropped


def _bad_reason(text):
    """None if the answer is fine, otherwise why it was rejected."""
    text = str(text).lower()
    if any(p in text for p in _BAD_PATTERNS):
        return "bot/removed"
    if len(text.split()) < MIN_ANSWER_WORDS:
        return "too short"
    return None


def is_good_answer(text):
    return _bad_reason(text) is None


def has_contradictory_negation(user_question, candidate_question):
    query = str(user_question).lower()
    candidate = str(candidate_question).lower()

    positive = ["like", "love", "enjoy", "prefer", "want"]
    negative = ["don't like", "do not like", "doesn't like", "does not like",
                "dont like", "didn't like", "did not like", "hate", "hates",
                "dislike", "dislikes"]

    if not any(p in query for p in positive):
        return False
    return any(n in candidate for n in negative)


# ============================================================
# SENTENCE-LEVEL EXTRACTION
# ============================================================

def extract_best_answer(question, text, max_sentences=3):
    text = clean_answer(text)
    sentences = [
        s.strip() for s in sent_tokenize(text)
        if len(s.strip().split()) >= 5
    ]
    if not sentences:
        return text

    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform([question] + sentences)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError:
        return text

    top = sorted(scores.argsort()[::-1][:max_sentences])
    return " ".join(sentences[i] for i in top)


def _jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ============================================================
# LOAD + PREPARE DATASET
# ============================================================

def _first_existing(columns, options):
    for col in options:
        if col in columns:
            return col
    return None


df = pd.read_csv(DATA_PATH)

if "question_title" not in df.columns or "body" not in df.columns:
    raise ValueError("Dataset needs 'question_title' and 'body' columns.")

df["question_title"] = df["question_title"].fillna("").astype(str)
df["body"] = df["body"].fillna("").astype(str)

# drop bot / removed / too-short answers once, up front
_before = len(df)
_reasons = df["body"].apply(_bad_reason)
_counts = _reasons.value_counts()
df = df[_reasons.isna()].reset_index(drop=True)
print(f"Dataset loaded: {len(df)} records ({_before - len(df)} of {_before} answers removed)")
for _why, _n in _counts.items():
    print(f"   removed - {_why}: {_n}")

df["title_clean"] = df["question_title"].apply(clean_answer)

if "question_nlp" not in df.columns:
    df["question_nlp"] = df["title_clean"].apply(preprocess_query)
df["question_nlp"] = df["question_nlp"].fillna("").astype(str)

if "subreddit" not in df.columns:
    df["subreddit"] = "unknown"

SCORE_COL = _first_existing(df.columns, SCORE_COLUMNS)
LINK_COL = _first_existing(df.columns, LINK_COLUMNS)

if SCORE_COL:
    df["_upvotes"] = pd.to_numeric(df[SCORE_COL], errors="coerce").fillna(0).clip(lower=0)
    _max_log = math.log1p(df["_upvotes"].max()) or 1.0
    df["_upvote_norm"] = np.log1p(df["_upvotes"]) / _max_log
    print(f"Upvote column detected: '{SCORE_COL}' | max {int(df['_upvotes'].max())}, "
          f"median {df['_upvotes'].median():.0f}, "
          f"{(df['_upvotes'] == 0).mean():.0%} of answers have 0 upvotes")
else:
    df["_upvotes"] = 0
    df["_upvote_norm"] = 0.0
    print("No upvote column found - upvote-aware ranking disabled.")


# ============================================================
# INDEXES
# ============================================================

# --- TF-IDF -------------------------------------------------
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["question_nlp"])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# --- BM25 ---------------------------------------------------
bm25 = None
if BM25Okapi is not None:
    bm25 = BM25Okapi([t.split() for t in df["question_nlp"]])
    print("BM25 index ready")
else:
    print("rank-bm25 not installed - BM25 disabled (pip install rank-bm25)")

# --- Sentence embeddings -----------------------------------
embedder = None
embeddings = None

if USE_EMBEDDINGS and SentenceTransformer is not None:
    try:
        embedder = SentenceTransformer(EMBED_MODEL)
        cache_ok = False
        if os.path.exists(EMBED_CACHE):
            cached = np.load(EMBED_CACHE)
            cache_ok = cached.shape[0] == len(df)
            if cache_ok:
                embeddings = cached
        if not cache_ok:
            print("Encoding questions (first run only, then cached)...")
            embeddings = embedder.encode(
                df["title_clean"].tolist(),
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
            os.makedirs(os.path.dirname(EMBED_CACHE) or ".", exist_ok=True)
            np.save(EMBED_CACHE, embeddings)
        print("Embeddings ready:", embeddings.shape)
    except Exception as exc:
        print("Embedding model unavailable, using TF-IDF/BM25 only:", exc)
        embedder, embeddings = None, None
else:
    print("sentence-transformers not installed - embeddings disabled "
          "(pip install sentence-transformers)")

# --- Cross-encoder reranker --------------------------------
reranker = None
if USE_RERANKER and CrossEncoder is not None and embeddings is not None:
    try:
        reranker = CrossEncoder(RERANK_MODEL)
        print("Cross-encoder reranker ready")
    except Exception as exc:
        print("Reranker unavailable:", exc)

HAS_EMBEDDINGS = embeddings is not None
THRESHOLDS = THRESHOLDS_EMBED if HAS_EMBEDDINGS else THRESHOLDS_TFIDF


# ============================================================
# SCORING PRIMITIVES (each returns an array over all rows)
# ============================================================

def _minmax(x):
    x = np.asarray(x, dtype="float64")
    span = x.max() - x.min()
    if span < 1e-12:
        return np.zeros_like(x)
    return (x - x.min()) / span


def _scores_tfidf(processed_query):
    vec = vectorizer.transform([processed_query])
    return cosine_similarity(vec, tfidf_matrix).flatten()


def _scores_bm25(processed_query):
    if bm25 is None:
        return None
    return np.asarray(bm25.get_scores(processed_query.split()))


def _scores_embedding(raw_query):
    if embeddings is None:
        return None
    q = embedder.encode([clean_answer(raw_query)], normalize_embeddings=True)[0]
    return embeddings @ q.astype("float32")


def _hybrid_scores(raw_query, processed_query):
    """Weighted blend of min-max normalized signals + the raw semantic cosine."""
    parts = {
        "tfidf": _scores_tfidf(processed_query),
        "bm25": _scores_bm25(processed_query),
        "embedding": _scores_embedding(raw_query),
    }
    available = {k: v for k, v in parts.items() if v is not None}
    total_w = sum(HYBRID_WEIGHTS[k] for k in available)
    blended = sum(
        (HYBRID_WEIGHTS[k] / total_w) * _minmax(v) for k, v in available.items()
    )
    semantic = parts["embedding"] if parts["embedding"] is not None else parts["tfidf"]
    return blended, semantic, parts


def rank_by_method(query, method="hybrid", k=10):
    """Return the top-k row positions for one retrieval method (used by evaluate.py)."""
    processed = preprocess_query(query)
    if method == "tfidf":
        scores = _scores_tfidf(processed)
    elif method == "bm25":
        scores = _scores_bm25(processed)
    elif method == "embedding":
        scores = _scores_embedding(query)
    else:
        scores, _, _ = _hybrid_scores(query, processed)
    if scores is None:
        return None
    return list(np.argsort(scores)[::-1][:k])


# ============================================================
# ANSWER SYNTHESIS (extractive, with citations)
# ============================================================

def synthesize_answer(question, ranked, max_sources=3, sentences_per_source=2):
    """
    Combine the best sentences from the top answers into one short answer.
    Every sentence is tagged with the source number, e.g. 'text [1]'.
    """
    pieces, sources, used = [], [], []

    for _, row in ranked.iterrows():
        if len(sources) >= max_sources:
            break

        snippet = extract_best_answer(question, row["body"],
                                      max_sentences=sentences_per_source)
        if not snippet or any(_jaccard(snippet, u) > 0.6 for u in used):
            continue  # skip near-duplicate answers

        n = len(sources) + 1
        used.append(snippet)
        pieces.append(f"{snippet} [{n}]")
        sources.append({
            "n": n,
            "title": row["title_clean"],
            "subreddit": row["subreddit"],
            "upvotes": int(row["_upvotes"]),
            "link": row[LINK_COL] if LINK_COL else None,
            "snippet": snippet,
            "relevance": round(float(row["relevance"]), 3),
        })

    return " ".join(pieces), sources


def _confidence_label(relevance, matched):
    if not matched:
        return "low"
    if relevance >= 0.75:
        return "high"
    return "medium"


# ============================================================
# MAIN SEARCH
# ============================================================

def search_questions(
    user_question,
    top_n=3,
    threshold=None,
    upvote_weight=DEFAULT_UPVOTE_WEIGHT,
    subreddit=None,
    use_reranker=True,
):
    """
    Returns a dict:
      matched (confident match), loose (weaker but related Reddit match),
      confidence, answer (synthesized, with [n] citations),
      sources (list of dicts), best_match (Series), related (DataFrame),
      best_similarity, best_relevance, method
    app.py can keep using: matched / best_match / related / best_similarity /
    best_relevance exactly as before.
    """
    if not SCORE_COL:
        upvote_weight = 0.0

    min_semantic = threshold if threshold is not None else THRESHOLDS["semantic"]
    min_relevance = THRESHOLDS["relevance"]

    processed = preprocess_query(user_question)
    intent = get_question_intent(user_question)
    query_words = get_content_words(user_question)

    blended, semantic, parts = _hybrid_scores(user_question, processed)

    if subreddit and subreddit != "All":
        mask = (df["subreddit"].str.lower() == subreddit.lower()).to_numpy()
        blended = np.where(mask, blended, -1.0)

    pool = min(CANDIDATE_POOL, len(df))
    top_idx = np.argsort(blended)[::-1][:pool]

    candidates = df.iloc[top_idx].copy()
    candidates["similarity"] = semantic[top_idx]
    candidates["hybrid"] = blended[top_idx]

    # ---- optional cross-encoder rerank -----------------------
    ce_scores = None
    if reranker is not None and use_reranker:
        pairs = [(user_question, t) for t in candidates["title_clean"]]
        raw = np.asarray(reranker.predict(pairs), dtype="float64")
        ce_scores = 1.0 / (1.0 + np.exp(-raw))  # sigmoid -> 0..1

    # ---- relevance ------------------------------------------
    relevance = []
    overlaps = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        title = row["title_clean"]
        cand_words = get_content_words(title)

        overlap = (
            len(query_words & cand_words) / len(query_words)
            if query_words else 0.0
        )
        cos = float(row["similarity"])

        cand_intent = get_question_intent(title)
        intent_score = 1.0 if (intent == "general" or cand_intent == intent) else 0.0

        if ce_scores is not None:
            rel = (0.40 * cos + 0.25 * float(ce_scores[i])
                   + 0.20 * overlap + 0.15 * intent_score)
        else:
            rel = 0.55 * cos + 0.30 * overlap + 0.15 * intent_score

        if has_contradictory_negation(user_question, title):
            rel *= 0.20

        relevance.append(rel)
        overlaps.append(overlap)

    candidates["relevance"] = relevance
    candidates["overlap"] = overlaps

    # ---- upvote-aware final ranking -------------------------
    candidates["final_score"] = (
        (1 - upvote_weight) * candidates["relevance"]
        + upvote_weight * candidates["_upvote_norm"]
    )
    candidates = candidates.sort_values("final_score", ascending=False)

    # related list: one row per distinct question title
    results = (
        candidates.drop_duplicates("title_clean").head(top_n).copy()
    )
    best = candidates.iloc[0].copy()

    best_cos = float(best["similarity"])
    best_rel = float(best["relevance"])

    matched = (
        best_cos >= min_semantic
        and best_rel >= min_relevance
        and not has_contradictory_negation(user_question, best["title_clean"])
    )

    # Reddit-first: not a confident match, but still loosely related to the
    # question -> show the closest Reddit thread instead of jumping to Wikipedia.
    loose = (
        not matched
        and best_rel >= LOOSE_THRESHOLDS["relevance"]
        and best_cos >= LOOSE_THRESHOLDS["semantic"]
        and float(best["overlap"]) > 0
        and not has_contradictory_negation(user_question, best["title_clean"])
    )

    # synthesize from candidates that are themselves relevant enough
    answer, sources = "", []
    if matched:
        usable = candidates[candidates["relevance"] >= min_relevance * 0.9]
        answer, sources = synthesize_answer(user_question, usable)
    elif loose:
        usable = candidates[candidates["relevance"] >= LOOSE_THRESHOLDS["relevance"]]
        answer, sources = synthesize_answer(user_question, usable, max_sources=2)

    best["question_title"] = best["title_clean"]
    best["body"] = extract_best_answer(user_question, best["body"])

    return {
        "matched": matched,
        "loose": loose,
        "confidence": _confidence_label(best_rel, matched),
        "answer": answer,
        "sources": sources,
        "best_match": best if (matched or loose) else None,
        "related": results,
        "best_similarity": best_cos,
        "best_relevance": best_rel,
        "method": (
            "hybrid+rerank" if ce_scores is not None
            else "hybrid" if HAS_EMBEDDINGS
            else "tfidf+bm25" if bm25 is not None
            else "tfidf"
        ),
    }


def get_subreddits():
    """For the UI filter dropdown."""
    return ["All"] + sorted(df["subreddit"].dropna().unique().tolist())


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    q = "Why did people attack Pearl Harbor?"
    res = search_questions(q)

    print("\nQuestion  :", q)
    print("Intent    :", get_question_intent(q))
    print("Method    :", res["method"])
    print("Matched   :", res["matched"], f"(confidence: {res['confidence']})")
    print("Similarity:", round(res["best_similarity"], 3))
    print("Relevance :", round(res["best_relevance"], 3))

    if res["matched"]:
        print("\nANSWER\n------")
        print(res["answer"])
        print("\nSOURCES\n-------")
        for s in res["sources"]:
            print(f"[{s['n']}] r/{s['subreddit']} | {s['upvotes']} upvotes | {s['title']}")
    else:
        print("\nNo sufficiently relevant forum discussion found -> use web fallback.")

    print("\nRELATED\n-------")
    for _, r in res["related"].iterrows():
        print(f"- {r['title_clean']}  (rel {r['relevance']:.2f}, {int(r['_upvotes'])} upvotes)")
