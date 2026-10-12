import csv
import os
import random
import time
from datetime import datetime

import streamlit as st

from src import qa_system, ui

st.set_page_config(page_title="Forum Answers", page_icon="💬", layout="wide")

FEEDBACK_PATH = "data/feedback.csv"

EXAMPLES = [
    "Why did Japan attack Pearl Harbor?",
    "How do vaccines work?",
    "What is the best way to learn programming?",
    "What are stars?",
]

st.session_state.setdefault("history", [])
st.session_state.setdefault("seed", 0)

st.markdown(ui.CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# Callbacks and small helpers
# ------------------------------------------------------------

def save_feedback(question, source, vote):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    is_new = not os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["time", "question", "source", "vote"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         qa_system._csv_safe(question), source, vote])
    st.session_state["voted"] = question


def feedback_row(question, source):
    if st.session_state.get("voted") == question:
        st.caption("Thanks for the feedback! 🙌 It now counts in the Insights tab.")
        return
    c0, c1, c2, _ = st.columns([2.4, 0.7, 0.7, 6])
    c0.markdown('<div class="hint" style="padding-top:.55rem">Was this helpful?</div>',
                unsafe_allow_html=True)
    c1.button("👍", key="fb_up", on_click=save_feedback, args=(question, source, "up"))
    c2.button("👎", key="fb_down", on_click=save_feedback, args=(question, source, "down"))


def set_example(text):
    st.session_state["question"] = text


def surprise_me():
    pool = question_pool()
    comm = st.session_state.get("subreddit_filter", "All")
    if comm != "All":
        pool = [p for p in pool if p[1] == comm] or pool
    if pool:
        st.session_state["question"] = random.choice(pool)[0]


def new_sample():
    st.session_state["seed"] += 1


def log_once(question, community, source, trust, seconds):
    key = (question, community)
    if st.session_state.get("last_logged") == key:
        return
    st.session_state["last_logged"] = key
    try:
        qa_system.log_query(question, community, source,
                            trust["score"], trust["label"], seconds)
    except OSError:
        pass  # never break the app because the log could not be written


# ------------------------------------------------------------
# Engine (loaded once per server process, not on every click)
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Loading search engine (the first run builds the index)...")
def load_engine():
    from src import retrieval, web_fallback
    return retrieval, web_fallback


@st.cache_data(show_spinner=False, ttl=3600)
def run_search(question, subreddit, upvote_weight, use_reranker):
    retrieval, _ = load_engine()
    return retrieval.search_questions(
        question,
        subreddit=subreddit,
        upvote_weight=upvote_weight,
        use_reranker=use_reranker,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def run_wikipedia(question):
    _, web = load_engine()
    try:
        return web.search_wikipedia(question) or [], None
    except Exception as exc:  # network down, API change, etc.
        return [], str(exc)


@st.cache_data(show_spinner=False)
def explore_terms(community):
    retrieval, _ = load_engine()
    return qa_system.community_terms(retrieval.df, community)


@st.cache_data(show_spinner=False)
def question_pool():
    """(title, community) for every dataset thread that is phrased as a question."""
    retrieval, _ = load_engine()
    d = retrieval.df
    q = d[d["title_clean"].str.contains(r"\?", regex=True)]
    q = q if len(q) else d
    return list(zip(q["title_clean"], q["subreddit"]))


retrieval, _web = load_engine()

MAX_UPVOTES = int(retrieval.df["_upvotes"].max())
HAS_UPVOTES = retrieval.SCORE_COL is not None and MAX_UPVOTES > 0
HAS_RERANKER = retrieval.reranker is not None


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### 💬 Forum Answers")
    subreddit = st.selectbox("Community", retrieval.get_subreddits(),
                             key="subreddit_filter")

    with st.expander("Advanced settings"):
        if HAS_UPVOTES:
            upvote_weight = st.slider(
                "Upvote influence",
                0.0, 0.6, float(retrieval.DEFAULT_UPVOTE_WEIGHT), 0.05,
                help="0 = rank by relevance only. Higher = prefer answers the "
                     "community voted up, among relevant results.",
            )
        else:
            upvote_weight = 0.0
            st.caption("ℹ️ The dataset has no upvote counts, so upvote ranking "
                       "and the 'community support' signal are off.")

        if HAS_RERANKER:
            use_reranker = st.checkbox("Cross-encoder reranking", value=True,
                                       help="Slower but more accurate ordering.")
        else:
            use_reranker = False
            st.caption("Reranker not loaded (lightweight mode).")


# ------------------------------------------------------------
# Hero + search
# ------------------------------------------------------------

st.markdown(
    ui.hero([
        (f"{len(retrieval.df):,}", "answers"),
        (str(retrieval.df["subreddit"].nunique()), "communities"),
        ("Reddit + Wikipedia", "cross-checked"),
    ]),
    unsafe_allow_html=True,
)

ask_col, surprise_col = st.columns([6, 1.3])
question = ask_col.text_input(
    "Ask a question",
    key="question",
    placeholder="Ask anything, e.g. Why did Japan attack Pearl Harbor?",
    label_visibility="collapsed",
).strip()
surprise_col.button("🎲 Surprise me", on_click=surprise_me, use_container_width=True)

example_cols = st.columns(len(EXAMPLES))
for col, example in zip(example_cols, EXAMPLES):
    col.button(example, key=f"ex_{example}", on_click=set_example,
               args=(example,), use_container_width=True)

if question and (not st.session_state["history"]
                 or st.session_state["history"][-1] != question):
    st.session_state["history"].append(question)

with st.sidebar:
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("**🕘 Recent questions**")
        for i, past in enumerate(reversed(st.session_state["history"][-6:])):
            st.button(past[:40] + ("…" if len(past) > 40 else ""),
                      key=f"hist_{i}", on_click=set_example, args=(past,),
                      use_container_width=True)


tab_ask, tab_explore, tab_lab, tab_insights, tab_about = st.tabs(
    ["💬 Ask", "🔎 Explore", "🧪 Method lab", "📊 Insights", "📘 How it works"]
)


# ============================================================
# TAB 1 - ASK
# ============================================================

with tab_ask:
    if not question:
        st.markdown(ui.section("How it works"), unsafe_allow_html=True)
        st.markdown(ui.steps(), unsafe_allow_html=True)

        head, shuffle = st.columns([5, 1.3])
        head.markdown(ui.section("From the archive - tap one to ask it"),
                      unsafe_allow_html=True)
        shuffle.button("🔀 Shuffle", on_click=new_sample, key="shuffle_archive",
                       use_container_width=True)
        pool = question_pool()
        picks = random.Random(st.session_state["seed"]).sample(pool, min(3, len(pool)))
        for i, (title, comm) in enumerate(picks):
            st.button(f"r/{comm} · {title[:95]}", key=f"arch_{i}",
                      on_click=set_example, args=(title,), use_container_width=True)
    else:
        started = time.perf_counter()
        query_words = qa_system.keywords(question)

        with st.spinner("Searching forum discussions..."):
            result = run_search(question, subreddit, upvote_weight, use_reranker)

        with st.spinner("Cross-checking with Wikipedia..."):
            web_results, web_error = run_wikipedia(question)
        top_wiki = web_results[0] if web_results else None

        # ====================================================
        # CASE 1: Reddit answer (confident match, or closest related thread)
        # ====================================================
        if result["matched"] or result.get("loose"):
            best = result["best_match"]
            answer = result["answer"] or best["body"]
            sources = result["sources"] or [{
                "n": 1,
                "title": best["question_title"],
                "subreddit": best["subreddit"],
                "upvotes": int(best["_upvotes"]),
                "link": None,
                "snippet": best["body"],
            }]
            plain_answer = qa_system.strip_citations(answer)

            wiki_sim = (qa_system.text_similarity(plain_answer, top_wiki["answer"])
                        if top_wiki else None)
            trust = qa_system.compute_trust(
                result["best_relevance"],
                [s["snippet"] for s in sources],
                [s["upvotes"] for s in sources] if HAS_UPVOTES else [],
                MAX_UPVOTES if HAS_UPVOTES else 0,
                wiki_sim,
            )

            if not result["matched"]:
                st.markdown(
                    ui.notice("Closest Reddit discussion. It is related to your question "
                              "but not an exact match, so check the trust score and "
                              "the Wikipedia comparison below."),
                    unsafe_allow_html=True)

            key_view = st.checkbox("Key points view", key="kp",
                                   help="One bullet per Reddit source.")
            body = (ui.points_html(answer, query_words) if key_view
                    else qa_system.highlight_html(answer, query_words))
            st.markdown(
                ui.answer_card("reddit", trust["label"], body,
                               "highlighted words match your question"),
                unsafe_allow_html=True)
            st.markdown(ui.verdict(trust), unsafe_allow_html=True)

            seconds = time.perf_counter() - started
            log_once(question, subreddit, "reddit", trust, seconds)
            feedback_row(question, "reddit")
            st.caption(f"Answered in {seconds:.1f}s · search method: {result['method']}")

            t_cmp, t_src, t_rel, t_how = st.tabs(
                ["⚖️ Compare with Wikipedia", "📚 Sources", "🧵 Related threads",
                 "🔬 How it was ranked"])

            with t_cmp:
                if top_wiki:
                    cmp = qa_system.compare_sources(plain_answer, top_wiki["answer"])
                    st.markdown(ui.duo(plain_answer, top_wiki) + ui.agreement(cmp),
                                unsafe_allow_html=True)
                elif web_error:
                    st.warning(f"Couldn't reach Wikipedia ({web_error}), so the "
                               "cross-check is skipped.")
                else:
                    st.info("Wikipedia had no matching article to compare against.")

            with t_src:
                st.markdown(ui.source_cards(sources, query_words), unsafe_allow_html=True)

            with t_rel:
                for i, (_, row) in enumerate(result["related"].iterrows(), start=1):
                    with st.expander(f"{i}. {row['title_clean']}"):
                        up = int(row["_upvotes"])
                        st.caption(
                            f"r/{row['subreddit']} · "
                            f"{'⬆ ' + format(up, ',') + ' upvotes' if up > 0 else 'upvotes n/a'} · "
                            f"relevance {row['relevance']:.2f}")
                        st.text(retrieval.clean_answer(row["body"])[:400] + "...")

            with t_how:
                st.write(
                    f"**Semantic similarity:** {result['best_similarity']:.3f}  \n"
                    f"**Relevance score:** {result['best_relevance']:.3f}  \n"
                    f"**Upvote influence:** {upvote_weight:.2f}  \n"
                    f"**Method:** {result['method']}")
                st.code(
                    "final = (1 - w) * relevance + w * log(1 + upvotes) / log(1 + max_upvotes)",
                    language="text")

        # ====================================================
        # CASE 2: nothing related in the forum at all -> Wikipedia fallback
        # ====================================================
        else:
            st.markdown(
                ui.notice("Nothing in the Reddit archive is related to this question "
                          f"(best similarity {result['best_similarity']:.2f}), "
                          "so Wikipedia is used as the fallback."),
                unsafe_allow_html=True)

            if top_wiki:
                trust = qa_system.wiki_only_trust(top_wiki.get("relevance", 0.0))
                st.markdown(
                    ui.answer_card("wiki", trust["label"],
                                   qa_system.highlight_html(top_wiki["answer"], query_words),
                                   "encyclopedia summary", title=top_wiki["title"]),
                    unsafe_allow_html=True)
                st.markdown(ui.verdict(trust), unsafe_allow_html=True)

                link = ui.safe_link(top_wiki.get("url"))
                if link:
                    st.markdown(f"[Read the full article ↗]({link})")

                seconds = time.perf_counter() - started
                log_once(question, subreddit, "wikipedia", trust, seconds)
                feedback_row(question, "wikipedia")
                st.caption(f"Answered in {seconds:.1f}s")

                t_other, t_forum = st.tabs(["📄 Other articles", "🧵 Closest forum threads"])
                with t_other:
                    others = web_results[1:]
                    if not others:
                        st.caption("No other articles found.")
                    for item in others:
                        item_link = ui.safe_link(item.get("url"))
                        st.markdown(f"- [{item['title']}]({item_link})" if item_link
                                    else f"- {item['title']}")
                with t_forum:
                    st.caption("These forum threads were below the match threshold.")
                    for _, row in result["related"].iterrows():
                        st.markdown(f"- **{row['title_clean']}** · r/{row['subreddit']} · "
                                    f"relevance {row['relevance']:.2f}")
            else:
                seconds = time.perf_counter() - started
                log_once(question, subreddit, "none",
                         {"score": 0, "label": "low"}, seconds)
                if web_error:
                    st.error(f"Couldn't reach Wikipedia right now ({web_error}). Please try again.")
                else:
                    st.error("Sorry, I couldn't find relevant information for this question.")


# ============================================================
# TAB 2 - EXPLORE
# ============================================================

with tab_explore:
    counts = retrieval.df["subreddit"].value_counts()
    with_votes = float((retrieval.df["_upvotes"] > 0).mean())

    st.markdown(ui.section("What is inside the dataset?"), unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Answers", f"{len(retrieval.df):,}")
    m2.metric("Communities", f"{len(counts)}")
    m3.metric("Answers with upvote data", f"{with_votes:.0%}")

    g1, g2 = st.columns(2)
    g1.markdown("**Answers per community**")
    g1.bar_chart(counts)
    g2.markdown("**How long are the answers? (words)**")
    g2.bar_chart(qa_system.length_distribution(retrieval.df))

    st.markdown(ui.section("Look inside a community"), unsafe_allow_html=True)
    comm = st.selectbox("Pick a community", list(counts.index), key="explore_comm")

    terms = explore_terms(comm)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Words that define this community**")
        if terms.empty:
            st.caption("Not enough answers to analyse.")
        else:
            st.bar_chart(terms.set_index("term")["weight"])
    with t2:
        st.markdown("**Random threads**")
        st.button("🎲 Show different threads", on_click=new_sample, key="new_sample")
        sub = retrieval.df[retrieval.df["subreddit"] == comm]
        sample = sub.sample(min(3, len(sub)), random_state=st.session_state["seed"])
        for _, row in sample.iterrows():
            with st.expander(row["title_clean"][:90]):
                st.write(retrieval.clean_answer(row["body"])[:500] + "…")
                st.button("Ask this question", key=f"ask_{row.name}",
                          on_click=set_example, args=(row["title_clean"],))


# ============================================================
# TAB 3 - METHOD LAB
# ============================================================

with tab_lab:
    st.markdown(ui.section("Same question, four retrieval methods"), unsafe_allow_html=True)
    st.caption("Each method ranks the forum threads differently. See what each one "
               "finds, and why the app blends them.")

    lab_q = st.text_input("Question to test", value=question or "How do vaccines work?",
                          key="lab_q").strip()

    METHODS = [
        ("tfidf", "TF-IDF", "Counts shared words, weighted by rarity."),
        ("bm25", "BM25", "Like TF-IDF, but handles answer length better."),
        ("embedding", "Embeddings", "Compares meaning, not just words."),
        ("hybrid", "Hybrid (used here)", "A weighted blend of the three."),
    ]

    if lab_q:
        rankings = {}
        for key, _, _ in METHODS:
            try:
                rankings[key] = retrieval.rank_by_method(lab_q, key, k=3)
            except Exception:
                rankings[key] = None

        hybrid_top = set(rankings.get("hybrid") or [])
        for col, (key, name, desc) in zip(st.columns(len(METHODS)), METHODS):
            with col:
                st.markdown(f"**{name}**")
                st.caption(desc)
                ranked = rankings.get(key)
                if not ranked:
                    st.info("Not available in this setup.")
                    continue
                cards = "".join(
                    '<div class="card"><div class="src-title">'
                    f'{ui.esc(retrieval.df.iloc[pos]["title_clean"][:90])}</div>'
                    f'<div class="meta">r/{ui.esc(retrieval.df.iloc[pos]["subreddit"])}</div></div>'
                    for pos in ranked)
                st.markdown(cards, unsafe_allow_html=True)
                if key != "hybrid" and hybrid_top:
                    st.caption(f"Overlap with hybrid top-3: {len(set(ranked) & hybrid_top)}/3")


# ============================================================
# TAB 4 - INSIGHTS
# ============================================================

with tab_insights:
    log = qa_system.read_log()
    fb = qa_system.read_feedback()

    st.markdown(ui.section("How the system is being used"), unsafe_allow_html=True)
    if log.empty:
        st.info("No questions logged yet. Ask a few questions and come back.")
    else:
        forum_rate = float((log["source"] == "reddit").mean())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Questions asked", f"{len(log):,}")
        k2.metric("Answered from Reddit", f"{forum_rate:.0%}")
        k3.metric("Average trust", f"{log['trust'].mean():.0f} / 100")
        k4.metric("Average speed", f"{log['seconds'].mean():.1f}s")

        i1, i2 = st.columns(2)
        i1.markdown("**Where answers came from**")
        i1.bar_chart(log["source"].value_counts())
        i2.markdown("**Trust level of answers**")
        i2.bar_chart(log["trust_label"].value_counts())

        st.markdown("**Trust score over time**")
        st.line_chart(log["trust"].reset_index(drop=True))

    st.markdown(ui.section("User feedback"), unsafe_allow_html=True)
    if fb.empty:
        st.info("No feedback yet. Use 👍 / 👎 under an answer.")
    else:
        ups = int((fb["vote"] == "up").sum())
        downs = int((fb["vote"] == "down").sum())
        f1, f2 = st.columns(2)
        f1.metric("Helpful rate", f"{ups / max(1, ups + downs):.0%}",
                  help=f"{ups} thumbs-up, {downs} thumbs-down")
        f2.markdown("**Votes by source**")
        f2.bar_chart(fb.groupby(["source", "vote"]).size().unstack(fill_value=0))
        with st.expander("Questions people rated 👎"):
            bad = fb[fb["vote"] == "down"]["question"].tolist()
            st.write(bad if bad else "None so far.")

    if not log.empty:
        with st.expander("Recent questions"):
            st.dataframe(log.tail(15).iloc[::-1], use_container_width=True)


# ============================================================
# TAB 5 - HOW IT WORKS
# ============================================================

with tab_about:
    st.markdown(ui.section("How Forum Answers works"), unsafe_allow_html=True)
    st.markdown(
        """
**Goal.** Give people the answer a knowledgeable online community would give, and be
honest about how far that answer can be trusted.

**Pipeline**
1. **Clean the data.** Bot messages, removed posts and very short answers are dropped.
2. **Retrieve.** The question is matched against forum thread titles using TF-IDF, BM25 and
   sentence embeddings, blended into one hybrid score (see the *Method lab* tab).
3. **Rerank.** A cross-encoder (when available) re-scores the best candidates, together with
   keyword overlap and question type (why / how / what ...).
4. **Extract.** The most relevant sentences from the top threads are combined into one answer,
   and every sentence carries a source number.
5. **Cross-check.** Wikipedia is searched for the same question and compared with the answer.
6. **Score trust.** The trust score combines four signals (below).
7. **Learn.** Every question and 👍 / 👎 vote is logged and summarised in the *Insights* tab.
"""
    )
    st.markdown("**Trust score**")
    st.code(
        "trust = 40% question match\n"
        "      + 20% sources agree with each other\n"
        "      + 20% community support (upvotes, log scale)\n"
        "      + 20% Wikipedia backs it up\n"
        "(a signal that is unavailable is dropped and the others are re-weighted)",
        language="text",
    )
    st.markdown("**Ranking formula**")
    st.code("final = (1 - w) * relevance + w * log(1 + upvotes) / log(1 + max_upvotes)",
            language="text")
    st.markdown(
        """
**Limitations (read these before relying on an answer)**
- Answers are *extracted* from Reddit, not generated, so they can be informal, partial or wrong.
- Reddit answers are opinions of volunteers, not verified facts. Wikipedia is a better check
  for facts, which is why the app compares the two.
- The trust score measures *agreement and relevance*, not truth. A confidently repeated
  mistake can still score well.
- Similarity is computed with TF-IDF word overlap, so it can miss paraphrases.
- If the dataset has no upvote counts, upvote ranking is switched off automatically.
"""
    )
