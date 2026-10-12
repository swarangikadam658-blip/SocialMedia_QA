import html
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    text = str(text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&#039;", "'")
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PREPARE SEARCH QUERY
# ============================================================

def prepare_search_query(question):
    question = str(question).strip()
    lower_question = question.lower()

    cleaned = re.sub(
        r"\b(what|why|when|where|who|which|how|"
        r"is|are|was|were|do|does|did|can|could|"
        r"would|will|should|the|a|an)\b",
        " ",
        lower_question
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Add context for astronomy questions.
    astronomy_terms = [
        "star", "stars", "planet", "planets",
        "galaxy", "galaxies", "black hole",
        "black holes", "universe", "nebula"
    ]

    if any(term in cleaned.split() for term in astronomy_terms):
        cleaned += " astronomy"

    # NOTE: we no longer append words like "definition explanation" here.
    # Wikipedia's search is full-text, so extra words pulled in unrelated
    # pages (e.g. "cat food definition" returned the Cheshire Cat article).
    return cleaned.strip()

# ============================================================
# RANK RESULTS
# ============================================================

def rank_results(question, results):
    if not results:
        return results

    documents = [question]

    for result in results:
        documents.append(
            result["title"] + " " + result["answer"]
        )

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(documents)

        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        ).flatten()

    except ValueError:
        return results

    for i in range(len(results)):
        results[i]["relevance"] = float(scores[i])

    results.sort(
        key=lambda item: item["relevance"],
        reverse=True
    )

    return results


# ============================================================
# SEARCH WIKIPEDIA
# ============================================================

def search_wikipedia(question):
    search_url = (
        "https://en.wikipedia.org/w/rest.php/v1/search/page"
    )

    search_query = prepare_search_query(question)

    params = {
        "q": search_query,
        "limit": 10
    }

    headers = {
        "User-Agent": "SocialMedia-QA-Student-Project/1.0"
    }

    try:
        response = requests.get(
            search_url,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()
        pages = data.get("pages", [])

        if not pages:
            return None

        # Fetch the article summaries in parallel (was one-by-one: ~10 requests
        # in a row made every Wikipedia answer take 10+ seconds).
        pages = [p for p in pages if p.get("title")][:6]

        def fetch_summary(page):
            title = page["title"]
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(title.replace(" ", "_"))
            )
            try:
                summary_response = requests.get(
                    summary_url,
                    headers=headers,
                    timeout=10
                )
                return summary_response.json() if summary_response.ok else {}
            except (requests.RequestException, ValueError):
                return {}

        with ThreadPoolExecutor(max_workers=6) as pool:
            summaries = list(pool.map(fetch_summary, pages))

        results = []

        for page, summary in zip(pages, summaries):
            title = page["title"]

            answer = summary.get("extract", "")

            if not answer:
                answer = page.get("excerpt", "")

            answer = clean_text(answer)

            if len(answer.split()) < 8:
                continue

            article_url = (
                "https://en.wikipedia.org/wiki/"
                + quote(title.replace(" ", "_"))
            )

            results.append({
                "title": title,
                "answer": answer,
                "url": article_url
            })

        if not results:
            return None

        results = rank_results(
            question,
            results
        )

        return results

    except requests.RequestException as e:
        print("Wikipedia search error:", e)
        return None

    except (ValueError, KeyError, TypeError) as e:
        print("Error processing Wikipedia results:", e)
        return None
