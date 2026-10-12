"""
ui.py - all the HTML + CSS for Forum Answers.

Pure functions that return HTML strings (no Streamlit import), so the look of
the app lives in one place and can be previewed in any browser. Charts are
built from plain HTML/CSS/SVG on purpose: nothing extra to install, they
animate, and they work in light and dark mode.

IMPORTANT: every builder returns ONE line of HTML with no blank lines or
leading spaces, because Streamlit's markdown turns indented / blank-line
separated HTML into a code block.
"""

import html
import re
from urllib.parse import quote_plus

from src import qa_system

BRAND = "#4f46e5"
REDDIT = "#e8590c"
WIKI = "#2563eb"

_CSS = """
:root {--brand:#4f46e5; --reddit:#e8590c; --wiki:#2563eb;
       --line:rgba(128,128,128,.24); --soft:rgba(128,128,128,.07);}
@property --p {syntax:'<integer>'; inherits:true; initial-value:0;}

/* ---------- keyframes ---------- */
@keyframes rise {from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:none;}}
@keyframes flow {0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;}}
@keyframes float {0%,100% {transform:translateY(0) rotate(-4deg);} 50% {transform:translateY(-12px) rotate(4deg);}}
@keyframes drift {0%,100% {transform:translateY(0);} 50% {transform:translateY(-6px);}}
@keyframes fillring {to {--p: var(--target);}}
@keyframes grow {from {transform:scaleX(0);}}
@keyframes popin {0% {opacity:0; transform:scale(.82);} 100% {opacity:1; transform:scale(1);}}
@keyframes fadeword {from {opacity:0; transform:translateY(5px);} to {opacity:1; transform:none;}}
@keyframes draw {from {stroke-dashoffset:1;} to {stroke-dashoffset:0;}}
@keyframes pulse {0%,100% {transform:scale(1);} 50% {transform:scale(1.18);}}
@keyframes sweep {from {opacity:0; transform:rotate(-140deg) scale(.9);} to {opacity:1; transform:none;}}
.rise {animation: rise .5s cubic-bezier(.2,.7,.2,1) both;}
.lift {transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;}
.lift:hover {transform:translateY(-3px); box-shadow:0 14px 28px -18px rgba(0,0,0,.55); border-color:var(--brand);}

#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stToolbar"] {display:none !important;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.2rem; padding-bottom:3rem; max-width:1080px;}

/* ---------- hero ---------- */
.hero {position:relative; overflow:hidden; color:#fff; border-radius:22px; padding:1.7rem 1.9rem 1.5rem;
       margin-bottom:1.1rem; background:linear-gradient(120deg,#4f46e5,#7c3aed,#db2777,#4f46e5);
       background-size:300% 300%; animation:flow 14s ease infinite;
       box-shadow:0 10px 30px -12px rgba(79,70,229,.55);}
.hero::before, .hero::after {content:""; position:absolute; border-radius:50%; background:rgba(255,255,255,.09);
       animation:drift 7s ease-in-out infinite;}
.hero::before {width:230px; height:230px; right:-70px; top:-100px;}
.hero::after {width:150px; height:150px; right:170px; bottom:-80px; animation-delay:-3s;}
.hero > * {position:relative; z-index:1;}
.hero h1 {color:#fff; margin:0 0 .25rem 0; font-size:2.1rem; line-height:1.15; padding:0;}
.hero p {color:#fff; opacity:.92; margin:0 0 1rem 0; font-size:1.02rem; max-width:44rem;}
.hero-emoji {position:absolute !important; right:2.2rem; top:50%; margin-top:-2.6rem; font-size:4.4rem; line-height:1;
       animation:float 5s ease-in-out infinite; filter:drop-shadow(0 10px 14px rgba(0,0,0,.28));}
.chip {display:inline-block; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28);
       color:#fff; border-radius:999px; padding:.18rem .8rem; margin:0 .45rem .3rem 0; font-size:.8rem;}
.chip b {font-weight:700;}

/* ---------- cards ---------- */
.card {border:1px solid var(--line); border-radius:16px; padding:1.05rem 1.25rem; margin:.45rem 0; background:var(--soft);}
.section-title {font-size:1.02rem; font-weight:700; margin:1.5rem 0 .6rem 0;}
.hint {font-size:.8rem; opacity:.7;}
.meta {font-size:.82rem; opacity:.75;}
a {text-decoration:none;}
.card a, .meta a {color:#6366f1 !important; font-weight:600;}
.card a:hover {text-decoration:underline;}

/* ---------- answer ---------- */
.answer {border-left:6px solid var(--accent, #4f46e5); padding:1.25rem 1.5rem;}
.answer-top {display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; margin-bottom:.7rem;}
.answer-body {font-size:1.12rem; line-height:1.72;}
.answer-body ul {margin:0; padding-left:1.2rem;}
.answer-body li {margin-bottom:.55rem;}
.w {display:inline-block; opacity:0; animation:fadeword .45s ease forwards;}
.badge {display:inline-block; padding:.16rem .7rem; border-radius:999px; font-size:.74rem; font-weight:700;
        color:#fff; letter-spacing:.01em;}
.b-reddit {background:#e8590c;} .b-wiki {background:#2563eb;} .b-num {background:#4f46e5;}
.b-high {background:#16a34a;} .b-medium {background:#d97706;} .b-low {background:#dc2626;}
mark {background:rgba(250,204,21,.5); color:inherit; padding:0 .15rem; border-radius:4px;}
.infochips {display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.9rem; padding-top:.8rem; border-top:1px dashed var(--line);}
.ichip {display:inline-flex; align-items:center; gap:.3rem; padding:.16rem .7rem; border-radius:999px;
        border:1px solid var(--line); font-size:.78rem; background:var(--soft);}
.ichip-topic {font-weight:700;}

/* ---------- trust strip ---------- */
.t-high {--c:#16a34a;} .t-medium {--c:#d97706;} .t-low {--c:#dc2626;}
.verdict {display:flex; flex-wrap:wrap; gap:1.5rem; align-items:center; padding:1.1rem 1.4rem;
          border:1px solid var(--line); border-radius:16px; background:var(--soft); margin:.6rem 0;}
.ring-wrap {--p:0; position:relative; width:94px; height:94px; flex:none;
            animation:fillring 1.3s cubic-bezier(.2,.7,.2,1) .25s forwards;}
.ring {position:absolute; inset:0; border-radius:50%;
       background:conic-gradient(var(--c) calc(var(--p) * 1%), rgba(128,128,128,.22) 0);
       -webkit-mask:radial-gradient(farthest-side, transparent calc(100% - 10px), #000 calc(100% - 9px));
       mask:radial-gradient(farthest-side, transparent calc(100% - 10px), #000 calc(100% - 9px));}
.ring-num {position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
           justify-content:center; line-height:1;}
.ring-num b {font-size:1.65rem; counter-reset:num var(--p);}
.ring-num b::after {content:counter(num);}
.ring-num span {font-size:.66rem; opacity:.65; margin-top:.15rem;}
.verdict-body {flex:1; min-width:260px;}
.verdict-title {font-weight:800; font-size:1.1rem; color:var(--c);}
.verdict-sub {font-size:.88rem; opacity:.8; margin:.1rem 0 .7rem 0;}
.signals {display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:.8rem 1.1rem;}
.signal-top {display:flex; justify-content:space-between; font-size:.78rem; margin-bottom:.25rem;}
.signal-top span {opacity:.8;}
.bar {height:7px; border-radius:99px; background:rgba(128,128,128,.22); overflow:hidden;}
.bar i {display:block; height:100%; border-radius:99px; background:var(--brand); transform-origin:left;
        animation:grow 1s cubic-bezier(.2,.7,.2,1) .3s both;}
.bar-lg {height:10px;}

/* ---------- reddit vs wikipedia ---------- */
.duo {display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:.4rem;}
.side {margin:0; border-top:4px solid var(--line);}
.side-reddit {border-top-color:#e8590c;} .side-wiki {border-top-color:#2563eb; animation-delay:.12s;}
.side-head {display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; margin-bottom:.5rem;}
.side-text {font-size:.95rem; line-height:1.6;}
.agree {margin:.9rem 0 .6rem 0;}
.agree-top {display:flex; justify-content:space-between; margin-bottom:.35rem; font-size:.9rem;}
.trio {display:grid; grid-template-columns:repeat(3, 1fr); gap:1rem; margin-top:.7rem;}
.trio-h {font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; opacity:.65; margin-bottom:.3rem;}
.pill {display:inline-block; padding:.1rem .62rem; margin:.14rem .25rem .14rem 0; border-radius:999px;
       font-size:.78rem; border:1px solid rgba(128,128,128,.4); animation:popin .4s ease both;}
.pill-shared {background:rgba(22,163,74,.2);} .pill-r {background:rgba(232,89,12,.18);}
.pill-w {background:rgba(37,99,235,.18);}

/* ---------- sources ---------- */
.cards3 {display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:1rem;}
.cards3 .card {margin:0;}
.cards3 .card:nth-child(2) {animation-delay:.1s;} .cards3 .card:nth-child(3) {animation-delay:.2s;}
.src-title {font-weight:700; margin:.4rem 0 .25rem 0; line-height:1.35;}
.snippet {font-size:.92rem; opacity:.92; line-height:1.55;}

/* ---------- landing ---------- */
.steps {display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:1rem; margin:.3rem 0 .5rem 0;}
.step {margin:0;}
.steps .step:nth-child(2) {animation-delay:.1s;} .steps .step:nth-child(3) {animation-delay:.2s;}
.step-n {display:inline-flex; width:1.7rem; height:1.7rem; border-radius:50%; background:var(--brand);
         color:#fff; font-weight:700; align-items:center; justify-content:center; margin-bottom:.45rem;}
.step b {display:block; margin-bottom:.15rem;}
.notice {border:1px solid var(--line); border-left:5px solid var(--wiki); border-radius:12px;
         padding:.7rem 1rem; margin:.3rem 0 .6rem 0; font-size:.92rem; background:var(--soft);}
.daily {display:flex; align-items:center; gap:1.2rem; border:1px solid var(--line); border-radius:18px;
        padding:1.15rem 1.5rem; margin:.4rem 0 .3rem 0;
        background:linear-gradient(135deg, rgba(79,70,229,.16), rgba(219,39,119,.13));}
.daily-emoji {font-size:2.7rem; line-height:1; animation:float 4.5s ease-in-out infinite;}
.daily-label {font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; opacity:.7; font-weight:700;}
.daily-q {font-size:1.25rem; font-weight:800; line-height:1.3; margin:.15rem 0 .35rem 0;}

/* ---------- charts ---------- */
.hbars {display:flex; flex-direction:column; gap:.6rem; margin:.3rem 0 .8rem 0;}
.hbar {display:grid; grid-template-columns:minmax(110px, 190px) 1fr 62px; align-items:center; gap:.7rem; font-size:.88rem;}
.hb-label {white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.hb-track {height:14px; border-radius:99px; background:rgba(128,128,128,.16); overflow:hidden;}
.hb-track i {display:block; height:100%; width:var(--w); border-radius:99px; background:var(--c);
             transform-origin:left; animation:grow .9s cubic-bezier(.2,.7,.2,1) both; animation-delay:calc(var(--i) * 70ms);}
.hbar:hover .hb-track i {filter:brightness(1.18);}
.hbar b {text-align:right; font-variant-numeric:tabular-nums;}
.donut-wrap {display:flex; flex-wrap:wrap; align-items:center; gap:1.4rem; margin:.3rem 0 .8rem 0;}
.donut {position:relative; width:150px; height:150px; flex:none; animation:sweep .9s cubic-bezier(.2,.7,.2,1) both;}
.donut-ring {position:absolute; inset:0; border-radius:50%;
       -webkit-mask:radial-gradient(farthest-side, transparent calc(100% - 22px), #000 calc(100% - 21px));
       mask:radial-gradient(farthest-side, transparent calc(100% - 22px), #000 calc(100% - 21px));}
.donut-center {position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; line-height:1.1;}
.donut-center b {font-size:1.6rem;} .donut-center span {font-size:.7rem; opacity:.65;}
.legend {display:flex; flex-direction:column; gap:.4rem; font-size:.88rem;}
.legend i {display:inline-block; width:.72rem; height:.72rem; border-radius:50%; margin-right:.5rem; background:var(--c);}
.spark {width:100%; height:auto; overflow:visible; margin:.2rem 0;}
.spark .line {fill:none; stroke:var(--brand); stroke-width:2.6; stroke-linecap:round; stroke-linejoin:round;
              stroke-dasharray:1; stroke-dashoffset:1; animation:draw 1.5s ease forwards;}
.spark .area {fill:var(--brand); opacity:.14; animation:rise 1.2s ease .6s both;}
.spark circle {fill:var(--brand); animation:popin .4s ease both;}
.bubbles {display:flex; flex-wrap:wrap; align-items:center; margin:.2rem 0 .6rem 0;}
.bub {display:inline-block; padding:.2rem .9rem; margin:.22rem; border-radius:999px; font-weight:600;
      font-size:var(--s); background:var(--bg); border:1px solid var(--bd);
      animation:popin .5s ease both, drift 5s ease-in-out infinite;
      animation-delay:calc(var(--i) * 60ms), calc(var(--i) * -450ms); transition:transform .15s ease;}
.bub:hover {transform:scale(1.14) rotate(-2deg);}

/* ---------- quiz ---------- */
.score {display:grid; grid-template-columns:repeat(3, 1fr); gap:.8rem; margin:.2rem 0 .8rem 0;}
.score div {border:1px solid var(--line); border-radius:14px; padding:.6rem .9rem; background:var(--soft); text-align:center;}
.score b {display:block; font-size:1.5rem; line-height:1.2;}
.score span {font-size:.74rem; opacity:.7; text-transform:uppercase; letter-spacing:.06em;}
.hot b {animation:pulse 1s ease-in-out infinite;}
.quizq {font-size:1.35rem; font-weight:800; line-height:1.35; margin:.3rem 0 0 0;}
.opt {display:flex; gap:.8rem; align-items:flex-start; border:1px solid var(--line); border-radius:14px;
      padding:.8rem 1rem; margin:.5rem 0; background:var(--soft); font-size:.95rem; line-height:1.55;}
.opt-l {flex:none; width:1.7rem; height:1.7rem; border-radius:50%; background:rgba(128,128,128,.3); color:inherit;
        display:inline-flex; align-items:center; justify-content:center; font-weight:800; font-size:.85rem;}
.opt-ok {border-color:#16a34a; background:rgba(22,163,74,.14);} .opt-ok .opt-l {background:#16a34a; color:#fff;}
.opt-bad {border-color:#dc2626; background:rgba(220,38,38,.12);} .opt-bad .opt-l {background:#dc2626; color:#fff;}
.result {border-left:6px solid var(--c); animation:popin .45s ease both;}
.result b {color:var(--c);}

/* ---------- streamlit widgets ---------- */
.stTextInput input {font-size:1.08rem; padding:.85rem 1rem;}
div[data-baseweb="input"] {border-radius:14px; border:2px solid rgba(79,70,229,.35); transition:all .2s ease;}
div[data-baseweb="input"]:focus-within {border-color:#4f46e5; box-shadow:0 0 0 4px rgba(79,70,229,.15);}
.stButton > button {border-radius:999px; border:1px solid rgba(128,128,128,.35); padding:.3rem 1rem;
                    font-size:.85rem; background:transparent; transition:all .15s ease;}
.stButton > button:hover {border-color:#4f46e5; color:#4f46e5; transform:translateY(-2px);
                          box-shadow:0 8px 16px -12px rgba(79,70,229,.8);}
.stButton > button:active {transform:scale(.97);}
.stTabs [data-baseweb="tab-list"] {gap:.3rem; border-bottom:1px solid var(--line);}
.stTabs button[data-baseweb="tab"] {font-weight:600; padding:.55rem 1rem; border-radius:10px 10px 0 0;}
div[data-testid="stMetric"] {border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem; background:var(--soft);}
div[data-testid="stExpander"] {border-radius:12px;}
@media (max-width:760px) {.duo, .trio {grid-template-columns:1fr;} .hero h1 {font-size:1.6rem;}
       .hero-emoji {font-size:2.6rem; right:1rem; top:1.2rem; margin-top:0;} .hbar {grid-template-columns:90px 1fr 50px;}}
@media (prefers-reduced-motion: reduce) {*, *::before, *::after {animation-duration:.001ms !important;
       animation-delay:0s !important; animation-iteration-count:1 !important; transition:none !important;}}
"""

# One line, no blank lines (see module docstring).
CSS = "<style>" + re.sub(r"/\*.*?\*/", "", _CSS, flags=re.S).replace("\n", " ") + "</style>"


# ============================================================
# helpers
# ============================================================

def esc(value):
    """Escape anything that comes from the dataset or the web before it goes in HTML."""
    return html.escape(str(value), quote=True)


def safe_link(url):
    if not isinstance(url, str):
        return None
    url = url.strip()
    if url.startswith("/r/"):
        url = "https://www.reddit.com" + url
    return url if url.startswith(("https://", "http://")) else None


def badge(text, kind):
    return f'<span class="badge b-{kind}">{esc(text)}</span>'


def pills(words, kind):
    if not words:
        return '<span class="meta">none</span>'
    return "".join(
        f'<span class="pill pill-{kind}" style="animation-delay:{i * 50}ms">{esc(w)}</span>'
        for i, w in enumerate(words))


def section(title):
    return f'<div class="section-title">{esc(title)}</div>'


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ============================================================
# topic themes: every community gets an emoji + colour
# ============================================================

_TOPICS = {
    "askscience": ("🔬", "#0ea5e9"), "science": ("🧪", "#06b6d4"), "space": ("🚀", "#7c3aed"),
    "askhistorians": ("🏛️", "#b45309"), "history": ("🏺", "#a16207"),
    "explainlikeimfive": ("🧸", "#ec4899"), "askreddit": ("💭", "#f97316"),
    "nostupidquestions": ("🤔", "#8b5cf6"), "personalfinance": ("💰", "#16a34a"),
    "lifeprotips": ("💡", "#eab308"), "todayilearned": ("🧠", "#14b8a6"),
    "askphilosophy": ("📜", "#6366f1"), "askengineers": ("⚙️", "#64748b"),
    "askculinary": ("🍳", "#ef4444"), "cooking": ("🍳", "#ef4444"),
    "technology": ("💻", "#3b82f6"), "programming": ("👩‍💻", "#2563eb"),
    "learnprogramming": ("👨‍💻", "#2563eb"), "askdocs": ("🩺", "#10b981"),
    "health": ("🩺", "#10b981"), "fitness": ("💪", "#f43f5e"), "movies": ("🎬", "#dc2626"),
    "music": ("🎵", "#d946ef"), "books": ("📚", "#92400e"), "gaming": ("🎮", "#7c3aed"),
    "legaladvice": ("⚖️", "#475569"), "askeconomics": ("📈", "#059669"),
    "askanthropology": ("🗿", "#b45309"), "askpolitics": ("🗳️", "#dc2626"),
}
_FALLBACK = [("📚", "#6366f1"), ("🌍", "#0891b2"), ("🎯", "#db2777"), ("🧩", "#7c3aed"), ("🌟", "#d97706")]


def topic(name):
    """(emoji, colour) for a community name; unknown names get a stable fallback."""
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if key in _TOPICS:
        return _TOPICS[key]
    return _FALLBACK[sum(ord(c) for c in key) % len(_FALLBACK)]


def topic_chip(name):
    emoji, color = topic(name)
    return (f'<span class="ichip ichip-topic" style="border-color:{color}; color:{color}; '
            f'background:{_rgba(color, .12)}">{emoji} r/{esc(name)}</span>')


def info_chips(items, topic_name=None):
    """items: list of (icon, text). Optionally starts with the topic chip."""
    body = (topic_chip(topic_name) if topic_name else "") + "".join(
        f'<span class="ichip">{icon} {esc(text)}</span>' for icon, text in items)
    return f'<div class="infochips">{body}</div>'


# ============================================================
# animation helper: words fade in one after another
# ============================================================

def reveal(html_text, max_words=140, step_ms=16):
    """Wrap each word of already-safe inline HTML so it fades in. Skips long texts."""
    tokens = re.split(r"(\s+)", html_text)
    words = [t for t in tokens if t and not t.isspace()]
    if len(words) > max_words:
        return html_text
    out, i = [], 0
    for t in tokens:
        if not t or t.isspace():
            out.append(t)
        else:
            out.append(f'<span class="w" style="animation-delay:{min(i * step_ms, 1500)}ms">{t}</span>')
            i += 1
    return "".join(out)


# ============================================================
# page pieces
# ============================================================

def hero(chips, emoji="💬"):
    """chips: list of (value, label)."""
    chip_html = "".join(f'<span class="chip"><b>{esc(v)}</b> {esc(l)}</span>' for v, l in chips)
    return (
        '<div class="hero">'
        f'<div class="hero-emoji">{emoji}</div>'
        '<h1>Forum Answers</h1>'
        "<p>Real answers from Reddit communities, cross-checked against Wikipedia "
        "and scored for trust, so you can see <em>why</em> an answer deserves belief.</p>"
        f"{chip_html}</div>"
    )


def steps():
    items = [
        ("1", "Ask", "Type a question, pick an example, or press 🎲 for a random one."),
        ("2", "Compare", "See what Reddit says next to Wikipedia and where they agree."),
        ("3", "Judge", "A trust score explains how far to rely on it. Rate it 👍 / 👎."),
    ]
    cards = "".join(
        f'<div class="card step rise lift"><div class="step-n">{n}</div><b>{esc(t)}</b>'
        f'<div class="snippet">{esc(d)}</div></div>' for n, t, d in items
    )
    return f'<div class="steps">{cards}</div>'


def notice(text):
    return f'<div class="notice rise">{esc(text)}</div>'


def daily_card(title, community):
    emoji, color = topic(community)
    return (
        f'<div class="daily rise" style="border-color:{_rgba(color, .5)}">'
        f'<div class="daily-emoji">{emoji}</div>'
        '<div><div class="daily-label">✨ Question of the day</div>'
        f'<div class="daily-q">{esc(title)}</div>{topic_chip(community)}</div></div>'
    )


def answer_card(source, label, body_html, hint, title=None, accent=None, foot_html=""):
    """source: 'reddit' or 'wiki'. body_html must already be safe HTML."""
    accent = accent or (REDDIT if source == "reddit" else WIKI)
    name = "Reddit" if source == "reddit" else "Wikipedia"
    extra = f'<span class="hint">{esc(title)}</span>' if title else ""
    return (
        f'<div class="card answer rise" style="--accent:{accent}">'
        f'<div class="answer-top">{badge(name, source)}{badge(label + " trust", label)}'
        f'{extra}<span class="hint">{esc(hint)}</span></div>'
        f'<div class="answer-body">{body_html}</div>{foot_html}</div>'
    )


def points_html(answer, query_words):
    items = "".join(f"<li>{qa_system.highlight_html(p, query_words)}</li>"
                    for p in qa_system.key_points(answer))
    return f"<ul>{items}</ul>"


VERDICT_TEXT = {
    "high": ("High trust", "Strong match. The sources line up with each other."),
    "medium": ("Moderate trust", "Reasonable match. Check the sources before relying on it."),
    "low": ("Low trust", "Weak match. Treat this answer with care."),
}


def verdict(trust):
    title, sub = VERDICT_TEXT[trust["label"]]
    signals = "".join(
        '<div class="signal">'
        f'<div class="signal-top"><span>{esc(p["name"])}</span><b>{round(p["value"] * 100)}%</b></div>'
        f'<div class="bar"><i style="width:{round(p["value"] * 100)}%"></i></div></div>'
        for p in trust["parts"]
    )
    score = int(trust["score"])
    return (
        f'<div class="verdict rise t-{trust["label"]}">'
        f'<div class="ring-wrap" style="--target:{score}" role="img" aria-label="Trust score {score} out of 100">'
        '<div class="ring"></div>'
        '<div class="ring-num"><b></b><span>/ 100</span></div></div>'
        '<div class="verdict-body">'
        f'<div class="verdict-title">{esc(title)}</div>'
        f'<div class="verdict-sub">{esc(sub)}</div>'
        f'<div class="signals">{signals}</div></div></div>'
    )


def duo(reddit_text, wiki):
    """Side-by-side Reddit / Wikipedia cards. wiki is a dict (title, answer, url)."""
    link = safe_link(wiki.get("url"))
    link_html = (f'<div class="meta" style="margin-top:.55rem"><a href="{esc(link)}" target="_blank" '
                 f'rel="noopener noreferrer">Read the article ↗</a></div>' if link else "")
    return (
        '<div class="duo">'
        '<div class="card side side-reddit rise lift">'
        f'<div class="side-head">{badge("Reddit", "reddit")}<span class="hint">what the community says</span></div>'
        f'<div class="side-text">{esc(qa_system.shorten(reddit_text))}</div></div>'
        '<div class="card side side-wiki rise lift">'
        f'<div class="side-head">{badge("Wikipedia", "wiki")}<span class="hint">{esc(wiki["title"])}</span></div>'
        f'<div class="side-text">{esc(qa_system.shorten(wiki["answer"]))}</div>{link_html}</div></div>'
    )


def agreement(cmp):
    pct = max(4, min(100, round(cmp["similarity"] / 0.40 * 100)))
    return (
        '<div class="agree">'
        f'<div class="agree-top"><b>{esc(cmp["verdict"])}</b><span class="hint">similarity {cmp["similarity"]:.2f}</span></div>'
        f'<div class="bar bar-lg"><i style="width:{pct}%"></i></div></div>'
        '<div class="trio">'
        f'<div><div class="trio-h">Both mention</div>{pills(cmp["shared"], "shared")}</div>'
        f'<div><div class="trio-h">Only the community</div>{pills(cmp["only_reddit"], "r")}</div>'
        f'<div><div class="trio-h">Only Wikipedia</div>{pills(cmp["only_wiki"], "w")}</div></div>'
    )


def source_cards(sources, query_words):
    cards = []
    for src in sources:
        link = safe_link(src.get("link"))
        if link:
            link_html = (f'<a href="{esc(link)}" target="_blank" rel="noopener noreferrer">'
                         f'View on Reddit ↗</a>')
        else:
            q = quote_plus(str(src["title"]))
            link_html = (f'<a href="https://www.reddit.com/search/?q={q}" target="_blank" '
                         f'rel="noopener noreferrer">Search on Reddit ↗</a>')
        upvotes = int(src["upvotes"])
        up = f"⬆ {upvotes:,} upvotes" if upvotes > 0 else "⬆ upvotes n/a"
        emoji, _ = topic(src["subreddit"])
        cards.append(
            '<div class="card rise lift">'
            f'{badge(src["n"], "num")}<span class="meta"> {emoji} r/{esc(src["subreddit"])} · {up}</span>'
            f'<div class="src-title">{esc(src["title"])}</div>'
            f'<div class="snippet">{qa_system.highlight_html(src["snippet"], query_words)}</div>'
            f'<div class="meta" style="margin-top:.55rem">{link_html}</div></div>'
        )
    return f'<div class="cards3">{"".join(cards)}</div>'


# ============================================================
# animated charts (pure HTML / CSS / SVG)
# ============================================================

def hbar_chart(items):
    """items: list of (label, value, colour). Bars grow in one after another."""
    if not items:
        return ""
    top = max(v for _, v, _ in items) or 1
    rows = []
    for i, (label, value, color) in enumerate(items):
        width = max(3, round(value / top * 100))
        rows.append(
            f'<div class="hbar" title="{esc(label)}: {value:,}">'
            f'<span class="hb-label">{esc(label)}</span>'
            f'<div class="hb-track"><i style="--w:{width}%; --c:{color}; --i:{i}"></i></div>'
            f'<b>{value:,}</b></div>')
    return f'<div class="hbars">{"".join(rows)}</div>'


def donut(items, center_label="total"):
    """items: list of (label, value, colour)."""
    total = sum(v for _, v, _ in items)
    stops, start = [], 0.0
    for _, v, color in items:
        end = start + (v / total * 100 if total else 0)
        stops.append(f"{color} {start:.2f}% {end:.2f}%")
        start = end
    gradient = ", ".join(stops) if total else "rgba(128,128,128,.25) 0 100%"
    legend = "".join(
        f'<div><i style="--c:{c}"></i>{esc(l)} <b>{v:,}</b> '
        f'<span class="hint">({(v / total * 100 if total else 0):.0f}%)</span></div>'
        for l, v, c in items)
    return (
        '<div class="donut-wrap">'
        '<div class="donut">'
        f'<div class="donut-ring" style="background:conic-gradient({gradient})"></div>'
        f'<div class="donut-center"><b>{total:,}</b><span>{esc(center_label)}</span></div></div>'
        f'<div class="legend">{legend}</div></div>'
    )


def sparkline(values, y_label=""):
    """Animated line chart. The line draws itself, then the area fades in."""
    values = [float(v) for v in values]
    if len(values) < 2:
        return '<div class="meta">Ask a few more questions to see a trend.</div>'
    w, h, pad = 320, 90, 8
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(values):
        x = pad + i * (w - 2 * pad) / (len(values) - 1)
        y = h - pad - ((v - lo) / span if hi != lo else 0.5) * (h - 2 * pad)
        pts.append((x, y, v))
    line = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts))
    area = f"{line} L{pts[-1][0]:.1f},{h - pad} L{pts[0][0]:.1f},{h - pad} Z"
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" style="animation-delay:{0.9 + i * 0.05:.2f}s">'
        f'<title>question {i + 1}: {v:.0f}</title></circle>' for i, (x, y, v) in enumerate(pts[-30:]))
    return (
        f'<svg class="spark" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(y_label)} trend">'
        f'<path class="area" d="{area}"/><path class="line" d="{line}" pathLength="1"/>{dots}</svg>'
        f'<div class="meta">min {lo:.0f} · max {hi:.0f} · latest {values[-1]:.0f}</div>'
    )


def bubbles(terms, color="#4f46e5"):
    """Floating word bubbles. terms: list of (word, weight); bigger weight = bigger bubble."""
    if not terms:
        return '<div class="meta">Not enough text to analyse.</div>'
    ws = [w for _, w in terms]
    lo, hi = min(ws), max(ws)
    span = (hi - lo) or 1.0
    out = []
    for i, (term, weight) in enumerate(terms):
        size = 0.85 + 0.95 * ((weight - lo) / span)
        out.append(
            f'<span class="bub" style="--s:{size:.2f}rem; --i:{i}; --bg:{_rgba(color, .15)}; '
            f'--bd:{_rgba(color, .5)}" title="importance {weight:.3f}">{esc(term)}</span>')
    return f'<div class="bubbles">{"".join(out)}</div>'


# ============================================================
# quiz
# ============================================================

def scoreboard(score, answered, streak, best):
    hot = " hot" if streak >= 3 else ""
    return (
        '<div class="score">'
        f'<div><b>{score}/{answered}</b><span>Score</span></div>'
        f'<div class="{hot.strip()}"><b>{"🔥 " if streak >= 3 else ""}{streak}</b><span>Streak</span></div>'
        f'<div><b>{best}</b><span>Best streak</span></div></div>'
    )


def quiz_question(question, community, hard):
    mode = "Hard · same topic" if hard else "Easy · mixed topics"
    return (
        '<div class="card rise">'
        f'{topic_chip(community)}<span class="ichip">🎯 {mode}</span>'
        '<div class="hint" style="margin-top:.7rem">Which Reddit answer belongs to this question?</div>'
        f'<div class="quizq">{esc(question)}</div></div>'
    )


def quiz_options(options, correct=None, picked=None):
    """Lettered answer cards. Pass correct/picked after the player answers to colour them."""
    out = []
    for i, text in enumerate(options):
        cls = "opt rise"
        if correct is not None:
            if i == correct:
                cls += " opt-ok"
            elif i == picked:
                cls += " opt-bad"
        out.append(f'<div class="{cls}" style="animation-delay:{i * 70}ms">'
                   f'<span class="opt-l">{"ABCDEFG"[i]}</span><div>{esc(text)}</div></div>')
    return "".join(out)


def quiz_result(ok, community, streak):
    color = "#16a34a" if ok else "#dc2626"
    if ok:
        msg = f"<b>Correct!</b> You know your r/{esc(community)}."
        if streak >= 3:
            msg += f" 🔥 {streak} in a row."
    else:
        msg = f"<b>Not quite.</b> The green card was the real answer from r/{esc(community)}."
    return f'<div class="card result" style="--c:{color}">{msg}</div>'
