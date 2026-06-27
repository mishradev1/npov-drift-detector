"""Project-wide configuration constants.

Deliberately free of any *topic- or subject-specific* configuration: the only
lexical lists here are GENERIC structural heading names (boilerplate sections,
and the kinds of headings that tend to carry alternative views). Per the build
spec, anything topic-specific must be learned from data, not hard-coded.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = DATA_DIR / "out"

# --- HTTP / API politeness -------------------------------------------------
# Wikimedia User-Agent policy asks for a descriptive UA with contact info.
USER_AGENT = (
    "npov-drift-detector/0.1 "
    "(research tool; viewpoint-drift detection; contact: nervywave@gmail.com)"
)
API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

# Article-type scoring. ORES is the classic endpoint; Lift Wing is the modern
# replacement. We try ORES first, then Lift Wing, then fall back to "unknown".
ORES_ARTICLETOPIC_URL = "https://ores.wikimedia.org/v3/scores/enwiki/{revid}/articletopic"
LIFTWING_ARTICLETOPIC_URL = (
    "https://api.wikimedia.org/service/lw/inference/v1/models/enwiki-articletopic:predict"
)

# maxlag: ask the API to back off when the replication lag exceeds this many
# seconds (standard polite-client behaviour).
DEFAULT_MAXLAG = 5
# Minimum seconds between live (non-cached) requests.
DEFAULT_MIN_INTERVAL = 0.2
DEFAULT_MAX_RETRIES = 5
# rvlimit: 500 is the max for non-bot clients.
REVISIONS_PER_REQUEST = 500
# How many revids to request content for in a single batch (non-bot max is 50;
# we stay well under to keep responses small).
CONTENT_BATCH_SIZE = 20

# --- Generic structural heading lists --------------------------------------
# Boilerplate sections are excluded from semantic/drift analysis (no prose
# content of analytic interest). Compared case-insensitively against headings.
BOILERPLATE_SECTIONS = frozenset(
    {
        "references",
        "external links",
        "see also",
        "further reading",
        "notes",
        "bibliography",
        "sources",
        "citations",
        "footnotes",
        "works cited",
        "notes and references",
        "external link",
    }
)

# Headings that GENERICALLY tend to host alternative views / criticism /
# reception across all topic types. This is a structural prior only; in later
# phases such sections are also identified by their stance/cluster composition
# (data-driven), not by these names alone.
ALT_VIEW_HEADINGS = frozenset(
    {
        "criticism",
        "criticisms",
        "controversy",
        "controversies",
        "reception",
        "reactions",
        "reaction",
        "opposition",
        "concerns",
        "debate",
        "debates",
        "legacy",
        "responses",
        "response",
        "analysis",
        "arguments",
        "criticism and controversy",
        "public opinion",
        "disputes",
    }
)
