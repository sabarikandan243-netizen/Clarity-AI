# ================================================================
# CLARITY AI v33
# HALLUCINATION-RESISTANT VERIFIER
# v32 bug-fixes + DATABASE / RAG / ROUTER merged into ONE file
# ================================================================

import os
import re
import json
import html as html_lib
import hashlib
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse

# ------------------------------------------------
# OPTIONAL LIBRARIES (nothing crashes if missing)
# ------------------------------------------------

try:
    import requests
except ImportError:
    requests = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    import tldextract
except ImportError:
    tldextract = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

try:
    import torch
except ImportError:
    torch = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ================================================================
# CONFIG
# ================================================================

VERSION = "33.0"

# Keys: environment variable first, else paste between the quotes.
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
TAVILY_API_KEY = "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"

# English -> fast models. Tamil / other scripts -> multilingual models.
NLI_MODEL_EN = "cross-encoder/nli-deberta-v3-base"
NLI_MODEL_MULTI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EMBED_MODEL_EN = "all-MiniLM-L6-v2"
EMBED_MODEL_MULTI = "paraphrase-multilingual-MiniLM-L12-v2"

MAX_SOURCES = 10
MAX_EVIDENCE = 5
MAX_NLI_CANDIDATES = 12
MAX_PAGE_CHARS = 60000
MAX_SENTENCES_PER_PAGE = 300
REQUEST_TIMEOUT = 12

SEMANTIC_MIN = 0.30      # embedding similarity floor
FALLBACK_MIN = 0.15      # token-overlap floor (no embedding model)
NLI_THRESHOLD = 0.60

# Database tool. "" = demo in-memory DB. Or a path to your .db file
# (opened READ-ONLY). Table must have a name column and a numeric column.
DB_PATH = ""
DB_TABLE = "products"
DB_NAME_COL = "name"
DB_VALUE_COL = "price"


# ================================================================
# LANGUAGE HELPERS
# ================================================================

# NOTE: Python's \w does NOT match Tamil vowel signs (combining marks),
# so Tamil words would get cut into pieces. This pattern fixes that.
TOKEN_RE = re.compile(r"[\w\u0B80-\u0BFF]+")


def is_non_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for c in letters if ord(c) > 0x24F)
    return (non_latin / len(letters)) > 0.30


# ================================================================
# SOURCE RELIABILITY
# ================================================================

SOURCE_WEIGHTS = {
    "database": 1.00,
    "gov": 1.00,
    "edu": 0.95,
    "research": 0.95,
    "document": 0.90,
    "reputable_news": 0.85,
    "organization": 0.85,
    "general": 0.60,
    "unknown": 0.45,
}

GOV_RE = re.compile(
    r"(^|\.)(gov|nic|gob)(\.[a-z]{2})?$|(^|\.)go\.(jp|kr|id|th)$"
)
EDU_RE = re.compile(r"(^|\.)edu(\.[a-z]{2})?$|(^|\.)ac\.[a-z]{2}$")

RESEARCH_DOMAINS = [
    "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov",
    "arxiv.org", "thelancet.com", "nejm.org"
]
NEWS_DOMAINS = [
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com",
    "theguardian.com", "nytimes.com", "thehindu.com",
    "indianexpress.com", "aljazeera.com", "washingtonpost.com"
]
ORG_DOMAINS = [
    "who.int", "un.org", "worldbank.org", "imf.org",
    "europa.eu", "oecd.org"
]

MULTI_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "gov.au",
    "edu.au", "co.in", "gov.in", "nic.in", "ac.in", "org.in",
    "net.in", "co.jp", "com.br", "co.nz", "co.za", "com.sg",
    "com.my", "com.cn"
}

_tld_extractor = None


def get_host(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _domain_in(host: str, domains: List[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def classify_source(url: str) -> str:
    if url.startswith("db://"):
        return "database"
    if url.startswith("doc://"):
        return "document"

    host = get_host(url)
    if not host:
        return "unknown"

    # Strict matching: "fake.gov.scam.com" is NOT gov.
    if GOV_RE.search(host):
        return "gov"
    if EDU_RE.search(host):
        return "edu"
    if _domain_in(host, RESEARCH_DOMAINS):
        return "research"
    if _domain_in(host, NEWS_DOMAINS):
        return "reputable_news"
    if _domain_in(host, ORG_DOMAINS):
        return "organization"
    return "general"


def source_weight(url: str) -> float:
    return SOURCE_WEIGHTS.get(
        classify_source(url), SOURCE_WEIGHTS["unknown"]
    )


def canonical_org(url: str) -> str:
    global _tld_extractor

    if url.startswith(("db://", "doc://")):
        return url

    host = get_host(url)
    if not host:
        return "unknown"

    if tldextract is not None:
        try:
            if _tld_extractor is None:
                _tld_extractor = tldextract.TLDExtract(
                    suffix_list_urls=()
                )
            registered = _tld_extractor(host).registered_domain
            if registered:
                return registered
        except Exception:
            pass

    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_SUFFIXES:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


# ================================================================
# DATA MODEL
# ================================================================

@dataclass
class Evidence:
    url: str
    title: str
    sentence: str
    context: str
    source_type: str
    reliability: float
    organization: str
    content_hash: str
    nli_label: str = "unknown"
    nli_score: float = 0.0
    semantic_score: float = 0.0
    final_score: float = 0.0


def compute_final_score(e: Evidence) -> float:
    return (
        e.reliability * 0.35
        + min(1.0, e.semantic_score) * 0.25
        + e.nli_score * 0.40
    )


# ================================================================
# TEXT HELPERS
# ================================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def content_hash_fixed(text: str) -> str:
    """Page-level hash (used to skip mirrored / identical pages)."""
    if not text:
        return ""

    text = normalize_text(text)
    clean = text[200:] if len(text) > 1000 else text
    clean = clean.lower()

    for phrase in [
        "cookie policy", "privacy policy", "subscribe",
        "sign up", "advertisement", "all rights reserved"
    ]:
        clean = clean.replace(phrase, "")

    clean = re.sub(r"\s+", " ", clean)[:2000]
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def sentence_hash(sentence: str) -> str:
    """Sentence-level hash."""
    clean = " ".join(TOKEN_RE.findall(sentence.lower()))
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def split_sentences(text: str, min_len: int = 20) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.strip()) >= min_len]


def token_similarity(a: str, b: str) -> float:
    a_tokens = set(TOKEN_RE.findall(a.lower()))
    b_tokens = set(TOKEN_RE.findall(b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    union = len(a_tokens | b_tokens)
    return len(a_tokens & b_tokens) / union if union else 0.0


STOP_CAPS = {
    "the", "this", "that", "these", "those", "there", "here", "and",
    "but", "for", "with", "from", "his", "her", "its", "their",
    "they", "she", "you", "our", "are", "was", "were", "has", "have",
    "had", "will", "can", "not", "all", "some", "one", "also", "when",
    "what", "which", "who", "how", "why", "after", "before", "while",
    "since", "about", "according", "however", "because"
}


def extract_entities(text: str) -> set:
    entities = set()

    for m in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text):
        if m.lower() not in STOP_CAPS:
            entities.add(m.lower())

    for n in extract_numbers(text):
        entities.add(str(n))

    for t in re.findall(r"[\u0B80-\u0BFF]{3,}", text):
        entities.add(t)

    return entities


def extract_numbers(text: str) -> List[float]:
    out = []
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


# ================================================================
# EMBEDDING / RANKING
# ================================================================

_embed_models: Dict[str, object] = {}


def load_embedding_model(multilingual: bool = False):
    name = EMBED_MODEL_MULTI if multilingual else EMBED_MODEL_EN
    if name in _embed_models:
        return _embed_models[name]

    model = None
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(name)
        except Exception:
            model = None

    _embed_models[name] = model
    return model


def rank_texts(
    claim: str,
    texts: List[str],
    top_k: int = 3,
    cache: Optional[dict] = None
) -> List[Tuple[float, int]]:
    if not texts:
        return []

    claim_ents = extract_entities(claim)
    multi = is_non_english(claim) or is_non_english(
        " ".join(texts[:10])
    )

    model = load_embedding_model(multi)

    if model is not None and util is not None:
        try:
            emb = cache.get(multi) if cache is not None else None
            if emb is None:
                emb = model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                if cache is not None:
                    cache[multi] = emb

            claim_emb = model.encode(
                claim,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            sims = util.cos_sim(claim_emb, emb)[0]

            ranked = []
            for i, t in enumerate(texts):
                score = float(sims[i])
                if score < SEMANTIC_MIN:
                    continue
                if claim_ents & extract_entities(t):
                    score += 0.10
                ranked.append((min(1.0, score), i))

            ranked.sort(reverse=True)
            return ranked[:top_k]

        except Exception:
            pass

    ranked = []
    for i, t in enumerate(texts):
        score = token_similarity(claim, t)
        if score < FALLBACK_MIN:
            continue
        if claim_ents & extract_entities(t):
            score += 0.15
        ranked.append((min(1.0, score), i))

    ranked.sort(reverse=True)
    return ranked[:top_k]


# ================================================================
# NLI
# ================================================================

_nli_cache: Dict[str, tuple] = {}


def _norm_label(label: str) -> str:
    l = str(label).lower()
    if l.startswith("entail"):
        return "entailment"
    if l.startswith("contra"):
        return "contradiction"
    return "neutral"


def load_nli_model(multilingual: bool = False):
    name = NLI_MODEL_MULTI if multilingual else NLI_MODEL_EN
    if name in _nli_cache:
        return _nli_cache[name]

    result = (None, None, None)

    if (
        AutoTokenizer is not None
        and AutoModelForSequenceClassification is not None
        and torch is not None
    ):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            model.eval()

            labels = {
                int(i): _norm_label(l)
                for i, l in model.config.id2label.items()
            }
            if "entailment" not in labels.values():
                labels = {
                    0: "contradiction",
                    1: "entailment",
                    2: "neutral"
                }
            result = (tok, model, labels)
        except Exception:
            result = (None, None, None)

    _nli_cache[name] = result
    return result


def nli_check(claim: str, premise: str) -> Tuple[str, float]:
    multi = is_non_english(claim) or is_non_english(premise)
    tok, model, labels = load_nli_model(multi)

    if tok is None:
        return "unknown", 0.0

    try:
        inputs = tok(
            premise[:2000],
            claim[:1000],
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0]

        p = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for i, name in labels.items():
            if i < len(probs):
                p[name] = float(probs[i])

        if p["entailment"] >= NLI_THRESHOLD:
            return "support", p["entailment"]
        if p["contradiction"] >= NLI_THRESHOLD:
            return "contradict", p["contradiction"]
        return "unknown", p["neutral"]

    except Exception:
        return "unknown", 0.0


def apply_nli(claim: str, evidences: List[Evidence]) -> bool:
    multi = is_non_english(claim) or any(
        is_non_english(e.sentence) for e in evidences[:5]
    )
    tok, _, _ = load_nli_model(multi)
    if tok is None:
        return False

    for e in evidences:
        label, score = nli_check(claim, e.context or e.sentence)
        e.nli_label = label
        e.nli_score = score
        e.final_score = compute_final_score(e)
    return True


# ================================================================
# WEB SEARCH (TAVILY API INTEGRATED)
# ================================================================

def search_brave(query: str) -> List[Dict]:
    # பெயர் search_brave-ஆ இருந்தாலும், உள்ளே Tavily API வேலை செய்யும்
    if not TAVILY_API_KEY or TAVILY_API_KEY == "PASTE_YOUR_NEW_TAVILY_KEY_HERE" or requests is None:
        return []

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": MAX_SOURCES
    }

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return []
        data = response.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("content", "")
            })
        return results
    except Exception:
        return []


def search_serpapi(query: str) -> List[Dict]:
    if not SERPAPI_KEY or requests is None:
        return []

    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google"
    }
    if is_non_english(query):
        params["hl"] = "ta"

    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code != 200:
            return []

        items = response.json().get("organic_results", [])
        return [
            {
                "title": i.get("title", ""),
                "url": i.get("link", ""),
                "description": i.get("snippet", "")
            }
            for i in items
        ]
    except Exception:
        return []


def search_web(query: str) -> List[Dict]:
    results = list(search_brave(query))

    if len(results) < 3:
        results.extend(search_serpapi(query))

    seen, unique = set(), []
    for r in results:
        url = r.get("url", "")
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        unique.append(r)

    return unique[:MAX_SOURCES]


# ================================================================
# PAGE EXTRACTION
# ================================================================

def extract_page_text(url: str) -> str:
    if requests is None:
        return ""

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": f"ClarityAI/{VERSION}"}
        )
        if response.status_code != 200:
            return ""

        ctype = response.headers.get("Content-Type", "").lower()
        if ctype and "html" not in ctype and "text" not in ctype:
            return ""

        raw = response.text[:500000]

        if trafilatura is not None:
            extracted = trafilatura.extract(
                raw, include_comments=False, include_tables=True
            )
            if extracted:
                return normalize_text(extracted)[:MAX_PAGE_CHARS]

        text = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html_lib.unescape(text)
        return normalize_text(text)[:MAX_PAGE_CHARS]

    except Exception:
        return ""


# ================================================================
# CLAIM HELPERS
# ================================================================

def decompose_claims(claim: str) -> List[str]:
    claim = normalize_text(claim)
    if not claim:
        return []
    parts = re.split(r"(?<=[.!?])\s+", claim)
    return [s.strip() for s in parts if s.strip()]


def is_question(text: str) -> bool:
    t = text.strip()
    return t.endswith("?") or t.endswith("\uFF1F")


HIGH_RISK_LATIN = [
    "medical", "medicine", "disease", "drug", "treatment", "suicide",
    "election", "president", "war", "legal", "law", "money",
    "investment", "stock", "bank", "vaccine", "cancer", "dose"
]
HIGH_RISK_TAMIL = [
    "மருந்து", "நோய்", "சிகிச்சை", "தேர்தல்", "சட்டம்",
    "முதலீடு", "வங்கி", "தடுப்பூசி"
]
_HIGH_RISK_RE = re.compile(
    r"\b(" + "|".join(HIGH_RISK_LATIN) + r")\b", re.I
)


def is_high_risk_claim(claim: str) -> bool:
    if _HIGH_RISK_RE.search(claim):
        return True
    return any(w in claim for w in HIGH_RISK_TAMIL)


def remove_duplicate_evidence(evidences: List[Evidence]) -> List[Evidence]:
    seen, unique = set(), []
    for e in evidences:
        if e.content_hash in seen:
            continue
        seen.add(e.content_hash)
        unique.append(e)
    return unique


def source_independence_score(evidences: List[Evidence]) -> float:
    if not evidences:
        return 0.0
    orgs = set(e.organization for e in evidences)
    return min(1.0, len(orgs) / 4.0)


def adversarial_check(claim: str, evidences: List[Evidence]) -> bool:
    if not evidences:
        return True
    return not any(e.reliability >= 0.80 for e in evidences)


# ================================================================
# DECISION ENGINE
# ================================================================

def decide(claim: str, evidences: List[Evidence], nli_ok: bool = True) -> Dict:

    if not evidences:
        return {
            "result": "UNKNOWN", "confidence": 0.0,
            "reason": "No usable evidence",
            "source_independence": 0.0, "evidence": []
        }

    evidences = remove_duplicate_evidence(
        sorted(evidences, key=lambda x: x.final_score, reverse=True)
    )

    if not nli_ok:
        return {
            "result": "NEEDS_REVIEW", "confidence": 0.0,
            "reason": "NLI model could not load.",
            "source_independence": 0.0,
            "evidence": evidences[:MAX_EVIDENCE]
        }

    decisive = [
        e for e in evidences
        if e.nli_label in ("support", "contradict")
    ][:MAX_EVIDENCE]

    if not decisive:
        return {
            "result": "NEEDS_REVIEW", "confidence": 0.0,
            "reason": "Evidence is inconclusive",
            "source_independence": 0.0,
            "evidence": evidences[:MAX_EVIDENCE]
        }

    support_score = sum(
        e.final_score for e in decisive if e.nli_label == "support"
    )
    contradict_score = sum(
        e.final_score for e in decisive if e.nli_label == "contradict"
    )
    total = support_score + contradict_score

    authoritative = all(e.source_type == "database" for e in decisive)
    independence = 1.0 if authoritative else source_independence_score(decisive)
    coverage = 1.0 if authoritative else min(1.0, len(decisive) / 3.0)

    reason = "Evidence-based verification"

    if support_score > contradict_score:
        result, ratio = "SUPPORTED", support_score / total
    elif contradict_score > support_score:
        result, ratio = "REFUTED", contradict_score / total
    else:
        result, ratio = "NEEDS_REVIEW", 0.0

    confidence = ratio * (0.60 + 0.40 * coverage) * (0.70 + 0.30 * independence)

    if result != "NEEDS_REVIEW" and ratio < 0.65:
        result = "NEEDS_REVIEW"
        reason = "Sources conflict with each other"

    if is_high_risk_claim(claim):
        confidence = min(confidence, 0.85)

    if adversarial_check(claim, decisive):
        confidence = min(confidence, 0.70)
        if confidence < 0.55:
            result = "NEEDS_REVIEW"
            reason = "Only weak sources found"

    return {
        "result": result,
        "confidence": round(confidence, 4),
        "reason": reason,
        "source_independence": round(independence, 3),
        "evidence": decisive
    }


# ================================================================
# TOOL 1: WEB
# ================================================================

def gather_web_candidates(claim: str) -> List[Evidence]:
    results = search_web(claim)
    if not results:
        return []

    def fetch(res):
        try:
            text = extract_page_text(res.get("url", ""))
            if not text:
                text = normalize_text(res.get("description", ""))
            return res, text
        except Exception:
            return res, ""

    with ThreadPoolExecutor(max_workers=5) as pool:
        fetched = list(pool.map(fetch, results))

    candidates, seen_pages = [], set()

    for res, text in fetched:
        if not text:
            continue

        page_hash = content_hash_fixed(text)
        if page_hash in seen_pages:
            continue
        seen_pages.add(page_hash)

        url = res.get("url", "")
        reliability = source_weight(url)
        sentences = split_sentences(text)[:MAX_SENTENCES_PER_PAGE]

        for sem, i in rank_texts(claim, sentences):
            context = " ".join(sentences[max(0, i - 1): i + 2])
            ev = Evidence(
                url=url,
                title=res.get("title", ""),
                sentence=sentences[i],
                context=context,
                source_type=classify_source(url),
                reliability=reliability,
                organization=canonical_org(url),
                content_hash=sentence_hash(sentences[i]),
                semantic_score=sem
            )
            ev.final_score = reliability * 0.4 + sem * 0.6
            candidates.append(ev)

    candidates.sort(key=lambda e: e.final_score, reverse=True)
    candidates = remove_duplicate_evidence(candidates)

    return candidates[:MAX_NLI_CANDIDATES]


def verify_web(claim: str) -> Dict:
    if requests is None:
        return {
            "result": "UNKNOWN", "confidence": 0.0,
            "reason": "'requests' missing. Run: pip install requests",
            "evidence": []
        }

    if not TAVILY_API_KEY or TAVILY_API_KEY == "PASTE_YOUR_NEW_TAVILY_KEY_HERE":
        return {
            "result": "UNKNOWN", "confidence": 0.0,
            "reason": "No Tavily API key set. Add it at the top of the file.",
            "evidence": []
        }

    candidates = gather_web_candidates(claim)
    if not candidates:
        return {
            "result": "UNKNOWN", "confidence": 0.0,
            "reason": "No usable evidence", "evidence": []
        }

    nli_ok = apply_nli(claim, candidates)
    return decide(claim, candidates, nli_ok)


# ================================================================
# TOOL 2: DATABASE
# ================================================================

_db_conn = None


def get_db():
    global _db_conn
    if _db_conn is not None:
        return _db_conn

    if DB_PATH:
        uri = Path(DB_PATH).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute(f"CREATE TABLE {DB_TABLE} ({DB_NAME_COL} TEXT, {DB_VALUE_COL} REAL)")
        conn.execute(
            f"INSERT INTO {DB_TABLE} VALUES ('Laptop', 50000), ('Phone', 20000)"
        )
        conn.commit()

    conn.execute("PRAGMA query_only = ON")
    _db_conn = conn
    return conn


_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|"
    r"pragma|replace|vacuum|truncate)\b",
    re.I
)


def sql_tool(query: str) -> Dict:
    q = query.strip().rstrip(";").strip()

    if ";" in q:
        return {"result": "ERROR", "reason": "Only one statement allowed"}
    if not re.match(r"(?is)^select\b", q) or _SQL_FORBIDDEN.search(q):
        return {"result": "ERROR", "reason": "Only SELECT queries are allowed"}

    try:
        rows = get_db().execute(q).fetchmany(50)
        return {"result": "DB_RESULT", "tool": "sql", "rows": rows}
    except Exception as e:
        return {"result": "ERROR", "reason": f"Database error: {e}"}


def verify_db(claim: str) -> Optional[Dict]:
    try:
        rows = get_db().execute(
            f"SELECT {DB_NAME_COL}, {DB_VALUE_COL} FROM {DB_TABLE} "
            f"WHERE instr(lower(?), lower({DB_NAME_COL})) > 0",
            (claim,)
        ).fetchall()
    except Exception as e:
        return {
            "result": "UNKNOWN", "confidence": 0.0,
            "reason": f"Database error: {e}", "evidence": []
        }

    if not rows:
        return None

    numbers = extract_numbers(claim)
    evidences = []

    for name, value in rows:
        sentence = f"{name} = {value}"

        if not numbers:
            label, score = "unknown", 0.0
        elif any(abs(n - float(value)) < 0.01 for n in numbers):
            label, score = "support", 1.0
        else:
            label, score = "contradict", 1.0

        ev = Evidence(
            url=f"db://{DB_TABLE}",
            title=f"Database: {DB_TABLE}",
            sentence=sentence,
            context=sentence,
            source_type="database",
            reliability=SOURCE_WEIGHTS["database"],
            organization=f"db://{DB_TABLE}",
            content_hash=sentence_hash(sentence),
            nli_label=label,
            nli_score=score,
            semantic_score=1.0
        )
        ev.final_score = compute_final_score(ev)
        evidences.append(ev)

    result = decide(claim, evidences, True)
    if not numbers:
        result["reason"] = "Item found in database, but no number to compare"
    return result


# ================================================================
# TOOL 3: DOCUMENTS (RAG)
# ================================================================

def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    sentences = split_sentences(text, min_len=1)
    chunks, current = [], ""

    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current)
            current = s
        else:
            current = (current + " " + s).strip()

    if current:
        chunks.append(current)
    return chunks


class DocumentStore:
    def __init__(self):
        self.chunks: List[Tuple[str, str]] = []
        self._cache: dict = {}

    def add_document(self, doc_id: str, text: str) -> int:
        self.chunks = [c for c in self.chunks if c[0] != doc_id]
        new = chunk_text(text)
        self.chunks.extend((doc_id, c) for c in new)
        self._cache = {}
        return len(new)

    def add_file(self, path: str) -> int:
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            if PdfReader is None:
                raise RuntimeError("PDF needs: pip install pypdf")
            reader = PdfReader(str(p))
            text = " ".join((pg.extract_text() or "") for pg in reader.pages)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
        return self.add_document(p.name, text)

    def search(self, claim: str, k: int = 3) -> List[Tuple[float, str, str]]:
        if not self.chunks:
            return []
        texts = [c[1] for c in self.chunks]
        ranked = rank_texts(claim, texts, top_k=k, cache=self._cache)
        return [(s, self.chunks[i][0], texts[i]) for s, i in ranked]


DOCS = DocumentStore()


def add_document(doc_id: str, text: str) -> int:
    return DOCS.add_document(doc_id, text)


def add_file(path: str) -> int:
    return DOCS.add_file(path)


def verify_docs(claim: str) -> Optional[Dict]:
    hits = DOCS.search(claim)
    if not hits:
        return None

    evidences = []
    for score, doc_id, chunk in hits:
        evidences.append(Evidence(
            url=f"doc://{doc_id}",
            title=f"Document: {doc_id}",
            sentence=chunk,
            context=chunk,
            source_type="document",
            reliability=SOURCE_WEIGHTS["document"],
            organization=f"doc://{doc_id}",
            content_hash=sentence_hash(chunk),
            semantic_score=score,
            final_score=SOURCE_WEIGHTS["document"] * 0.4 + score * 0.6
        ))

    nli_ok = apply_nli(claim, evidences)
    return decide(claim, evidences, nli_ok)


# ================================================================
# TOOL 4: ROUTER
# ================================================================

_DB_RE = re.compile(
    r"\b(database|db|sql|price list|company price|inventory)\b|தரவுத்தளம்",
    re.I
)
_DOC_RE = re.compile(
    r"\b(pdf|my documents?|my notes|my files?|uploaded|in the document)\b"
    r"|ஆவணம்|என் குறிப்பு",
    re.I
)

_PREFIXES = {
    "sql:": "sql",
    "db:": "database",
    "doc:": "rag",
    "web:": "web",
}


def parse_route(text: str) -> Tuple[Optional[str], str]:
    t = text.strip()
    lower = t.lower()
    for prefix, tool in _PREFIXES.items():
        if lower.startswith(prefix):
            return tool, t[len(prefix):].strip()
    return None, t


def route_claim(claim: str) -> str:
    if _DB_RE.search(claim):
        return "database"
    if _DOC_RE.search(claim):
        return "rag"
    return "web"


def verify_with_tool(claim: str, tool: str) -> Dict:
    used = tool
    verification = None

    if tool == "database":
        verification = verify_db(claim)
        if verification is None:
            used = "web (no matching item in database)"
    elif tool == "rag":
        verification = verify_docs(claim)
        if verification is None:
            used = "web (no relevant document found)"

    if verification is None:
        verification = verify_web(claim)
        if tool == "web":
            used = "web"

    verification["tool"] = used
    return verification


# ================================================================
# FINAL RESULT
# ================================================================

def build_final_result(claim: str, verification: Dict) -> Dict:
    evidence = verification.get("evidence", [])

    support = [e for e in evidence if e.nli_label == "support"]
    contradict = [e for e in evidence if e.nli_label == "contradict"]

    def fmt(e):
        return {
            "text": e.sentence,
            "source": e.url,
            "score": round(e.final_score, 3)
        }

    return {
        "claim": claim,
        "result": verification.get("result", "UNKNOWN"),
        "probability": round(verification.get("confidence", 0.0) * 100, 2),
        "reason": verification.get("reason", ""),
        "tool": verification.get("tool", ""),
        "source_independence": verification.get("source_independence", 0.0),
        "supporting_evidence": [fmt(e) for e in support],
        "contradicting_evidence": [fmt(e) for e in contradict],
        "answer": [e.sentence for e in support[:3]],
    }


# ================================================================
# CLARITY AI MAIN
# ================================================================

def clarity_ai(text: str) -> Dict:
    forced_tool, body = parse_route(text)

    if forced_tool == "sql":
        return sql_tool(body)

    claims = decompose_claims(body)
    if not claims:
        return {"result": "UNKNOWN", "reason": "Empty claim"}

    results = []

    for atomic in claims:
        if is_question(atomic):
            results.append({
                "claim": atomic,
                "result": "NOT_A_CLAIM",
                "probability": 0.0,
                "reason": (
                    "This is a question. Give a statement to verify, "
                    "e.g. 'Paris is the capital of France.'"
                ),
                "tool": "",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "answer": []
            })
            continue

        tool = forced_tool or route_claim(atomic)
        verification = verify_with_tool(atomic, tool)
        results.append(build_final_result(atomic, verification))

    if len(results) == 1:
        return results[0]

    n = len(results)
    supported = sum(1 for r in results if r["result"] == "SUPPORTED")
    refuted = sum(1 for r in results if r["result"] == "REFUTED")

    if supported == n:
        overall = "SUPPORTED"
    elif refuted == n:
        overall = "REFUTED"
    elif refuted > 0 and supported > 0:
        overall = "MIXED"
    elif refuted > 0:
        overall = "PARTIALLY_REFUTED"
    elif supported > 0:
        overall = "PARTIALLY_SUPPORTED"
    else:
        overall = "NEEDS_REVIEW"

    return {
        "result": overall,
        "probability": min(r["probability"] for r in results),
        "claims": results
    }


# ================================================================
# BENCHMARK
# ================================================================

def run_benchmark():
    add_document("doc1", "The capital of Tamil Nadu is Chennai.")

    tests = [
        ("Paris is the capital of France.", "SUPPORTED"),
        ("Chennai is the capital of France.", "REFUTED"),
        ("db: The Laptop price is 50000.", "SUPPORTED"),
        ("db: The Laptop price is 45000.", "REFUTED"),
        ("doc: Chennai is the capital of Tamil Nadu.", "SUPPORTED"),
        ("சென்னை தமிழ்நாட்டின் தலைநகரம்.", "SUPPORTED"),
    ]

    print("=" * 70)
    print(f"CLARITY AI v{VERSION} BENCHMARK")
    print("=" * 70)

    passed = 0
    for claim, expected in tests:
        try:
            r = clarity_ai(claim)
            got = r.get("result")
            ok = (got == expected)
            passed += ok
            print(f"\nCLAIM   : {claim}")
            print(f"EXPECTED: {expected} | GOT: {got} | "
                  f"{r.get('probability', 'N/A')}% | tool: {r.get('tool', '')}")
            print("PASS" if ok else f"FAIL ({r.get('reason', '')})")
        except Exception as e:
            print(f"\nCLAIM: {claim}\nERROR: {e}")

    print("\n" + "=" * 70)
    print(f"{passed}/{len(tests)} passed")


# ================================================================
# CLI
# ================================================================

HELP = """Commands:
  <claim>                  verify a claim (auto-routed)
  db: <claim>              force database    doc: <claim>   force documents
  web: <claim>             force web         sql: SELECT ...  read-only SQL
  /doc <id> <text>         add a document    /file <path>   add .txt/.md/.pdf
  /bench                   run benchmark     EXIT           quit"""

if __name__ == "__main__":

    print("=" * 70)
    print(f"CLARITY AI v{VERSION}")
    print("Hallucination-Resistant Verifier")
    print("=" * 70)
    print(HELP)
    print()

    if not TAVILY_API_KEY or TAVILY_API_KEY == "PASTE_YOUR_NEW_TAVILY_KEY_HERE":
        print("[!] No Tavily API key set -> web checks will return UNKNOWN.")
        print()

    while True:
        try:
            line = input("CLAIM > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not line:
            continue

        cmd = line.lower()

        if cmd == "exit":
            break

        if cmd == "/bench":
            run_benchmark()
            continue

        if cmd.startswith("/doc "):
            parts = line.split(None, 2)
            if len(parts) < 3:
                print("Usage: /doc <id> <text>")
            else:
                print(f"Added {add_document(parts[1], parts[2])} chunk(s).")
            continue

        if cmd.startswith("/file "):
            try:
                print(f"Added {add_file(line[6:].strip())} chunk(s).")
            except Exception as e:
                print("ERROR:", e)
            continue

        print("\nChecking...\n")

        try:
            print(json.dumps(clarity_ai(line), indent=2, ensure_ascii=False))
        except Exception as e:
            print("ERROR:", e)

        print("\n" + "-" * 70)
