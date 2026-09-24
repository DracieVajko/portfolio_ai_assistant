from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import feedparser
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-source news params. Precedence (highest first):
# 1. explicit call-site args, 2. settings market_regime news dict,
# 3. built-in defaults below (kept for backward compatibility).
# settings keys: max_age_hours, min_relevance_score, max_articles_per_symbol.
# ---------------------------------------------------------------------------
DEFAULT_MAX_AGE_HOURS = 48
DEFAULT_MIN_RELEVANCE = 60
DEFAULT_MAX_ARTICLES = 5

NEWS_CACHE_TTL_SECONDS = 21600
NEWS_CACHE_FILENAME = "news_google_rss.json"

_MACRO_KEYS = ("SPY", "BTC", "TRUMP")

MACRO_QUERY_MAP = {
    "SPY": ["S&P 500 market outlook", "S&P 500 Fed rate outlook"],
    "BTC": ["Bitcoin price outlook", "Bitcoin ETF flows"],
    "TRUMP": ["Trump tariffs trade policy", "Trump economy markets", "Trump regulation stocks"],
}

_POS_KW = [
    "beat", "raise", "upgrade", "strong", "growth", "record", "bullish",
    "outperform", "buy", "surge", "rally", "gain", "profit", "positive",
    "optimistic", "confident",
]
_NEG_KW = [
    "miss", "cut", "downgrade", "weak", "decline", "loss", "bearish",
    "underperform", "sell", "risk", "fall", "drop", "crash",
    "plunge", "warn", "concern", "negative", "pessimistic",
]

_FILE_LOCK = threading.Lock()


def _news_cache_path():
    override = os.getenv("NEWS_CACHE_PATH", "").strip()
    if override:
        return Path(override)
    try:
        root = Path(__file__).resolve().parents[2]
        return root / "data" / "cache" / NEWS_CACHE_FILENAME
    except Exception:
        return Path("data") / "cache" / NEWS_CACHE_FILENAME


def _normalize_query(query):
    return re.sub(r"\s+", " ", str(query or "").strip().lower())


normalize_query = _normalize_query
def resolve_news_params(settings=None, settings_dict=None, **overrides):
    """Single source for news thresholds.

    Precedence: explicit overrides, settings market_regime news, defaults.
    """
    mr_news = {}
    src = settings_dict if settings_dict is not None else settings
    try:
        if src is not None:
            if isinstance(src, dict):
                mr = src.get("market_regime", {}) or {}
                if isinstance(mr, dict):
                    mr_news = dict(mr.get("news", {}) or {})
                if not mr_news and isinstance(src.get("news"), dict):
                    mr_news = dict(src.get("news") or {})
            else:
                mr = getattr(src, "market_regime", None)
                if isinstance(mr, dict):
                    mr_news = dict(mr.get("news", {}) or {})
                elif mr is not None:
                    mr_news = dict(getattr(mr, "news", {}) or {})
                if not mr_news and hasattr(src, "__dict__"):
                    try:
                        d = dict(getattr(src, "__dict__", {}) or {})
                        mr2 = d.get("market_regime", {}) or {}
                        if isinstance(mr2, dict):
                            mr_news = dict(mr2.get("news", {}) or {})
                    except Exception:
                        pass
    except Exception:
        mr_news = {}
    max_age = overrides.get("max_age_hours", mr_news.get("max_age_hours", DEFAULT_MAX_AGE_HOURS))
    min_rel = overrides.get("min_relevance", mr_news.get("min_relevance_score", mr_news.get("min_relevance", DEFAULT_MIN_RELEVANCE)))
    max_art = overrides.get("max_articles", mr_news.get("max_articles_per_symbol", mr_news.get("max_articles", DEFAULT_MAX_ARTICLES)))
    try:
        max_age = int(max_age)
    except (TypeError, ValueError):
        max_age = DEFAULT_MAX_AGE_HOURS
    try:
        min_rel = int(min_rel)
    except (TypeError, ValueError):
        min_rel = DEFAULT_MIN_RELEVANCE
    try:
        max_art = int(max_art)
    except (TypeError, ValueError):
        max_art = DEFAULT_MAX_ARTICLES
    return {"max_age_hours": max_age, "min_relevance": min_rel, "max_articles": max_art}


def _item_to_cache_dict(item):
    return {
        "title": item.title, "url": item.url, "source": item.source,
        "published_dt": item.published_dt.isoformat(), "published_str": item.published_str,
        "relevance_score": item.relevance_score, "content_hash": item.content_hash,
        "query": item.query,
    }


def _item_from_cache_dict(d):
    try:
        dt = datetime.fromisoformat(str(d.get("published_dt", "")).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return NewsItem(
            title=str(d.get("title", "")), url=str(d.get("url", "")),
            source=str(d.get("source", "unknown")), published_dt=dt,
            published_str=str(d.get("published_str", dt.date().isoformat())),
            relevance_score=int(d.get("relevance_score", 0) or 0),
            content_hash=str(d.get("content_hash", "")), query=str(d.get("query", "")),
        )
    except Exception:
        return None


def _clone_items(items):
    return [replace(it) for it in items]


def _read_file_cache():
    path = _news_cache_path()
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _load_file_cache_entry(normalized_query):
    with _FILE_LOCK:
        data = _read_file_cache()
        entry = data.get(normalized_query)
        if not isinstance(entry, dict):
            return None
        try:
            fetched_at = datetime.fromisoformat(str(entry.get("fetched_at", "")).replace("Z", "+00:00"))
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        except Exception:
            return None
        age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
        if age > NEWS_CACHE_TTL_SECONDS:
            return None
        raw_items = entry.get("items", [])
        if not isinstance(raw_items, list):
            return None
        items = [it for d in raw_items if isinstance(d, dict) for it in [_item_from_cache_dict(d)] if it is not None]
        return items if items else None


def _save_file_cache_entry(normalized_query, items):
    path = _news_cache_path()
    try:
        with _FILE_LOCK:
            data = _read_file_cache()
            data[normalized_query] = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "items": [_item_to_cache_dict(it) for it in items],
            }
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("News file cache save failed for %r: %s", normalized_query, exc)


def clear_news_file_cache():
    """Remove the JSON file cache (tests / forced refresh)."""
    try:
        with _FILE_LOCK:
            path = _news_cache_path()
            if path.exists():
                path.unlink()
    except Exception:
        pass


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published_dt: datetime
    published_str: str
    relevance_score: int
    content_hash: str
    query: str


class StrictNewsFetcher:
    """
    Fetches news with STRICT temporal validation.
    - Parses actual publication date (not feed date)
    - Deduplicates by content hash across ALL queries
    - Filters by relevance score
    - Returns only verified recent items
    """

    def __init__(
        self,
        max_age_hours=48,
        min_relevance=60,
        timeout=8,
        max_articles_per_symbol=5,
        enable_file_cache=True,
        cache_path=None,
    ):
        self.max_age = timedelta(hours=max_age_hours)
        self.max_age_hours = max_age_hours
        self.min_relevance = min_relevance
        self.timeout = timeout
        self.max_articles_per_symbol = max_articles_per_symbol
        self.enable_file_cache = enable_file_cache
        self._cache_path_override = Path(cache_path) if cache_path else None
        self._seen_hashes = set()
        self._lock = threading.Lock()
        self._query_cache = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "PortfolioAI/2.0"})

    @classmethod
    def from_settings(cls, settings=None, **overrides):
        """Build a fetcher from settings market_regime news (see precedence note)."""
        params = resolve_news_params(settings, **overrides)
        timeout = overrides.get("timeout", 8)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 8
        return cls(
            max_age_hours=params["max_age_hours"],
            min_relevance=params["min_relevance"],
            timeout=timeout,
            max_articles_per_symbol=params["max_articles"],
        )

    def fetch_for_symbol(
        self,
        symbol,
        name,
        limit=5,
        extra_queries=None,
        sector=None,
    ):
        """Fetch and strictly filter news for a symbol."""
        queries = self._build_queries(symbol, name, extra_queries, sector=sector)

        all_items = []
        for query in queries:
            items = self._fetch_google_news(query)
            for item in items:
                item.query = query
            all_items.extend(items)

        # Deduplicate by content hash (global across all queries)
        unique_items = self._deduplicate(all_items)

        # Strict temporal filter
        cutoff = datetime.now(timezone.utc) - self.max_age
        recent_items = [item for item in unique_items if item.published_dt >= cutoff]

        # Relevance scoring
        scored = self._score_relevance(recent_items, symbol, name)
        filtered = [item for item in scored if item.relevance_score >= self.min_relevance]

        # Sort by relevance * recency
        filtered.sort(key=lambda x: (x.relevance_score, x.published_dt), reverse=True)

        return filtered[:limit]

    def fetch_market_news(
        self,
        queries,
        limit_per_query=3,
    ):
        """Fetch news for multiple market-wide queries."""
        results = {}
        for query in queries:
            items = self._fetch_google_news(query)
            for item in items:
                item.query = query
            # Deduplicate globally
            unique = self._deduplicate(items)
            cutoff = datetime.now(timezone.utc) - self.max_age
            recent = [item for item in unique if item.published_dt >= cutoff]
            scored = self._score_generic_relevance(recent)
            filtered = [item for item in scored if item.relevance_score >= self.min_relevance]
            filtered.sort(key=lambda x: (x.relevance_score, x.published_dt), reverse=True)
            results[query] = filtered[:limit_per_query]
        return results

    def _build_queries(
        self,
        symbol,
        name,
        extra_queries,
        sector=None,
    ):
        base = [
            f"\"{name}\" {symbol} stock earnings",
            f"\"{name}\" {symbol} guidance",
            f"{symbol} stock price target analyst",
            f"{symbol} earnings call transcript",
        ]
        if sector:
            try:
                from investment_engine.research.sector_templates import build_dynamic_search_queries
                base.extend(build_dynamic_search_queries(symbol, sector))
            except Exception as exc:
                logger.debug("Dynamic sector queries unavailable for %s/%s: %s", symbol, sector, exc)
        if extra_queries:
            base.extend(extra_queries)
        return base

    def _fetch_google_news(self, query):
        nq = _normalize_query(query)
        with self._lock:
            cached = self._query_cache.get(nq)
            if cached is not None:
                return _clone_items(cached)
        if self.enable_file_cache:
            file_items = _load_file_cache_entry(nq)
            if file_items is not None:
                with self._lock:
                    self._query_cache[nq] = _clone_items(file_items)
                return _clone_items(file_items)
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as e:
            logger.warning("News fetch failed for '%s': %s", query, e)
            return []

        items = []
        for entry in feed.entries:
            try:
                pub_dt = self._parse_date(entry)
                if not pub_dt:
                    continue

                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                source = entry.get("source", {}).get("title", "unknown").strip()

                if title and link:
                    content_hash = hashlib.md5(f"{title}{link}".encode()).hexdigest()[:16]
                    items.append(
                        NewsItem(
                            title=title,
                            url=link,
                            source=source,
                            published_dt=pub_dt,
                            published_str=pub_dt.date().isoformat(),
                            relevance_score=0,
                            content_hash=content_hash,
                            query=query,
                        )
                    )
            except Exception as e:
                logger.debug("Failed parsing news entry: %s", e)
                continue
        with self._lock:
            self._query_cache[nq] = _clone_items(items)
        if self.enable_file_cache and items:
            _save_file_cache_entry(nq, items)
        return items

    def _parse_date(self, entry):
        """Parse publication date from multiple possible fields."""
        for field in ("published_parsed", "updated_parsed", "created_parsed"):
            if hasattr(entry, field) and getattr(entry, field):
                try:
                    return datetime(*getattr(entry, field)[:6], tzinfo=timezone.utc)
                except Exception:
                    continue
        for field in ("published", "updated", "created"):
            val = entry.get(field, "")
            if val:
                try:
                    return parsedate_to_datetime(val).astimezone(timezone.utc)
                except Exception:
                    continue
        return None

    def _deduplicate(self, items):
        unique = []
        with self._lock:
            for item in items:
                if item.content_hash not in self._seen_hashes:
                    self._seen_hashes.add(item.content_hash)
                    unique.append(item)
        return unique

    def _score_relevance(
        self,
        items,
        symbol,
        name,
    ):
        symbol_upper = symbol.upper()
        name_upper = name.upper()

        keywords_high = {
            symbol_upper: 30,
            name_upper: 30,
            "earnings": 25,
            "guidance": 25,
            "upgrade": 20,
            "downgrade": 20,
            "target": 15,
            "beat": 20,
            "miss": 20,
            "raise": 15,
            "cut": 15,
        }
        keywords_med = {
            "revenue": 10,
            "profit": 10,
            "margin": 10,
            "dividend": 10,
            "buyback": 10,
            "acquisition": 15,
            "merger": 15,
            "partnership": 10,
        }
        keywords_low = {
            "stock": 5,
            "shares": 5,
            "market": 3,
            "trading": 3,
            "investor": 3,
        }

        for item in items:
            text = f"{item.title} {item.source}".upper()
            score = 0
            for kw, val in keywords_high.items():
                if kw in text:
                    score += val
            for kw, val in keywords_med.items():
                if kw in text:
                    score += val
            for kw, val in keywords_low.items():
                if kw in text:
                    score += val
            # Penalize generic/noisy sources
            source_upper = item.source.upper()
            if source_upper in {"STOCKTWITS", "REDDIT", "TWITTER", "X.COM"}:
                score = int(score * 0.7)
            elif source_upper in {"YAHOO FINANCE", "GOOGLE FINANCE"}:
                score = int(score * 0.9)
            item.relevance_score = min(score, 100)
        return items

    def _score_generic_relevance(self, items):
        keywords = {
            "fed": 20,
            "federal reserve": 20,
            "inflation": 15,
            "interest rate": 15,
            "cpi": 15,
            "ppi": 15,
            "gdp": 15,
            "unemployment": 10,
            "earnings": 15,
            "guidance": 15,
            "recession": 15,
            "rally": 10,
            "selloff": 10,
            "crash": 10,
        }
        for item in items:
            text = f"{item.title} {item.source}".upper()
            score = 0
            for kw, val in keywords.items():
                if kw in text:
                    score += val
            item.relevance_score = min(score, 100)
        return items

    def reset_run_state(self):
        """Clear per-run dedupe and query caches (call at each report run start)."""
        with self._lock:
            self._seen_hashes.clear()
            self._query_cache.clear()

    def clear_cache(self):
        """Clear deduplication cache (call between report runs)."""
        self.reset_run_state()

    # -----------------------------------------------------------------
    # Slovak news sites (sme.sk, pravda.sk, aktuality.sk)
    # -----------------------------------------------------------------

    def fetch_slovak_news(self, symbols: list[str] | None = None,
                          companies: list[str] | None = None) -> list[NewsItem]:
        """Fetch news from Slovak news sites (sme.sk, pravda.sk, aktuality.sk).

        Filters by symbols/company names if provided.
        """
        import feedparser as _fp
        cutoff = datetime.now(timezone.utc) - self.max_age
        all_items: list[NewsItem] = []
        _slovak_sites = [
            ("SME", "https://www.sme.sk/rss/aktuality/"),
            ("Pravda", "https://www.pravda.sk/rss/"),
            ("Aktuality", "https://www.aktuality.sk/rss/"),
        ]
        for source_name, rss_url in _slovak_sites:
            try:
                resp = self._session.get(rss_url, timeout=self.timeout)
                resp.raise_for_status()
                feed = _fp.parse(resp.content)
            except Exception as exc:
                logger.debug("Slovak news fetch failed for %s: %s", source_name, exc)
                continue
            for entry in feed.entries:
                try:
                    pub_dt = self._parse_date(entry)
                    if pub_dt is None or pub_dt < cutoff:
                        continue
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "").strip()
                    if not title or not link:
                        continue
                    if symbols or companies:
                        text = title.upper()
                        if symbols and not any(s.upper() in text for s in symbols):
                            continue
                        if companies and not any(c.upper() in text for c in companies):
                            continue
                    all_items.append(NewsItem(
                        title=title, url=link, source=source_name,
                        published_dt=pub_dt,
                        published_str=pub_dt.strftime("%Y-%m-%d %H:%M"),
                        relevance_score=0,
                        content_hash=hashlib.md5(f"{title}{link}".encode()).hexdigest()[:16],
                        query="slovak_news",
                    ))
                except Exception:
                    continue
        return all_items

    # -----------------------------------------------------------------
    # Reddit search
    # -----------------------------------------------------------------

    def fetch_reddit_search(self, queries: list[str], limit_per_query: int = 5
                            ) -> list[NewsItem]:
        """Search Reddit for relevant news and discussions."""
        cutoff = datetime.now(timezone.utc) - self.max_age
        all_items: list[NewsItem] = []
        for query in queries:
            try:
                url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&limit={limit_per_query}&t=day"
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                continue
            for post in data.get("data", {}).get("children", []):
                try:
                    d = post.get("data", {})
                    title = d.get("title", "").strip()
                    permalink = d.get("permalink", "")
                    link = f"https://reddit.com{permalink}" if permalink else d.get("url", "")
                    created_utc = d.get("created_utc", 0)
                    pub_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                    if pub_dt < cutoff:
                        continue
                    source = f"r/{d.get('subreddit', '')}"
                    all_items.append(NewsItem(
                        title=title, url=link, source=source,
                        published_dt=pub_dt,
                        published_str=pub_dt.strftime("%Y-%m-%d %H:%M"),
                        relevance_score=0,
                        content_hash=hashlib.md5(f"{title}{link}".encode()).hexdigest()[:16],
                        query=query,
                    ))
                except Exception:
                    continue
        return all_items

    # -----------------------------------------------------------------
    # Trump tracking with categorized keywords
    # -----------------------------------------------------------------

    def fetch_trump_tracking(self) -> dict[str, list[NewsItem]]:
        """Fetch Trump-specific news categorized by topic area.

        Categories: tariffs, greenland, mining, rare_earths, regulation, economy.
        """
        _trump_categories = {
            "tariffs": ["Trump tariffs trade policy", "Trump import duty stock market"],
            "greenland": ["Trump Greenland rare earth minerals", "Trump Arctic strategic minerals"],
            "mining": ["Trump mining drilling policy", "Trump mineral extraction stocks"],
            "rare_earths": ["Trump rare earth critical minerals", "Trump strategic minerals supply chain"],
            "regulation": ["Trump regulation deregulation stocks", "Trump SEC policy market"],
            "economy": ["Trump economy markets fiscal policy", "Trump tax trade GDP inflation"],
        }
        results: dict[str, list[NewsItem]] = {}
        for category, queries in _trump_categories.items():
            items = self.fetch_market_news(queries, limit_per_query=3)
            results[category] = items
        return results


def reset_news_state(fetcher=None):
    """Per-report-run reset hook: clears _seen_hashes and per-run query cache.

    Call at run start: reset_news_state(fetcher). None is a no-op
    (file cache is TTL-based, not per-run).
    """
    if fetcher is not None:
        try:
            fetcher.reset_run_state()
        except Exception:
            pass


def fetch_macro_news(
    fetcher=None,
    symbols=("SPY", "BTC", "TRUMP"),
    limit_per_query=3,
):
    """Macro path: route SPY/BTC/TRUMP through fetch_market_news."""
    fetcher = fetcher or StrictNewsFetcher()
    out = {}
    for sym in symbols:
        key = str(sym or "").strip().upper()
        queries = MACRO_QUERY_MAP.get(key, [key])
        try:
            res = fetcher.fetch_market_news(queries, limit_per_query=limit_per_query)
        except Exception as exc:
            logger.debug("Macro news fetch failed for %s: %s", key, exc)
            continue
        merged = []
        for q_items in res.values():
            merged.extend(q_items)
        merged.sort(key=lambda x: (x.relevance_score, x.published_dt), reverse=True)
        if merged:
            out[key] = merged[:limit_per_query]
    return out


def analyze_news_sentiment(news_items):
    """Aggregate sentiment from news headlines."""
    if not news_items:
        return {
            "sentiment": "NEUTRAL",
            "score": 50,
            "count": 0,
            "positive_signals": 0,
            "negative_signals": 0,
            "key_topics": [],
            "latest_headline": None,
        }

    positive_kw = [
        "beat", "raise", "upgrade", "strong", "growth", "record", "bullish",
        "outperform", "buy", "surge", "rally", "gain", "profit", "positive",
        "optimistic", "confident",
    ]
    negative_kw = [
        "miss", "cut", "downgrade", "weak", "decline", "loss", "bearish",
        "underperform", "sell", "risk", "fall", "drop", "crash",
        "plunge", "warn", "concern", "negative", "pessimistic",
    ]

    pos_count = neg_count = 0
    for item in news_items:
        title = item.title.lower()
        pos_count += sum(1 for kw in positive_kw if kw in title)
        neg_count += sum(1 for kw in negative_kw if kw in title)

    total = pos_count + neg_count
    if total == 0:
        sentiment_score = 50
    else:
        sentiment_score = 50 + (pos_count - neg_count) / total * 50

    # Extract key topics (simple noun phrase extraction)
    topics = extract_key_topics(news_items)

    return {
        "sentiment": "POSITIVE" if sentiment_score > 60 else "NEGATIVE" if sentiment_score < 40 else "NEUTRAL",
        "score": round(sentiment_score, 1),
        "count": len(news_items),
        "positive_signals": pos_count,
        "negative_signals": neg_count,
        "key_topics": topics,
        "latest_headline": news_items[0].title if news_items else None,
    }


def extract_key_topics(news_items, max_topics=5):
    """Extract key topics from news headlines."""
    # Simple approach: find capitalized words/phrases that appear multiple times
    from collections import Counter
    import re

    all_text = " ".join(item.title for item in news_items)
    # Find potential entities (capitalized words, 2-3 word phrases)
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", all_text)
    # Filter common words
    stopwords = {
        "The", "A", "An", "And", "Or", "But", "In", "On", "At", "To", "For",
        "Of", "With", "By", "From", "As", "Is", "Was", "Are", "Were", "Be",
        "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did", "Will",
        "Would", "Could", "Should", "May", "Might", "Must", "Can", "Stock",
        "Stocks", "Market", "Markets", "Trading", "Trader", "Investor", "Investors",
        "Company", "Companies", "Business", "Report", "Reports", "News", "Analysis",
    }
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    counted = Counter(filtered).most_common(max_topics)
    return [topic for topic, _ in counted]


def _item_title(item):
    if isinstance(item, dict):
        return str(item.get("title", "") or "")
    return str(getattr(item, "title", "") or "")


def _item_url(item):
    if isinstance(item, dict):
        return str(item.get("url", "") or "")
    return str(getattr(item, "url", "") or "")


def _item_source(item):
    if isinstance(item, dict):
        return str(item.get("source", "") or "unknown")
    return str(getattr(item, "source", "") or "unknown")


def _item_published_dt(item):
    if isinstance(item, dict):
        val = item.get("published_dt", item.get("published"))
        if isinstance(val, datetime):
            return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
        try:
            parsed = parsedate_to_datetime(str(val or "").strip())
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
        try:
            parsed = datetime.fromisoformat(str(val or "").strip().replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None
    val2 = getattr(item, "published_dt", None)
    if isinstance(val2, datetime):
        return val2 if val2.tzinfo else val2.replace(tzinfo=timezone.utc)
    return None


def _format_age(published_dt, now=None):
    if published_dt is None:
        return "unknown age"
    ref = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    dt = published_dt if published_dt.tzinfo else published_dt.replace(tzinfo=timezone.utc)
    secs = max((ref - dt).total_seconds(), 0)
    if secs < 3600:
        return f"{max(int(secs // 60), 1)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def _annotate_news_item(item, now=None):
    """Lite per-item enrichment: domain, tier, sentiment/confidence, age."""
    title = _item_title(item)
    url = _item_url(item)
    source = _item_source(item)
    try:
        domain = (urlparse(url).netloc or "").lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        domain = domain or "unknown"
    except Exception:
        domain = "unknown"
    try:
        tier = source_tier(source, url)
    except Exception:
        tier = 3
    low = title.lower()
    pos = sum(1 for kw in _POS_KW if kw in low)
    neg = sum(1 for kw in _NEG_KW if kw in low)
    total = pos + neg
    if total == 0:
        sentiment = "NEUTRAL"
        confidence = 0.0
    elif pos > neg:
        sentiment = "POSITIVE"
        confidence = abs(pos - neg) / total
    elif neg > pos:
        sentiment = "NEGATIVE"
        confidence = abs(pos - neg) / total
    else:
        sentiment = "NEUTRAL"
        confidence = 0.0
    pub_dt = _item_published_dt(item)
    age = _format_age(pub_dt, now)
    return {
        "domain": domain, "tier": tier, "sentiment": sentiment,
        "confidence": round(float(confidence), 4), "age": age, "published_dt": pub_dt,
    }


def format_headline(item, now=None):
    """One-line headline: title (domain, tierN, SENTIMENT conf, age)."""
    ann = _annotate_news_item(item, now)
    title = _item_title(item).strip()
    return f"{title} ({ann['domain']}, tier{ann['tier']}, {ann['sentiment']} {ann['confidence']:.2f}, {ann['age']})"


def _sort_tier_recency(items):
    def _key(it):
        ann = _annotate_news_item(it)
        dt = ann.get("published_dt")
        ts = dt.timestamp() if isinstance(dt, datetime) else 0.0
        return (int(ann.get("tier", 3)), -ts)
    return sorted(items, key=_key)


def news_headlines_block(news_by_symbol, per_symbol=3, now=None):
    """Headline block owned by this module: tier-then-recency, capped, URL kept."""
    per_symbol = max(1, min(int(per_symbol or 3), 3))
    lines = []
    for sym in sorted((news_by_symbol or {}).keys()):
        entries = list((news_by_symbol or {}).get(sym) or [])
        entries = [e for e in entries if _item_url(e).strip()]
        if not entries:
            continue
        for item in _sort_tier_recency(entries)[:per_symbol]:
            lines.append(f"- {sym}: {format_headline(item, now)} -- {_item_url(item).strip()}")
    return "\n".join(lines) if lines else "None"


headlines_block = news_headlines_block


def symbol_sentiment_line(symbol, items):
    """One-line sentiment for prompt injection.

    Example: Sentiment: POSITIVE 72 (5+/1-, topics: Blackwell, guidance).
    """
    norm = []
    now = datetime.now(timezone.utc)
    for it in items or []:
        if isinstance(it, NewsItem):
            norm.append(it)
        elif isinstance(it, dict):
            dt = _item_published_dt(it) or now
            norm.append(
                NewsItem(
                    title=str(it.get("title", "")), url=str(it.get("url", "")),
                    source=str(it.get("source", "unknown")), published_dt=dt,
                    published_str=dt.date().isoformat(),
                    relevance_score=int(it.get("relevance_score", 0) or 0),
                    content_hash=str(it.get("content_hash", "")),
                    query=str(it.get("query", "")),
                )
            )
    agg = analyze_news_sentiment(norm)
    topics = agg.get("key_topics", []) or []
    topics_s = ", ".join(topics) if topics else "n/a"
    try:
        score_n = int(round(float(agg.get("score", 50))))
    except (TypeError, ValueError):
        score_n = 50
    return (
        f"Sentiment: {agg.get('sentiment', 'NEUTRAL')} {score_n} "
        f"({agg.get('positive_signals', 0)}+/{agg.get('negative_signals', 0)}-, topics: {topics_s})"
    )


def validate_idea_with_news(
    ticker,
    name="",
    *,
    fetcher=None,
    limit=3,
    min_relevance=30,
    max_age_hours=72,
    names=None,
):
    """Validate one discovery idea against recent news.

    main.py hookup signature:
        validate_idea_with_news(ticker, name="", *, fetcher=None, limit=3,
                                min_relevance=30, max_age_hours=72) -> dict
    Returns ticker, name, supported, evidence, urls, url, confidence, items.
    Matching uses _record_links (symbol-key plus company-name), never a bare
    literal TICKER-substring scan.
    """
    sym = str(ticker or "").strip().upper()
    company = str(name or sym)
    if not sym or sym == "UNKNOWN":
        return {"ticker": sym, "name": company, "supported": False, "evidence": [],
                "urls": [], "url": None, "confidence": "none", "items": []}
    fetcher = fetcher or StrictNewsFetcher(min_relevance=min_relevance)
    try:
        fetched = fetcher.fetch_for_symbol(sym, company, limit=max(1, min(int(limit or 3), 3)))
    except Exception as exc:
        logger.debug("validate_idea_with_news fetch failed for %s: %s", sym, exc)
        return {"ticker": sym, "name": company, "supported": False, "evidence": [],
                "urls": [], "url": None, "confidence": "none", "items": []}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    universe_names = dict(names or {})
    universe_names.setdefault(sym, company)
    kept = []
    for item in fetched or []:
        try:
            rel = int(getattr(item, "relevance_score", 0) or 0)
        except (TypeError, ValueError):
            rel = 0
        if rel < min_relevance:
            continue
        pub = getattr(item, "published_dt", None)
        if not isinstance(pub, datetime) or pub < cutoff:
            continue
        links = _record_links(
            getattr(item, "title", ""), sym,
            {sym}, set(), set(), set(), universe_names,
        )
        if links:
            kept.append(item)
    kept = _sort_tier_recency(kept)[: max(1, min(int(limit or 3), 3))]
    evidence = [k.title for k in kept]
    urls = [k.url for k in kept if k.url]
    return {
        "ticker": sym, "name": company, "supported": bool(kept),
        "evidence": evidence, "urls": urls, "url": urls[0] if urls else None,
        "confidence": "high" if len(kept) >= 2 else ("medium" if kept else "none"),
        "items": news_items_to_dict(kept),
    }


def news_items_to_dict(items):
    """Convert NewsItem list to dict for JSON serialization (enriched)."""
    out = []
    for item in items:
        ann = _annotate_news_item(item)
        out.append(
            {
                "title": item.title, "url": item.url, "source": item.source,
                "published": item.published_str,
                "published_dt": item.published_dt.isoformat(),
                "relevance_score": item.relevance_score,
                "content_hash": item.content_hash, "query": item.query,
                "domain": ann["domain"], "tier": ann["tier"],
                "item_sentiment": ann["sentiment"],
                "item_confidence": ann["confidence"], "age": ann["age"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Decision-news validation (daily brief): filtering, scoring, quality ranking,
# deduplication. Only fetched article records are used, never LLM metadata.
# ---------------------------------------------------------------------------

_TIER_1_SOURCES = frozenset({
    "REUTERS", "ASSOCIATED PRESS", "AP", "BLOOMBERG", "WALL STREET JOURNAL",
    "FINANCIAL TIMES", "CNBC",
})
_OFFICIAL_RE = None
_INDEX_CLICKBAIT_RE = None


def source_tier(source, url=""):
    """Quality tier: 0 official IR/regulatory/exchange, 1 wire, 2 pub, 3 other."""
    global _OFFICIAL_RE
    if _OFFICIAL_RE is None:
        _OFFICIAL_RE = re.compile(
            r"investor.?relations|\bir\b|sec\.gov|exchange|regulatory|regulatory filing", re.IGNORECASE)
    s = (source or "").strip()
    if _OFFICIAL_RE.search(f"{s} {url or ''}"):
        return 0
    if s.upper() in _TIER_1_SOURCES:
        return 1
    if s and s.lower() not in ("unknown", "google news"):
        return 2
    return 3


def canonical_url(url):
    """Canonical URL for dedupe: lowercase scheme/host/path, no query/fragment."""
    from urllib.parse import urlparse
    try:
        parts = urlparse((url or "").strip())
        if not parts.scheme.startswith("http") or not parts.netloc:
            return ""
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path.rstrip('/') or '/'}"
    except Exception:
        return ""


def news_dedupe_key(title, publisher, pub_date, url=None):
    """Dedupe key: canonical URL when available, else title+publisher+date."""
    curl = canonical_url(url)
    if curl:
        return f"url:{curl}"
    norm = lambda v: re.sub(r"\s+", " ", str(v or "").strip().lower())
    return f"meta:{norm(title)}|{norm(publisher)}|{norm(pub_date)}"


def _is_index_clickbait(title):
    global _INDEX_CLICKBAIT_RE
    if _INDEX_CLICKBAIT_RE is None:
        _INDEX_CLICKBAIT_RE = re.compile(
            r"s&\s?p\s*500|\bspy\b|\bvoo\b|\bqqq\b|dow\s*jones|index\s*(fund|etf)", re.IGNORECASE)
    return bool(_INDEX_CLICKBAIT_RE.search(title or ""))


def _record_published_dt(record):
    """Parse a fetched record publication time (datetime or ISO string)."""
    val = (record or {}).get("published_dt", (record or {}).get("published"))
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    try:
        parsed = parsedate_to_datetime(str(val or "").strip())
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(str(val or "").strip().replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _record_links(
    title,
    symbol_key,
    owned,
    constituents,
    watchlist,
    earnings,
    names,
):
    """Held exposures a record links to (explicit references only).

    Single-letter tickers (C, O) link via symbol key only, never title scan.
    """
    links = set()
    key = (symbol_key or "").strip().upper()
    universe = set(owned) | set(constituents) | set(watchlist) | set(earnings)
    if key and key in universe:
        links.add(key)
    upper_title = (title or "").upper()
    for disp in universe:
        if len(disp) >= 2 and re.search(r"\b" + re.escape(disp) + r"\b", upper_title):
            links.add(disp)
    for disp, company in (names or {}).items():
        if disp in universe and company and len(str(company)) >= 5 \
                and str(company).upper() in upper_title:
            links.add(disp)
    return links


def filter_decision_news(
    news_by_symbol,
    *,
    owned=None,
    constituents=None,
    watchlist=None,
    earnings=None,
    names=None,
    relevance_min=30,
    max_age_hours=72,
    now=None,
    macro_keys=("MACRO", "CRYPTO-MACRO", "TRUMP", "SPY", "BTC"),
    include_macro_unlinked=True,
    min_tier_for_unlinked=1,
):
    """Central decision-news filter for the daily brief.

    Keeps fetched records newer than max_age_hours tied to an owned position,
    a held PIE constituent, a watchlist candidate, or an earnings event.
    Macro records (SPY, BTC, TRUMP) are included even without held links
    if from tier 0-1 sources (official/wire).
    """
    owned = {str(x).strip().upper() for x in (owned or set()) if x}
    constituents = {str(x).strip().upper() for x in (constituents or set()) if x}
    watchlist = {str(x).strip().upper() for x in (watchlist or set()) if x}
    earnings = {str(x).strip().upper() for x in (earnings or set()) if x}
    held = set(owned) | set(constituents)
    now_utc = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    if isinstance(news_by_symbol, dict):
        flat = [(k, r) for k, recs in news_by_symbol.items() for r in (recs or [])]
    else:
        flat = [(r.get("symbol_key", r.get("symbol", "")), r) for r in (news_by_symbol or [])]

    best = {}
    for symbol_key, record in flat:
        if not isinstance(record, dict):
            continue
        title = str(record.get("title", "") or "").strip()
        url = str(record.get("url", "") or "").strip()
        source = str(record.get("source", "") or "unknown").strip()
        if not title or not canonical_url(url):
            continue
        try:
            relevance = int(record.get("relevance_score", 0) or 0)
        except (TypeError, ValueError):
            relevance = 0
        if relevance < relevance_min:
            continue
        pub_dt = _record_published_dt(record)
        if pub_dt is None or (now_utc - pub_dt) >= timedelta(hours=max_age_hours):
            continue
        links = _record_links(title, symbol_key, owned, constituents, watchlist, earnings, names or {})
        sym_upper = str(symbol_key or "").strip().upper()
        is_macro = sym_upper in macro_keys
        if not links and is_macro and include_macro_unlinked:
            tier = source_tier(source, url)
            if tier <= min_tier_for_unlinked:
                links = {"MACRO"}
            else:
                continue
        elif not links:
            continue
        tier = source_tier(source, url)
        key = news_dedupe_key(title, source, pub_dt.date().isoformat(), url)
        cand = {"title": title, "url": url, "source": source, "published_dt": pub_dt,
                "tier": tier, "links": sorted(links)}
        prev = best.get(key)
        if prev is None or tier < prev["tier"] or (tier == prev["tier"] and pub_dt > prev["published_dt"]):
            best[key] = cand
    ranked = sorted(best.values(), key=lambda r: (r["tier"], -r["published_dt"].timestamp()))
    out = []
    for r in ranked:
        age_h = (now_utc - r["published_dt"]).total_seconds() / 3600.0
        age = f"{int(age_h)}h ago" if age_h >= 1 else f"{max(int(age_h * 60), 1)}m ago"
        linked = ", ".join(r["links"])
        if r["links"] and set(r["links"]) & set(earnings):
            why = f"Why it matters: earnings window approaching for {linked} held in the portfolio."
        elif len([t for t in r["links"] if t in held]) >= 2:
            why = f"Why it matters: broad-market development touching {linked} held positions."
        elif "MACRO" in r["links"]:
            why = f"Why it matters: macro development relevant to portfolio positioning."
        else:
            why = f"Why it matters: directly references {linked} held in the portfolio."
        out.append({"title": r["title"], "url": r["url"], "source": r["source"],
                    "published": r["published_dt"].strftime("%Y-%m-%d %H:%M UTC"),
                    "age": age, "tier": r["tier"], "links": r["links"], "why": why})
    return out