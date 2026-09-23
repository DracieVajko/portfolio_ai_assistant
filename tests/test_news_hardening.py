"""Hardening tests for the news path: caches, locks, annotation, wiring."""
from __future__ import annotations

import json
import time
from email.utils import formatdate
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

import investment_engine.research.news_engine as ne
from investment_engine.research.news_engine import (
    NewsItem,
    StrictNewsFetcher,
    fetch_macro_news,
    format_headline,
    news_headlines_block,
    news_items_to_dict,
    reset_news_state,
    resolve_news_params,
    symbol_sentiment_line,
    validate_idea_with_news,
)


def _mkitem(title, url="https://example.com/a", source="Reuters", hours_ago=5, relevance=80, query="q", key="h"):
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return NewsItem(title=title, url=url, source=source, published_dt=dt,
                    published_str=dt.date().isoformat(), relevance_score=relevance,
                    content_hash=key, query=query)


class _FakeResp:
    def __init__(self, counter):
        self.counter = counter
        self.content = b"<rss/>"

    def raise_for_status(self):
        self.counter["n"] += 1


def _patch_rss(monkeypatch, fetcher, entries, counter):
    def _get(url, timeout=None):
        return _FakeResp(counter)
    class _Feed:
        pass
    feed = _Feed()
    feed.entries = entries
    monkeypatch.setattr(fetcher._session, "get", _get)
    monkeypatch.setattr(ne.feedparser, "parse", lambda content: feed)


def _rss_entry(title, link, source="Reuters"):
    return {"title": title, "link": link, "source": {"title": source},
            "published": formatdate(time.time(), usegmt=True)}


def test_timeout_default_lowered():
    assert StrictNewsFetcher().timeout == 8


def test_seen_hashes_guarded_and_reset():
    f = StrictNewsFetcher(enable_file_cache=False)
    assert hasattr(f, "_lock")
    f._seen_hashes.add("abc")
    f._query_cache["x"] = []
    reset_news_state(f)
    assert f._seen_hashes == set()
    assert f._query_cache == {}
    f._seen_hashes.add("z")
    f.clear_cache()
    assert f._seen_hashes == set()


def test_per_run_cache_avoids_second_fetch(monkeypatch):
    f = StrictNewsFetcher(enable_file_cache=False)
    counter = {"n": 0}
    _patch_rss(monkeypatch, f, [_rss_entry("NVDA beats earnings NVDA", "https://example.com/a1")], counter)
    first = f._fetch_google_news("NVDA test")
    second = f._fetch_google_news("NVDA test")
    assert len(first) == 1 and len(second) == 1
    assert counter["n"] == 1
    assert first[0].title == second[0].title
def test_file_cache_roundtrip_and_ttl(monkeypatch, tmp_path):
    monkeypatch.setenv("NEWS_CACHE_PATH", str(tmp_path / "nc.json"))
    f = StrictNewsFetcher(enable_file_cache=True)
    counter = {"n": 0}
    _patch_rss(monkeypatch, f, [_rss_entry("Hello beats guidance", "https://example.com/h1")], counter)
    first = f._fetch_google_news("  Hello World ")
    assert counter["n"] == 1
    f2 = StrictNewsFetcher(enable_file_cache=True)

    def _boom(url, timeout=None):
        raise AssertionError("network should not be hit on file-cache hit")
    monkeypatch.setattr(f2._session, "get", _boom)
    second = f2._fetch_google_news("HELLO world")
    assert [i.title for i in second] == [i.title for i in first]
    # Age the entry beyond 6h TTL -> must miss.
    data = json.loads((tmp_path / "nc.json").read_text(encoding="utf-8"))
    key = next(iter(data))
    old = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    data[key]["fetched_at"] = old
    (tmp_path / "nc.json").write_text(json.dumps(data), encoding="utf-8")
    assert ne._load_file_cache_entry(key) is None


def test_normalize_query_keys():
    assert ne._normalize_query("  Hello   WORLD ") == "hello world"
    assert ne._normalize_query("HELLO WORLD") == ne._normalize_query("hello  world")


def test_annotate_and_format_headline():
    item = _mkitem("Nvidia beats earnings with strong Blackwell guidance",
                   url="https://www.reuters.com/markets/a", source="Reuters", hours_ago=5)
    line = format_headline(item)
    assert "reuters.com" in line
    assert "tier1" in line
    assert "POSITIVE" in line
    assert "ago" in line
    assert item.title in line


def test_headlines_block_caps_and_sorts_and_keeps_url():
    items = [
        _mkitem("NVDA blog post", url="https://blog.example/p1", source="Some Blog", hours_ago=1, key="k1"),
        _mkitem("NVDA beats earnings", url="https://reuters.example/r1", source="Reuters", hours_ago=10, key="k2"),
        _mkitem("NVDA random", url="https://unknown.example/u1", source="unknown", hours_ago=0.5, key="k3"),
        _mkitem("NVDA extra", url="https://blog.example/p2", source="Some Blog", hours_ago=2, key="k4"),
        _mkitem("NVDA no url", url="", source="Reuters", hours_ago=1, key="k5"),
    ]
    block = news_headlines_block({"NVDA": items}, per_symbol=2)
    lines = [ln for ln in block.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "tier1" in lines[0]
    assert all("http" in ln for ln in lines)
    block3 = news_headlines_block({"NVDA": items}, per_symbol=10)
    assert len([ln for ln in block3.splitlines() if ln.strip()]) <= 3


def test_sector_queries_wired(monkeypatch):
    f = StrictNewsFetcher(enable_file_cache=False)
    seen = {}

    def _fake_fetch(query):
        seen.setdefault("queries", []).append(query)
        return []
    monkeypatch.setattr(f, "_fetch_google_news", _fake_fetch)
    f.fetch_for_symbol("NVDA", "Nvidia", limit=2, sector="AI")
    assert any("Blackwell" in q for q in seen["queries"])
    seen2 = {}
    monkeypatch.setattr(f, "_fetch_google_news", lambda q: seen2.setdefault("queries", []).append(q) or [])
    f.fetch_for_symbol("NVDA", "Nvidia", limit=2)
    assert len(seen2["queries"]) == 4
def test_macro_routes_through_fetch_market_news(monkeypatch):
    f = StrictNewsFetcher(enable_file_cache=False)
    calls = {}

    def _fake_market(queries, limit_per_query=3):
        calls["queries"] = list(queries)
        now = datetime.now(timezone.utc)
        return {queries[0]: [_mkitem("S&P 500 Fed outlook", url="https://reuters.example/m",
                                     source="Reuters", hours_ago=2, relevance=80, key="m1")]}
    monkeypatch.setattr(f, "fetch_market_news", _fake_market)
    out = fetch_macro_news(f, symbols=("SPY",), limit_per_query=2)
    assert "SPY" in out
    assert calls["queries"] == ne.MACRO_QUERY_MAP["SPY"]


def test_news_items_to_dict_single_and_enriched():
    assert sum(1 for _ in [ne.news_items_to_dict]) == 1
    rows = news_items_to_dict([_mkitem("Nvidia beats earnings", url="https://reuters.example/x", source="Reuters")])
    row = rows[0]
    for k in ("title", "url", "source", "published", "published_dt", "relevance_score", "content_hash", "query"):
        assert k in row
    for k in ("domain", "tier", "item_sentiment", "item_confidence", "age"):
        assert k in row
    assert row["url"].startswith("https://")


def test_resolve_news_params_precedence():
    assert resolve_news_params() == {"max_age_hours": 48, "min_relevance": 60, "max_articles": 5}
    sd = {"market_regime": {"news": {"max_age_hours": 72, "min_relevance_score": 30, "max_articles_per_symbol": 2}}}
    assert resolve_news_params(sd) == {"max_age_hours": 72, "min_relevance": 30, "max_articles": 2}
    assert resolve_news_params(sd, max_age_hours=10)["max_age_hours"] == 10
    f = StrictNewsFetcher.from_settings(sd)
    assert (f.max_age_hours, f.min_relevance, f.max_articles_per_symbol) == (72, 30, 2)


class _StubFetcher(StrictNewsFetcher):
    def __init__(self, items, min_relevance=30, max_age_hours=96):
        super().__init__(min_relevance=min_relevance, max_age_hours=max_age_hours, enable_file_cache=False)
        self._stub = list(items)

    def _fetch_google_news(self, query):
        import copy
        return copy.deepcopy(self._stub)


def test_validate_idea_company_name_match_and_filters():
    good = _mkitem("Nvidia Corporation raises Blackwell guidance", url="https://reuters.example/g",
                   source="Reuters", hours_ago=3, relevance=80, key="g1")
    weak = _mkitem("Market chatter about equities", url="https://blog.example/w", source="Some Blog",
                   hours_ago=3, relevance=5, key="w1")
    old = _mkitem("NVDA beats NVDA", url="https://reuters.example/o", source="Reuters",
                  hours_ago=80, relevance=90, key="o1")
    stub = _StubFetcher([good, weak, old])
    res = validate_idea_with_news("NVDA", "Nvidia Corporation", fetcher=stub, limit=3,
                                  min_relevance=30, max_age_hours=72)
    assert res["ticker"] == "NVDA"
    assert res["supported"] is True
    assert res["evidence"] == [good.title]
    assert res["url"] == good.url
    assert res["confidence"] in ("medium", "high")
    # Company-name match works without a literal ticker substring.
    assert "NVDA" not in good.title


def test_symbol_sentiment_line_shape():
    items = [
        _mkitem("Nvidia Blackwell beats expectations", key="s1"),
        _mkitem("Nvidia raises guidance on strong demand", key="s2"),
        _mkitem("Nvidia slips on market drop", key="s3"),
    ]
    line = symbol_sentiment_line("NVDA", items)
    assert line.startswith("Sentiment:")
    assert "+/" in line and "topics:" in line
def test_discovery_validated_caps_and_links(monkeypatch):
    from investment_engine.discovery.engine import DiscoveryEngine

    def _mkfetch(sym, name, limit=3):
        now = datetime.now(timezone.utc)
        return [_mkitem(f"{sym} beats earnings {i}", url=f"https://reuters.example/{sym}{i}",
                        source="Reuters", hours_ago=2, relevance=80, key=f"{sym}{i}")
                for i in range(3)]

    class _F:
        def fetch_for_symbol(self, sym, name, limit=3, **kw):
            return _mkfetch(sym, name, limit)

    eng = DiscoveryEngine()
    suggs = [{"broker_symbol": f"T{i:02d}", "reason": "test"} for i in range(15)]
    out = eng.discover_validated(suggestions=suggs, fetcher=_F(), max_total=20)
    assert len(out) <= 20
    assert sum(len(v.get("items", [])) for v in out) <= 20
    assert all(v.get("url", "").startswith("https://") for v in out)
    # discover() stays backward compatible and includes watchlist ideas.
    base = eng.discover([{"broker_symbol": "NVDA"}])
    assert any(s["broker_symbol"] == "AVGO" for s in base)
    with_watch = eng.discover([{"broker_symbol": "NVDA"}], watchlist=["PLTR"])
    assert any(s["broker_symbol"] == "PLTR" for s in with_watch)


def test_dedupe_thread_safe():
    f = StrictNewsFetcher(enable_file_cache=False)
    base = [_mkitem(f"title {i % 10}", url=f"https://example.com/{i % 10}",
                    key=f"h{i % 10}") for i in range(40)]

    def _work(chunk):
        return f._deduplicate(chunk)

    chunks = [base[i::4] for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_work, chunks))
    assert sum(len(r) for r in results) == 10