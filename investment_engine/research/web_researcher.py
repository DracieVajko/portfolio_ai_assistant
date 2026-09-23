from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EarningsEvent:
    symbol: str
    company: str
    date: str  # ISO format
    time: str  # "bmo" | "amc" | "unknown"
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    market_cap: Optional[str] = None
    eps_surprise_pct: Optional[str] = None
    revenue_surprise_pct: Optional[str] = None
    one_day_change_pct: Optional[str] = None
    source: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketMover:
    symbol: str
    name: str
    price: float
    change_pct: float
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    source: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradingViewSymbolData:
    """Complete data for a single symbol from TradingView."""
    symbol: str
    exchange: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    key_facts_today: Optional[str] = None
    technicals_rating: Optional[str] = None  # Strong sell, Sell, Neutral, Buy, Strong buy
    analyst_rating: Optional[str] = None     # Strong sell, Sell, Neutral, Buy, Strong buy
    # Financials (4 quadrants)
    financials_performance: Optional[dict] = None
    financials_revenue_to_profit: Optional[dict] = None
    financials_debt_coverage: Optional[dict] = None
    financials_earnings: Optional[dict] = None
    # Market movers categories
    source: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinQuotaData:
    symbol: str
    metrics: list[str]
    news: list[str]
    source: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Web Researcher
# =============================================================================

class WebResearcher:
    """Playwright-based web researcher for financial data sources."""

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 90000,  # Increased for slow sites
        cache_dir: str = "data/cache/web",
        max_concurrent: int = 2,  # Lower for heavy sites
        stealth: bool = True,
    ):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.stealth = stealth
        self._browser = None
        self._context = None
        self._semaphore = None
        self._playwright = None

    async def _get_browser(self):
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins,site-per-process",
                    ] if self.stealth else [],
                )
                self._context = await self._browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    locale="en-US",
                    timezone_id="America/New_York",
                )
                if self.stealth:
                    await self._context.add_init_script("""
                        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                        window.chrome = {runtime: {}};
                        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
                        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                    """)
                self._semaphore = asyncio.Semaphore(self.max_concurrent)
            except ImportError:
                raise RuntimeError("playwright not installed. Run: pip install playwright && playwright install chromium")
        return self._browser

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None
        if hasattr(self, "_playwright") and self._playwright:
            await self._playwright.stop()

    async def __aenter__(self):
        await self._get_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}_{date.today().isoformat()}.json"

    def _load_cache(self, name: str, max_age_hours: int = 6) -> Optional[list]:
        path = self._cache_path(name)
        if not path.exists():
            return None
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if (datetime.now() - mtime).total_seconds() > max_age_hours * 3600:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data
        except Exception:
            return None

    def _save_cache(self, name: str, data: Any):
        path = self._cache_path(name)
        try:
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save cache %s: %s", name, e)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    async def _safe_goto(self, page, url: str, wait_until: str = "domcontentloaded"):
        """Navigate with retry and popup handling."""
        for attempt in range(3):
            try:
                await page.goto(url, wait_until=wait_until, timeout=self.timeout_ms)
                return True
            except Exception as e:
                logger.warning("Navigation attempt %d failed for %s: %s", attempt + 1, url, e)
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
        return False

    async def _dismiss_popups(self, page):
        """Dismiss common popups (cookies, login, etc.)."""
        popup_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Agree")',
            'button:has-text("Allow")',
            'button:has-text("Close")',
            '[aria-label="Close"]',
            '.close-button',
            '.popup-close',
            'button:has-text("No thanks")',
            'button:has-text("Maybe later")',
            'button:has-text("Sign in with Google")',  # earningshub
        ]
        for selector in popup_selectors:
            try:
                btn = await page.query_selector(selector)
                if btn and await btn.is_visible():
                    await btn.click(timeout=1000)
                    await asyncio.sleep(0.5)
            except Exception:
                pass

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from text like '$321.93M', '-$0.05', '+4.35%'."""
        if not text:
            return None
        text = text.strip().replace(",", "").replace("$", "").replace("%", "").replace("+", "")
        # Handle M/B/K suffixes
        mult = 1
        if text.endswith("M"):
            mult = 1_000_000
            text = text[:-1]
        elif text.endswith("B"):
            mult = 1_000_000_000
            text = text[:-1]
        elif text.endswith("K"):
            mult = 1_000
            text = text[:-1]
        try:
            return float(text) * mult
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    # 1. FINVIZ Earnings - Click "This Week" tab
    # -------------------------------------------------------------------------
    async def fetch_finviz_earnings(self, days_ahead: int = 7) -> list[EarningsEvent]:
        """Fetch earnings from finviz.com calendar - clicks 'This Week' tab."""
        cached = self._load_cache("finviz_earnings", max_age_hours=6)
        if cached:
            return [EarningsEvent(**e) for e in cached]

        async with self._semaphore:
            page = await self._context.new_page()
            try:
                url = "https://finviz.com/calendar/earnings"
                await self._safe_goto(page, url)
                await page.wait_for_selector("table", timeout=30000)
                await self._dismiss_popups(page)

                # Click "This Week" tab if not active
                try:
                    this_week_tab = await page.query_selector('text="This Week"')
                    if this_week_tab:
                        await this_week_tab.click(timeout=5000)
                        await asyncio.sleep(2)  # Wait for table to update
                except Exception:
                    pass

                # Wait for table to render
                await page.wait_for_selector("table tbody tr", timeout=20000)

                events = []
                table = await page.query_selector("table")
                if table:
                    rows = await table.query_selector_all("tbody tr")
                    for row in rows:
                        try:
                            cells = await row.query_selector_all("td")
                            if len(cells) < 14:  # Full Finviz table has 14 columns
                                continue
                            
                            symbol = (await cells[0].inner_text()).strip().upper().replace('\n', '')
                            company = (await cells[1].inner_text()).strip()
                            time_str = (await cells[2].inner_text()).strip().lower()
                            market_cap = (await cells[3].inner_text()).strip()
                            eps_est = (await cells[4].inner_text()).strip()
                            eps_act = (await cells[5].inner_text()).strip()
                            eps_sur = (await cells[6].inner_text()).strip()
                            rev_est = (await cells[9].inner_text()).strip()
                            rev_act = (await cells[10].inner_text()).strip()
                            rev_sur = (await cells[11].inner_text()).strip()
                            one_day_chg = (await cells[13].inner_text()).strip()

                            # Use today's date for This Week view
                            parsed_date = date.today()

                            events.append(EarningsEvent(
                                symbol=symbol,
                                company=company,
                                date=parsed_date.isoformat(),
                                time=time_str if time_str in ("bmo", "amc") else "unknown",
                                eps_estimate=self._parse_number(eps_est),
                                eps_actual=self._parse_number(eps_act),
                                revenue_estimate=self._parse_number(rev_est),
                                revenue_actual=self._parse_number(rev_act),
                                market_cap=market_cap,
                                eps_surprise_pct=eps_sur,
                                revenue_surprise_pct=rev_sur,
                                one_day_change_pct=one_day_chg,
                                source="finviz.com",
                                fetched_at=datetime.now(timezone.utc).isoformat(),
                            ))
                        except Exception as e:
                            logger.debug("Finviz row parse error: %s", e)
                            continue

                self._save_cache("finviz_earnings", [e.to_dict() for e in events])
                return events
            finally:
                await page.close()

    # -------------------------------------------------------------------------
    # 2. EarningsHub - with popup handling
    # -------------------------------------------------------------------------
    async def fetch_earnings_hub(self, days_ahead: int = 7) -> list[EarningsEvent]:
        """Fetch earnings from earningshub.com - handles Google login popup."""
        cached = self._load_cache("earnings_hub", max_age_hours=12)
        if cached:
            return [EarningsEvent(**e) for e in cached]

        async with self._semaphore:
            page = await self._context.new_page()
            try:
                url = "https://earningshub.com/earnings-calendar/this-week"
                await self._safe_goto(page, url)
                await asyncio.sleep(3)  # Let JS render
                await self._dismiss_popups(page)

                events = []
                tables = await page.query_selector_all("table")
                for table in tables:
                    text = await table.inner_text()
                    if "earnings" in text.lower() or "symbol" in text.lower() or "ticker" in text.lower():
                        rows = await table.query_selector_all("tbody tr")
                        for row in rows:
                            try:
                                cells = await row.query_selector_all("td")
                                if len(cells) < 3:
                                    continue
                                symbol = (await cells[0].inner_text()).strip().upper()
                                company = (await cells[1].inner_text()).strip()
                                date_str = (await cells[2].inner_text()).strip()
                                time_str = (await cells[3].inner_text()).strip().lower() if len(cells) > 3 else "unknown"

                                try:
                                    parsed_date = datetime.strptime(f"{date_str} {datetime.now().year}", "%b %d %Y").date()
                                except ValueError:
                                    continue

                                if (parsed_date - date.today()).days > days_ahead:
                                    continue

                                events.append(EarningsEvent(
                                    symbol=symbol,
                                    company=company,
                                    date=parsed_date.isoformat(),
                                    time=time_str if time_str in ("bmo", "amc") else "unknown",
                                    source="earningshub.com",
                                    fetched_at=datetime.now(timezone.utc).isoformat(),
                                ))
                            except Exception:
                                continue
                        break

                self._save_cache("earnings_hub", [e.to_dict() for e in events])
                return events
            finally:
                await page.close()

    # -------------------------------------------------------------------------
    # 3. ALL Earnings (Finviz + EarningsHub only)
    # -------------------------------------------------------------------------
    async def fetch_all_earnings(self, days_ahead: int = 7) -> list[EarningsEvent]:
        """Fetch and merge earnings from Finviz and EarningsHub only."""
        tasks = [
            self.fetch_finviz_earnings(days_ahead),
            self.fetch_earnings_hub(days_ahead),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events: list[EarningsEvent] = []
        source_names = ["finviz.com", "earningshub.com"]
        for i, result in enumerate(results):
            if isinstance(result, list):
                all_events.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Earnings fetch failed for %s: %s", source_names[i], result)

        # Deduplicate by (symbol, date)
        seen = set()
        unique = []
        for e in all_events:
            key = (e.symbol, e.date)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return sorted(unique, key=lambda x: (x.date, x.symbol))

    # -------------------------------------------------------------------------
    # 4. TRADINGVIEW - Symbol Detail Pages
    # -------------------------------------------------------------------------
    async def fetch_tradingview_symbol(
        self, 
        symbol: str, 
        exchange: str = "NYSE",
        include_financials: bool = True
    ) -> TradingViewSymbolData:
        """Fetch complete data for a single symbol from TradingView."""
        cache_name = f"tv_symbol_{exchange}_{symbol}"
        cached = self._load_cache(cache_name, max_age_hours=12)
        if cached:
            return TradingViewSymbolData(**cached)

        async with self._semaphore:
            page = await self._context.new_page()
            try:
                # TradingView symbol URL format
                url = f"https://www.tradingview.com/symbols/{exchange}-{symbol}/"
                await self._safe_goto(page, url)
                await asyncio.sleep(3)
                await self._dismiss_popups(page)

                data = TradingViewSymbolData(
                    symbol=symbol,
                    exchange=exchange,
                    source=f"tradingview.com/symbols/{exchange}-{symbol}",
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )

                # 1. Price & Change (header)
                try:
                    price_el = await page.query_selector('[data-symbol] .price, .js-symbol-header .price, .symbol-header .price')
                    if not price_el:
                        price_el = await page.query_selector('.tv-symbol-price-quote__value, [data-field="close"]')
                    if price_el:
                        price_text = (await price_el.inner_text()).strip()
                        data.price = self._parse_number(price_text)
                    
                    change_el = await page.query_selector('[data-field="ch"], .tv-symbol-price-quote__change')
                    if change_el:
                        change_text = (await change_el.inner_text()).strip()
                        data.change_pct = self._parse_number(change_text)
                except Exception as e:
                    logger.debug("Price extraction failed for %s: %s", symbol, e)

                # 2. Key Facts Today (carousel)
                try:
                    # Look for carousel with "Key facts" text
                    key_facts_el = await page.query_selector('text="Key facts" >> .. >> .. >> div')
                    if not key_facts_el:
                        key_facts_el = await page.query_selector('.key-facts, [data-key-facts], .carousel-item:has-text("Key facts")')
                    if not key_facts_el:
                        # Try finding by the specific text pattern
                        all_text = await page.inner_text("body")
                        match = re.search(r'Key facts today[:\s]*([^.]+\.[^.]+\.)', all_text)
                        if match:
                            data.key_facts_today = match.group(1).strip()
                    if key_facts_el:
                        data.key_facts_today = (await key_facts_el.inner_text()).strip()[:500]
                except Exception as e:
                    logger.debug("Key facts extraction failed for %s: %s", symbol, e)

                # 3. Technicals Rating (gauge)
                try:
                    # Look for the gauge with "Technicals" label
                    tech_section = await page.query_selector('text="Technicals" >> .. >> ..')
                    if tech_section:
                        gauge_text = await tech_section.inner_text()
                        # Extract rating: Strong sell, Sell, Neutral, Buy, Strong buy
                        for rating in ["Strong buy", "Buy", "Neutral", "Sell", "Strong sell"]:
                            if rating.lower() in gauge_text.lower():
                                data.technicals_rating = rating
                                break
                except Exception as e:
                    logger.debug("Technicals extraction failed for %s: %s", symbol, e)

                # 4. Analyst Rating (gauge)
                try:
                    analyst_section = await page.query_selector('text="Analyst rating" >> .. >> ..')
                    if analyst_section:
                        gauge_text = await analyst_section.inner_text()
                        for rating in ["Strong buy", "Buy", "Neutral", "Sell", "Strong sell"]:
                            if rating.lower() in gauge_text.lower():
                                data.analyst_rating = rating
                                break
                except Exception as e:
                    logger.debug("Analyst rating extraction failed for %s: %s", symbol, e)

                # 5. Financials (4 quadrants) - if requested
                if include_financials:
                    data.financials_performance = await self._extract_financials_quadrant(page, "Performance")
                    data.financials_revenue_to_profit = await self._extract_financials_quadrant(page, "Revenue to profit")
                    data.financials_debt_coverage = await self._extract_financials_quadrant(page, "Debt level")
                    data.financials_earnings = await self._extract_financials_quadrant(page, "Earnings")

                self._save_cache(cache_name, data.to_dict())
                return data
            finally:
                await page.close()

    async def _extract_financials_quadrant(self, page, quadrant_name: str) -> Optional[dict]:
        """Extract data from one financial quadrant."""
        try:
            # Find quadrant by heading
            heading = await page.query_selector(f'text="{quadrant_name}"')
            if not heading:
                return None
            container = await heading.evaluate_handle('el => el.closest("section, div, article")')
            if not container:
                return None
            
            text = await container.inner_text()
            return {"raw_text": text[:2000], "extracted_at": datetime.now(timezone.utc).isoformat()}
        except Exception:
            return None

    async def fetch_tradingview_symbols_batch(
        self, 
        symbols: list[tuple[str, str]],  # [(symbol, exchange), ...]
        include_financials: bool = True
    ) -> list[TradingViewSymbolData]:
        """Fetch multiple symbols with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_one(sym_exch):
            async with semaphore:
                return await self.fetch_tradingview_symbol(sym_exch[0], sym_exch[1], include_financials)
        
        tasks = [fetch_one(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, TradingViewSymbolData):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.warning("TradingView fetch failed for %s: %s", symbols[i], result)
        
        return valid_results

    # -------------------------------------------------------------------------
    # 5. TRADINGVIEW - Market Movers (multiple categories)
    # -------------------------------------------------------------------------
    TV_MOVER_CATEGORIES = {
        "gainers": "market-movers-gainers",
        "losers": "market-movers-losers",
        "ath": "market-movers-ath",
        "atl": "market-movers-atl",
        "52wk_high": "market-movers-52wk-high",
        "52wk_low": "market-movers-52wk-low",
        "most_volatile": "market-movers-most-volatile",
        "best_performing": "market-movers-best-performing",
    }

    async def fetch_tradingview_movers_category(self, category: str, market: str = "stocks-usa") -> list[MarketMover]:
        """Fetch movers for a specific category."""
        if category not in self.TV_MOVER_CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Valid: {list(self.TV_MOVER_CATEGORIES.keys())}")

        cache_name = f"tv_movers_{market}_{category}"
        cached = self._load_cache(cache_name, max_age_hours=2)
        if cached:
            return [MarketMover(**m) for m in cached]

        async with self._semaphore:
            page = await self._context.new_page()
            try:
                url = f"https://www.tradingview.com/markets/{market}/{self.TV_MOVER_CATEGORIES[category]}/"
                await self._safe_goto(page, url)
                await page.wait_for_selector("table, [data-qa]", timeout=30000)
                await self._dismiss_popups(page)

                movers = []
                # Try multiple row selectors
                rows = await page.query_selector_all("tbody tr, [data-rowkey], .tv-data-table__row")
                for row in rows[:30]:
                    try:
                        cells = await row.query_selector_all("td")
                        if len(cells) < 3:
                            continue
                        symbol = (await cells[0].inner_text()).strip().upper()
                        name = (await cells[1].inner_text()).strip()
                        price_str = (await cells[2].inner_text()).strip().replace(",", "")
                        change_str = (await cells[3].inner_text()).strip().replace("%", "").replace("+", "")

                        price = self._parse_number(price_str) or 0.0
                        change_pct = self._parse_number(change_str) or 0.0

                        movers.append(MarketMover(
                            symbol=symbol,
                            name=name,
                            price=price,
                            change_pct=change_pct,
                            source=f"tradingview.com/markets/{market}/{self.TV_MOVER_CATEGORIES[category]}",
                            fetched_at=datetime.now(timezone.utc).isoformat(),
                        ))
                    except Exception:
                        continue

                self._save_cache(cache_name, [m.to_dict() for m in movers])
                return movers
            finally:
                await page.close()

    async def fetch_all_tradingview_movers(self, market: str = "stocks-usa") -> dict[str, list[MarketMover]]:
        """Fetch all mover categories."""
        results = {}
        for category in self.TV_MOVER_CATEGORIES:
            try:
                results[category] = await self.fetch_tradingview_movers_category(category, market)
                await asyncio.sleep(1)  # Be polite
            except Exception as e:
                logger.warning("TradingView movers %s failed: %s", category, e)
                results[category] = []
        return results

    # -------------------------------------------------------------------------
    # 6. FINQUOTA - Main site + News
    # -------------------------------------------------------------------------
    async def fetch_finquota(self, symbol: str) -> FinQuotaData:
        """Fetch data for a symbol from finquota.com (main + news)."""
        cached = self._load_cache(f"finquota_{symbol}", max_age_hours=12)
        if cached:
            return FinQuotaData(**cached)

        async with self._semaphore:
            page = await self._context.new_page()
            try:
                metrics = []
                news = []

                # 1. Main symbol page
                url = f"https://finquota.com/{symbol.lower()}"
                await self._safe_goto(page, url)
                await asyncio.sleep(2)
                await self._dismiss_popups(page)

                # Extract metrics
                metric_elements = await page.query_selector_all(".metric, .value, [data-testid], table tr, .stat, .financial-row")
                for m in metric_elements:
                    text = (await m.inner_text()).strip()
                    if text and 5 < len(text) < 300:
                        metrics.append(text)

                # 2. News page
                try:
                    news_url = "https://finquota.com/news/"
                    await self._safe_goto(page, news_url)
                    await asyncio.sleep(2)
                    
                    news_elements = await page.query_selector_all("article, .news-item, .post, [data-testid*='news']")
                    for n in news_elements[:20]:
                        text = (await n.inner_text()).strip()
                        if text and len(text) > 20:
                            news.append(text[:500])
                except Exception:
                    pass

                data = FinQuotaData(
                    symbol=symbol,
                    metrics=list(set(metrics))[:50],  # Dedup
                    news=news[:20],
                    source="finquota.com",
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )

                self._save_cache(f"finquota_{symbol}", data.to_dict())
                return data
            finally:
                await page.close()

    # -------------------------------------------------------------------------
    # Recording / Script Generation
    # -------------------------------------------------------------------------
    async def record_session(self, output_path: str, start_url: str = "https://www.tradingview.com/"):
        """Record a browser session and generate a Playwright script."""
        await self._get_browser()
        page = await self._context.new_page()

        await self._context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            await page.goto(start_url, wait_until="networkidle")
            print(f"Recording started. Interact with the page at {start_url}")
            print("Press Ctrl+C when done to save the trace and script.")
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\nStopping recording...")
        finally:
            trace_path = output_path.replace(".py", ".zip")
            await self._context.tracing.stop(path=trace_path)
            print(f"Trace saved to {trace_path}")

            script = self._generate_script_from_trace(trace_path, output_path)
            print(f"Script template saved to {output_path}")

        await page.close()

    def _generate_script_from_trace(self, trace_path: str, output_path: str) -> str:
        now = datetime.now().isoformat()
        script = '"""' + '\n'
        script += f'Auto-generated Playwright script from recorded session.\n'
        script += f'Trace: {trace_path}\n'
        script += f'Generated: {now}\n'
        script += '"""' + '\n\n'
        script += 'from playwright.async_api import async_playwright\n'
        script += 'import asyncio\n\n\n'
        script += 'async def main():\n'
        script += '    async with async_playwright() as p:\n'
        script += '        browser = await p.chromium.launch(headless=False)\n'
        script += '        context = await browser.new_context(\n'
        script += '            viewport={"width": 1920, "height": 1080},\n'
        script += '            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",\n'
        script += '        )\n'
        script += '        page = await context.new_page()\n\n'
        script += '        # TODO: Add recorded actions here\n'
        script += '        await browser.close()\n\n\n'
        script += 'if __name__ == "__main__":\n'
        script += '    asyncio.run(main())\n'
        Path(output_path).write_text(script, encoding="utf-8")
        return script


# -------------------------------------------------------------------------
# Synchronous wrappers
# -------------------------------------------------------------------------
def run_web_research(coro):
    """Run async web research in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def fetch_finviz_earnings_sync(days_ahead: int = 7) -> list[EarningsEvent]:
    async def _run():
        async with WebResearcher() as researcher:
            return await researcher.fetch_finviz_earnings(days_ahead)
    return run_web_research(_run())


def fetch_all_earnings_sync(days_ahead: int = 7) -> list[EarningsEvent]:
    async def _run():
        async with WebResearcher() as researcher:
            return await researcher.fetch_all_earnings(days_ahead)
    return run_web_research(_run())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "record":
        output = sys.argv[2] if len(sys.argv) > 2 else "recorded_session.py"
        start_url = sys.argv[3] if len(sys.argv) > 3 else "https://www.tradingview.com/"
        asyncio.run(WebResearcher(headless=False).record_session(output, start_url))
    else:
        earnings = fetch_all_earnings_sync(7)
        print(f"Fetched {len(earnings)} earnings events:")
        for e in earnings[:10]:
            print(f"  {e.symbol} ({e.company}) - {e.date} {e.time} [{e.source}]")