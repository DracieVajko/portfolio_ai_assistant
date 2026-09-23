from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from investment_engine.research.news_engine import StrictNewsFetcher, analyze_news_sentiment, NewsItem
from investment_engine.research.technical_analysis import TechnicalAnalyzer, IndicatorConfig, FinVizEnrichment
from investment_engine.research.peak_valley import PeakValleyDetector, PriceStructure, FibonacciLevels

logger = logging.getLogger(__name__)


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe."""
    name: str
    interval: str
    period: str
    lookback_bars: int
    enabled: bool = True
    weight: float = 1.0


@dataclass
class RegimeThresholds:
    """Thresholds for regime classification."""
    # PEAK_HOLD
    peak_hold_rsi_daily_min: float = 60
    peak_hold_rsi_weekly_min: float = 55
    peak_hold_price_above_sma20_daily: bool = True
    peak_hold_price_above_sma50_weekly: bool = True
    peak_hold_macd_hist_declining: bool = True
    peak_hold_consolidation_days_min: int = 5
    peak_hold_consolidation_range_pct: float = 2.5

    # DECLINING
    declining_price_below_sma20_daily: bool = True
    declining_price_below_sma50_daily: bool = True
    declining_macd_bearish_cross: bool = True
    declining_rsi_daily_max: float = 50
    declining_adx_min: float = 25
    declining_lower_highs: bool = True
    declining_lower_lows: bool = True

    # MUST_BUY
    must_buy_near_200dma_tolerance_pct: float = 2.0
    must_buy_near_52w_low_tolerance_pct: float = 5.0
    must_buy_rsi_daily_max: float = 35
    must_buy_rsi_weekly_max: float = 40
    must_buy_volume_spike_threshold: float = 2.0
    must_buy_supertrend_flip_bullish: bool = True
    must_buy_macd_weekly_bullish: bool = True

    # Confidence thresholds
    min_timeframes_agree: int = 2
    confidence_threshold: float = 0.75


@dataclass
class RegimeResult:
    """Complete regime analysis result."""
    symbol: str
    regime: str  # PEAK_HOLD, DECLINING, MUST_BUY, NEUTRAL
    confidence: float
    primary_signal: str
    timeframes: dict[str, dict]
    price_structure: PriceStructure
    finviz_data: dict
    news_sentiment: dict
    implications: dict
    warnings: list[str]
    generated_at: str
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "regime": self.regime,
            "confidence": round(self.confidence, 3),
            "primary_signal": self.primary_signal,
            "timeframes": self.timeframes,
            "price_structure": self.price_structure.to_dict(),
            "finviz_data": self.finviz_data,
            "news_sentiment": self.news_sentiment,
            "implications": self.implications,
            "warnings": self.warnings,
            "generated_at": self.generated_at,
        }


class EXI2RegimeAnalyzer:
    """
    Multi-timeframe EXI2 (iShares Dow Jones Global Titans 50 UCITS ETF)
    market regime analyzer for portfolio allocation timing.
    """

    DEFAULT_TIMEFRAMES = {
        "intraday_15m": TimeframeConfig("intraday_15m", "15m", "1d", 96, True, 0.5),
        "intraday_1h": TimeframeConfig("intraday_1h", "1h", "5d", 120, True, 0.7),
        "daily": TimeframeConfig("daily", "1d", "3mo", 63, True, 1.5),
        "weekly": TimeframeConfig("weekly", "1wk", "2y", 104, True, 2.0),
        "monthly": TimeframeConfig("monthly", "1mo", "max", 120, True, 1.0),
    }

    REGIME_ACTIONS = {
        "PEAK_HOLD": {
            "portfolio_action": "REDUCE_RISK",
            "tech_allocation": "REDUCE_NEW_ENTRY",
            "crypto_allocation": "REDUCE_NEW_ENTRY",
            "dca_multiplier": 0.5,
            "cash_target_pct": 15,
            "message": "Market overheated. Avoid new tech positions. Let DCA run at reduced pace.",
        },
        "DECLINING": {
            "portfolio_action": "PREPARE_ACCUMULATION",
            "tech_allocation": "PREPARE_WATCHLIST",
            "crypto_allocation": "PREPARE_WATCHLIST",
            "dca_multiplier": 1.0,
            "cash_target_pct": 25,
            "message": "Downtrend confirmed. Build watchlists. Raise cash for deployment.",
        },
        "MUST_BUY": {
            "portfolio_action": "AGGRESSIVE_ACCUMULATE",
            "tech_allocation": "AGGRESSIVE_ACCUMULATE",
            "crypto_allocation": "SELECTIVE_BUY",
            "dca_multiplier": 2.0,
            "cash_target_pct": 5,
            "message": "Major support hit with oversold conditions. Deploy cash aggressively.",
        },
        "NEUTRAL": {
            "portfolio_action": "NORMAL_DCA",
            "tech_allocation": "NORMAL_DCA",
            "crypto_allocation": "NORMAL_DCA",
            "dca_multiplier": 1.0,
            "cash_target_pct": 10,
            "message": "No clear regime signal. Continue normal DCA schedule.",
        },
    }

    def __init__(
        self,
        symbol: str = "EXI2.DE",
        timeframes: dict[str, TimeframeConfig] | None = None,
        thresholds: RegimeThresholds | None = None,
        news_fetcher: StrictNewsFetcher | None = None,
        cache_ttl_seconds: int = 3600,
    ):
        self.symbol = symbol
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES
        self.thresholds = thresholds or RegimeThresholds()
        self.news_fetcher = news_fetcher or StrictNewsFetcher(max_age_hours=48, min_relevance=60)
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, RegimeResult]] = {}

        self.analyzer = TechnicalAnalyzer()
        self.peak_detector = PeakValleyDetector(
            prominence_pct=2.0,
            min_distance=5,
            cluster_threshold_pct=1.0,
        )
        self.finviz = FinVizEnrichment() if FinVizEnrichment else None

    def analyze(self, force_refresh: bool = False) -> RegimeResult:
        """Run complete regime analysis."""
        cache_key = f"{self.symbol}_{int(time.time() // max(self.cache_ttl, 1))}"
        
        if not force_refresh and cache_key in self._cache:
            _, cached = self._cache[cache_key]
            return cached

        logger.info("Starting EXI2 regime analysis for %s", self.symbol)

        try:
            # 1. Fetch multi-timeframe data
            tf_data = self._fetch_multi_timeframe()
            if not tf_data:
                return self._empty_result("No data fetched")

            # 2. Technical analysis per timeframe
            tf_analysis = self.analyzer.get_multi_timeframe_analysis(tf_data)

            # 3. Price structure on daily
            daily_df = tf_data.get("daily")
            price_structure = PriceStructure(
                peaks=[], valleys=[], current_trend="UNKNOWN",
                key_levels={"support": [], "resistance": []},
                nearest_support=None, nearest_resistance=None, trend_quality=0.0
            )
            if daily_df is not None and not daily_df.empty:
                price_structure = self.peak_detector.detect(daily_df["close"])

            # 4. FinViz enrichment
            finviz_data = {}
            if self.finviz:
                try:
                    finviz_data = self.finviz.get_quote_data("EXI2")
                    finviz_data["sector_breadth"] = self.finviz.get_sector_breadth()
                except Exception as e:
                    logger.warning("FinViz enrichment failed: %s", e)

            # 5. News sentiment
            news_items = self.news_fetcher.fetch_for_symbol(
                "EXI2", "iShares Dow Jones Global Titans 50", limit=10
            )
            news_sentiment = analyze_news_sentiment(news_items)

            # 6. Classify regime
            regime, confidence, primary_signal = self._classify_regime(tf_analysis, price_structure, news_sentiment)

            # 7. Get implications
            implications = self.REGIME_ACTIONS.get(regime, self.REGIME_ACTIONS["NEUTRAL"]).copy()
            implications["regime"] = regime
            implications["confidence"] = confidence

            # 8. Generate warnings
            warnings = self._generate_warnings(regime, tf_analysis, price_structure, news_sentiment)

            result = RegimeResult(
                symbol=self.symbol,
                regime=regime,
                confidence=confidence,
                primary_signal=primary_signal,
                timeframes=tf_analysis,
                price_structure=price_structure,
                finviz_data=finviz_data,
                news_sentiment=news_sentiment,
                implications=implications,
                warnings=warnings,
                generated_at=datetime.now(timezone.utc).isoformat(),
                cache_key=cache_key,
            )

            self._cache[cache_key] = (time.time(), result)
            return result
        except Exception as e:
            logger.exception("Regime analysis failed for %s", self.symbol)
            return self._empty_result(f"Analysis failed: {e}")

    def _fetch_multi_timeframe(self) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all enabled timeframes."""
        results = {}
        for tf_name, config in self.timeframes.items():
            if not config.enabled:
                continue
            try:
                df = yf.download(
                    self.symbol,
                    period=config.period,
                    interval=config.interval,
                    progress=False,
                    auto_adjust=True,
                    actions=True,
                    threads=True,
                )
                if not df.empty:
                    # Handle MultiIndex columns from yfinance 1.6+
                    if isinstance(df.columns, pd.MultiIndex):
                        # Flatten columns: take the first level (Open, High, Low, Close, Volume)
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna()
                    df.columns = [c.lower() for c in df.columns]
                    # Validate
                    if self._validate_dataframe(df, config.lookback_bars):
                        results[tf_name] = df.tail(config.lookback_bars)
                    else:
                        logger.warning("Data validation failed for %s", tf_name)
            except Exception as e:
                logger.warning("Failed to fetch %s for %s: %s", tf_name, self.symbol, e)
        return results

    def _validate_dataframe(self, df: pd.DataFrame, min_bars: int) -> bool:
        if df.empty or len(df) < min_bars:
            return False
        if df.isnull().any().any():
            return False
        if (df["high"] < df["low"]).any():
            return False
        if (df["close"] <= 0).any():
            return False
        return True

    def _classify_regime(
        self,
        tf_analysis: dict[str, dict],
        price_structure: PriceStructure,
        news_sentiment: dict,
    ) -> tuple[str, float, str]:
        """Classify market regime based on multi-timeframe analysis."""
        scores = {
            "PEAK_HOLD": 0.0,
            "DECLINING": 0.0,
            "MUST_BUY": 0.0,
            "NEUTRAL": 0.5,  # Base score
        }
        signals = []

        # Get key timeframe data
        daily = tf_analysis.get("daily", {})
        weekly = tf_analysis.get("weekly", {})
        monthly = tf_analysis.get("monthly", {})
        intraday = tf_analysis.get("intraday_1h", {})

        # --- PEAK_HOLD Detection ---
        peak_score = 0
        peak_checks = 0

        # RSI conditions
        daily_ind = daily.get("indicators", {})
        weekly_ind = weekly.get("indicators", {})
        if daily_ind.get("RSI_14", 0) > self.thresholds.peak_hold_rsi_daily_min:
            peak_score += 1
            signals.append("RSI daily > 60")
        peak_checks += 1
        if weekly_ind.get("RSI_14", 0) > self.thresholds.peak_hold_rsi_weekly_min:
            peak_score += 1
            signals.append("RSI weekly > 55")
        peak_checks += 1

        # Price above MAs
        if self.thresholds.peak_hold_price_above_sma20_daily:
            if daily_ind.get("CLOSE", 0) > daily_ind.get("SMA_20", 0):
                peak_score += 1
                signals.append("Price > SMA20 daily")
            peak_checks += 1
        if self.thresholds.peak_hold_price_above_sma50_weekly:
            if weekly_ind.get("CLOSE", 0) > weekly_ind.get("SMA_50", 0):
                peak_score += 1
                signals.append("Price > SMA50 weekly")
            peak_checks += 1

        # MACD histogram declining
        if self.thresholds.peak_hold_macd_hist_declining:
            macd_hist = daily_ind.get("MACD_HIST", 0)
            # Check if histogram was higher 3 bars ago (approximation)
            if macd_hist > 0:
                peak_score += 0.5
                signals.append("MACD hist positive but check declining")
            peak_checks += 1

        # Consolidation detection
        if price_structure.current_trend == "CONSOLIDATING_AT_TOP":
            peak_score += 2
            signals.append("Consolidating at top")
        peak_checks += 2

        if peak_checks > 0:
            scores["PEAK_HOLD"] = peak_score / peak_checks

        # --- DECLINING Detection ---
        dec_score = 0
        dec_checks = 0

        if self.thresholds.declining_price_below_sma20_daily:
            if daily_ind.get("CLOSE", 0) < daily_ind.get("SMA_20", 0):
                dec_score += 1
                signals.append("Price < SMA20 daily")
            dec_checks += 1

        if self.thresholds.declining_price_below_sma50_daily:
            if daily_ind.get("CLOSE", 0) < daily_ind.get("SMA_50", 0):
                dec_score += 1
                signals.append("Price < SMA50 daily")
            dec_checks += 1

        if self.thresholds.declining_macd_bearish_cross:
            if daily_ind.get("MACD", 0) < daily_ind.get("MACD_SIGNAL", 0):
                dec_score += 1.5
                signals.append("MACD bearish cross daily")
            dec_checks += 1

        if self.thresholds.declining_rsi_daily_max:
            if daily_ind.get("RSI_14", 50) < self.thresholds.declining_rsi_daily_max:
                dec_score += 1
                signals.append("RSI daily < 50")
            dec_checks += 1

        if self.thresholds.declining_adx_min:
            if daily_ind.get("ADX", 0) > self.thresholds.declining_adx_min:
                dec_score += 1
                signals.append("ADX > 25 (trending)")
            dec_checks += 1

        if self.thresholds.declining_lower_highs:
            if price_structure.current_trend == "DOWN_TREND":
                dec_score += 2
                signals.append("Lower highs confirmed")
            dec_checks += 2

        if dec_checks > 0:
            scores["DECLINING"] = dec_score / dec_checks

        # --- MUST_BUY Detection ---
        buy_score = 0
        buy_checks = 0

        current = daily_ind.get("CLOSE", 0)
        sma200 = daily_ind.get("SMA_200", 0)
        low_52w = daily_ind.get("LOW_52W", daily_ind.get("DC_LOWER", 0)) if "DC_LOWER" in daily_ind else 0

        if self.thresholds.must_buy_near_200dma_tolerance_pct and sma200 > 0:
            dist = abs(current - sma200) / sma200 * 100
            if dist <= self.thresholds.must_buy_near_200dma_tolerance_pct:
                buy_score += 2
                signals.append(f"At 200DMA ({dist:.1f}%)")
            buy_checks += 2

        if self.thresholds.must_buy_near_52w_low_tolerance_pct and low_52w > 0:
            dist = (current - low_52w) / low_52w * 100
            if dist <= self.thresholds.must_buy_near_52w_low_tolerance_pct:
                buy_score += 2
                signals.append(f"Near 52W low ({dist:.1f}%)")
            buy_checks += 2

        if self.thresholds.must_buy_rsi_daily_max:
            if daily_ind.get("RSI_14", 50) < self.thresholds.must_buy_rsi_daily_max:
                buy_score += 1.5
                signals.append(f"RSI daily < {self.thresholds.must_buy_rsi_daily_max}")
            buy_checks += 1

        if self.thresholds.must_buy_rsi_weekly_max:
            if weekly_ind.get("RSI_14", 50) < self.thresholds.must_buy_rsi_weekly_max:
                buy_score += 1.5
                signals.append(f"RSI weekly < {self.thresholds.must_buy_rsi_weekly_max}")
            buy_checks += 1

        if self.thresholds.must_buy_volume_spike_threshold:
            vol_ratio = daily_ind.get("VOLUME_RATIO", 1)
            if vol_ratio > self.thresholds.must_buy_volume_spike_threshold:
                buy_score += 1
                signals.append(f"Volume spike {vol_ratio:.1f}x")
            buy_checks += 1

        if self.thresholds.must_buy_supertrend_flip_bullish:
            if daily_ind.get("SUPERT_DIR", -1) == 1:
                buy_score += 1
                signals.append("Supertrend flipped bullish")
            buy_checks += 1

        if self.thresholds.must_buy_macd_weekly_bullish:
            if weekly_ind.get("MACD", 0) > weekly_ind.get("MACD_SIGNAL", 0):
                buy_score += 1
                signals.append("MACD weekly bullish")
            buy_checks += 1

        if buy_checks > 0:
            scores["MUST_BUY"] = buy_score / buy_checks

        # --- Select regime with highest score ---
        # Require minimum confidence and timeframe agreement
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        # Check timeframe agreement
        agreeing_tfs = self._count_agreeing_timeframes(tf_analysis, best_regime)
        if agreeing_tfs < self.thresholds.min_timeframes_agree:
            best_regime = "NEUTRAL"
            best_score = 0.5

        confidence = min(best_score, 1.0)
        primary_signal = "; ".join(signals[:5])  # Top 5 signals

        return best_regime, confidence, primary_signal

    def _count_agreeing_timeframes(self, tf_analysis: dict, regime: str) -> int:
        """Count timeframes agreeing with regime."""
        count = 0
        for tf_name, tf_data in tf_analysis.items():
            trend = tf_data.get("trend", "")
            momentum = tf_data.get("momentum", "")

            if regime == "PEAK_HOLD":
                if "UP" in trend or "CONSOLIDATING_AT_TOP" in trend:
                    count += 1
            elif regime == "DECLINING":
                if "DOWN" in trend:
                    count += 1
            elif regime == "MUST_BUY":
                if "OVERSOLD" in momentum or "CONSOLIDATING_AT_BOTTOM" in trend:
                    count += 1
        return count

    def _generate_warnings(
        self,
        regime: str,
        tf_analysis: dict,
        price_structure: PriceStructure,
        news_sentiment: dict,
    ) -> list[str]:
        """Generate risk warnings."""
        warnings = []

        # Low confidence warning
        if regime != "NEUTRAL":
            pass  # Confidence is in implications

        # Conflicting timeframes
        daily_trend = tf_analysis.get("daily", {}).get("trend", "")
        weekly_trend = tf_analysis.get("weekly", {}).get("trend", "")
        if "UP" in daily_trend and "DOWN" in weekly_trend:
            warnings.append("Daily/Weekly trend conflict - mixed signals")

        # News sentiment divergence
        if regime == "MUST_BUY" and news_sentiment.get("sentiment") == "NEGATIVE":
            warnings.append("Negative news sentiment despite oversold technicals")

        if regime == "PEAK_HOLD" and news_sentiment.get("sentiment") == "POSITIVE":
            warnings.append("Positive news may extend peak - be patient")

        # Low trend quality
        if price_structure.trend_quality < 0.3:
            warnings.append("Low trend quality - choppy price action")

        # Volume declining on peaks
        daily_ind = tf_analysis.get("daily", {}).get("indicators", {})
        if regime == "PEAK_HOLD" and daily_ind.get("VOLUME_RATIO", 1) < 0.8:
            warnings.append("Declining volume on consolidation - distribution risk")

        return warnings

    def _empty_result(self, reason: str) -> RegimeResult:
        return RegimeResult(
            symbol=self.symbol,
            regime="ERROR",
            confidence=0.0,
            primary_signal=reason,
            timeframes={},
            price_structure=PriceStructure(
                peaks=[], valleys=[], current_trend="UNKNOWN",
                key_levels={"support": [], "resistance": []},
                nearest_support=None, nearest_resistance=None, trend_quality=0.0
            ),
            finviz_data={},
            news_sentiment={},
            implications=self.REGIME_ACTIONS["NEUTRAL"],
            warnings=[reason],
            generated_at=datetime.now(timezone.utc).isoformat(),
            cache_key="error",
        )

    def get_regime_summary(self) -> dict[str, Any]:
        """Get concise regime summary for prompt injection."""
        result = self.analyze()
        return {
            "regime": result.regime,
            "confidence": result.confidence,
            "price": result.timeframes.get("daily", {}).get("indicators", {}).get("CLOSE", 0),
            "rsi_daily": result.timeframes.get("daily", {}).get("indicators", {}).get("RSI_14", 0),
            "rsi_weekly": result.timeframes.get("weekly", {}).get("indicators", {}).get("RSI_14", 0),
            "trend_daily": result.timeframes.get("daily", {}).get("trend", "UNKNOWN"),
            "trend_weekly": result.timeframes.get("weekly", {}).get("trend", "UNKNOWN"),
            "action": result.implications.get("portfolio_action", "UNKNOWN"),
            "cash_target": result.implications.get("cash_target_pct", 10),
            "dca_mult": result.implications.get("dca_multiplier", 1.0),
            "message": result.implications.get("message", ""),
            "warnings": result.warnings,
        }