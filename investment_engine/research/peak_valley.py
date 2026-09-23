from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class Extremum:
    """Represents a peak or valley in price."""
    idx: int
    date: pd.Timestamp
    price: float
    prominence: float
    width: int
    type: str  # "peak" or "valley"

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "price": round(self.price, 4),
            "prominence": round(self.prominence, 4),
            "width": self.width,
            "type": self.type,
        }


@dataclass
class KeyLevel:
    """Support or resistance level from clustered extrema."""
    price: float
    strength: int  # number of touches
    level_type: str  # "support" or "resistance"
    touches: list[Extremum] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": round(self.price, 4),
            "strength": self.strength,
            "type": self.level_type,
            "touches": [t.to_dict() for t in self.touches],
        }


@dataclass
class PriceStructure:
    """Complete price structure analysis."""
    peaks: list[Extremum]
    valleys: list[Extremum]
    current_trend: str
    key_levels: dict[str, list[KeyLevel]]
    nearest_support: float | None
    nearest_resistance: float | None
    trend_quality: float  # 0-1 score

    def to_dict(self) -> dict[str, Any]:
        return {
            "peaks": [p.to_dict() for p in self.peaks],
            "valleys": [v.to_dict() for v in self.valleys],
            "current_trend": self.current_trend,
            "key_levels": {
                "support": [l.to_dict() for l in self.key_levels.get("support", [])],
                "resistance": [l.to_dict() for l in self.key_levels.get("resistance", [])],
            },
            "nearest_support": round(self.nearest_support, 4) if self.nearest_support else None,
            "nearest_resistance": round(self.nearest_resistance, 4) if self.nearest_resistance else None,
            "trend_quality": round(self.trend_quality, 3),
        }


class PeakValleyDetector:
    """
    Detects significant peaks and valleys using scipy.signal.find_peaks.
    Clusters extrema into support/resistance levels.
    """

    def __init__(
        self,
        prominence_pct: float = 2.0,
        min_distance: int = 5,
        cluster_threshold_pct: float = 1.0,
        min_prominence_bars: int = 3,
    ):
        """
        Args:
            prominence_pct: Minimum prominence as % of price std
            min_distance: Minimum bars between extrema
            cluster_threshold_pct: Max % diff to cluster levels
            min_prominence_bars: Minimum width of peak/valley
        """
        self.prominence_pct = prominence_pct
        self.min_distance = min_distance
        self.cluster_threshold_pct = cluster_threshold_pct
        self.min_prominence_bars = min_prominence_bars

    def detect(self, close: pd.Series) -> PriceStructure:
        """Detect peaks, valleys, and key levels from price series."""
        if len(close) < self.min_distance * 3:
            logger.warning("Insufficient data for peak/valley detection")
            return PriceStructure(
                peaks=[], valleys=[], current_trend="UNKNOWN",
                key_levels={"support": [], "resistance": []},
                nearest_support=None, nearest_resistance=None, trend_quality=0.0
            )

        prices = close.values
        prominence = prices.std() * (self.prominence_pct / 100)

        # Detect peaks (highs)
        peak_idx, peak_props = find_peaks(
            prices,
            prominence=prominence,
            distance=self.min_distance,
            width=self.min_prominence_bars,
        )

        # Detect valleys (lows) - invert prices
        valley_idx, valley_props = find_peaks(
            -prices,
            prominence=prominence,
            distance=self.min_distance,
            width=self.min_prominence_bars,
        )

        peaks = self._format_extrema(peak_idx, peak_props, close, "peak")
        valleys = self._format_extrema(valley_idx, valley_props, close, "valley")

        # Determine trend
        current_trend = self._determine_trend(peaks, valleys, close)

        # Identify key levels (support/resistance)
        key_levels = self._identify_key_levels(peaks, valleys, close)

        # Find nearest support/resistance
        current_price = close.iloc[-1]
        support_levels = key_levels.get("support", [])
        resistance_levels = key_levels.get("resistance", [])
        
        # Nearest support: closest support level below current (or closest overall if none below)
        support_below = [l for l in support_levels if l.price < current_price]
        if support_below:
            nearest_support = max(support_below, key=lambda l: l.price).price
        elif support_levels:
            nearest_support = min(support_levels, key=lambda l: abs(l.price - current_price)).price
        else:
            nearest_support = None
        
        # Nearest resistance: closest resistance level above current (or closest overall if none above)
        resistance_above = [l for l in resistance_levels if l.price > current_price]
        if resistance_above:
            nearest_resistance = min(resistance_above, key=lambda l: l.price).price
        elif resistance_levels:
            nearest_resistance = min(resistance_levels, key=lambda l: abs(l.price - current_price)).price
        else:
            nearest_resistance = None

        # Calculate trend quality
        trend_quality = self._calculate_trend_quality(peaks, valleys, close)

        return PriceStructure(
            peaks=peaks,
            valleys=valleys,
            current_trend=current_trend,
            key_levels=key_levels,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            trend_quality=trend_quality,
        )

    def _format_extrema(self, indices: np.ndarray, props: dict, close: pd.Series, typ: str) -> list[Extremum]:
        """Format scipy find_peaks output into Extremum objects."""
        extrema = []
        for j, i in enumerate(indices):
            if i < len(close):
                extrema.append(Extremum(
                    idx=int(i),
                    date=close.index[i],
                    price=float(close.iloc[i]),
                    prominence=float(props["prominences"][j]),
                    width=int(props["widths"][j]),
                    type=typ,
                ))
        return extrema

    def _determine_trend(self, peaks: list[Extremum], valleys: list[Extremum], close: pd.Series) -> str:
        """Classify current trend structure."""
        if not peaks or not valleys:
            return "UNKNOWN"

        current = close.iloc[-1]
        
        # Use recent extrema (last 10 of each) for trend determination
        recent_peaks = sorted(peaks, key=lambda p: p.date)[-10:]
        recent_valleys = sorted(valleys, key=lambda v: v.date)[-10:]
        
        if not recent_peaks or not recent_valleys:
            return "UNKNOWN"

        # Overall range from recent extrema
        recent_peak_max = max(p.price for p in recent_peaks)
        recent_valley_min = min(v.price for v in recent_valleys)
        recent_peak_min = min(p.price for p in recent_peaks)
        recent_valley_max = max(v.price for v in recent_valleys)
        
        peak_range = (recent_peak_max - recent_valley_min) / recent_valley_min * 100 if recent_valley_min > 0 else 0
        
        # Check for higher highs / higher lows (uptrend)
        if len(recent_peaks) >= 2 and len(recent_valleys) >= 2:
            # Check last few for trend direction
            hh = recent_peaks[-1].price > recent_peaks[-2].price
            hl = recent_valleys[-1].price > recent_valleys[-2].price
            if hh and hl:
                return "UP_TREND"

            lh = recent_peaks[-1].price < recent_peaks[-2].price
            ll = recent_valleys[-1].price < recent_valleys[-2].price
            if lh and ll:
                return "DOWN_TREND"

        # Consolidation detection: check where current price sits in the recent range
        # Use a shorter lookback for consolidation (last 3 of each) to focus on recent action
        consolidation_peaks = sorted(peaks, key=lambda p: p.date)[-3:]
        consolidation_valleys = sorted(valleys, key=lambda v: v.date)[-3:]
        
        if consolidation_peaks and consolidation_valleys:
            cons_peak_max = max(p.price for p in consolidation_peaks)
            cons_valley_min = min(v.price for v in consolidation_valleys)
            cons_range = (cons_peak_max - cons_valley_min) / cons_valley_min * 100 if cons_valley_min > 0 else 0
            
            # Only consider it consolidation if the range is tight (< 8%)
            if cons_range < 8:
                # Position in the consolidation range
                if cons_peak_max > cons_valley_min:
                    range_position = (current - cons_valley_min) / (cons_peak_max - cons_valley_min)
                else:
                    range_position = 0.5
                
                # Near top of consolidation range (>= 80th percentile)
                if range_position >= 0.8:
                    return "CONSOLIDATING_AT_TOP"
                # Near bottom of consolidation range (<= 20th percentile)
                elif range_position <= 0.2:
                    return "CONSOLIDATING_AT_BOTTOM"
                # Tight consolidation range overall
                elif cons_range < 5:
                    return "TIGHT_RANGE"

        # Fallback to overall range position
        range_position = (current - recent_valley_min) / (recent_peak_max - recent_valley_min) if recent_peak_max > recent_valley_min else 0.5
        
        if range_position >= 0.8:
            return "CONSOLIDATING_AT_TOP"
        elif range_position <= 0.2:
            return "CONSOLIDATING_AT_BOTTOM"
        elif peak_range < 5:
            return "TIGHT_RANGE"

        return "TRANSITIONAL"

    def _identify_key_levels(
        self,
        peaks: list[Extremum],
        valleys: list[Extremum],
        close: pd.Series,
    ) -> dict[str, list[KeyLevel]]:
        """Cluster extrema into support/resistance levels."""
        all_extrema = peaks + valleys
        if not all_extrema:
            return {"support": [], "resistance": []}

        current = close.iloc[-1]

        # Separate by type
        peak_prices = [p.price for p in peaks]
        valley_prices = [v.price for v in valleys]

        # Cluster peaks -> resistance
        resistance_clusters = self._cluster_levels(peak_prices, peaks, "resistance")
        # Cluster valleys -> support
        support_clusters = self._cluster_levels(valley_prices, valleys, "support")

        # Sort by proximity to current price (closest first)
        resistance_clusters.sort(key=lambda x: abs(x.price - current))
        support_clusters.sort(key=lambda x: abs(x.price - current))

        return {
            "support": support_clusters[:10],
            "resistance": resistance_clusters[:10],
        }

    def _cluster_levels(
        self,
        prices: list[float],
        extrema: list[Extremum],
        level_type: str,
    ) -> list[KeyLevel]:
        """Cluster price levels within threshold%."""
        if not prices:
            return []

        # Sort prices with their extrema
        paired = sorted(zip(prices, extrema), key=lambda x: x[0])
        clustered_prices = [p[0] for p in paired]
        clustered_extrema = [p[1] for p in paired]

        clusters = []
        current_cluster_prices = [clustered_prices[0]]
        current_cluster_extrema = [clustered_extrema[0]]

        for price, extr in zip(clustered_prices[1:], clustered_extrema[1:]):
            cluster_center = np.mean(current_cluster_prices)
            if abs(price - cluster_center) / cluster_center * 100 <= self.cluster_threshold_pct:
                current_cluster_prices.append(price)
                current_cluster_extrema.append(extr)
            else:
                # Finalize cluster
                clusters.append(KeyLevel(
                    price=np.mean(current_cluster_prices),
                    strength=len(current_cluster_prices),
                    level_type=level_type,
                    touches=current_cluster_extrema,
                ))
                current_cluster_prices = [price]
                current_cluster_extrema = [extr]

        # Don't forget last cluster
        clusters.append(KeyLevel(
            price=np.mean(current_cluster_prices),
            strength=len(current_cluster_prices),
            level_type=level_type,
            touches=current_cluster_extrema,
        ))

        return clusters

    def _calculate_trend_quality(
        self,
        peaks: list[Extremum],
        valleys: list[Extremum],
        close: pd.Series,
    ) -> float:
        """Calculate trend quality score 0-1."""
        if len(peaks) < 2 or len(valleys) < 2:
            return 0.0

        # Check consistency of higher highs / higher lows
        peak_trend = sum(1 for i in range(1, len(peaks)) if peaks[i].price > peaks[i-1].price)
        valley_trend = sum(1 for i in range(1, len(valleys)) if valleys[i].price > valleys[i-1].price)

        peak_consistency = peak_trend / (len(peaks) - 1) if len(peaks) > 1 else 0
        valley_consistency = valley_trend / (len(valleys) - 1) if len(valleys) > 1 else 0

        # Prominence factor
        avg_prominence = np.mean([p.prominence for p in peaks + valleys])
        price_range = close.max() - close.min()
        prominence_factor = min(avg_prominence / (price_range * 0.1), 1.0) if price_range > 0 else 0

        return (peak_consistency + valley_consistency + prominence_factor) / 3

    def detect_divergences(
        self,
        close: pd.Series,
        indicator: pd.Series,
        lookback: int = 20,
    ) -> dict[str, Any]:
        """Detect bullish/bearish divergences between price and indicator."""
        if len(close) < lookback or len(indicator) < lookback:
            return {"bullish": False, "bearish": False, "details": []}

        recent_close = close.tail(lookback)
        recent_ind = indicator.tail(lookback)

        # Find peaks in both
        prom = recent_close.std() * 0.01
        price_peaks, _ = find_peaks(recent_close.values, prominence=prom, distance=5)
        ind_peaks, _ = find_peaks(recent_ind.values, prominence=prom, distance=5)

        # Find valleys in both
        price_valleys, _ = find_peaks(-recent_close.values, prominence=prom, distance=5)
        ind_valleys, _ = find_peaks(-recent_ind.values, prominence=prom, distance=5)

        divergences = []

        # Bearish divergence: price higher high, indicator lower high
        if len(price_peaks) >= 2 and len(ind_peaks) >= 2:
            if (recent_close.iloc[price_peaks[-1]] > recent_close.iloc[price_peaks[-2]] and
                recent_ind.iloc[ind_peaks[-1]] < recent_ind.iloc[ind_peaks[-2]]):
                divergences.append({
                    "type": "BEARISH_DIVERGENCE",
                    "price_peak_1": float(recent_close.iloc[price_peaks[-2]]),
                    "price_peak_2": float(recent_close.iloc[price_peaks[-1]]),
                    "ind_peak_1": float(recent_ind.iloc[ind_peaks[-2]]),
                    "ind_peak_2": float(recent_ind.iloc[ind_peaks[-1]]),
                })

        # Bullish divergence: price lower low, indicator higher low
        if len(price_valleys) >= 2 and len(ind_valleys) >= 2:
            if (recent_close.iloc[price_valleys[-1]] < recent_close.iloc[price_valleys[-2]] and
                recent_ind.iloc[ind_valleys[-1]] > recent_ind.iloc[ind_valleys[-2]]):
                divergences.append({
                    "type": "BULLISH_DIVERGENCE",
                    "price_valley_1": float(recent_close.iloc[price_valleys[-2]]),
                    "price_valley_2": float(recent_close.iloc[price_valleys[-1]]),
                    "ind_valley_1": float(recent_ind.iloc[ind_valleys[-2]]),
                    "ind_valley_2": float(recent_ind.iloc[ind_valleys[-1]]),
                })

        return {
            "bullish": any(d["type"] == "BULLISH_DIVERGENCE" for d in divergences),
            "bearish": any(d["type"] == "BEARISH_DIVERGENCE" for d in divergences),
            "details": divergences,
        }


class FibonacciLevels:
    """Calculate Fibonacci retracement and extension levels."""

    @staticmethod
    def retracement(high: float, low: float) -> dict[str, float]:
        """Standard Fibonacci retracement levels."""
        diff = high - low
        return {
            "0.0": high,
            "0.236": high - diff * 0.236,
            "0.382": high - diff * 0.382,
            "0.5": high - diff * 0.5,
            "0.618": high - diff * 0.618,
            "0.786": high - diff * 0.786,
            "1.0": low,
        }

    @staticmethod
    def extension(high: float, low: float, base: float | None = None) -> dict[str, float]:
        """Fibonacci extension levels."""
        if base is None:
            base = low
        diff = high - low
        return {
            "1.0": base + diff * 1.0,
            "1.272": base + diff * 1.272,
            "1.618": base + diff * 1.618,
            "2.0": base + diff * 2.0,
            "2.618": base + diff * 2.618,
        }

    @classmethod
    def from_swing(cls, peaks: list[Extremum], valleys: list[Extremum]) -> dict[str, float] | None:
        """Calculate Fib levels from most recent significant swing."""
        if not peaks or not valleys:
            return None

        last_peak = max(peaks, key=lambda p: p.date)
        last_valley = max(valleys, key=lambda v: v.date)

        if last_peak.date > last_valley.date:
            # Peak after valley - down move, retrace up
            return cls.retracement(last_peak.price, last_valley.price)
        else:
            # Valley after peak - up move, retrace down
            return cls.retracement(last_valley.price, last_peak.price)


def analyze_price_structure(close: pd.Series, config: dict | None = None) -> PriceStructure:
    """Convenience function for quick analysis."""
    cfg = config or {}
    detector = PeakValleyDetector(
        prominence_pct=cfg.get("prominence_pct", 2.0),
        min_distance=cfg.get("min_distance", 5),
        cluster_threshold_pct=cfg.get("cluster_threshold_pct", 1.0),
    )
    return detector.detect(close)