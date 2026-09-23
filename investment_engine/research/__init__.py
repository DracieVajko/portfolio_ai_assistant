# Investment Engine Research Package
from investment_engine.research.news_engine import StrictNewsFetcher, NewsItem, analyze_news_sentiment
from investment_engine.research.technical_analysis import TechnicalAnalyzer, IndicatorConfig, FinVizEnrichment, compute_derived_metrics
from investment_engine.research.peak_valley import PeakValleyDetector, PriceStructure, Extremum, KeyLevel, FibonacciLevels, analyze_price_structure
from investment_engine.research.market_regime import EXI2RegimeAnalyzer, RegimeResult, RegimeThresholds, TimeframeConfig

__all__ = [
    "StrictNewsFetcher",
    "NewsItem", 
    "analyze_news_sentiment",
    "TechnicalAnalyzer",
    "IndicatorConfig",
    "FinVizEnrichment",
    "compute_derived_metrics",
    "PeakValleyDetector",
    "PriceStructure",
    "Extremum",
    "KeyLevel",
    "FibonacciLevels",
    "analyze_price_structure",
    "EXI2RegimeAnalyzer",
    "RegimeResult",
    "RegimeThresholds",
    "TimeframeConfig",
]