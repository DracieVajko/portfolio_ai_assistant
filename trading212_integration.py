"""
Trading212 Integration Module  –  Read-Only (GET only)

Spája trading212_auth.py + trading212_portfolio.py do jedného
entry pointu pre portfolio_ai_assistant.py.

Zmeny oproti predchádzajúcej verzii:
  - Odstránené buy/sell metódy (API key je read-only)
  - Opravené parsovanie T212 cash response (polia: free, invested, pieCash, result, total)
  - Opravený name mismatch: place_order → neexistuje, vyhodené
  - Pridané odysseus/Ollama AI analýzy
"""

import os
import json
import logging
import socket
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from trading212_auth import Trade212Client
from trading212_portfolio import PortfolioMonitor

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('Trading212Integration')


# ===========================================================================
# ODYSSEUS / OLLAMA  AI BRIDGE
# ===========================================================================

class OllamaClient:
    """
    Volá Ollama API (alebo kompatibilný server ako odysseus).

    Priorita URL:
      1. odysseus_url  (WSL server s načítanými skills)
      2. ollama_url    (priamo Windows Ollama)
    """

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434",
                 odysseus_url: str = None,
                 model: str = "dolphin3:latest",
                 timeout: int = 300):
        self.ollama_url   = ollama_url.rstrip("/")
        self.odysseus_url = odysseus_url.rstrip("/") if odysseus_url else None
        self.model        = model
        self.timeout      = timeout

    def _active_url(self) -> str:
        """Vracia odysseus URL ak je dostupný, inak ollama."""
        if self.odysseus_url:
            try:
                r = requests.get(f"{self.odysseus_url}/api/tags", timeout=10)
                if r.status_code == 200:
                    logger.info(f"[OllamaClient] Používam odysseus: {self.odysseus_url}")
                    return self.odysseus_url
            except Exception:
                pass
            logger.warning(f"[OllamaClient] Odysseus nedostupný, fallback na Ollama")
        return self.ollama_url

    def _resolve_model(self, base_url: str) -> str:
        """Zistí dostupné modely a vyberie najlepší."""
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return self.model
            names = [m.get("name", "") for m in r.json().get("models", [])]
            if not names:
                return self.model
            # Preferuj nakonfigurovaný model
            for candidate in [self.model, "dolphin3:latest", "exaone3.5:7.8b"]:
                if any(candidate in n for n in names):
                    return candidate
            return names[0]
        except Exception:
            return self.model

    def chat(self, prompt: str, system: str = None,
             temperature: float = 0.15, num_ctx: int = 16384) -> str:
        base = self._active_url()
        model = self._resolve_model(base)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   model,
            "stream":  False,
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        try:
            r = requests.post(f"{base}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            return f"ERROR [OllamaClient.chat]: {e}"

    def is_available(self) -> bool:
        base = self._active_url()
        try:
            return requests.get(f"{base}/api/tags", timeout=4).status_code == 200
        except Exception:
            return False


# ===========================================================================
# PORTFOLIO AI ANALYST
# ===========================================================================

SYSTEM_PROMPT = (
    "Si prísny, faktický investičný analytik. "
    "NIKDY nevymýšľaš pravdepodobnosti, ceny ani názvy spoločností. "
    "NIKDY neprepisuješ vypočítané skóre. "
    "Pracuješ výhradne so štruktúrovanými dátami ktoré dostaneš. "
    "Ak sú dáta nekvalitné alebo chýbajú, povieš 'neoverené'. "
    "Odpovedáš v slovenčine. Buď stručný, konkrétny a akčný."
)


def build_portfolio_prompt(summary: Dict[str, Any], metrics: Dict[str, Any],
                            order_history: List[Dict], dividends: List[Dict]) -> str:
    """Zostaví prompt pre AI analýzu portfólia."""
    ts = summary.get("timestamp", "")
    total = summary.get("total_equity", 0)
    cash = summary.get("cash", 0)
    pnl = summary.get("total_pnl", 0)
    pnl_pct = summary.get("total_pnl_pct", 0)
    n = summary.get("positions_count", 0)

    lines = [
        f"=== TRADING212 PORTFÓLIO – {ts} ===",
        f"Celková hodnota účtu : {total:,.2f} EUR",
        f"Dostupná hotovosť    : {cash:,.2f} EUR",
        f"Celkový P&L          : {pnl:,.2f} EUR ({pnl_pct:.1f}%)",
        f"Počet pozícií        : {n}",
        "",
        "--- POZÍCIE (top 10 podľa hodnoty) ---",
    ]

    positions = sorted(summary.get("positions", []),
                       key=lambda p: p.get("value", 0), reverse=True)[:10]
    for p in positions:
        sym   = p.get("symbol", "?")
        val   = p.get("value", 0)
        qty   = p.get("quantity", 0)
        pnlp  = p.get("pnl", 0)
        pnlpp = p.get("pnl_pct", 0)
        lines.append(
            f"  {sym:<12} qty={qty:>8.4f}  hodnota={val:>10.2f}  "
            f"P&L={pnlp:>+8.2f} ({pnlpp:>+5.1f}%)"
        )

    # Najlepší / najhorší
    best  = metrics.get("best_performer")
    worst = metrics.get("worst_performer")
    if best:
        lines += ["", f"Najlepší výkon: {best['symbol']}  {best['pnl_pct']:+.1f}%"]
    if worst:
        lines += [f"Najhorší výkon: {worst['symbol']}  {worst['pnl_pct']:+.1f}%"]

    # Posledné obchody
    if order_history:
        lines += ["", "--- POSLEDNÉ OBCHODY (max 5) ---"]
        for o in order_history[:5]:
            lines.append(
                f"  {o.get('side','?')} {o.get('symbol','?')}  "
                f"qty={o.get('quantity',0):.4f}  "
                f"@{o.get('price',0):.2f}  [{o.get('timestamp','')}]"
            )

    # Dividendy
    if dividends:
        total_div = sum(float(d.get("amount", 0)) for d in dividends[:20])
        lines += ["", f"Dividendy (posledných {min(20, len(dividends))}): {total_div:.2f} EUR"]

    lines += [
        "",
        "=== ÚLOHA ===",
        "Analyzuj portfólio a daj konkrétne odporúčania:",
        "1. Pozície ktoré sú problematické (veľké straty, nadmerná koncentrácia)",
        "2. Pozície kde je potenciál rastu alebo akcia",
        "3. Diverzifikácia – je portfólio vyvážené?",
        "4. Hotovosť – má zmysel ju investovať alebo držať?",
        "5. Celkové hodnotenie a top 2-3 konkrétne akcie na ďalší týždeň",
    ]
    return "\n".join(lines)


# ===========================================================================
# TRADING212 INTEGRATION
# ===========================================================================

class Trading212Integration:
    """
    Hlavný entry point pre Trading212 portfólio.
    Read-only – žiadne buy/sell operácie.
    Autentifikácia: HTTP Basic Auth (api_key:api_secret).
    """

    def __init__(self, api_key: str = None, api_secret: str = None,
                 account_id: str = None,
                 config_file: str = "api.env",
                 ollama_url: str = "http://127.0.0.1:11434",
                 odysseus_url: str = None,
                 ai_model: str = "dolphin3:latest"):

        self.config = self._load_env(config_file)

        self.api_key    = api_key    or self.config.get("TRADING212_API_KEY")    or os.getenv("TRADING212_API_KEY")
        self.api_secret = api_secret or self.config.get("TRADING212_API_SECRET") or os.getenv("TRADING212_API_SECRET")
        self.account_id = account_id or self.config.get("TRADING212_ACCOUNT_ID") or os.getenv("TRADING212_ACCOUNT_ID")

        if not self.api_key or not self.api_secret:
            logger.warning("TRADING212_API_KEY alebo TRADING212_API_SECRET nie sú nastavené – integrácia vypnutá")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Trading212 inicializovaný (credentials loaded)")

        if self.enabled:
            self.monitor = PortfolioMonitor(self.api_key, self.api_secret, self.account_id)
        else:
            self.monitor = None

        # Odysseus/Ollama
        _odysseus = odysseus_url or self.config.get("ODYSSEUS_URL") or os.getenv("ODYSSEUS_URL")
        _ollama   = ollama_url   or self.config.get("OLLAMA_URL")   or "http://127.0.0.1:11434"
        _model    = ai_model     or self.config.get("OLLAMA_MODEL")  or "dolphin3:latest"
        self.ai = OllamaClient(
            ollama_url=_ollama,
            odysseus_url=_odysseus,
            model=_model,
        )

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_env(config_file: str) -> Dict[str, str]:
        cfg: Dict[str, str] = {}
        path = Path(config_file)
        if not path.exists():
            return cfg
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
        return cfg

    def is_enabled(self) -> bool:
        return self.enabled

    # -----------------------------------------------------------------------
    # Portfolio data  (read-only)
    # -----------------------------------------------------------------------

    def get_account_summary(self, use_cache: bool = True) -> Dict[str, Any]:
        if not self.enabled:
            return {"error": "Trading212 nie je povolený", "status": "disabled"}
        return self.monitor.get_portfolio_summary(use_cache=use_cache)

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return self.monitor.get_positions()

    def get_cash(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return self.monitor.get_account_cash()

    def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return self.monitor.get_order_history(limit)

    def get_dividends(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return self.monitor.get_dividends(limit)

    def get_pies(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return self.monitor.get_pies()

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return self.monitor.calculate_portfolio_metrics()

    # -----------------------------------------------------------------------
    # AI Analysis
    # -----------------------------------------------------------------------

    def analyze_portfolio(self, temperature: float = 0.15, num_ctx: int = 16384) -> str:
        """
        Komplexná AI analýza portfólia (odysseus → Ollama fallback).
        """
        if not self.enabled:
            return "ERROR: Trading212 integrácia nie je povolená."

        if not self.ai.is_available():
            return "ERROR: Ollama/odysseus server nedostupný."

        logger.info("Zbieranie portfólio dát pre AI analýzu…")
        summary  = self.get_account_summary()
        metrics  = self.get_portfolio_metrics()
        orders   = self.get_order_history(limit=20)
        divs     = self.get_dividends(limit=20)

        if summary.get("status") == "failed":
            return f"ERROR: {summary.get('error')}"

        prompt = build_portfolio_prompt(summary, metrics, orders, divs)
        logger.info("Volám AI model…")
        return self.ai.chat(prompt, system=SYSTEM_PROMPT,
                            temperature=temperature, num_ctx=num_ctx)

    # -----------------------------------------------------------------------
    # Health & Export
    # -----------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled"}
        try:
            cash = self.monitor.get_account_cash()
            if cash.get("status") == "failed":
                return {"status": "error", "message": cash.get("error"), "enabled": True}
            return {
                "status":    "healthy",
                "enabled":   True,
                "cash_free": cash.get("free", 0),
                "cash_total": cash.get("total", 0),
                "ai_available": self.ai.is_available(),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "enabled": True}

    def export_portfolio(self, filepath: str = "trading212_portfolio.json") -> bool:
        if not self.enabled:
            return False
        try:
            data = {
                "timestamp":         datetime.now().isoformat(),
                "account_summary":   self.get_account_summary(),
                "portfolio_metrics": self.get_portfolio_metrics(),
                "positions":         self.get_positions(),
                "order_history":     self.get_order_history(10),
                "dividends":         self.get_dividends(10),
                "pies":              self.get_pies(),
            }
            Path(filepath).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            logger.info(f"Export: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Export zlyhal: {e}")
            return False


# ===========================================================================
# CONVENIENCE + CLI
# ===========================================================================

def create_integration(config_file: str = "api.env",
                       odysseus_url: str = None) -> Optional[Trading212Integration]:
    t = Trading212Integration(config_file=config_file, odysseus_url=odysseus_url)
    if not t.is_enabled():
        logger.error("Trading212 integrácia zlyhala – skontroluj TRADING212_API_KEY v api.env")
        return None
    return t


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading212 Integration CLI")
    parser.add_argument("--summary",   action="store_true", help="Prehľad účtu")
    parser.add_argument("--positions", action="store_true", help="Pozície")
    parser.add_argument("--cash",      action="store_true", help="Hotovosť")
    parser.add_argument("--orders",    action="store_true", help="História obchodov")
    parser.add_argument("--dividends", action="store_true", help="Dividendy")
    parser.add_argument("--pies",      action="store_true", help="Pies")
    parser.add_argument("--health",    action="store_true", help="Health check")
    parser.add_argument("--analyze",   action="store_true", help="AI analýza portfólia")
    parser.add_argument("--export",    type=str,            help="Export do JSON súboru")
    parser.add_argument("--odysseus",  type=str,            help="Odysseus URL (WSL)")
    parser.add_argument("--config",    type=str, default="api.env")
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    t = create_integration(config_file=args.config, odysseus_url=args.odysseus)
    if not t:
        exit(1)

    if args.health:
        print(json.dumps(t.health_check(), indent=2, default=str))

    elif args.summary:
        print(json.dumps(t.get_account_summary(), indent=2, default=str))

    elif args.positions:
        print(json.dumps(t.get_positions(), indent=2, default=str))

    elif args.cash:
        print(json.dumps(t.get_cash(), indent=2, default=str))

    elif args.orders:
        print(json.dumps(t.get_order_history(), indent=2, default=str))

    elif args.dividends:
        print(json.dumps(t.get_dividends(), indent=2, default=str))

    elif args.pies:
        print(json.dumps(t.get_pies(), indent=2, default=str))

    elif args.analyze:
        print("\n" + "="*60)
        print(t.analyze_portfolio())
        print("="*60)

    elif args.export:
        ok = t.export_portfolio(args.export)
        print(f"Export {'OK' if ok else 'FAILED'}: {args.export}")

    else:
        # Default: health + krátky summary
        h = t.health_check()
        print(f"Status  : {h.get('status')}")
        print(f"Cash    : {h.get('cash_free', 0):.2f} EUR (free) / {h.get('cash_total', 0):.2f} EUR (total)")
        print(f"AI      : {'dostupná' if h.get('ai_available') else 'nedostupná'}")
