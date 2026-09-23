"""
Trading212 API Authentication Module

Trading212 API v0 uses a SINGLE API key.
Auth header: Authorization: <api_key>   (no Bearer, no Basic, just the key)

Skutočné T212 response fieldy:
  /equity/account/cash  → { free, invested, pieCash, result, total }
  /equity/account/info  → { currencyCode, id }
  /equity/portfolio     → [ { ticker, quantity, averagePrice, currentPrice,
                               ppl, fxPpl, initialFillDate, maxBuy, maxSell,
                               pieQuantity } ]
  /equity/history/orders?limit=N → { items: [...] }
"""

import os
import requests
import time
import json
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path


# ===========================================================================
# TRADING212 AUTHENTICATION
# ===========================================================================

class Trading212Auth:
    """Secure API authentication handler for Trading212."""

    def __init__(self, api_key: str, api_secret: str = None, account_id: str = None):
        """
        T212 API v0 (current) uses HTTP Basic Auth: base64(api_key:api_secret).
        Both api_key and api_secret are required.
        """
        self.api_key = api_key
        self.api_secret = api_secret or ""
        self.account_id = account_id
        self.base_url = "https://live.trading212.com"
        self.api_base = f"{self.base_url}/api/v0"
        self.session = requests.Session()
        self._rate_limit_timestamp = 0
        self._request_count = 0

    def generate_auth_header(self) -> Dict[str, str]:
        """
        T212 Basic Auth: Authorization: Basic base64(api_key:api_secret)
        api_key = username, api_secret = password.
        """
        if not self.api_key:
            raise ValueError("API key is required")
        import base64
        creds = base64.b64encode(f"{self.api_key}:{self.api_secret}".encode()).decode()
        return {
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/json",
        }

    def verify_credentials(self) -> bool:
        try:
            headers = self.generate_auth_header()
            response = self.session.get(
                f"{self.api_base}/equity/account/cash",
                headers=headers,
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            print(f"[Trading212Auth] Credential verification error: {e}")
            return False

    def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retry_count: int = 5,
    ) -> Dict[str, Any]:
        headers = self.generate_auth_header()
        url = f"{self.api_base}/{endpoint.lstrip('/')}"

        for attempt in range(retry_count):
            try:
                self._handle_rate_limit()

                kwargs = dict(headers=headers, timeout=20)
                if params:
                    kwargs["params"] = params

                if method.upper() == "GET":
                    response = self.session.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=data, **kwargs)
                elif method.upper() == "DELETE":
                    response = self.session.delete(url, **kwargs)
                else:
                    response = self.session.request(method, url, json=data, **kwargs)

                if response.status_code == 429:
                    # Better 429 handling with exponential backoff + jitter
                    wait = int(response.headers.get("Retry-After", 2 ** attempt * 10))
                    wait = min(wait, 120)  # Cap at 2 minutes
                    print(f"[Trading212Auth] Rate limited (429), waiting {wait}s... (attempt {attempt + 1}/{retry_count})")
                    time.sleep(wait)
                    continue

                if response.status_code == 401:
                    return {"error": "Unauthorized – check TRADING212_API_KEY", "status": "auth_failed"}

                if response.status_code == 403:
                    return {"error": "Forbidden – API key has no permission for this endpoint (read-only key?)", "status": "forbidden"}

                if response.status_code == 404:
                    return {"error": f"404 Not Found: {endpoint}", "status": "not_found"}

                if response.status_code in (200, 201):
                    try:
                        return response.json()
                    except Exception:
                        return {"success": True, "status": "ok"}

                if response.status_code >= 500:
                    if attempt < retry_count - 1:
                        time.sleep(2 ** attempt)
                        continue

                try:
                    return response.json()
                except Exception:
                    return {"error": f"HTTP {response.status_code}", "status": "error"}

            except requests.Timeout:
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": f"{type(e).__name__}: {e}", "status": "error"}

        return {"error": f"Request failed after {retry_count} retries: {endpoint}", "status": "error"}

    def _handle_rate_limit(self, max_requests: int = 50):
        current_time = time.time()
        if current_time - self._rate_limit_timestamp > 60:
            self._rate_limit_timestamp = current_time
            self._request_count = 0
        self._request_count += 1
        if self._request_count > max_requests:
            sleep_time = 60 - (current_time - self._rate_limit_timestamp)
            if sleep_time > 0:
                print(f"[Trading212Auth] Rate limit – sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self._rate_limit_timestamp = time.time()
                self._request_count = 0


class Trade212Client:
    """Trading212 API Client – read-only operations (GET only)."""

    def __init__(self, api_key: str, api_secret: str = None, account_id: str = None):
        """Both api_key and api_secret required for Basic Auth."""
        self.auth = Trading212Auth(api_key, api_secret, account_id)
        self.account_id = account_id

    # ------------------------------------------------------------------
    # ACCOUNT
    # ------------------------------------------------------------------

    def get_account_cash(self) -> Dict[str, Any]:
        """
        GET /api/v0/equity/account/cash
        Response: { free, invested, pieCash, result, total }
        """
        return self.auth.make_request("GET", "/equity/account/cash")

    def get_account_info(self) -> Dict[str, Any]:
        """
        GET /api/v0/equity/account/info
        Response: { currencyCode, id }
        """
        return self.auth.make_request("GET", "/equity/account/info")

    # ------------------------------------------------------------------
    # PORTFOLIO
    # ------------------------------------------------------------------

    def get_positions(self) -> Any:
        """
        GET /api/v0/equity/portfolio
        Response: list of position objects:
          { ticker, quantity, averagePrice, currentPrice,
            ppl, fxPpl, initialFillDate, maxBuy, maxSell, pieQuantity }
        """
        return self.auth.make_request("GET", "/equity/portfolio")

    # ------------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------------

    def get_order_history(self, limit: int = 50, ticker: str = None) -> Dict[str, Any]:
        """GET /api/v0/equity/history/orders"""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self.auth.make_request("GET", "/equity/history/orders", params=params)

    def get_dividends(self, limit: int = 50) -> Dict[str, Any]:
        """GET /api/v0/history/dividends"""
        return self.auth.make_request("GET", "/history/dividends", params={"limit": limit})

    def get_transactions(self, limit: int = 50) -> Dict[str, Any]:
        """GET /api/v0/history/transactions"""
        return self.auth.make_request("GET", "/history/transactions", params={"limit": limit})

    def get_all_transaction_history(self, limit: int = 50, max_pages: int = 40) -> Dict[str, Any]:
        """GET /api/v0/equity/history/transactions with cursor pagination.

        Read-only: follows cursor / nextPagePath until exhausted or max_pages.
        Returns {"pages": [...raw page payloads...], "exhausted": bool}.
        Sanitization happens downstream; this method never logs payloads.
        """
        from urllib.parse import parse_qs, urlparse
        pages: list[Any] = []
        cursor: str | None = None
        seen: set[str] = set()
        exhausted = False
        while len(pages) < max_pages:
            params: Dict[str, Any] = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            payload = self.auth.make_request(
                "GET", "/equity/history/transactions", params=params)
            pages.append(payload)
            if not isinstance(payload, dict) or payload.get("error"):
                break
            items = payload.get("items")
            if not items:
                exhausted = True
                break
            nxt: str | None = None
            for key in ("cursor", "nextCursor", "next_cursor", "pageCursor"):
                if payload.get(key):
                    nxt = str(payload[key])
                    break
            if nxt is None:
                nxt_path = payload.get("nextPagePath") or payload.get("next_page")
                if isinstance(nxt_path, str) and nxt_path:
                    try:
                        qs = parse_qs(urlparse(nxt_path).query)
                        for ck in ("cursor", "page", "offset"):
                            if qs.get(ck):
                                nxt = qs[ck][0]
                                break
                    except Exception:
                        nxt = None
            if not nxt or nxt in seen:
                exhausted = True
                break
            seen.add(nxt)
            cursor = nxt
        return {"pages": pages, "exhausted": exhausted}

    # ------------------------------------------------------------------
    # PIES
    # ------------------------------------------------------------------

    def get_pies(self) -> Any:
        """GET /api/v0/equity/pies"""
        return self.auth.make_request("GET", "/equity/pies")

    def get_pie(self, pie_id: str) -> Dict[str, Any]:
        """GET /api/v0/equity/pies/{id}"""
        return self.auth.make_request("GET", f"/equity/pies/{pie_id}")

    # ------------------------------------------------------------------
    # MARKET DATA
    # ------------------------------------------------------------------

    def search_instrument(self, query: str) -> Dict[str, Any]:
        """GET /api/v0/equity/metadata/instruments?search=<query>"""
        return self.auth.make_request("GET", "/equity/metadata/instruments",
                                      params={"search": query})


# ===========================================================================
# DIAGNOSTIC DUMP (placed at bottom to avoid import-order issues)
# ===========================================================================

def dump_raw_t212_diagnostics(client: "Trade212Client", run_id: str = None) -> Dict[str, Any]:
    """
    Diagnostic-only debug dump: calls /equity/account/cash and /equity/portfolio
    once each, dumps full raw JSON responses, logs all keys/types, computes delta combinations.
    Does NOT alter any existing reconciliation logic.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    timestamp = datetime.now().isoformat()
    dump_dir = Path("reports")
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"raw_endpoint_dump_{run_id}.json"
    
    diagnostic = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "cash_endpoint": {},
        "portfolio_endpoint": {},
        "delta_combinations": {},
    }
    
    # 1. Call /equity/account/cash
    print("\n" + "="*60)
    print("RAW CASH ENDPOINT FIELDS:")
    print("="*60)
    try:
        cash_resp = client.get_account_cash()
        diagnostic["cash_endpoint"] = cash_resp
        if isinstance(cash_resp, dict):
            for k, v in cash_resp.items():
                print(f"  {k}: {v} (type: {type(v).__name__})")
            diagnostic["cash_keys"] = {k: type(v).__name__ for k, v in cash_resp.items()}
        else:
            print(f"  Response type: {type(cash_resp).__name__}, value: {cash_resp}")
            diagnostic["cash_keys"] = f"Non-dict response: {type(cash_resp).__name__}"
    except Exception as e:
        print(f"  ERROR fetching cash: {e}")
        diagnostic["cash_error"] = str(e)
        cash_resp = {}
    
    # 2. Call /equity/portfolio
    print("\n" + "="*60)
    print("RAW PORTFOLIO ENDPOINT FIELDS (first 3 positions):")
    print("="*60)
    try:
        portfolio_resp = client.get_positions()
        diagnostic["portfolio_endpoint"] = portfolio_resp
        if isinstance(portfolio_resp, list):
            print(f"  Total positions: {len(portfolio_resp)}")
            diagnostic["portfolio_count"] = len(portfolio_resp)
            for i, pos in enumerate(portfolio_resp[:3]):
                print(f"  Position {i+1}:")
                if isinstance(pos, dict):
                    for k, v in pos.items():
                        print(f"    {k}: {v} (type: {type(v).__name__})")
                    diagnostic[f"position_{i}_keys"] = {k: type(v).__name__ for k, v in pos.items()}
        else:
            print(f"  Response type: {type(portfolio_resp).__name__}, value: {portfolio_resp}")
            diagnostic["portfolio_error"] = f"Non-list response: {type(portfolio_resp).__name__}"
    except Exception as e:
        print(f"  ERROR fetching portfolio: {e}")
        diagnostic["portfolio_error"] = str(e)
        portfolio_resp = []
    
    # 3. Compute delta combinations
    print("\n" + "="*60)
    print("ALL DELTA COMBINATIONS:")
    print("="*60)
    
    def _safe_float(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default
    
    cash = cash_resp if isinstance(cash_resp, dict) else {}
    portfolio = portfolio_resp if isinstance(portfolio_resp, list) else []
    
    # Extract cash fields
    free = _safe_float(cash.get("free"))
    invested = _safe_float(cash.get("invested"))
    pie_cash = _safe_float(cash.get("pieCash"))
    blocked = _safe_float(cash.get("blocked"))
    result = _safe_float(cash.get("result"))
    ppl = _safe_float(cash.get("ppl"))
    total = _safe_float(cash.get("total"))
    
    # Sum positions
    positions_sum = sum(_safe_float(p.get("value", p.get("value_eur", p.get("quantity", 0) * p.get("currentPrice", p.get("currentPrice", 0))))) for p in portfolio if isinstance(p, dict))
    
    # Compute all combinations
    deltas = {}
    deltas["total"] = total
    deltas["free"] = free
    deltas["invested"] = invested
    deltas["pieCash"] = pie_cash
    deltas["blocked"] = blocked
    deltas["result"] = result
    deltas["ppl"] = ppl
    deltas["positions_sum"] = positions_sum
    
    deltas["total_minus_free_plus_pieCash"] = total - (free + pie_cash)
    deltas["total_minus_invested_plus_free_plus_pieCash"] = total - (invested + free + pie_cash)
    deltas["total_minus_positions_sum"] = total - positions_sum
    deltas["invested_minus_positions_sum"] = invested - positions_sum
    
    # Any "blocked", "pending", "result", "ppl" fields summed separately
    extra_fields = {}
    for field in ["blocked", "pending", "result", "ppl", "unrealizedPnl", "unrealized_pnl"]:
        if field in cash:
            extra_fields[field] = _safe_float(cash.get(field))
    deltas["extra_cash_fields"] = extra_fields
    
    diagnostic["delta_combinations"] = deltas
    
    for k, v in deltas.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.2f}")
    
    # Save full dump to file
    try:
        dump_path = Path("reports") / f"raw_endpoint_dump_{run_id}.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps({
            "run_id": run_id,
            "timestamp_utc": datetime.now().isoformat(),
            "cash_endpoint": cash_resp,
            "portfolio_endpoint": portfolio_resp,
            "delta_combinations": deltas,
        }, indent=2, default=str), encoding="utf-8")
        print(f"\nFull raw dump saved to: {dump_path}")
    except Exception as e:
        print(f"  ERROR saving dump: {e}")
    
    print("="*60)
    return diagnostic