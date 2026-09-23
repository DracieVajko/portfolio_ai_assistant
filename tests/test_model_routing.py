"""Stage->model routing + LM Studio unload (offline, mocked transport)."""

from __future__ import annotations

from investment_engine.main import _model_for_stage
from investment_engine.providers import lmstudio as lms_mod


def _settings(decision="deepseek-r1-finance-reasoning-14b", writer="google/gemma-4-12b"):
    from types import SimpleNamespace

    return SimpleNamespace(decision_model=decision, writer_model=writer)


def _provider():
    return lms_mod.LMStudioProvider(
        base_url="http://x:1234/v1", model="m",
        max_context_tokens=32768, max_output_tokens=8192)


class _FakeResp:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}

    def raise_for_status(self):
        pass


class _FakeResp:
    status_code = 200
    text = '{"ok": true}'

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}

    def raise_for_status(self):
        pass


def _mock_transport(monkeypatch):
    monkeypatch.setattr(lms_mod.requests, "post", lambda *a, **k: _FakeResp())
    monkeypatch.setattr(lms_mod.requests, "delete", lambda *a, **k: _FakeResp())
    monkeypatch.setattr(
        lms_mod.LMStudioProvider, "_warmup_model", lambda self, model: None)


def _capture(monkeypatch, **context):
    captured = {}

    class CapResp(_FakeResp):
        pass

    orig = CapResp.json

    def fake_post(url, json=None, timeout=None):
        captured.update(json or {})
        return CapResp()

    monkeypatch.setattr(lms_mod.requests, "post", fake_post)
    monkeypatch.setattr(
        lms_mod.LMStudioProvider, "_warmup_model", lambda self, model: None)
    _provider().generate("hi", stage=context.pop("_stage", "decision"), context=context)
    return captured


def test_summary_routes_to_writer():
    s = _settings()
    assert _model_for_stage(s, "summary") == "google/gemma-4-12b"


def test_reasoning_stages_route_to_decision():
    s = _settings()
    for stage in ("decision", "discovery", "ai_recommendations", "think", "research"):
        assert _model_for_stage(s, stage) == "deepseek-r1-finance-reasoning-14b"


def test_writer_fallback_to_decision_when_empty():
    s = _settings(writer="")
    assert _model_for_stage(s, "summary") == "deepseek-r1-finance-reasoning-14b"


def test_generate_uses_routed_model(monkeypatch):
    """_generate_with_fallback passes the routed model into provider context."""
    from investment_engine.main import _generate_with_fallback

    seen = {}

    class FakeProvider:
        name = "fake"

        def generate(self, prompt, *, stage, context=None):
            seen.update(context or {})
            return "ok content"

    _generate_with_fallback(FakeProvider(), "hi", _settings(), tokens=10, stage="summary")
    assert seen.get("model") == "google/gemma-4-12b"
    _generate_with_fallback(FakeProvider(), "hi", _settings(), tokens=10, stage="decision")
    assert seen.get("model") == "deepseek-r1-finance-reasoning-14b"


def test_payload_carries_reasoning_preset(monkeypatch):
    out = _capture(monkeypatch, _stage="decision")
    assert out["temperature"] == 0.6
    assert out["top_p"] == 0.95
    assert out["top_k"] == 40
    assert out["repeat_penalty"] == 1.0


def test_payload_carries_summary_preset(monkeypatch):
    out = _capture(monkeypatch, _stage="summary")
    assert out["temperature"] == 0.2
    assert out["repeat_penalty"] == 1.05


def test_reasoning_token_floor(monkeypatch):
    out = _capture(monkeypatch, _stage="decision", max_output_tokens=400)
    assert out["max_tokens"] == 2048  # floor applied


def test_no_floor_for_summary(monkeypatch):
    out = _capture(monkeypatch, _stage="summary", max_output_tokens=400)
    assert out["max_tokens"] == 400


def test_unload_never_raises_and_tries_variants(monkeypatch):
    calls = []
    _mock_transport(monkeypatch)
    orig_post = lms_mod.requests.post

    def spy_post(url, json=None, timeout=None):
        calls.append(("POST", url))
        return orig_post(url, json=json, timeout=timeout)

    monkeypatch.setattr(lms_mod.requests, "post", spy_post)
    assert _provider().unload_model("m") is True
    assert any("unload" in u for _, u in calls)


def test_unload_unsupported_server_returns_false(monkeypatch):
    class BadResp(_FakeResp):
        text = '{"error":"Unexpected endpoint or method. (POST /x)"}'

    monkeypatch.setattr(lms_mod.requests, "post", lambda *a, **k: BadResp())
    monkeypatch.setattr(lms_mod.requests, "delete", lambda *a, **k: BadResp())
    assert _provider().unload_model("m") is False  # no raise, clear False


def test_unload_clears_warmup_cache(monkeypatch):
    p = _provider()
    lms_mod._model_warmed.add((p.base_url, "m"))

    def _down(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(lms_mod.requests, "post", _down)
    monkeypatch.setattr(lms_mod.requests, "delete", _down)
    assert p.unload_model("m") is False
    assert (p.base_url, "m") not in lms_mod._model_warmed


def test_generate_retries_once_on_400_then_succeeds(monkeypatch):
    from investment_engine.providers import lmstudio as lms_mod

    calls = []

    class LoadingResp:
        status_code = 400
        text = '{"error": "model loading"}'

        def raise_for_status(self):
            raise lms_mod.requests.exceptions.HTTPError("400")

    class OkResp:
        status_code = 200
        text = '{"ok": true}'

        def json(self):
            return {"choices": [{"message": {"content": "recovered"}}]}

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        calls.append(url)
        return LoadingResp() if len(calls) == 1 else OkResp()

    monkeypatch.setattr(lms_mod.requests, "post", fake_post)
    monkeypatch.setattr(
        lms_mod.LMStudioProvider, "_warmup_model", lambda self, model: None)
    monkeypatch.setattr(lms_mod.time, "sleep", lambda s: calls.append(f"sleep:{s}"))
    p = lms_mod.LMStudioProvider(base_url="http://x:1234/v1", model="m")
    assert p.generate("hi", stage="decision", context={}) == "recovered"
    assert any(str(c).startswith("sleep:") for c in calls)


def test_unload_lmstudio_models_helper(monkeypatch):
    monkeypatch.setattr(
        lms_mod, "_unload_via_api", lambda base_url, m: m != "bad"
    )
    res = lms_mod.unload_lmstudio_models("http://x:1234/v1", ["a", "bad", "a"])
    assert res == {"a": True, "bad": False}
