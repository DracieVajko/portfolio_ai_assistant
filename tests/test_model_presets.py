"""Model presets: selection, merge precedence, payload wiring (mocked transport)."""

from __future__ import annotations

from investment_engine.providers import presets as P


def test_summary_preset_values():
    p = P.preset_for("summary")
    assert p["temperature"] == 0.2
    assert p["top_p"] == 0.9
    assert p["min_tokens_floor"] == 0


def test_reasoning_preset_values():
    for stage in ("decision", "discovery", "ai_recommendations", "think"):
        p = P.preset_for(stage)
        assert p["temperature"] == 0.6
        assert p["top_p"] == 0.95
        assert p["min_tokens_floor"] == 2048


def test_merge_context_wins():
    merged = P.merge_sampling(P.preset_for("decision"), {"temperature": 0.9, "top_k": 10})
    assert merged["temperature"] == 0.9
    assert merged["top_k"] == 10
    assert merged["top_p"] == 0.95  # preset kept


def test_merge_ignores_unknown_keys():
    merged = P.merge_sampling(P.preset_for("summary"), {"seed": 42, "foo": "bar"})
    assert "seed" not in merged and "foo" not in merged


def test_writer_stages_single_source():
    from investment_engine import main as engine_main

    assert engine_main._WRITER_STAGES is P.WRITER_STAGES


def _capture_payload(monkeypatch, **context):
    from investment_engine.providers import lmstudio as lms_mod

    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        captured.update(json or {})
        return FakeResp()

    monkeypatch.setattr(lms_mod.requests, "post", fake_post)
    monkeypatch.setattr(
        lms_mod.LMStudioProvider, "_warmup_model", lambda self, model: None
    )
    p = lms_mod.LMStudioProvider(
        base_url="http://x:1234/v1", model="m",
        max_context_tokens=32768, max_output_tokens=8192)
    p.generate("hi", stage=context.pop("_stage", "decision"), context=context)
    return captured


def test_payload_carries_reasoning_preset(monkeypatch):
    out = _capture_payload(monkeypatch)
    assert out["temperature"] == 0.6
    assert out["top_p"] == 0.95
    assert out["top_k"] == 40
    assert out["repeat_penalty"] == 1.0


def test_payload_carries_summary_preset(monkeypatch):
    from investment_engine.providers import lmstudio as lms_mod

    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        lms_mod.requests, "post",
        lambda url, json=None, timeout=None: (captured.update(json or {}), FakeResp())[1],
    )
    monkeypatch.setattr(
        lms_mod.LMStudioProvider, "_warmup_model", lambda self, model: None
    )

    p = lms_mod.LMStudioProvider(
        base_url="http://x:1234/v1", model="m",
        max_context_tokens=32768, max_output_tokens=8192)
    p.generate("hi", stage="summary", context={})
    assert captured["temperature"] == 0.2
    assert captured["repeat_penalty"] == 1.05


def test_reasoning_token_floor(monkeypatch):
    out = _capture_payload(monkeypatch, max_output_tokens=400)
    assert out["max_tokens"] == 2048  # floor applied


def test_no_floor_for_summary(monkeypatch):
    from investment_engine.providers import lmstudio as lms_mod

    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        lms_mod.requests, "post",
        lambda url, json=None, timeout=None: (captured.update(json or {}), FakeResp())[1],
    )
    monkeypatch.setattr(
        lms_mod.LMStudioProvider, "_warmup_model", lambda self, model: None
    )

    p = lms_mod.LMStudioProvider(
        base_url="http://x:1234/v1", model="m",
        max_context_tokens=32768, max_output_tokens=8192)
    p.generate("hi", stage="summary",
               context={"max_output_tokens": 400})
    assert captured["max_tokens"] == 400
