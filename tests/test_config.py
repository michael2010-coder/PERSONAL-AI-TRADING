"""Config loading, and the two locks in front of live trading."""
import os

import pytest

import main as cli
from trading.config import load_config


def write(tmp_path, body):
    path = os.path.join(str(tmp_path), "config.yaml")
    with open(path, "w") as fh:
        fh.write(body)
    return path


BASE = """
crypto:
  exchange: binance
  mode: {mode}
  symbol: BTC/USDT
  timeframe: 1h
risk:
  allocated_capital_usdt: 100.0
  max_position_size_pct: 10.0
  stop_loss_pct: 2.0
  take_profit_pct: 4.0
  max_daily_loss_pct: 5.0
strategy:
  ema_fast: 12
  ema_slow: 26
"""


def test_the_shipped_config_loads_and_is_on_testnet():
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    assert cfg.crypto.mode == "testnet", "the committed default must never be live"
    assert cfg.risk.stop_loss_pct > 0
    assert not cfg.is_live


def test_a_bad_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, BASE.format(mode="yolo")))


def test_a_typo_in_a_risk_key_is_rejected_not_ignored(tmp_path):
    body = BASE.format(mode="testnet").replace("stop_loss_pct: 2.0", "stop_los_pct: 2.0")
    with pytest.raises(ValueError):
        load_config(write(tmp_path, body))


def test_secrets_are_named_in_config_never_stored_in_it():
    with open(os.path.join(cli.ROOT, "config.yaml")) as fh:
        text = fh.read()
    assert "api_key_env" in text
    for leak in ("api_key:", "api_secret:", "secret:"):
        assert leak not in text


def test_live_mode_without_the_flag_refuses_to_trade(tmp_path, monkeypatch, caplog):
    path = write(tmp_path, BASE.format(mode="live"))
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    assert cli.main(["--config", path, "trade", "--once"]) == 2
    assert "Refusing to trade" in caplog.text


def test_live_mode_without_keys_refuses_to_trade(tmp_path, monkeypatch):
    path = write(tmp_path, BASE.format(mode="live"))
    monkeypatch.setenv("BINANCE_API_KEY", "")
    monkeypatch.setenv("BINANCE_API_SECRET", "")
    assert cli.main(["--config", path, "trade", "--once",
                     "--i-understand-this-is-live"]) == 2


def test_the_live_flag_alone_does_not_make_a_testnet_config_live(tmp_path, monkeypatch, caplog):
    path = write(tmp_path, BASE.format(mode="testnet"))
    monkeypatch.setenv("BINANCE_API_KEY", "")
    monkeypatch.setenv("BINANCE_API_SECRET", "")
    # No testnet keys either, so it stops -- the point is that it never reached
    # a live exchange on the strength of the flag alone.
    assert cli.main(["--config", path, "trade", "--once",
                     "--i-understand-this-is-live"]) == 2
    assert "Staying on testnet" in caplog.text


def test_the_shipped_evidence_gate_asks_for_100k_analogues_at_80_percent():
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    assert cfg.evidence.enabled
    assert cfg.evidence.min_similar_situations >= 100_000
    assert cfg.evidence.min_success_rate == 0.80
    assert cfg.evidence.confidence_z > 0, "the gate must use a confidence bound"


def test_the_evidence_gate_is_optional_but_must_be_switched_off_deliberately(tmp_path):
    body = BASE.format(mode="paper") + "\nevidence:\n  enabled: false\n"
    cfg = load_config(write(tmp_path, body))
    assert not cfg.evidence.enabled
    assert cfg.evidence.min_similar_situations == 100_000   # the default stands


def test_a_missing_library_is_a_clear_error_not_a_silent_pass(tmp_path):
    from trading.analogues import EvidenceGate, EvidenceParams
    from trading.risk import RiskParams

    params = EvidenceParams(library=os.path.join(str(tmp_path), "nope.npz"))
    with pytest.raises(FileNotFoundError) as err:
        EvidenceGate.load(params, RiskParams())
    assert "build_corpus" in str(err.value)


def _write_validation(tmp_path, monkeypatch, payload):
    path = os.path.join(str(tmp_path), "validation.json")
    with open(path, "w") as fh:
        import json
        json.dump(payload, fh)
    monkeypatch.setattr(cli, "_validation_path", lambda: path)
    return path


def _passing(cfg_bits=None):
    from datetime import datetime, timezone
    return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "passed": True, "reasons": [],
            "config": cfg_bits or {"stop_loss_pct": 2.0, "take_profit_pct": 4.0}}


def test_live_needs_a_validation_on_record(tmp_path, monkeypatch):
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    monkeypatch.setattr(cli, "_validation_path",
                        lambda: os.path.join(str(tmp_path), "nothing.json"))
    assert "no validation on record" in cli._validation_blocks_live(cfg)


def test_a_failed_validation_blocks_live(tmp_path, monkeypatch):
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    _write_validation(tmp_path, monkeypatch,
                      {"at": _passing()["at"], "passed": False,
                       "reasons": ["average return across windows is -4.39%"],
                       "config": {"stop_loss_pct": 2.0, "take_profit_pct": 4.0}})
    blocker = cli._validation_blocks_live(cfg)
    assert "FAILED" in blocker and "-4.39%" in blocker


def test_a_stale_validation_blocks_live(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(timespec="seconds")
    _write_validation(tmp_path, monkeypatch,
                      {"at": old, "passed": True, "reasons": [],
                       "config": {"stop_loss_pct": 2.0, "take_profit_pct": 4.0}})
    assert "days old" in cli._validation_blocks_live(cfg)


def test_changing_the_risk_settings_invalidates_the_validation(tmp_path, monkeypatch):
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    _write_validation(tmp_path, monkeypatch,
                      _passing({"stop_loss_pct": 2.0, "take_profit_pct": 1.5}))
    assert "settings changed" in cli._validation_blocks_live(cfg)


def test_a_partial_validation_record_fails_closed(tmp_path, monkeypatch):
    """An old record that only captured the stop and target is not proof about
    the timeframe, the threshold or the evidence gate. Refuse it."""
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    _write_validation(tmp_path, monkeypatch, _passing(
        {"stop_loss_pct": cfg.risk.stop_loss_pct, "take_profit_pct": cfg.risk.take_profit_pct}))
    blocker = cli._validation_blocks_live(cfg)
    assert blocker is not None
    assert "not recorded" in blocker


def test_live_is_refused_when_validation_fails_even_with_the_flag_and_keys(
        tmp_path, monkeypatch, caplog):
    path = write(tmp_path, BASE.format(mode="live"))
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    _write_validation(tmp_path, monkeypatch,
                      {"at": _passing()["at"], "passed": False, "reasons": ["it lost money"],
                       "config": {"stop_loss_pct": 2.0, "take_profit_pct": 4.0}})
    assert cli.main(["--config", path, "trade", "--once",
                     "--i-understand-this-is-live"]) == 2
    assert "refusing to trade live" in caplog.text


def test_a_pass_earned_on_one_config_does_not_unlock_a_different_one(tmp_path, monkeypatch):
    """The bug this guards: validate on the profitable 4h settings, then run the
    losing 1h settings live because the stop and target happened to match."""
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    validated = cli._validated_shape(cfg, True)
    validated["timeframe"] = "4h"           # the pass was earned on 4h
    validated["buy_threshold"] = 2.0
    _write_validation(tmp_path, monkeypatch, {"at": _passing()["at"], "passed": True,
                                              "reasons": [], "config": validated})
    blocker = cli._validation_blocks_live(cfg)
    assert blocker is not None
    assert "timeframe" in blocker


def test_an_exact_match_still_clears(tmp_path, monkeypatch):
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    _write_validation(tmp_path, monkeypatch,
                      {"at": _passing()["at"], "passed": True, "reasons": [],
                       "config": cli._validated_shape(cfg, True)})
    assert cli._validation_blocks_live(cfg) is None


def test_turning_the_evidence_gate_down_invalidates_the_pass(tmp_path, monkeypatch):
    cfg = load_config(os.path.join(cli.ROOT, "config.yaml"))
    validated = cli._validated_shape(cfg, True)
    validated["min_success_rate"] = 0.34
    _write_validation(tmp_path, monkeypatch, {"at": _passing()["at"], "passed": True,
                                              "reasons": [], "config": validated})
    assert "min_success_rate" in cli._validation_blocks_live(cfg)


def test_the_trend_config_selects_the_trend_strategy():
    from trading.config import make_strategy
    from trading.strategy import TrendStrategy
    cfg = load_config(os.path.join(cli.ROOT, "config.trend.yaml"))
    assert cfg.strategy_kind == "trend"
    assert isinstance(make_strategy(cfg), TrendStrategy)
    assert cfg.trend.ma_days == 100


def test_the_trend_drawdown_ceiling_clears_its_own_expected_drawdown():
    """Measured drawdown is ~45%. A ceiling below that would halt the strategy
    at the worst possible moment and lock the loss in."""
    cfg = load_config(os.path.join(cli.ROOT, "config.trend.yaml"))
    assert cfg.portfolio.max_total_drawdown_pct > 46.0


def test_the_trend_config_does_not_take_profits_or_stop_for_the_day():
    """Both would break a strategy whose entire edge is holding winners."""
    cfg = load_config(os.path.join(cli.ROOT, "config.trend.yaml"))
    assert cfg.risk.take_profit_pct >= 1000
    assert cfg.supervisor.daily_profit_target_pct == 0
    assert cfg.portfolio.compounding == "full"
