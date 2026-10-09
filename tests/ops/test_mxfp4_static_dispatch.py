import os

import pytest

from lumen.ops.quantize import linear as linear_mod
from lumen.ops.quantize import mxfp4_autotune
from lumen.ops.quantize import ops as ops_mod


def test_static_fallback_is_memoized_only_in_memory(monkeypatch):
    key = (1024, 4096, 14336)
    candidates = [
        ("asm", lambda: pytest.fail("static policy ran a candidate")),
        ("plain", lambda: pytest.fail("static policy ran a candidate")),
    ]

    mxfp4_autotune.clear()
    monkeypatch.setattr(mxfp4_autotune, "_CACHE_PATH", "")
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", False)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    try:
        assert (
            mxfp4_autotune.pick_backend(key, candidates, fallback="plain")
            == "plain"
        )
        assert mxfp4_autotune.cached(key) == "plain"
        assert mxfp4_autotune.cached_static(key) == "plain"
        assert key not in mxfp4_autotune._choice
        assert mxfp4_autotune._static_choice[key] == "plain"
    finally:
        mxfp4_autotune.clear()


def test_autotune_off_ignores_persisted_and_in_process_choices(
    monkeypatch, tmp_path
):
    key = (1024, 4096, 14336)
    candidates = [
        ("asm", lambda: pytest.fail("disabled autotune ran a candidate")),
        ("plain", lambda: pytest.fail("disabled autotune ran a candidate")),
    ]

    mxfp4_autotune.clear()
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", False)
    monkeypatch.setattr(mxfp4_autotune, "_CACHE_PATH", str(tmp_path / "history.json"))
    monkeypatch.setattr(
        mxfp4_autotune,
        "_load_cache",
        lambda: pytest.fail("disabled autotune loaded its persisted cache"),
    )
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    mxfp4_autotune._choice[key] = "asm"
    try:
        mxfp4_autotune.configure()
        assert mxfp4_autotune.cached(key) is None
        assert (
            mxfp4_autotune.pick_backend(key, candidates, fallback="plain")
            == "plain"
        )
        assert mxfp4_autotune.cached(key) == "plain"
        assert mxfp4_autotune._choice[key] == "asm"
        assert mxfp4_autotune._static_choice[key] == "plain"
    finally:
        mxfp4_autotune.clear()


def test_static_policy_enables_first_use_shuffle(monkeypatch):
    key = (1024, 4096, 14336)
    mxfp4_autotune.clear()
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", False)
    monkeypatch.setattr(mxfp4_autotune, "is_capturing", lambda: False)
    monkeypatch.setattr(
        linear_mod, "_mxfp4_static_backend_for_shape", lambda shape: "shuffled"
    )
    monkeypatch.setattr(
        ops_mod, "mxfp4_data_shuffle_supported", lambda rows, cols: True
    )
    try:
        assert linear_mod._mxfp4_can_fuse_b_shuffle(key, 4096, 7168)
        assert mxfp4_autotune.cached(key) == "shuffled"
        assert mxfp4_autotune.cached_static(key) == "shuffled"
        assert key not in mxfp4_autotune._choice
    finally:
        mxfp4_autotune.clear()


def test_static_dispatch_hit_skips_runtime_probes(monkeypatch):
    key = (1024, 4096, 14336)

    class Operand:
        def __init__(self, shape):
            self.shape = shape

    a_fp4 = Operand((key[0], key[2] // 2))
    w_fp4 = Operand((key[1], key[2] // 2))
    mxfp4_autotune.clear()
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_probe_backends",
        lambda: pytest.fail("static dispatch repeated backend probes"),
    )
    try:
        mxfp4_autotune.remember_static(key, "plain")
        assert linear_mod._mxfp4_choose_backend(a_fp4, w_fp4, None, None) == "plain"
    finally:
        mxfp4_autotune.clear()


def test_stale_static_backend_is_revalidated(monkeypatch):
    key = (1024, 4096, 14336)

    class Operand:
        def __init__(self, shape):
            self.shape = shape

    a_fp4 = Operand((key[0], key[2] // 2))
    w_fp4 = Operand((key[1], key[2] // 2))
    mxfp4_autotune.clear()
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", False)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    monkeypatch.setattr(
        linear_mod, "_mxfp4_backend_legality", lambda *_args: (False, False)
    )
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_backend_candidates",
        lambda *_args: (
            [("plain", lambda: pytest.fail("static policy ran a candidate"))],
            "plain",
            False,
        ),
    )
    try:
        mxfp4_autotune.remember_static(key, "asm")
        assert linear_mod._mxfp4_choose_backend(a_fp4, w_fp4, None, None) == "plain"
        assert mxfp4_autotune.cached_static(key) == "plain"
    finally:
        mxfp4_autotune.clear()


def test_static_plain_is_rejected_for_shuffled_operand(monkeypatch):
    key = (1024, 4096, 14336)

    class Operand:
        def __init__(self, shape):
            self.shape = shape

    a_fp4 = Operand((key[0], key[2] // 2))
    w_fp4 = Operand((key[1], key[2] // 2))
    linear_mod._mark_mxfp4_data_shuffled(w_fp4)
    mxfp4_autotune.clear()
    monkeypatch.setattr(mxfp4_autotune, "AUTOTUNE_ENABLED", False)
    monkeypatch.setattr(mxfp4_autotune, "_capturing", lambda: False)
    monkeypatch.setattr(
        linear_mod,
        "_mxfp4_backend_candidates",
        lambda *_args: (
            [("shuffled", lambda: pytest.fail("static policy ran a candidate"))],
            "shuffled",
            False,
        ),
    )
    try:
        mxfp4_autotune.remember_static(key, "plain")
        assert (
            linear_mod._mxfp4_choose_backend(a_fp4, w_fp4, None, None)
            == "shuffled"
        )
        assert mxfp4_autotune.cached_static(key) == "shuffled"
    finally:
        mxfp4_autotune.clear()


def test_tuned_config_cannot_change_after_dispatch(monkeypatch, tmp_path):
    key = (1024, 4096, 14336)
    tuned = tmp_path / "late.csv"
    tuned.write_text("gfx\n")

    mxfp4_autotune.clear()
    monkeypatch.delenv(mxfp4_autotune.AITER_TUNED_CONFIG_ENV, raising=False)
    monkeypatch.setattr(
        mxfp4_autotune, "_aiter_default_tuned_config", lambda: None
    )
    try:
        mxfp4_autotune.remember_static(key, "plain")
        with pytest.raises(RuntimeError, match="before the first backend decision"):
            mxfp4_autotune.configure(tuned_config=str(tuned))
        assert mxfp4_autotune.AITER_TUNED_CONFIG_ENV not in os.environ
    finally:
        mxfp4_autotune.clear()


def test_wgrad_target_rows_match_power2_dispatch_key(monkeypatch):
    monkeypatch.setattr(linear_mod, "_MXFP4_WGRAD_TILE_PAD", True)
    monkeypatch.setattr(linear_mod, "_MXFP4_WGRAD_POWER2_BUCKET", True)

    rows = [1, 255, 256, 257, 542, 992, 1024, 1025, 32768]
    targets = [linear_mod._mxfp4_wgrad_target_rows(value) for value in rows]

    assert targets == [256, 256, 256, 512, 1024, 1024, 1024, 2048, 32768]
    for value in range(1, 32769):
        expected = max(256, 1 << (value - 1).bit_length())
        assert linear_mod._mxfp4_wgrad_target_rows(value) == expected
