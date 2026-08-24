from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from core.config import DEFAULT_CONFIG, load_config

VALID = yaml.safe_load(DEFAULT_CONFIG.read_text())


def write_cfg(tmp_path, mutate=None):
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if mutate:
        mutate(data)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_base_config_loads():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.chunk.target_tokens == 400
    assert cfg.retrieve.k == 20
    # tau is a calibrated value, not an architectural one — it has already moved 0.35 -> 0.50
    # -> 0.45 as the golden set grew, and pinning the literal here just breaks this test on
    # every retune. §6's structural choices above are pinned; the tuned one is range-checked.
    assert 0.0 < cfg.retrieve.tau <= 1.0
    assert cfg.retrieve.n_context <= cfg.retrieve.k
    assert cfg.models.generator.precision == "int4"


def test_config_hash_is_stable_and_sensitive(tmp_path):
    a = load_config(write_cfg(tmp_path))
    b = load_config(write_cfg(tmp_path))
    assert a.config_hash == b.config_hash

    def bump(d):
        d["chunk"]["target_tokens"] = 512

    c = load_config(write_cfg(tmp_path, bump))
    assert c.config_hash != a.config_hash
    assert c.chunk_config_hash != a.chunk_config_hash


def test_chunk_hash_ignores_unrelated_tuning(tmp_path):
    """Retuning tau must not invalidate the chunk cache."""

    def bump_tau(d):
        # Derived, not a literal: pinning a number here makes the test silently become a
        # no-op the day base.yaml is retuned to that same number, which is what happened when
        # tau moved to 0.50. Halving is distinct for any tau > 0 and stays inside [0, 1];
        # 1 - tau is not — it is a fixed point at exactly 0.5.
        d["retrieve"]["tau"] = round(d["retrieve"]["tau"] / 2, 4)

    a = load_config(write_cfg(tmp_path))
    b = load_config(write_cfg(tmp_path, bump_tau))
    assert b.chunk_config_hash == a.chunk_config_hash
    assert b.config_hash != a.config_hash


class TestMalformedConfig:
    def test_unknown_key_rejected(self, tmp_path):
        def add(d):
            d["chunk"]["nonsense"] = 1

        with pytest.raises(Exception, match="[Ee]xtra"):
            load_config(write_cfg(tmp_path, add))

    def test_n_context_cannot_exceed_k(self, tmp_path):
        def bad(d):
            d["retrieve"]["n_context"] = 50

        with pytest.raises(Exception, match="cannot exceed"):
            load_config(write_cfg(tmp_path, bad))

    def test_overlap_must_be_less_than_target(self, tmp_path):
        def bad(d):
            d["chunk"]["overlap"] = 400

        with pytest.raises(Exception, match="chunking cannot advance"):
            load_config(write_cfg(tmp_path, bad))

    def test_bad_device_rejected(self, tmp_path):
        def bad(d):
            d["models"]["embedder"]["device"] = "TPU"

        with pytest.raises(ValidationError):
            load_config(write_cfg(tmp_path, bad))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


class TestV2Arms:
    """§10 arms must default off, so an unmodified base.yaml is the measured V1 baseline."""

    def test_every_v2_arm_is_off_by_default(self):
        cfg = load_config(DEFAULT_CONFIG)
        assert cfg.retrieve.v2_arms == ()
        assert not cfg.retrieve.rerank.enabled
        assert not cfg.retrieve.hybrid.enabled
        assert not cfg.retrieve.rewrite.enabled

    def test_v2_arms_names_what_is_live(self, tmp_path):
        cfg = load_config(DEFAULT_CONFIG)
        on = cfg.model_copy(
            update={
                "retrieve": cfg.retrieve.model_copy(
                    update={
                        "rerank": cfg.retrieve.rerank.model_copy(update={"enabled": True}),
                        "hybrid": cfg.retrieve.hybrid.model_copy(update={"enabled": True}),
                    }
                )
            }
        )
        assert on.retrieve.v2_arms == ("rerank", "hybrid")

    def test_enabling_rerank_without_a_reranker_model_is_refused(self, tmp_path):
        def turn_on(d):
            d["retrieve"]["rerank"] = {"enabled": True}
            d["models"].pop("reranker", None)

        with pytest.raises(ValueError, match="models.reranker is not configured"):
            load_config(write_cfg(tmp_path, turn_on))

    def test_turning_an_arm_on_changes_config_hash_but_not_chunk_hash(self, tmp_path):
        """Flipping a retrieval arm must never invalidate the ingest cache."""

        def turn_on(d):
            d["retrieve"]["hybrid"] = {"enabled": True}

        a = load_config(write_cfg(tmp_path))
        b = load_config(write_cfg(tmp_path, turn_on))
        assert b.config_hash != a.config_hash
        assert b.chunk_config_hash == a.chunk_config_hash
