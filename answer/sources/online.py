"""Tier 2, online documents — INTERFACE ONLY. ARCHITECTURE.md §9, §11.

Deliberately not built: it would cost a day and undercut the "nothing leaves your device"
claim, and §11 puts it out of scope. This exists so the architecture diagram is honest. It
imports nothing that can reach a network.

Do not implement this. Do not add a config key for it.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from core.schema import Retrieved


@runtime_checkable
class OnlineSource(Protocol):
    """The shape a Tier 2 provider would have to satisfy to slot in beside Tier 1."""

    enabled: bool

    def fetch(self, query: str, k: int) -> list[Retrieved]:
        """Return up to `k` hits from network documents, rank 1 = best."""
        ...


class DisabledOnlineSource:
    """The only Tier 2 implementation that ships: one that refuses.

    `enabled` is a class constant, so a caller can branch without constructing anything.
    """

    enabled: ClassVar[bool] = False

    def fetch(self, query: str, k: int) -> list[Retrieved]:
        raise NotImplementedError(
            "Tier 2 (online documents) is interface-only — ARCHITECTURE.md §9, §11. "
            "Nothing leaves your device. Answers come from the local index (Tier 1) or, "
            "on explicit opt-in, model general knowledge (Tier 3)."
        )
