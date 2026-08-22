"""Tier 2, online documents — INTERFACE ONLY. ARCHITECTURE.md §9, §11.

Deliberately not built. §9: building it would cost a day and undercut the "nothing leaves
your device" claim, which is the strongest asset in the pitch. §11 lists online document
fetching as out of scope. This module exists so the architecture diagram is honest and the
roadmap is concrete — it imports nothing that can reach a network, and it never will while
§11 stands.

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

    Permission-gated in the roadmap; permanently off here. `enabled` is a class constant so
    a caller can branch on it without constructing anything or triggering the error.
    """

    enabled: ClassVar[bool] = False

    def fetch(self, query: str, k: int) -> list[Retrieved]:
        raise NotImplementedError(
            "Tier 2 (online documents) is interface-only — ARCHITECTURE.md §9, §11. "
            "Nothing leaves your device. Answers come from the local index (Tier 1) or, "
            "on explicit opt-in, model general knowledge (Tier 3)."
        )
