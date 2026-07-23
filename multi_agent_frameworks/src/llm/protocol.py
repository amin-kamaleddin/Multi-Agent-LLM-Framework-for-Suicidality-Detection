"""Minimal model protocol shared by production chat models and test doubles."""

from __future__ import annotations

from typing import Any, Protocol


class InvokableModel(Protocol):
    def invoke(self, messages: list[dict[str, str]]) -> Any:
        """Return a model response for the supplied chat messages."""
