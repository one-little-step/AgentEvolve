"""Content-hash keyed JSON cache for RHO preprocessing stages.

Comprehension and difficulty judging are the only RHO stages that re-run over
the same historical corpus every round, so they are the only stages worth
caching. Keys always include the trace content hash, which is why a stale trace
can never produce a false hit.

A ``None`` root disables the cache entirely: a measurement run must be able to
spend nothing on capture, and creating directories for a run that asked for no
cache would violate that.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping


@dataclass(slots=True)
class JsonDiskCache:
    """A tiny filename-per-key JSON cache with hit/miss counters.

    Counters are per-instance and feed the per-round cost accounting in the run
    manifest, so they must only ever be incremented by ``get``.
    """

    root: Path | None
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    def _path(self, key: str) -> Path | None:
        if self.root is None:
            return None
        digest = sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, key: str) -> dict | None:
        """Return the cached mapping for ``key`` or ``None`` on any miss."""
        path = self._path(key)
        if path is None or not path.exists():
            self.misses += 1
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt or unreadable entry must not abort a round; recompute.
            self.misses += 1
            return None
        if not isinstance(value, dict):
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: Mapping[str, object]) -> None:
        """Store ``value`` under ``key``; a ``None`` root makes this a no-op."""
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(value), default=str)
        # Write-then-rename so a concurrent reader never observes a partial
        # file: parallel RHO batches share one cache root.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
