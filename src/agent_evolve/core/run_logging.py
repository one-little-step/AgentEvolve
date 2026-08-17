"""Configurable capture of run logs for workers, analyzer, editor, pipeline.

Why this exists: the rollout process pool sent every child's ``stderr`` to
``DEVNULL``, which is the only channel CUGA writes routing decisions to. A
finished run therefore could not answer "did this rollout get a planning pass?"
without a paid re-run.

Opt-in, and off means absent (no directory, no stub file) so a measurement run
can turn capture off completely. Per-channel because ``workers`` is the
expensive channel and an operator debugging the editor should not pay for it.
Agent-neutral: lives in ``core``, imports no adapter and no CUGA symbol.
"""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

#: The four things a run produces logs from. ``pipeline`` is the orchestration
#: layer itself (iterations, selection, acceptance), kept separate from the three
#: components so a run can record decisions without recording rollouts.
ALL_LOG_CHANNELS: tuple[str, ...] = ("pipeline", "workers", "analyzer", "editor")

#: Record names come from candidate/task/worker ids -- all caller-supplied -- so
#: a name is sanitized rather than trusted: ``../../x`` must stay a file name.
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str, *, fallback: str = "unnamed") -> str:
    cleaned = _UNSAFE_NAME.sub("-", str(name)).strip("-.")
    return cleaned[:120] or fallback


@dataclass(frozen=True, slots=True)
class LogCaptureConfig:
    """Whether to capture, where to, and which channels.

    Validated at construction: "enabled with nowhere to write" would silently
    drop everything, which is the one failure mode a logging feature must not
    have.
    """

    enabled: bool = False
    root: Path | None = None
    channels: tuple[str, ...] = ALL_LOG_CHANNELS

    def __post_init__(self) -> None:
        unknown = [c for c in self.channels if c not in ALL_LOG_CHANNELS]
        if unknown:
            raise ValueError(
                f"unknown log channel(s): {sorted(unknown)}; "
                f"valid channels are {list(ALL_LOG_CHANNELS)}"
            )
        if self.enabled and self.root is None:
            raise ValueError(
                "log capture is enabled but no root was given; a capture with "
                "nowhere to write would discard every line"
            )
        if self.root is not None and not isinstance(self.root, Path):
            object.__setattr__(self, "root", Path(self.root))

    def wants(self, channel: str) -> bool:
        return self.enabled and channel in self.channels

    def channel_dir(self, channel: str) -> Path | None:
        """The directory for ``channel``, or ``None`` when not capturing.

        Creates nothing: a channel that is never written leaves no trace, so
        creation is deferred to the first record.
        """
        if not self.wants(channel) or self.root is None:
            return None
        return self.root / channel

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "root": str(self.root) if self.root is not None else None,
            "channels": list(self.channels) if self.enabled else [],
        }


@dataclass
class RunLogSink:
    """Writes one channel's logs, or nothing at all.

    A disabled sink is a working object whose every write returns ``None``, so
    call sites need no ``if capture_enabled`` branches -- the decision lives
    here. One sink per channel; the caller closes it in a ``finally``.
    """

    config: LogCaptureConfig
    channel: str
    _streams: dict[str, IO[str]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.channel not in ALL_LOG_CHANNELS:
            raise ValueError(
                f"unknown log channel: {self.channel!r}; "
                f"valid channels are {list(ALL_LOG_CHANNELS)}"
            )

    @property
    def active(self) -> bool:
        return self.config.wants(self.channel)

    def write_record(self, name: str, record: Mapping[str, Any]) -> Path | None:
        """Append one JSONL record; return its file, or ``None`` if inactive.

        Appends rather than overwrites: one attempt produces several records
        (request, response, outcome) and losing the earlier ones would leave the
        least informative one.
        """
        directory = self.config.channel_dir(self.channel)
        if directory is None:
            return None
        path = directory / f"{_safe_name(name)}.jsonl"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "channel": self.channel,
            **dict(record),
        }
        line = json.dumps(payload, default=str)
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return path

    def open_stream(self, name: str) -> IO[str] | None:
        """A writable file for a child's ``stderr``, or ``None`` if inactive.

        One file per name so interleaved worker output stays attributable; the
        handle goes straight to ``Popen(stderr=...)``, which costs nothing in the
        rollout's latency path.
        """
        directory = self.config.channel_dir(self.channel)
        if directory is None:
            return None
        with self._lock:
            existing = self._streams.get(name)
            if existing is not None:
                return existing
            directory.mkdir(parents=True, exist_ok=True)
            handle = (directory / f"{_safe_name(name)}.log").open(
                "a", encoding="utf-8", errors="replace"
            )
            self._streams[name] = handle
            return handle

    def close(self) -> None:
        """Flush and close every stream. Idempotent; never raises.

        Teardown must not fail a run that already produced its numbers.
        """
        with self._lock:
            streams, self._streams = dict(self._streams), {}
        for handle in streams.values():
            try:
                handle.flush()
                handle.close()
            except Exception:  # noqa: BLE001 - teardown must not fail a run
                pass

    def __enter__(self) -> "RunLogSink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_sinks(
    config: LogCaptureConfig, channels: Iterable[str] | None = None
) -> dict[str, RunLogSink]:
    """One sink per channel, active or not, so call sites need no branches."""
    wanted = tuple(channels) if channels is not None else ALL_LOG_CHANNELS
    return {
        channel: RunLogSink(config=config, channel=channel) for channel in wanted
    }
