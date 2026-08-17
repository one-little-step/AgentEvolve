"""Configurable run-log capture: enabled persists, disabled writes nothing.

Why these tests exist
---------------------
Worker CUGA logs were *lost*: the process pool sent child ``stderr`` to
``DEVNULL``, so the one channel that carries ``is_autonomous_subtask`` and
``Routing to:`` never reached disk and a routing question could only be answered
by a manual re-run. Capture is therefore a research instrument, not tidiness.

Two properties are load-bearing and both are asserted here:

* **Configurable.** Disabled must mean *nothing on disk* -- not an empty tree,
  not a stub file. A capture that always writes cannot be turned off for a
  measurement run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_evolve.core.run_logging import (
    ALL_LOG_CHANNELS,
    LogCaptureConfig,
    RunLogSink,
)


# --------------------------------------------------------------------------- #
# configuration surface
# --------------------------------------------------------------------------- #


def test_capture_is_disabled_by_default(tmp_path):
    """Opt-in: an unconfigured run must behave exactly as it did before."""
    config = LogCaptureConfig()
    assert config.enabled is False
    assert config.wants("workers") is False
    assert config.channel_dir("workers") is None


def test_enabling_without_a_root_is_refused():
    """"Enabled" with nowhere to write would silently drop every line."""
    with pytest.raises(ValueError, match="root"):
        LogCaptureConfig(enabled=True, root=None)


def test_an_unknown_channel_is_refused(tmp_path):
    """A typo'd channel would disable capture for the channel that matters."""
    with pytest.raises(ValueError, match="unknown log channel"):
        LogCaptureConfig(enabled=True, root=tmp_path, channels=("wokers",))


def test_channels_can_be_narrowed_to_one(tmp_path):
    """Worker logs are the expensive channel; capturing only them is valid."""
    config = LogCaptureConfig(enabled=True, root=tmp_path, channels=("workers",))
    assert config.wants("workers") is True
    assert config.wants("editor") is False


def test_every_named_channel_is_wanted_by_default(tmp_path):
    """The user asked for workers, judge, editor AND the whole pipeline."""
    config = LogCaptureConfig(enabled=True, root=tmp_path)
    for channel in ALL_LOG_CHANNELS:
        assert config.wants(channel) is True
    assert set(ALL_LOG_CHANNELS) == {"pipeline", "workers", "analyzer", "editor"}


# --------------------------------------------------------------------------- #
# the sink: records
# --------------------------------------------------------------------------- #


def test_a_disabled_sink_creates_no_file_and_no_directory(tmp_path):
    """Disabled means absent. An empty dir is still an observable side effect."""
    root = tmp_path / "logs"
    sink = RunLogSink(LogCaptureConfig(enabled=False), channel="analyzer")
    assert sink.write_record("group-1", {"prompt": "hello"}) is None
    sink.close()
    assert not root.exists()
    assert list(tmp_path.iterdir()) == []


def test_an_enabled_sink_persists_one_jsonl_record_per_call(tmp_path):
    """Append, never overwrite: a second call must not erase the first."""
    sink = RunLogSink(
        LogCaptureConfig(enabled=True, root=tmp_path), channel="analyzer"
    )
    path = sink.write_record("cand-A__t1", {"event": "request", "n": 1})
    sink.write_record("cand-A__t1", {"event": "response", "n": 2})
    sink.close()

    assert path is not None
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert [r["event"] for r in lines] == ["request", "response"]
    # Every record is timestamped and self-describing: a log that cannot be
    # ordered or attributed is not evidence.
    assert all(r["channel"] == "analyzer" for r in lines)
    assert all(r["timestamp"].endswith("+00:00") for r in lines)


def test_a_record_for_an_unwanted_channel_is_dropped(tmp_path):
    """Narrowing channels must actually stop the writes."""
    sink = RunLogSink(
        LogCaptureConfig(enabled=True, root=tmp_path, channels=("workers",)),
        channel="editor",
    )
    assert sink.write_record("attempt-1", {"prompt": "x"}) is None
    sink.close()
    assert not (tmp_path / "editor").exists()


def test_a_record_name_cannot_escape_its_channel_directory(tmp_path):
    """Names come from candidate/task ids; a '/' must not become a path."""
    sink = RunLogSink(
        LogCaptureConfig(enabled=True, root=tmp_path), channel="pipeline"
    )
    path = sink.write_record("../../etc/passwd", {"event": "x"})
    sink.close()
    assert path is not None
    assert (tmp_path / "pipeline") in path.parents


# --------------------------------------------------------------------------- #
# the sink: subprocess stream capture (the channel that was lost)
# --------------------------------------------------------------------------- #


def test_a_disabled_sink_hands_back_no_stream(tmp_path):
    """Callers fall back to DEVNULL, preserving today's behaviour exactly."""
    sink = RunLogSink(LogCaptureConfig(enabled=False), channel="workers")
    assert sink.open_stream("w0001") is None
    sink.close()


def test_child_stderr_is_captured(tmp_path):
    """End to end on a real child: the lost channel, recovered.

    Uses a trivial Python child rather than CUGA: the property under test is
    that the pipe is drained to disk, which is independent of who writes to it.
    """
    sink = RunLogSink(
        LogCaptureConfig(enabled=True, root=tmp_path), channel="workers"
    )
    stream = sink.open_stream("w0001")
    assert stream is not None

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('is_autonomous_subtask=True\\n')",
        ],
        stdout=subprocess.DEVNULL,
        stderr=stream,
        text=True,
    )
    process.wait(timeout=30)
    sink.close()

    written = (tmp_path / "workers" / "w0001.log").read_text()
    assert "is_autonomous_subtask=True" in written


def test_two_workers_never_share_a_log_file(tmp_path):
    """Interleaved worker output cannot be attributed after the fact."""
    sink = RunLogSink(
        LogCaptureConfig(enabled=True, root=tmp_path), channel="workers"
    )
    a = sink.open_stream("w0001")
    b = sink.open_stream("w0002")
    sink.close()
    assert a is not None and b is not None
    assert (tmp_path / "workers" / "w0001.log").exists()
    assert (tmp_path / "workers" / "w0002.log").exists()


def test_close_is_idempotent(tmp_path):
    """Teardown runs on both the happy and the failing path."""
    sink = RunLogSink(
        LogCaptureConfig(enabled=True, root=tmp_path), channel="workers"
    )
    sink.open_stream("w0001")
    sink.close()
    sink.close()
