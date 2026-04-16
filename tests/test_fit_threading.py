"""Tests for fit() threading: heuristic, parallel path, errors, determinism."""

from __future__ import annotations

import sys

import pytest

from relin.fit import _resolveWorkerCount


def test_resolveWorkerCountExplicitIntIsReturnedAsIs():
    # Explicit wins over heuristic — no clamping, no size check.
    assert _resolveWorkerCount(1, 10, 10) == 1
    assert _resolveWorkerCount(4, 10, 10) == 4
    assert _resolveWorkerCount(16, 10, 10) == 16
    # Even on a "large" frame, explicit 1 is honored.
    assert _resolveWorkerCount(1, 5000, 5000) == 1


def test_resolveWorkerCountSmallFrameDefaultsToOne():
    # With H*W < _SMALL_FRAME_PIXEL_LIMIT (1_000_000), None → 1 worker
    # regardless of os.cpu_count().
    assert _resolveWorkerCount(None, 100, 100) == 1
    assert _resolveWorkerCount(None, 1000, 999) == 1  # 999_000 < 1_000_000


def test_resolveWorkerCountLargeFrameCapsAtEight(monkeypatch):
    # When H*W >= 1_000_000 and os.cpu_count() > 8, cap at 8.
    fitModule = sys.modules["relin.fit"]
    monkeypatch.setattr(fitModule.os, "cpu_count", lambda: 16)
    assert _resolveWorkerCount(None, 2000, 500) == 8  # 1_000_000 exactly
    assert _resolveWorkerCount(None, 4096, 4096) == 8


def test_resolveWorkerCountLargeFrameUncappedBelowEight(monkeypatch):
    # When H*W is large but os.cpu_count() < 8, use cpu_count.
    fitModule = sys.modules["relin.fit"]
    monkeypatch.setattr(fitModule.os, "cpu_count", lambda: 4)
    assert _resolveWorkerCount(None, 4096, 4096) == 4


def test_resolveWorkerCountHandlesNoneCpuCount(monkeypatch):
    # os.cpu_count() can return None on some platforms; fall back to 1.
    fitModule = sys.modules["relin.fit"]
    monkeypatch.setattr(fitModule.os, "cpu_count", lambda: None)
    assert _resolveWorkerCount(None, 4096, 4096) == 1


def test_resolveWorkerCountInvalidRaises():
    with pytest.raises(ValueError, match="workers"):
        _resolveWorkerCount(0, 10, 10)
    with pytest.raises(ValueError, match="workers"):
        _resolveWorkerCount(-3, 10, 10)
