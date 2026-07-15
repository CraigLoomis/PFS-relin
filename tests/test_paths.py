"""Tests for input/output path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from fitLinearity.paths import (
    DEFAULT_OUTPUT_ROOT,
    dataRoot,
    inputDir,
    outputDir,
    repoRoot,
)


def testRepoRootHoldsThePackage():
    assert (repoRoot() / "python" / "fitLinearity" / "paths.py").is_file()


def testDataRootDefaultsToSiblingJhuData(monkeypatch):
    monkeypatch.delenv("FITLINEARITY_DATA", raising=False)
    assert dataRoot() == repoRoot().parent / "jhu-data"


def testDataRootHonorsEnvVar(monkeypatch, tmp_path):
    monkeypatch.setenv("FITLINEARITY_DATA", str(tmp_path))
    assert dataRoot() == tmp_path


def testInputDirIsRootSlashDet(monkeypatch, tmp_path):
    monkeypatch.setenv("FITLINEARITY_DATA", str(tmp_path))
    assert inputDir("18734") == tmp_path / "18734"


def testInputDirAcceptsExplicitRoot(tmp_path):
    assert inputDir("18660", root=tmp_path) == tmp_path / "18660"


def testDefaultOutputRoot():
    assert DEFAULT_OUTPUT_ROOT == Path("/work/cloomis/outputs/fitLinearity")


def testOutputDirIsDetSlashTagAndIsCreated(tmp_path):
    out = outputDir("18734", "o5_dev0.45", root=tmp_path)
    assert out == tmp_path / "18734" / "o5_dev0.45"
    assert out.is_dir()


def testOutputDirIsIdempotent(tmp_path):
    outputDir("18734", "o4", root=tmp_path)
    out = outputDir("18734", "o4", root=tmp_path)
    assert out.is_dir()


def testOutputDirNeverWritesUnderTheDataRoot(monkeypatch, tmp_path):
    dataDir = tmp_path / "jhu-data"
    outRoot = tmp_path / "outputs"
    monkeypatch.setenv("FITLINEARITY_DATA", str(dataDir))
    out = outputDir("18734", "o4", root=outRoot)
    assert dataDir not in out.parents
    assert not dataDir.exists()


def testInputDirRejectsEmptyDet():
    with pytest.raises(ValueError, match="detector id must be non-empty"):
        inputDir("")
