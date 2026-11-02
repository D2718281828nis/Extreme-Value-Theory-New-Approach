"""Эталон не правили мимо сверки с R1: MD5 файлов совпадают с записанными в reference_results/R1_SYNC.json.

Сама сверка с R1 (scripts/check_sync_with_r1.py) требует локальной копии dissertation-calculations; этот тест — нет,
поэтому работает в CI. Изменили эталон или синхронизированный код — повторите сверку с --write.
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNC = json.loads((ROOT / "reference_results" / "R1_SYNC.json").read_text(encoding="utf-8"))


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("system", ["brain", "aviation", "finance"])
def test_folds_match_the_r1_sync_record(system):
    ref = ROOT / "reference_results" / system
    assert md5(ref / "folds.csv") == SYNC[system]["folds_md5"]
    assert json.loads((ref / "results.json").read_text(encoding="utf-8"))["folds_md5"] == SYNC[system]["folds_md5"]


@pytest.mark.parametrize("path", sorted(SYNC["code_md5"]))
def test_synced_code_is_unchanged_since_the_r1_sync(path):
    assert md5(ROOT / path) == SYNC["code_md5"][path]
