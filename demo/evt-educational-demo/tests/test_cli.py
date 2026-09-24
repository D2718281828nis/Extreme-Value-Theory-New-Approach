from pathlib import Path
import os
import subprocess
import sys


def test_cli_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(
        [sys.executable, str(root / "scripts/run_demo.py"), "--seed", "42",
         "--bootstrap", "5", "--output-dir", str(tmp_path)],
        check=True, env=env, capture_output=True, text=True,
    )
    assert (tmp_path / "data/synthetic_data.csv").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) >= 4
