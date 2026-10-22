"""Embed demo/replay_data.json into demo/template.html -> demo/index.html (one self-contained file)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEMO, RESULTS


def main():
    data = (DEMO / "replay_data.json").read_text()
    gap = pd.read_csv(RESULTS / "handover_gap.csv").handover_gap.mean() * 100
    html = (DEMO / "template.html").read_text().replace("__DATA__", data).replace("__GAP__", f"{gap:.1f}")
    (DEMO / "index.html").write_text(html)
    print(f"demo/index.html written: {len(html) / 1e6:.2f} MB, {len(json.loads(data))} matches")


if __name__ == "__main__":
    main()
