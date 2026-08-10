from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from .pretraining import run_standard_pretraining_batch_census
from .study import atomic_write_json


PROGRESS_PREFIX = "AUTOSCALER_PROGRESS "


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed standard-text pretraining worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    rank = int(os.environ.get("RANK", "0"))

    def progress(event: Dict[str, Any]) -> None:
        if rank == 0:
            print(PROGRESS_PREFIX + json.dumps(event, sort_keys=True), flush=True)

    result = run_standard_pretraining_batch_census(
        config, device="cuda", progress=progress
    )
    if rank == 0:
        atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
