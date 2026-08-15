from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from .pretraining import run_standard_pretraining_batch_census
from .forecast_campaigns import run_real_text_scaling_campaign
from .study import atomic_write_json


PROGRESS_PREFIX = "AUTOSCALER_PROGRESS "


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed standard-text pretraining worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        choices=("standard_pretraining_census", "real_text_scaling_ladder"),
        default="standard_pretraining_census",
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    rank = int(os.environ.get("RANK", "0"))

    def progress(event: Dict[str, Any]) -> None:
        if rank == 0:
            print(PROGRESS_PREFIX + json.dumps(event, sort_keys=True), flush=True)

    if args.campaign == "real_text_scaling_ladder":
        result = run_real_text_scaling_campaign(
            config, device="cuda", progress=progress
        )
    else:
        result = run_standard_pretraining_batch_census(
            config, device="cuda", progress=progress
        )
    if rank == 0:
        atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
