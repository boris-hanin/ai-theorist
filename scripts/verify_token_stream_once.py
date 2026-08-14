#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_theorist.autoscaler.tokenization import (
    write_token_stream_verification_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fully hash a token stream and write a reusable local receipt."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = write_token_stream_verification_receipt(
        args.manifest, args.output
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
