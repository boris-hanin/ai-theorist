#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.jiang_chizat import (
    JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
)
from ai_theorist.autoscaler.study import atomic_write_json
from ai_theorist.autoscaler.tokenization import (
    load_token_stream_manifest,
    load_tokenizer_manifest,
)


PAPER_SOURCE = "https://arxiv.org/abs/2505.01618v4"
PAPER_SOURCE_TAR_SHA256 = (
    "dbf47c765f119357c998df1ff569be8950e98cf607f847936f0bb37cb5206fab"
)
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SLIMPAJAMA6B_SOURCE_REVISION = "b5f90f419b7489cdba26fdbc8c022fcb5562f968"
SLIMPAJAMA6B_PARQUET_REVISION = "c4f51dc260275e8e01aa0fbf46c64832dbee5369"
GPT2_ASSET_SHA256 = {
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}


def _load(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregister the SlimPajama/GPT-2 paper-coordinate rerun."
    )
    parser.add_argument("corpus_manifest", type=Path)
    parser.add_argument("jiang_config", type=Path)
    parser.add_argument("completep_anchor_config", type=Path)
    parser.add_argument("jiang_qualification", type=Path)
    parser.add_argument("completep_qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = _load(args.corpus_manifest, "corpus manifest")
    jiang_config = _load(args.jiang_config, "Jiang config")
    completep_config = _load(args.completep_anchor_config, "CompleteP config")
    jiang_qualification = _load(args.jiang_qualification, "Jiang qualification")
    completep_qualification = _load(
        args.completep_qualification, "CompleteP qualification"
    )
    jiang = compile_real_text_scaling_plan(jiang_config)
    completep = compile_real_text_scaling_plan(completep_config)
    stream = load_token_stream_manifest(
        Path(str(corpus["token_stream_manifest_path"])), verify_files=True
    )
    tokenizer = load_tokenizer_manifest(
        Path(str(corpus["tokenizer_manifest_path"])), verify_assets=True
    )
    tokenizer_assets = {
        str(row["name"]): str(row["sha256"])
        for row in tokenizer["assets"]
    }
    jiang_fixed = jiang["fixed_budget_contract"]
    completep_fixed = completep["fixed_budget_contract"]
    jiang_reference = jiang["scales"][
        int(jiang["architecture_contract"]["reference_scale_index"])
    ]
    completep_reference = completep["scales"][0]

    gates = {
        "corpus_complete": corpus.get("status") == "complete",
        "source_is_preserved_slimpajama6b_sample": corpus.get("source", {}).get(
            "dataset"
        )
        == "DKYoon/SlimPajama-6B",
        "source_revision_is_exact": corpus.get("source", {}).get("revision")
        == SLIMPAJAMA6B_SOURCE_REVISION,
        "parquet_revision_is_exact": corpus.get("source", {}).get(
            "parquet_revision"
        )
        == SLIMPAJAMA6B_PARQUET_REVISION,
        "source_backend_is_immutable_parquet": corpus.get("source", {}).get(
            "acquisition_backend"
        )
        == "parquet",
        "separate_parquet_inventories_are_recorded": set(
            corpus.get("source", {}).get("parquet_inventory_fingerprints", {})
        )
        == {"train", "validation"}
        and all(
            isinstance(value, str) and len(value) == 64
            for value in corpus.get("source", {})
            .get("parquet_inventory_fingerprints", {})
            .values()
        ),
        "dedicated_train_and_validation_splits": corpus["splits"]["train"].get(
            "source_split"
        )
        == "train"
        and corpus["splits"]["validation"].get("source_split") == "validation",
        "gpt2_tokenizer_id": tokenizer.get("id") == "gpt2_openai",
        "gpt2_revision_exact": tokenizer.get("revision") == GPT2_REVISION,
        "gpt2_vocab_and_eot_exact": tokenizer.get("vocab_size") == 50_257
        and tokenizer.get("special_token_ids", {}).get("document_separator")
        == 50_256,
        "gpt2_asset_hashes_exact": tokenizer_assets == GPT2_ASSET_SHA256,
        "document_eos_packing": stream.get("packing", {}).get("contract")
        == "document_eos_concatenation_v1",
        "unique_train_stream_covers_budget": int(
            stream["splits"]["train"]["tokens"]
        )
        >= 299_892_736,
        "shared_immutable_token_stream": jiang["dataset_identity"]["fingerprint"]
        == completep["dataset_identity"]["fingerprint"]
        == stream["fingerprint"],
        "shared_paper_fixed_token_coordinates": jiang_fixed == completep_fixed
        and jiang_fixed["batch_examples"] == 128
        and jiang_fixed["batch_tokens"] == 262_144
        and jiang_fixed["optimizer_steps"] == 1_144
        and jiang_fixed["presented_tokens"] == 299_892_736,
        "shared_context_2048": int(jiang_config["architecture"]["context_length"])
        == int(completep_config["architecture"]["context_length"])
        == 2_048,
        "shared_paper_schedule": jiang["schedule"] == completep["schedule"]
        and jiang["schedule"]["family"] == "linear_warmup_decay"
        and jiang["schedule"]["warmup_fraction"] == 0.1
        and jiang["schedule"]["terminal_fraction"] == 0.0,
        "shared_adamw_core": all(
            plan["optimizer_contract"]["name"] == "adamw"
            and plan["optimizer_contract"]["beta1"] == 0.9
            and plan["optimizer_contract"]["beta2"] == 0.95
            and plan["optimizer_contract"]["epsilon"] == 1e-16
            and plan["optimizer_contract"]["weight_decay"] == 0.0
            for plan in (jiang, completep)
        ),
        "jiang_exact_rho32": all(
            float(row["rho_lm_over_d"]) == 32.0 for row in jiang["scales"]
        ),
        "jiang_reference_exact": (
            int(jiang_reference["depth"]),
            int(jiang_reference["hidden_width"]),
            int(jiang_reference["width"]),
        )
        == (4, 2560, 320),
        "jiang_all_per_group_rules_frozen": jiang["optimizer_contract"].get(
            "learning_rate_multipliers"
        )
        == JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
        "jiang_one_seed_dense_lr_scan": jiang["seeds"] == [11]
        and len(jiang["learning_rates"]) == 23
        and jiang["learning_rates"] == sorted(set(jiang["learning_rates"])),
        "completep_exact_reference_geometry": (
            int(completep_reference["depth"]), int(completep_reference["width"])
        )
        == (2, 256),
        "completep_exact_architecture_coordinates": completep[
            "architecture_contract"
        ]["position_encoding"]
        == "alibi"
        and completep["architecture_contract"]["attention_scale"] == "QK^T/N"
        and completep["architecture_contract"]["tied_embeddings"] is False
        and completep_config["architecture"]["activation"] == "relu_squared"
        and completep_config["architecture"]["initialization_std"] == 0.02,
        "completep_published_lr_is_frozen_anchor": 0.00390625
        in completep["learning_rates"]
        and completep["seeds"] == [11],
        "jiang_runtime_qualified": jiang_qualification.get("status") == "passed"
        and {row["plan_fingerprint"] for row in jiang_qualification["campaigns"]}
        == {jiang["fingerprint"]},
        "completep_anchor_runtime_qualified": completep_qualification.get("status")
        == "passed"
        and completep_qualification.get("plan_fingerprint")
        == completep["fingerprint"],
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("paper rerun preregistration failed: " + ", ".join(failed))

    payload = {
        "schema_version": 1,
        "status": "preregistered",
        "paper": {
            "url": PAPER_SOURCE,
            "arxiv_source_tar_sha256": PAPER_SOURCE_TAR_SHA256,
            "reported_training_coordinates": {
                "dataset": "SlimPajama",
                "accessible_snapshot": "DKYoon/SlimPajama-6B",
                "tokenizer": "GPT-2",
                "context_length": 2_048,
                "batch_examples": 128,
                "optimizer_steps": 1_144,
                "training_tokens": "300M (299,892,736 update-aligned tokens)",
                "weight_decay": 0.0,
                "schedule": "10% linear warmup then linear decay to zero",
            },
        },
        "claim_scope": {
            "jiang": (
                "Jiang-Chizat rho=32 fixed-token scaling scan in the CompleteP "
                "paper's SlimPajama corpus family, tokenizer, context, batch, "
                "schedule, and AdamW core coordinates; architecture-specific "
                "Jiang rules remain unchanged"
            ),
            "completep_anchor": (
                "literal CompleteP N=256,L=2 paper-coordinate calibration at the "
                "published eta=2^-8 at fixed seed 11"
            ),
            "not_claimed": (
                "This is not a byte-identical reconstruction of Cerebras' removed "
                "627B repository or its unpublished example order. Jiang raw loss "
                "also is not expected to reproduce CompleteP raw loss because Jiang "
                "retains tied embeddings, learned positions, GELU, QK/d_head, and "
                "its mean-field FFN architecture"
            ),
        },
        "corpus": {
            "manifest": str(args.corpus_manifest),
            "manifest_sha256": _sha256(args.corpus_manifest),
            "manifest_fingerprint": corpus["manifest_fingerprint"],
            "dataset_fingerprint": stream["fingerprint"],
            "source_revision": corpus["source"]["revision"],
            "parquet_revision": corpus["source"]["parquet_revision"],
            "parquet_inventory_fingerprints": corpus["source"][
                "parquet_inventory_fingerprints"
            ],
            "tokenizer_fingerprint": tokenizer["fingerprint"],
            "training_tokens": stream["splits"]["train"]["tokens"],
            "validation_tokens": stream["splits"]["validation"]["tokens"],
        },
        "jiang": {
            "config": str(args.jiang_config),
            "config_sha256": _sha256(args.jiang_config),
            "plan_fingerprint": jiang["fingerprint"],
            "learning_rates": jiang["learning_rates"],
            "seeds": jiang["seeds"],
            "scales": jiang["scales"],
            "hidden_test": "largest S8 rung withheld from the fit",
        },
        "completep_anchor": {
            "config": str(args.completep_anchor_config),
            "config_sha256": _sha256(args.completep_anchor_config),
            "plan_fingerprint": completep["fingerprint"],
            "scale": completep_reference,
            "learning_rate": 0.00390625,
            "seeds": [11],
        },
        "qualification": {
            "jiang_sha256": _sha256(args.jiang_qualification),
            "completep_anchor_sha256": _sha256(args.completep_qualification),
        },
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
