import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def reject_nonstandard_constant(value):
    raise ValueError("non-standard JSON constant: %s" % value)


def test_all_json_artifacts_are_standard_json():
    failures = []
    for path in ROOT.rglob("*.json"):
        # Runtime and build directories are mutable while campaigns and web
        # builds are active. This test guards repository inputs and fixtures,
        # not partially copied result/cache files.
        relative = path.relative_to(ROOT)
        if {"node_modules", ".next", ".wrangler", "dist", "runs", "work"} & set(relative.parts):
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_constant)
        except Exception as exc:  # collect every invalid source artifact in one report
            failures.append("%s: %s" % (relative, exc))
    assert not failures, "\n".join(failures)


def test_round_011_has_a_scoreboard():
    report = (ROOT / "rounds/011-graph-transformer/results.md").read_text()
    assert "Overall verdict: FAILED" in report
    for prediction in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        assert prediction in report
