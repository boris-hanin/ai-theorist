import json
import io
import math
import sys

import jsonio

from ai_theorist.autoscaler.cli import _print
from ai_theorist.autoscaler.study import json_safe


def test_jsonio_converts_nonfinite_values_to_null(tmp_path):
    path = tmp_path / "artifact.json"
    jsonio.dump({"finite": 1.5, "bad": [math.inf, -math.inf, math.nan]}, path)
    assert json.loads(path.read_text()) == {
        "finite": 1.5, "bad": [None, None, None]}


def test_shared_json_safety_converts_nested_nonfinite_values_to_null():
    payload = {"finite": 1.5, "bad": [math.inf, -math.inf, math.nan]}
    assert json_safe(payload) == {
        "finite": 1.5,
        "bad": [None, None, None],
    }


def test_cli_print_uses_strict_json_safety(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    _print({"diagnostic": math.nan})
    assert json.loads(output.getvalue()) == {"diagnostic": None}
