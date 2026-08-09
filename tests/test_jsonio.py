import json
import math

import jsonio


def test_jsonio_converts_nonfinite_values_to_null(tmp_path):
    path = tmp_path / "artifact.json"
    jsonio.dump({"finite": 1.5, "bad": [math.inf, -math.inf, math.nan]}, path)
    assert json.loads(path.read_text()) == {
        "finite": 1.5, "bad": [None, None, None]}
