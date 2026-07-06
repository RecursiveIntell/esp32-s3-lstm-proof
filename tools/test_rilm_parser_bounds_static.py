#!/usr/bin/env python3
"""Static guard for bounded RILM parsing diagnostics.

If the coordinator resets after `RILM version=... tensors=...`, we need the
firmware parser to prove which tensor/offset it reached instead of trusting raw
payload lengths and silently wandering through mapped flash.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "main.cpp"


def function_body(src: str, name: str) -> str:
    m = re.search(rf"bool\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", src)
    assert m, f"missing function {name}"
    i = m.end()
    depth = 1
    j = i
    while j < len(src) and depth:
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
        j += 1
    assert depth == 0, f"unterminated function {name}"
    return src[i:j-1]


def test_rilm_parser_has_bounds_checks_and_tensor_logging() -> None:
    body = function_body(MAIN.read_text(), "load_model_partition")
    assert "RILM_TENSOR" in body
    assert "RILM_BOUNDS_ERROR" in body
    assert "rilm_offset" in body
    assert "payload_len" in body
    assert "model.len" in body
    assert "Serial.flush" in body


if __name__ == "__main__":
    test_rilm_parser_has_bounds_checks_and_tensor_logging()
    print("PASS rilm parser bounds static")
