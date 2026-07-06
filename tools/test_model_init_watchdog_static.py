#!/usr/bin/env python3
"""Static regression tests for H512 model-init watchdog friendliness.

The coordinator has repeatedly watchdog-reset while cloning/converting the H512
model. Heavy init loops must periodically pump/yield so firmware reaches
MODEL_READY reliably before any performance receipt is attempted.
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


def test_model_init_has_watchdog_pump_helper() -> None:
    src = MAIN.read_text()
    assert "model_init_pump_watchdog" in src


def test_clone_payload_loop_pumps_watchdog() -> None:
    body = function_body(MAIN.read_text(), "clone_payloads_to_psram")
    assert "model_init_pump_watchdog" in body


def test_clone_payload_loop_uses_chunked_copy() -> None:
    body = function_body(MAIN.read_text(), "clone_payloads_to_psram")
    assert "memcpy(copy, t->payload, t->payload_len)" not in body
    assert "model_init_copy_payload" in body


def test_int4_conversion_loop_pumps_watchdog() -> None:
    body = function_body(MAIN.read_text(), "convert_wih_to_int4")
    assert "model_init_pump_watchdog" in body


def test_model_init_brackets_core_watchdogs() -> None:
    src = MAIN.read_text()
    assert "model_init_suspend_watchdogs" in src
    assert "disableLoopWDT" in src
    assert "disableCore0WDT" in src
    assert "disableCore1WDT" in src
    assert "model_init_resume_watchdogs" in src
    assert "enableLoopWDT" in src
    assert "enableCore0WDT" in src
    assert "enableCore1WDT" in src


if __name__ == "__main__":
    test_model_init_has_watchdog_pump_helper()
    test_clone_payload_loop_pumps_watchdog()
    test_clone_payload_loop_uses_chunked_copy()
    test_int4_conversion_loop_pumps_watchdog()
    test_model_init_brackets_core_watchdogs()
    print("PASS model init watchdog static")
