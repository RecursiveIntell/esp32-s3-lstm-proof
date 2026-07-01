#!/usr/bin/env python3
"""Deterministic local sentinel policy for the ESP32/S3 physical-AI endpoint.

This is the product-shape that makes the current 3.69 tok/s compute useful:
rules decide, receipts prove, tiny LSTM/status text decorates, bigger model only
wakes for ambiguous or high-risk states.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

@dataclass
class SentinelReceipt:
    schema: str
    device_id: str
    observed_at_unix: int
    temp_c: float | None
    humidity_pct: float | None
    age_s: float
    decision: str
    route_reason: str
    confidence: float
    local_status: str
    ai_route: bool
    suggested_lstm_prompt: str
    suggested_lstm_chars: int
    blockers: list[str]


def fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def evaluate(temp_c: float | None, humidity: float | None, age_s: float, device_id: str) -> SentinelReceipt:
    blockers: list[str] = []
    ai_route = False
    decision = "normal"
    reason = "local_confident"
    confidence = 0.92

    if temp_c is None or humidity is None:
        decision = "sensor_missing"
        reason = "sensor_missing"
        confidence = 0.99
        ai_route = True
        blockers.append("missing_sensor_value")
        status = "sensor missing. no claim."
        prompt = "missing sensor. action is "
    elif age_s > 120:
        decision = "stale"
        reason = "stale_reading"
        confidence = 0.98
        ai_route = True
        blockers.append("stale_sensor_reading")
        status = "sensor stale. wait."
        prompt = "stale data. action is "
    else:
        temp_f = fahrenheit(temp_c)
        hot = temp_f >= 82.0
        cold = temp_f <= 60.0
        humid = humidity >= 65.0
        dry = humidity <= 25.0
        if hot and humid:
            decision = "escalate_heat_humidity"
            reason = "temperature_and_humidity_out_of_range"
            confidence = 0.90
            ai_route = True
            status = f"hot humid {temp_f:.0f}f {humidity:.0f}%."
            prompt = "high heat and humidity. action is "
        elif hot:
            decision = "warm"
            reason = "temperature_out_of_range"
            confidence = 0.88
            ai_route = False
            status = f"warm room {temp_f:.0f}f."
            prompt = "hot room. action is "
        elif cold:
            decision = "cold"
            reason = "temperature_out_of_range"
            confidence = 0.88
            ai_route = False
            status = f"cold room {temp_f:.0f}f."
            prompt = "cold room. action is "
        elif humid:
            decision = "humid"
            reason = "humidity_out_of_range"
            confidence = 0.88
            ai_route = False
            status = f"humid room {humidity:.0f}%."
            prompt = "humid room. action is "
        elif dry:
            decision = "dry"
            reason = "humidity_out_of_range"
            confidence = 0.86
            ai_route = False
            status = f"dry room {humidity:.0f}%."
            prompt = "dry room. action is "
        else:
            status = f"room normal {temp_f:.0f}f {humidity:.0f}%."
            prompt = "normal room. action is "

    return SentinelReceipt(
        schema="ri_esp32_local_sentinel_v1",
        device_id=device_id,
        observed_at_unix=int(time.time()),
        temp_c=temp_c,
        humidity_pct=humidity,
        age_s=age_s,
        decision=decision,
        route_reason=reason,
        confidence=confidence,
        local_status=status,
        ai_route=ai_route,
        suggested_lstm_prompt=prompt,
        suggested_lstm_chars=24 if not ai_route else 16,
        blockers=blockers,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp-c", type=float, default=None)
    ap.add_argument("--humidity", type=float, default=None)
    ap.add_argument("--age-s", type=float, default=0.0)
    ap.add_argument("--device-id", default="esp32s3-lstm-node")
    args = ap.parse_args()
    receipt = evaluate(args.temp_c, args.humidity, args.age_s, args.device_id)
    print(json.dumps(asdict(receipt), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
