#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_platformio_layer_shard_envs():
    platformio = read("platformio.ini")
    expected = {
        "cluster_coord_ap_layer_shard_h512": [
            "CLUSTER_ROLE_COORD=1",
            "CLUSTER_BOARD_ID=0",
            "CLUSTER_WIFI_AP_MODE=1",
        ],
        "cluster_worker1_ap_layer_shard_h512": [
            "CLUSTER_ROLE_WORKER=1",
            "CLUSTER_BOARD_ID=1",
        ],
        "cluster_worker2_ap_layer_shard_h512": [
            "CLUSTER_ROLE_WORKER=1",
            "CLUSTER_BOARD_ID=2",
        ],
    }
    for env, role_flags in expected.items():
        marker = f"[env:{env}]"
        assert marker in platformio
        section = platformio.split(marker, 1)[1].split("\n[env:", 1)[0]
        assert "extends = env:esp32s3_tinystories_h512" in section
        for flag in role_flags:
            assert f"-D {flag}" in section
        for flag in (
            "CLUSTER_WIFI_LAYER_SHARD=1",
            "CLUSTER_WIFI_DEMO=1",
            "CLUSTER_ENABLE_OTA=1",
            "CLUSTER_ENABLE_HTTP_UPDATE=1",
        ):
            assert f"-D {flag}" in section


def test_main_layer_shard_scaffold_markers():
    main_cpp = read("src/main.cpp")
    for marker in (
        "#ifndef CLUSTER_WIFI_LAYER_SHARD",
        '"layer_shard"',
        "CLUSTER_MSG_LSTM_STATE_FORWARD_REQUEST",
        "CLUSTER_MSG_LSTM_STATE_FORWARD_RESULT",
        "CLUSTER_LAYER_SHARD_STATE_REQUEST",
        "CLUSTER_LAYER_SHARD_STATE_RESULT",
    ):
        assert marker in main_cpp


def main():
    test_platformio_layer_shard_envs()
    test_main_layer_shard_scaffold_markers()
    print("PASS layer shard static")


if __name__ == "__main__":
    main()
