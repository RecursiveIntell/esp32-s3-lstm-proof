#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def load_tool(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_gen_mode_is_available_in_update_tools():
    expected = {
        "coord": "cluster_coord_ap_local_gen",
        "worker1": "cluster_worker1_ap_local_gen",
        "worker2": "cluster_worker2_ap_local_gen",
    }
    for tool_name in ("ota_cluster_wifi", "flash_cluster_wifi"):
        tool = load_tool(tool_name)
        assert tool.ROLE_ENVS["local_gen"] == expected

    relay = load_tool("relay_worker_update")
    assert relay.ROLE_ENVS["local_gen"] == {
        "worker1": "cluster_worker1_ap_local_gen",
        "worker2": "cluster_worker2_ap_local_gen",
    }


def test_h512_local_gen_mode_is_available_in_update_tools():
    expected = {
        "coord": "cluster_coord_ap_local_gen_h512",
        "worker1": "cluster_worker1_ap_local_gen_h512",
        "worker2": "cluster_worker2_ap_local_gen_h512",
    }
    for tool_name in ("ota_cluster_wifi", "flash_cluster_wifi"):
        tool = load_tool(tool_name)
        assert tool.ROLE_ENVS["local_gen_h512"] == expected

    relay = load_tool("relay_worker_update")
    assert relay.ROLE_ENVS["local_gen_h512"] == {
        "worker1": "cluster_worker1_ap_local_gen_h512",
        "worker2": "cluster_worker2_ap_local_gen_h512",
    }


def main():
    test_local_gen_mode_is_available_in_update_tools()
    test_h512_local_gen_mode_is_available_in_update_tools()
    print("PASS cluster mode tools")


if __name__ == "__main__":
    main()
