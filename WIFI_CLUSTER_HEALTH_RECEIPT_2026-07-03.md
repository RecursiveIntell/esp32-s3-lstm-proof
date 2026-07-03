# ESP32-S3 WiFi cluster health receipt — 2026-07-03

Status: PASS live WiFi health check from host joined to `RI-ESP-CLUSTER`.

Host network sequence:

```text
nmcli connection up RI-ESP-CLUSTER
host ip: 192.168.4.4/24
curl http://192.168.4.1:8080/health
curl http://192.168.4.2:8080/health
curl http://192.168.4.3:8080/health
nmcli connection up PurpleMama_5G
host restored ip: 192.168.50.181/24
```

Live health responses:

```text
192.168.4.1 /health:
ok=1 board_id=0 role=coord mode=lstm_shard ip=192.168.4.1

192.168.4.2 /health:
ok=1 board_id=1 role=worker mode=lstm_shard ip=192.168.4.2

192.168.4.3 /health:
ok=1 board_id=2 role=worker mode=lstm_shard ip=192.168.4.3
```

Interpretation:

- Coordinator AP is live at `192.168.4.1`.
- Worker1 is live at `192.168.4.2`.
- Worker2 is live at `192.168.4.3`.
- All three HTTP update/health servers respond on port `8080`.
- The host can temporarily join the ESP32 cluster AP, query all nodes, and restore normal WiFi afterward.

Claim boundary:

- Safe: all three boards are currently reachable over the cluster WiFi AP and report `mode=lstm_shard`.
- Not claimed here: aggregate benchmark performance, fresh distributed generation pass, or updated firmware deployment.
