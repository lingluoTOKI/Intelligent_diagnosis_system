"""Persisted network settings and LAN discovery without an external connection."""
from pathlib import Path
import ipaddress
import json
import socket
import psutil
try:
    from configs.runtime_config import PROJECT_ROOT
except ModuleNotFoundError:
    from runtime_config import PROJECT_ROOT

SYSTEM_CONFIG_PATH = PROJECT_ROOT / "configs" / "system_config.json"
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(value) for value in
                         ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))


def get_ipv4_interfaces(addresses=None, stats=None):
    addresses = psutil.net_if_addrs() if addresses is None else addresses
    stats = psutil.net_if_stats() if stats is None else stats
    candidates = []
    for name, entries in addresses.items():
        lower = name.casefold()
        if name not in stats or not stats[name].isup:
            continue
        if lower == "meta" or any(token in lower for token in
            ("vmware", "vethernet", "hyper-v", "virtual", "wintun", "tap-", "vpn", "loopback", "docker", "wireguard")):
            continue
        for entry in entries:
            if entry.family != socket.AF_INET:
                continue
            try:
                address = ipaddress.IPv4Address(entry.address)
            except ipaddress.AddressValueError:
                continue
            if not any(address in network for network in PRIVATE_NETWORKS):
                continue
            score = 2 if any(token in lower for token in ("wlan", "wi-fi", "wifi", "无线")) else 1
            candidates.append({"interface": name, "ip": str(address), "score": score})
    return sorted(candidates, key=lambda item: (-item["score"], item["interface"], item["ip"]))


def read_system_config(path=None):
    source = Path(path or SYSTEM_CONFIG_PATH)
    if path is None and not source.is_file():
        source = PROJECT_ROOT / "configs/system_config.example.json"
    return json.loads(source.read_text(encoding="utf-8-sig"))


def get_local_ip(preferred_interface=None):
    config = read_system_config()
    network = config.get("network", {})
    if not config.get("startup", {}).get("auto_detect_ip", True):
        return str(ipaddress.IPv4Address(network.get("pc_ip", "127.0.0.1")))
    candidates = get_ipv4_interfaces()
    preferred = preferred_interface or network.get("preferred_interface")
    for candidate in candidates:
        if candidate["interface"] == preferred:
            return candidate["ip"]
    return candidates[0]["ip"] if candidates else "127.0.0.1"


def save_network_config(pc_ip, auto_detect, preferred_interface="", path=None, board_ip=None):
    ip = str(ipaddress.IPv4Address(pc_ip))
    board = str(ipaddress.IPv4Address(board_ip)) if board_ip is not None else None
    target = Path(path or SYSTEM_CONFIG_PATH)
    data = read_system_config(target)
    data.setdefault("network", {}).update(pc_ip=ip, preferred_interface=preferred_interface)
    if board is not None:
        data["network"]["board_ip"] = board
    data.setdefault("startup", {})["auto_detect_ip"] = bool(auto_detect)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
