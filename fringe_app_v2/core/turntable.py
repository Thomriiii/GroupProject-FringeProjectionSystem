"""HTTP client for the ESP8266 wireless turntable controller."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


class TurntableError(OSError):
    pass


class TurntableClient:
    """
    Thin HTTP client for the ESP8266 turntable.

    Endpoints:
      POST /rotate   {"degrees": float, "relative": bool=true}
      GET  /position -> {"degrees": float}
      GET  /status   -> {"ip", "rssi", "pos_deg"}
    """

    def __init__(self, ip: str, port: int = 80, timeout_s: float = 10.0) -> None:
        self.ip = ip
        self.port = port
        self.base = f"http://{ip}:{port}"
        self.timeout = timeout_s

    # ── low-level ────────────────────────────────────────────────────────────

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise TurntableError(f"GET {url} failed: {exc}") from exc

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise TurntableError(f"POST {url} failed: {exc}") from exc

    # ── public API ────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return {"ip", "rssi", "pos_deg"} from the controller."""
        return self._get("/status")

    def get_position(self) -> float:
        """Return the current dead-reckoned plate angle in degrees."""
        return float(self._get("/position")["degrees"])

    def rotate(self, degrees: float, relative: bool = True) -> dict[str, Any]:
        """
        Rotate the turntable.

        Args:
            degrees:  Angle to rotate (relative=True) or target angle (relative=False).
            relative: If True, move relative to current position.
        Returns:
            {"ok": True, "target_deg": float, "pos_deg": float}
        """
        resp = self._post("/rotate", {"degrees": float(degrees), "relative": bool(relative)})
        if not resp.get("ok"):
            raise TurntableError(f"rotate command rejected: {resp}")
        return resp

    def go_to(self, angle_deg: float, settle_s: float = 0.0) -> float:
        """Move to absolute angle. Returns final position."""
        resp = self.rotate(angle_deg, relative=False)
        if settle_s > 0:
            time.sleep(settle_s)
        return float(resp["pos_deg"])

    def home(self, settle_s: float = 0.0) -> float:
        """Return to 0°."""
        return self.go_to(0.0, settle_s=settle_s)

    def ping(self) -> bool:
        """Return True if the controller responds."""
        try:
            self.status()
            return True
        except TurntableError:
            return False


# ── discovery ─────────────────────────────────────────────────────────────────

def _nmap_port80_hosts(cidr: str, timeout_s: float = 20) -> list[str]:
    """
    Use nmap to find hosts with TCP port 80 open on the given CIDR range.
    Returns a list of IP address strings.
    """
    try:
        result = subprocess.run(
            ["nmap", "-p", "80", "--open", "-T4", "-n", "-oG", "-", cidr],
            capture_output=True, text=True, timeout=timeout_s,
        )
        ips: list[str] = []
        for line in result.stdout.splitlines():
            if "80/open" in line and line.startswith("Host:"):
                # e.g. "Host: 192.168.0.42 ()	Ports: 80/open/tcp//http///"
                parts = line.split()
                if len(parts) >= 2:
                    ips.append(parts[1])
        return ips
    except Exception:
        return []


def discover_turntable(
    subnet: str = "192.168.0",
    port: int = 80,
    timeout_s: float = 0.5,
) -> str | None:
    """
    Find the ESP8266 turntable on the local network.

    Strategy:
      1. Run nmap -p 80 --open on the /24 subnet (fast ~5 s).
      2. For each host found with port 80 open, check GET /status for "pos_deg".

    Returns the IP string of the first matching host, or None.
    """
    cidr = f"{subnet}.0/24"
    print(f"[turntable] nmap scanning {cidr} for port 80 ...")
    candidates = _nmap_port80_hosts(cidr, timeout_s=25)
    print(f"[turntable] nmap found {len(candidates)} host(s) with port 80 open: {candidates}")

    for ip in candidates:
        try:
            client = TurntableClient(ip, port=port, timeout_s=timeout_s)
            d = client.status()
            if "pos_deg" in d:
                print(f"[turntable] Found turntable at {ip}")
                return ip
        except Exception:
            pass

    print("[turntable] No turntable found")
    return None


def turntable_from_config(config: dict[str, Any]) -> TurntableClient | None:
    """
    Build a TurntableClient from application config, or None if disabled/unconfigured.
    """
    cfg = config.get("turntable") or {}
    if not cfg.get("enabled", False):
        return None
    ip = cfg.get("ip")
    if not ip:
        return None
    return TurntableClient(
        ip=str(ip),
        port=int(cfg.get("port", 80)),
        timeout_s=float(cfg.get("timeout_s", 10.0)),
    )


def get_local_subnet() -> str:
    """
    Guess the local /24 subnet prefix (e.g. '192.168.0') from the Pi's IP.
    Falls back to '192.168.0'.
    """
    try:
        result = subprocess.run(
            ["ip", "route"],
            capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.splitlines():
            # e.g. "192.168.0.0/24 dev eth0 ..."
            parts = line.split()
            if parts and "/" in parts[0] and not parts[0].startswith("169."):
                ip_cidr = parts[0]
                ip_part = ip_cidr.split("/")[0]
                segs = ip_part.split(".")
                if len(segs) == 4 and segs[0] not in ("127", "10"):
                    return ".".join(segs[:3])
    except Exception:
        pass
    return "192.168.0"
