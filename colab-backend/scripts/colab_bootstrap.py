"""
Single-cell Google Colab L4 bootstrap script for Lyrion AI Music Studio.

Usage inside a Google Colab notebook cell (Runtime -> L4 GPU):
    !git clone <your-repo-url> /content/lyrion || true
    %cd /content/lyrion/colab-backend
    !python scripts/colab_bootstrap.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: str, check: bool = True) -> None:
    print(f"[Bootstrap] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=check)


def mount_google_drive() -> Path:
    drive_mount = Path("/content/drive")
    if "google.colab" in sys.modules or Path("/content").exists():
        if not (drive_mount / "MyDrive").exists():
            try:
                from google.colab import drive  # type: ignore

                drive.mount(str(drive_mount), force_remount=False)
            except ImportError:
                print("[Bootstrap] google.colab not detected; using local ./data directory.")
                return Path.cwd() / "data" / "AIMusicStudio"
        target = drive_mount / "MyDrive" / "AIMusicStudio"
        target.mkdir(parents=True, exist_ok=True)
        return target
    target = Path.cwd() / "data" / "AIMusicStudio"
    target.mkdir(parents=True, exist_ok=True)
    return target


def install_dependencies() -> None:
    print("[Bootstrap] Installing system audio tools (ffmpeg, fluidsynth) and cloudflared...")
    run("apt-get update -qq && apt-get install -y -qq ffmpeg fluidsynth curl > /dev/null", check=False)

    cloudflared_bin = Path("/usr/local/bin/cloudflared")
    if not cloudflared_bin.exists():
        run(
            "curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
        )

    print("[Bootstrap] Installing Python backend packages...")
    run(f"{sys.executable} -m pip install -q -r requirements.txt")

    # Install ACE-Step 1.5 if missing
    check_ace = subprocess.run(
        [sys.executable, "-c", "import acestep"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check_ace.returncode != 0:
        print("[Bootstrap] Installing ACE-Step from GitHub...")
        run(
            f"{sys.executable} -m pip install -q git+https://github.com/ace-step/ACE-Step.git",
            check=False,
        )
        verify_ace = subprocess.run(
            [sys.executable, "-c", "import acestep"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if verify_ace.returncode != 0:
            print(
                "[Bootstrap] WARNING: 'acestep' failed to import after pip install! "
                "Real GPU generation/repaint will fail unless ACE-Step is installed manually "
                "(or set MOCK_GPU=true).",
                flush=True,
            )


def start_cloudflared(port: int = 8000) -> tuple[subprocess.Popen[str], str]:
    log_path = Path("/tmp/cloudflared_lyrion.log")
    if log_path.exists():
        log_path.unlink()

    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    tunnel_url = ""
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    for _ in range(45):
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            match = pattern.search(content)
            if match:
                tunnel_url = match.group(0)
                break
        time.sleep(1.0)

    return proc, tunnel_url


def main() -> None:
    drive_root = mount_google_drive()
    os.environ.setdefault("MOCK_GPU", "false")
    os.environ["DRIVE_ROOT"] = str(drive_root)

    if Path("/content").exists():
        install_dependencies()

    port = int(os.environ.get("PORT", "8000"))
    tunnel_proc, tunnel_url = start_cloudflared(port=port)

    print("\n" + "=" * 72)
    print("  LYRION AI MUSIC STUDIO — COLAB L4 BACKEND READY")
    print("=" * 72)
    print(f"  Drive Storage : {drive_root}")
    print(f"  GPU Mode      : {'MOCK_GPU' if os.environ.get('MOCK_GPU') == 'true' else 'CUDA bfloat16'}")
    print(f"  Public URL    : {tunnel_url or '(Wait 5s and check /tmp/cloudflared_lyrion.log)'}")
    print("  Paste the Public URL into the Lyrion Frontend top bar!")
    print("=" * 72 + "\n", flush=True)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ],
            check=True,
        )
    finally:
        tunnel_proc.terminate()


if __name__ == "__main__":
    main()
