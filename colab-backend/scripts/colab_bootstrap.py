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
import secrets
import subprocess
import sys
import time
from pathlib import Path

ACESTEP_REPO_DIR = Path("/content/ACE-Step-1.5")


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

    # Install ACE-Step 1.5 without letting its pyproject cu128/uv-path pins break Colab PyTorch
    probe_env = os.environ.copy()
    if ACESTEP_REPO_DIR.exists():
        existing_pp = probe_env.get("PYTHONPATH", "")
        probe_env["PYTHONPATH"] = f"{ACESTEP_REPO_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(ACESTEP_REPO_DIR)

    check_ace = subprocess.run(
        [sys.executable, "-c", "import acestep"],
        env=probe_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check_ace.returncode != 0:
        print("[Bootstrap] Cloning & installing ACE-Step 1.5 (preserving Colab PyTorch/CUDA)...")
        if not ACESTEP_REPO_DIR.exists():
            run(
                f"git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {ACESTEP_REPO_DIR}",
                check=False,
            )
        run(
            f"{sys.executable} -m pip install -q "
            "transformers diffusers accelerate einops loguru diskcache "
            "vector-quantize-pytorch numba huggingface_hub safetensors omegaconf",
            check=False,
        )
        nano_vllm_dir = ACESTEP_REPO_DIR / "acestep" / "third_parts" / "nano-vllm"
        if nano_vllm_dir.exists():
            run(
                f"{sys.executable} -m pip install -q --no-deps --ignore-requires-python -e {nano_vllm_dir}",
                check=False,
            )
        if ACESTEP_REPO_DIR.exists():
            run(
                f"{sys.executable} -m pip install -q --no-deps --ignore-requires-python -e {ACESTEP_REPO_DIR}",
                check=False,
            )
            (ACESTEP_REPO_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
            existing_pp = probe_env.get("PYTHONPATH", "")
            if str(ACESTEP_REPO_DIR) not in existing_pp:
                probe_env["PYTHONPATH"] = (
                    f"{ACESTEP_REPO_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(ACESTEP_REPO_DIR)
                )

        verify_ace = subprocess.run(
            [sys.executable, "-c", "import acestep"],
            env=probe_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if verify_ace.returncode != 0:
            print(
                "[Bootstrap] WARNING: 'acestep' failed to import after install! "
                "Real GPU generation/repaint will fail unless ACE-Step 1.5 is installed manually "
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

    api_token = os.environ.get("LYRION_API_TOKEN", "").strip()
    if not api_token:
        api_token = secrets.token_urlsafe(24)
        os.environ["LYRION_API_TOKEN"] = api_token

    if Path("/content").exists():
        install_dependencies()

    if ACESTEP_REPO_DIR.exists():
        os.environ.setdefault("ACESTEP_PROJECT_ROOT", str(ACESTEP_REPO_DIR))
        existing_pp = os.environ.get("PYTHONPATH", "")
        if str(ACESTEP_REPO_DIR) not in existing_pp:
            os.environ["PYTHONPATH"] = (
                f"{ACESTEP_REPO_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(ACESTEP_REPO_DIR)
            )

    port = int(os.environ.get("PORT", "8000"))
    tunnel_proc, tunnel_url = start_cloudflared(port=port)

    print("\n" + "=" * 72)
    print("  LYRION AI MUSIC STUDIO — COLAB L4 BACKEND READY")
    print("=" * 72)
    print(f"  Drive Storage : {drive_root}")
    print(f"  GPU Mode      : {'MOCK_GPU' if os.environ.get('MOCK_GPU') == 'true' else 'CUDA bfloat16'}")
    print(f"  Public URL    : {tunnel_url or '(Wait 5s and check /tmp/cloudflared_lyrion.log)'}")
    print(f"  API Token     : {api_token}")
    print("  Paste both Public URL and API Token into the Lyrion Frontend top bar!")
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
