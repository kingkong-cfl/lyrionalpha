"""
Single-cell Google Colab L4 bootstrap script for Lyrion AI Music Studio.

Usage inside Google Colab notebook cells (Runtime -> L4 GPU):
    # Cell 1: Mount Drive inside the IPython kernel first
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

    # Cell 2: Clone repo & run bootstrap via %run (or !python if Drive is already mounted)
    !git clone https://github.com/kingkong-cfl/lyrionalpha.git /content/lyrion || true
    %cd /content/lyrion/colab-backend
    %run scripts/colab_bootstrap.py
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

ACESTEP_SMOKE_CMD = (
    "import re, transformers, diffusers, huggingface_hub; "
    "ver = lambda s: (tuple(int(x) for x in re.findall(r'\\d+', s)[:3]) + (0, 0, 0))[:3]; "
    "assert (4, 51, 0) <= ver(transformers.__version__) < (4, 58, 0), f'incompatible transformers=={transformers.__version__}'; "
    "assert ver(diffusers.__version__) >= (0, 35, 0), f'incompatible diffusers=={diffusers.__version__}'; "
    "assert (0, 34, 0) <= ver(huggingface_hub.__version__) < (1, 0, 0), f'incompatible huggingface_hub=={huggingface_hub.__version__}'; "
    "from diffusers.models import AutoencoderOobleck; "
    "from transformers import AutoModel; "
    "from acestep.handler import AceStepHandler; "
    "from acestep.inference import generate_music"
)


def parse_version_tuple(version_str: str) -> tuple[int, int, int]:
    parts = [int(x) for x in re.findall(r"\d+", version_str)[:3]]
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def check_acestep_versions(transformers_v: str, diffusers_v: str, hf_v: str) -> bool:
    t_ver = parse_version_tuple(transformers_v)
    d_ver = parse_version_tuple(diffusers_v)
    h_ver = parse_version_tuple(hf_v)
    return (4, 51, 0) <= t_ver < (4, 58, 0) and d_ver >= (0, 35, 0) and (0, 34, 0) <= h_ver < (1, 0, 0)


def stream_command(
    cmd: str | list[str],
    *,
    check: bool = True,
    shell: bool = False,
    log_path: Path | None = None,
) -> int:
    """Run a subprocess with unbuffered merged stdout/stderr streamed line-by-line into IPython."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    log_fh = log_path.open("a", encoding="utf-8") if log_path is not None else None
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                print(line, end="", flush=True)
                if log_fh is not None:
                    log_fh.write(line)
                    log_fh.flush()
    finally:
        if log_fh is not None:
            log_fh.close()
    ret = proc.wait()
    if check and ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return ret


def run(cmd: str, check: bool = True) -> None:
    print(f"[Bootstrap] $ {cmd}", flush=True)
    stream_command(cmd, check=check, shell=True)


def mount_google_drive() -> Path:
    drive_mount = Path("/content/drive")
    if "google.colab" in sys.modules or Path("/content").exists():
        if not (drive_mount / "MyDrive").exists():
            try:
                from google.colab import drive  # type: ignore

                drive.mount(str(drive_mount), force_remount=False)
            except (ImportError, AttributeError, Exception) as exc:
                print(
                    "[Bootstrap] WARNING: Could not auto-mount Google Drive "
                    f"({type(exc).__name__}: {exc}). Mount Drive in a notebook cell first via "
                    "`from google.colab import drive; drive.mount('/content/drive')` or run via "
                    "`%run scripts/colab_bootstrap.py`. Falling back to local ./data/AIMusicStudio.",
                    flush=True,
                )
                fallback = Path.cwd() / "data" / "AIMusicStudio"
                fallback.mkdir(parents=True, exist_ok=True)
                return fallback
        if (drive_mount / "MyDrive").exists():
            target = drive_mount / "MyDrive" / "AIMusicStudio"
            target.mkdir(parents=True, exist_ok=True)
            return target
        print(
            "[Bootstrap] WARNING: /content/drive/MyDrive still unavailable; "
            "falling back to local ./data/AIMusicStudio.",
            flush=True,
        )
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
        [sys.executable, "-c", ACESTEP_SMOKE_CMD],
        env=probe_env,
        capture_output=True,
        text=True,
    )
    if check_ace.returncode != 0:
        print("[Bootstrap] Cloning & installing ACE-Step 1.5 (preserving Colab PyTorch/CUDA)...")
        if not ACESTEP_REPO_DIR.exists():
            run(
                f"git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {ACESTEP_REPO_DIR}",
                check=False,
            )
        run(
            f'{sys.executable} -m pip install -q '
            '"transformers>=4.51.0,<4.58.0" '
            '"diffusers>=0.35.0" '
            '"huggingface-hub>=0.34.0,<1.0" '
            '"vector-quantize-pytorch>=1.27.15" '
            'accelerate einops loguru diskcache numba safetensors omegaconf',
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
            [sys.executable, "-c", ACESTEP_SMOKE_CMD],
            env=probe_env,
            capture_output=True,
            text=True,
        )
        if verify_ace.returncode != 0:
            err_detail = (verify_ace.stderr or verify_ace.stdout or "").strip()
            print(
                "[Bootstrap] WARNING: ACE-Step 1.5 smoke import failed after install! "
                "Real GPU generation/repaint may fail unless resolved (or set MOCK_GPU=true).\n"
                f"[Bootstrap] Import stderr:\n{err_detail}",
                flush=True,
            )


def configure_model_caches(drive_root: Path) -> None:
    """Optionally cache HuggingFace / Torch / ACE-Step checkpoints on Drive to skip cold downloads."""
    enabled = os.environ.get("LYRION_CACHE_MODELS_TO_DRIVE", "false").lower() in ("1", "true", "yes")
    if not enabled or not Path("/content/drive/MyDrive").exists():
        return
    cache_root = drive_root / "_model_cache"
    hf_cache = cache_root / "huggingface"
    torch_cache = cache_root / "torch"
    ckpt_cache = cache_root / "checkpoints"
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    ckpt_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TORCH_HOME", str(torch_cache))
    if ACESTEP_REPO_DIR.exists():
        ckpt_link = ACESTEP_REPO_DIR / "checkpoints"
        if ckpt_link.is_dir() and not ckpt_link.is_symlink() and not any(ckpt_link.iterdir()):
            ckpt_link.rmdir()
        if not ckpt_link.exists() and not ckpt_link.is_symlink():
            ckpt_link.symlink_to(ckpt_cache)
    print(f"[Bootstrap] Persistent Drive model cache active at {cache_root}", flush=True)


def start_cloudflared(port: int = 8000) -> tuple[subprocess.Popen[str], str]:
    log_path = Path("/tmp/cloudflared_lyrion.log")
    if log_path.exists():
        log_path.unlink()

    log_file = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_file.close()

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
        configure_model_caches(drive_root)

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
        log_file_path = Path("/tmp/lyrion_backend.log")
        if log_file_path.exists():
            log_file_path.unlink()
        stream_command(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--log-level",
                "info",
            ],
            check=True,
            shell=False,
            log_path=log_file_path,
        )
    finally:
        tunnel_proc.terminate()


if __name__ == "__main__":
    main()
