from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


PACKAGES = [
    "gradio",
    "huggingface_hub",
    "matplotlib",
    "mediapipe",
    "numpy",
    "opencv-python-headless",
    "omegaconf",
    "pandas",
    "scikit-learn",
    "torch",
    "torchaudio",
]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "packages": {name: package_version(name) for name in PACKAGES},
    }
    payload_str = json.dumps(payload, sort_keys=True)
    payload["environment_hash"] = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
